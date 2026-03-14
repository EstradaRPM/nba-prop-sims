#!/usr/bin/env python3
"""
build_backtest.py — NBA prop simulator historical backtesting pipeline

Collects three data sources for a date range, runs Monte Carlo simulation
on each prop, and outputs a structured backtest_results.json for calibration
analysis.

Data sources:
  1. The Odds API (historical events + player prop odds)
  2. NBA Stats API (actual game results via nba_api)
  3. ETR projection CSVs (user-downloaded, one file per date)

Usage:
  # Estimate credit cost before spending (no API calls made)
  python scripts/build_backtest.py --dry-run --start 2026-03-01 --end 2026-03-07

  # Full run — 7 days
  python scripts/build_backtest.py \\
    --api-key YOUR_ODDS_API_KEY \\
    --start 2026-03-01 \\
    --end 2026-03-07 \\
    --etr-dir ~/Downloads/etr_csvs \\
    --output backtest_results.json

  # With archived CV snapshots for historical accuracy
  python scripts/build_backtest.py \\
    --api-key YOUR_ODDS_API_KEY \\
    --start 2026-03-01 \\
    --end 2026-03-07 \\
    --etr-dir ~/Downloads/etr_csvs \\
    --cv-snapshots-dir cv_snapshots \\
    --output backtest_results.json

ETR CSV format (case-insensitive column matching):
  Required: player name column ("Name", "Player", "PLAYER")
  Stat columns (any case): "Pts"/"PTS", "Reb"/"REB", "Ast"/"AST",
                            "Stl"/"STL", "Blk"/"BLK", "3PM"/"Threes"/"FG3M"
  Optional: "Team" (for disambiguation)

  Files must be named YYYY-MM-DD.csv (e.g. 2026-03-01.csv) or
  YYYY-MM-DD_*.csv (e.g. 2026-03-01_etr_nba.csv).

Credit cost reference (The Odds API):
  10 credits per region × per market × per event
  6 markets × 1 region = 60 credits/game
  ~7 games/day × 60 = ~420 credits/day
  7-day run ≈ 2,940 credits  (~15% of 20k monthly budget)
  14-day run ≈ 5,880 credits (~29% of 20k monthly budget)
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    from nba_api.stats.endpoints import LeagueGameLog
except ImportError:
    print("ERROR: 'nba_api' not installed. Run: pip install nba_api", file=sys.stderr)
    sys.exit(1)


# ── Config ───────────────────────────────────────────────────────────────────

SEASON = "2025-26"

# The Odds API base URL
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Player prop market keys → internal stat keys
MARKET_TO_STAT: dict[str, str] = {
    "player_points":   "pts",
    "player_rebounds": "reb",
    "player_assists":  "ast",
    "player_threes":   "threes",
    "player_blocks":   "blk",
    "player_steals":   "stl",
}
MARKETS = list(MARKET_TO_STAT.keys())

# Simulation parameters
N_SIMS = 50_000

# Fallback CV% by stat (mirrors STD_RATIOS × 100 in index.html)
FALLBACK_CV: dict[str, float] = {
    "pts":    30.0,
    "reb":    42.0,
    "ast":    40.0,
    "stl":    65.0,
    "blk":    75.0,
    "threes": 52.0,
}

# Stats using log-normal distribution (mirrors JS LOG_NORMAL_STATS + KDE_STATS routing)
LOG_NORMAL_STATS = {"pts", "reb", "ast", "pra", "pr", "pa", "ra"}

# Stats using NegBin / Poisson
POISSON_STATS = {"stl", "blk", "threes"}

# Edge tier thresholds (vig-free edge %)
EDGE_TIERS = [
    ("STRONG",   5.0),
    ("SOLID",    3.0),
    ("MARGINAL", 1.5),
    ("NO EDGE",  0.0),
]

# Preferred bookmaker order for line selection (first available wins)
BOOKMAKER_PRIORITY = [
    "draftkings", "fanduel", "betmgm", "caesars", "pointsbet",
    "bet365", "betonlineag", "bovada",
]

# Rate limiting
ODDS_API_DELAY   = 0.3   # seconds between Odds API calls
NBA_API_DELAY    = 0.65  # seconds between NBA Stats API calls (matches compute_cv.py)

# Pre-game snapshot time: 6:30 PM ET
# EST = UTC-5 → 23:30 UTC | EDT = UTC-4 → 22:30 UTC
# Use 23:00 UTC as a stable conservative pre-game anchor (works for both)
SNAPSHOT_HOUR_UTC = 23   # override with --snapshot-hour if needed


# ── Utilities ────────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """NFD unicode normalization + strip — matches JS and compute_cv.py."""
    return unicodedata.normalize("NFD", name.strip())


def date_range(start: date, end: date):
    """Yield each date from start to end inclusive."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def american_to_prob(odds: int) -> float:
    """Convert American odds integer to implied probability (vig included)."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def vig_free_prob(over_odds: int | None, under_odds: int | None) -> tuple[float | None, float | None]:
    """
    Normalize both sides to remove vig.
    Returns (over_prob, under_prob) or (None, None) if both sides missing.
    Falls back to raw implied if only one side present.
    """
    if over_odds is None and under_odds is None:
        return None, None
    if over_odds is None:
        p = american_to_prob(under_odds)
        return 1.0 - p, p
    if under_odds is None:
        p = american_to_prob(over_odds)
        return p, 1.0 - p
    p_over  = american_to_prob(over_odds)
    p_under = american_to_prob(under_odds)
    total   = p_over + p_under
    return p_over / total, p_under / total


def get_edge_label(edge_pct: float) -> str:
    for label, threshold in EDGE_TIERS:
        if edge_pct >= threshold:
            return label
    return "NO EDGE"


def snapshot_utc(game_date: date, hour_utc: int = SNAPSHOT_HOUR_UTC) -> str:
    """ISO 8601 UTC timestamp for the pre-game snapshot on a given date."""
    dt = datetime(game_date.year, game_date.month, game_date.day,
                  hour_utc, 0, 0, tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Monte Carlo Simulation (Python port of index.html engine) ─────────────────

def simulate_log_normal(mean: float, cv_pct: float, n: int) -> np.ndarray:
    """
    Log-normal sampler parameterized from mean and CV%.
    Mirrors simulateStatLogNormal() in index.html.
    """
    sigma_n = cv_pct / 100.0
    mu_l    = math.log(mean) - 0.5 * math.log(1.0 + sigma_n ** 2)
    sigma_l = math.sqrt(math.log(1.0 + sigma_n ** 2))
    return np.random.lognormal(mu_l, sigma_l, n)


def simulate_neg_binomial(mean: float, cv_pct: float, n: int) -> np.ndarray:
    """
    NegBin sampler for overdispersed count stats (stl, blk, threes).
    Falls back to Poisson if CV ≤ 1/sqrt(mean) (underdispersed).
    Mirrors simulateStatNegBin() in index.html.
    """
    cv = cv_pct / 100.0
    poisson_cv = 1.0 / math.sqrt(max(mean, 1e-9))
    if cv <= poisson_cv:
        return np.random.poisson(mean, n).astype(float)
    r = 1.0 / (cv ** 2 - 1.0 / max(mean, 1e-9))
    r = max(r, 0.01)  # numerical guard
    p = r / (r + mean)
    return np.random.negative_binomial(int(round(r * 1000)) / 1000.0, p, n).astype(float)


def simulate_poisson(mean: float, n: int) -> np.ndarray:
    """Poisson sampler for discrete count stats with no CV data."""
    return np.random.poisson(max(mean, 0.0), n).astype(float)


def run_simulation(
    stat_key: str,
    mean: float,
    cv_pct: float | None,
    prop_line: float,
) -> dict | None:
    """
    Simulate one prop and return probability estimates.
    Returns None if inputs are invalid.
    """
    if mean <= 0 or prop_line < 0:
        return None

    effective_cv = cv_pct if cv_pct is not None else FALLBACK_CV.get(stat_key)
    if effective_cv is None or effective_cv <= 0:
        return None

    if stat_key in LOG_NORMAL_STATS:
        samples = simulate_log_normal(mean, effective_cv, N_SIMS)
        model = "LogNormal"
    elif stat_key in POISSON_STATS:
        if cv_pct is not None:
            samples = simulate_neg_binomial(mean, effective_cv, N_SIMS)
            model = "NegBin"
        else:
            samples = simulate_poisson(mean, N_SIMS)
            model = "Poisson"
    else:
        samples = simulate_log_normal(mean, effective_cv, N_SIMS)
        model = "LogNormal"

    over_prob  = float(np.mean(samples > prop_line))
    under_prob = float(np.mean(samples < prop_line))
    push_prob  = float(np.mean(samples == prop_line))

    return {
        "model":      model,
        "cv_used":    round(effective_cv, 1),
        "cv_source":  "historical" if cv_pct is not None else "fallback",
        "over_prob":  round(over_prob, 4),
        "under_prob": round(under_prob, 4),
        "push_prob":  round(push_prob, 4),
    }


# ── The Odds API ─────────────────────────────────────────────────────────────

def odds_api_get(
    url: str,
    params: dict,
    dry_run: bool = False,
    label: str = "",
) -> dict | None:
    """
    Make a GET request to The Odds API with rate limiting and error handling.
    Returns parsed JSON or None on failure. Skips actual call in dry_run mode.
    """
    if dry_run:
        print(f"  [DRY RUN] Would call: {url}")
        return None

    try:
        resp = requests.get(url, params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used       = resp.headers.get("x-requests-used", "?")
        print(f"    API: {label}  |  remaining={remaining}  used={used}")
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 422:
            print(f"    WARN 422 (no snapshot available at this timestamp): {label}")
            return None
        print(f"    ERROR {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        return None
    except requests.RequestException as exc:
        print(f"    REQUEST ERROR: {exc}", file=sys.stderr)
        return None
    finally:
        time.sleep(ODDS_API_DELAY)


def fetch_historical_events(
    game_date: date,
    api_key: str,
    dry_run: bool = False,
) -> list[dict]:
    """
    Fetch all NBA events for a given date from the historical events endpoint.
    Returns list of {id, home_team, away_team, commence_time}.
    """
    snapshot = snapshot_utc(game_date)
    url    = f"{ODDS_API_BASE}/historical/sports/basketball_nba/events"
    params = {"apiKey": api_key, "date": snapshot}

    print(f"  Fetching events for {game_date} (snapshot: {snapshot})")
    data = odds_api_get(url, params, dry_run=dry_run, label=f"events/{game_date}")

    if data is None:
        return []

    events = data.get("data", [])
    # Only return events that actually tip off on this date (filter out next-day games)
    target_prefix = game_date.strftime("%Y-%m-%d")
    day_events = [
        e for e in events
        if e.get("commence_time", "").startswith(target_prefix)
    ]
    print(f"    Found {len(day_events)} NBA games on {game_date}")
    return day_events


def fetch_event_prop_odds(
    event_id: str,
    game_date: date,
    api_key: str,
    dry_run: bool = False,
    bookmaker: str | None = None,
) -> list[dict]:
    """
    Fetch historical player prop odds for a single event.
    Returns list of:
      {player, stat, line, over_odds, under_odds, bookmaker}

    Cost: 10 credits per region × per market = 60 credits total for 6 markets.
    """
    snapshot = snapshot_utc(game_date)
    url    = f"{ODDS_API_BASE}/historical/sports/basketball_nba/events/{event_id}/odds"
    params = {
        "apiKey":  api_key,
        "date":    snapshot,
        "regions": "us",
        "markets": ",".join(MARKETS),
        "oddsFormat": "american",
    }

    label = f"event_odds/{event_id[:8]}…/{game_date}"
    data  = odds_api_get(url, params, dry_run=dry_run, label=label)
    if data is None:
        return []

    event_data  = data.get("data", {})
    bookmakers  = event_data.get("bookmakers", [])

    # Select preferred bookmaker
    selected_bm = None
    priority = [bookmaker] + BOOKMAKER_PRIORITY if bookmaker else BOOKMAKER_PRIORITY
    bm_by_key   = {bm["key"]: bm for bm in bookmakers}
    for key in priority:
        if key in bm_by_key:
            selected_bm = bm_by_key[key]
            break
    if selected_bm is None and bookmakers:
        selected_bm = bookmakers[0]  # fallback: first available
    if selected_bm is None:
        return []

    props: list[dict] = []
    for market in selected_bm.get("markets", []):
        stat_key = MARKET_TO_STAT.get(market["key"])
        if stat_key is None:
            continue

        # Group outcomes by player (description field)
        by_player: dict[str, dict] = {}
        for outcome in market.get("outcomes", []):
            player = normalize_name(outcome.get("description", ""))
            if not player:
                continue
            side  = outcome.get("name", "").lower()   # "over" or "under"
            price = outcome.get("price")
            point = outcome.get("point")
            if player not in by_player:
                by_player[player] = {"line": None, "over_odds": None, "under_odds": None}
            if point is not None:
                by_player[player]["line"] = point
            if side == "over" and price is not None:
                by_player[player]["over_odds"] = int(price)
            elif side == "under" and price is not None:
                by_player[player]["under_odds"] = int(price)

        for player, data_p in by_player.items():
            if data_p["line"] is None:
                continue
            props.append({
                "player":     player,
                "stat":       stat_key,
                "line":       float(data_p["line"]),
                "over_odds":  data_p["over_odds"],
                "under_odds": data_p["under_odds"],
                "bookmaker":  selected_bm["key"],
            })

    return props


# ── NBA Stats — Actual Results ────────────────────────────────────────────────

def fetch_actual_results(
    start_date: date,
    end_date: date,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Fetch actual player game stats for a date range via NBA Stats API.
    Returns: {normalized_player_name: {date_str: {stat_key: value}}}

    Uses LeagueGameLog for efficiency (one call covers all players in range).
    """
    print(f"\nFetching actual NBA results: {start_date} → {end_date}")

    # NBA Stats API date format: MM/DD/YYYY
    date_from = start_date.strftime("%m/%d/%Y")
    date_to   = end_date.strftime("%m/%d/%Y")

    try:
        log = LeagueGameLog(
            season=SEASON,
            date_from_nullable=date_from,
            date_to_nullable=date_to,
            player_or_team_abbreviation="P",
            season_type_all_star="Regular Season",
            timeout=60,
        )
        df = log.get_data_frames()[0]
    except Exception as exc:
        print(f"  ERROR fetching LeagueGameLog: {exc}", file=sys.stderr)
        return {}

    if df.empty:
        print("  No games found in date range")
        return {}

    results: dict[str, dict[str, dict[str, float]]] = {}

    for _, row in df.iterrows():
        raw_name  = str(row.get("PLAYER_NAME") or "")
        player    = normalize_name(raw_name)
        game_date = str(row.get("GAME_DATE") or "")  # format: "MAR 01, 2026" or "2026-03-01"

        # Normalize date string to YYYY-MM-DD
        try:
            if "," in game_date:
                # "MAR 01, 2026" format
                dt = datetime.strptime(game_date.strip(), "%b %d, %Y")
            else:
                dt = datetime.strptime(game_date.strip()[:10], "%Y-%m-%d")
            date_key = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_key = game_date[:10]

        stats = {
            "pts":    float(row.get("PTS") or 0),
            "reb":    float(row.get("REB") or 0),
            "ast":    float(row.get("AST") or 0),
            "stl":    float(row.get("STL") or 0),
            "blk":    float(row.get("BLK") or 0),
            "threes": float(row.get("FG3M") or 0),
        }

        if player not in results:
            results[player] = {}
        results[player][date_key] = stats

    print(f"  Loaded actual results for {len(results)} players across {len(df)} game entries")
    time.sleep(NBA_API_DELAY)
    return results


# ── ETR CSV Loading ───────────────────────────────────────────────────────────

# Common ETR CSV column name variants (lowercase for case-insensitive matching)
ETR_COL_ALIASES: dict[str, list[str]] = {
    "player": ["name", "player", "playername", "player name"],
    "team":   ["team", "tm"],
    "pts":    ["pts", "points", "proj pts", "proj_pts", "projected pts"],
    "reb":    ["reb", "rebounds", "proj reb", "proj_reb", "projected reb"],
    "ast":    ["ast", "assists", "proj ast", "proj_ast", "projected ast"],
    "stl":    ["stl", "steals", "proj stl", "proj_stl"],
    "blk":    ["blk", "blocks", "proj blk", "proj_blk"],
    "threes": ["3pm", "threes", "fg3m", "3 pm", "three pm", "proj 3pm",
               "proj threes", "3-pt made", "3ptm"],
}


def _detect_col(headers_lower: list[str], aliases: list[str]) -> int | None:
    """Return column index of first matching alias, or None."""
    for alias in aliases:
        for i, h in enumerate(headers_lower):
            if h.strip().lower() == alias:
                return i
    return None


def load_etr_csv(csv_path: Path) -> dict[str, dict[str, float]]:
    """
    Load one ETR projection CSV. Returns {normalized_player_name: {stat: value}}.
    Returns empty dict on any parse failure.
    """
    try:
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)
    except Exception as exc:
        print(f"  WARN: Could not read ETR CSV {csv_path.name}: {exc}", file=sys.stderr)
        return {}

    headers_lower = [h.lower().strip() for h in headers]

    # Detect column indices
    col_map: dict[str, int | None] = {
        stat: _detect_col(headers_lower, aliases)
        for stat, aliases in ETR_COL_ALIASES.items()
    }

    if col_map["player"] is None:
        print(f"  WARN: Could not find player name column in {csv_path.name}. "
              f"Headers: {headers[:8]}", file=sys.stderr)
        return {}

    found_stats = [k for k, v in col_map.items() if v is not None and k != "player"]
    print(f"  ETR {csv_path.name}: found columns: player + {found_stats}")

    projections: dict[str, dict[str, float]] = {}

    try:
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if not row or col_map["player"] is None:
                    continue
                try:
                    raw_name = row[col_map["player"]]
                except IndexError:
                    continue

                if not raw_name or raw_name.lower() in ("name", "player", ""):
                    continue

                player = normalize_name(raw_name)
                player_stats: dict[str, float] = {}

                for stat in ("pts", "reb", "ast", "stl", "blk", "threes"):
                    idx = col_map.get(stat)
                    if idx is None:
                        continue
                    try:
                        val = float(str(row[idx]).replace(",", "").strip())
                        if val > 0:
                            player_stats[stat] = val
                    except (ValueError, IndexError):
                        pass

                if player_stats:
                    projections[player] = player_stats

    except Exception as exc:
        print(f"  WARN: Error parsing {csv_path.name}: {exc}", file=sys.stderr)

    return projections


def find_etr_csv_for_date(etr_dir: Path, target_date: date) -> Path | None:
    """
    Find an ETR CSV file for a given date. Accepts patterns:
      YYYY-MM-DD.csv
      YYYY-MM-DD_*.csv
    """
    date_str = target_date.strftime("%Y-%m-%d")
    for path in sorted(etr_dir.glob(f"{date_str}*.csv")):
        return path  # first match
    return None


# ── CV Snapshot Loading ───────────────────────────────────────────────────────

def load_cv_snapshot(
    cv_snapshots_dir: Path | None,
    target_date: date,
    fallback_path: Path = Path("cv_data.json"),
) -> dict:
    """
    Load the cv_data.json snapshot closest to (but not after) target_date.
    Falls back to current cv_data.json if no snapshots are available.
    """
    if cv_snapshots_dir and cv_snapshots_dir.exists():
        # Find most recent snapshot on or before target_date
        candidates = sorted(cv_snapshots_dir.glob("????-??-??.json"))
        best = None
        for path in candidates:
            try:
                snap_date = date.fromisoformat(path.stem)
                if snap_date <= target_date:
                    best = path
            except ValueError:
                pass
        if best:
            with open(best, encoding="utf-8") as f:
                data = json.load(f)
            print(f"  CV snapshot: {best.name} (for game date {target_date})")
            return data

    # Fallback to current cv_data.json
    if fallback_path.exists():
        with open(fallback_path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  CV snapshot: cv_data.json (current, no historical snapshot found)")
        return data

    print("  WARN: No CV data found — will use fallback STD_RATIOS only", file=sys.stderr)
    return {"players": {}}


def get_cv_for_player(
    cv_data: dict,
    player_name: str,
    stat_key: str,
    window: str = "last20",
) -> float | None:
    """
    Look up CV% for a player/stat/window from cv_data.json.
    Returns None if not found.
    """
    player_data = cv_data.get("players", {}).get(player_name)
    if not player_data:
        return None
    return player_data.get("cv", {}).get(stat_key, {}).get(window)


# ── Fuzzy Name Matching ───────────────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance for fuzzy name matching."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[lb]


def fuzzy_match_player(
    query: str,
    candidates: set[str],
    max_distance: int = 2,
) -> str | None:
    """
    Find best fuzzy match for a player name in a set of candidates.
    Returns None if no match within max_distance edits.
    """
    best_name = None
    best_dist = max_distance + 1
    for candidate in candidates:
        d = _edit_distance(query.lower(), candidate.lower())
        if d < best_dist:
            best_dist = d
            best_name = candidate
    return best_name if best_dist <= max_distance else None


# ── Record Builder ────────────────────────────────────────────────────────────

def build_record(
    game_date: date,
    player_norm: str,
    stat_key: str,
    prop_line: float,
    over_odds: int | None,
    under_odds: int | None,
    bookmaker: str,
    etr_mean: float | None,
    cv_pct: float | None,
    actual_value: float | None,
) -> dict | None:
    """
    Build one backtest record by running simulation and computing edge metrics.
    Returns None if we can't compute a meaningful result.
    """
    if etr_mean is None or etr_mean <= 0:
        return None

    sim = run_simulation(stat_key, etr_mean, cv_pct, prop_line)
    if sim is None:
        return None

    # Vig-free book probabilities
    vf_over, vf_under = vig_free_prob(over_odds, under_odds)

    # Edge calculations (vig-free when both sides present, raw otherwise)
    over_edge  = None
    under_edge = None
    if vf_over is not None:
        over_edge  = round((sim["over_prob"]  - vf_over)  * 100, 2)
        under_edge = round((sim["under_prob"] - vf_under) * 100, 2)

    # Determine actual outcome
    hit_over  = None
    hit_under = None
    if actual_value is not None:
        hit_over  = actual_value > prop_line
        hit_under = actual_value < prop_line

    # Best side
    best_side  = None
    best_edge  = None
    edge_label = "NO EDGE"
    if over_edge is not None and under_edge is not None:
        if over_edge >= 1.5 or under_edge >= 1.5:
            if over_edge >= under_edge:
                best_side  = "OVER"
                best_edge  = over_edge
            else:
                best_side  = "UNDER"
                best_edge  = under_edge
            edge_label = get_edge_label(best_edge)

    return {
        "date":         game_date.isoformat(),
        "player":       player_norm,
        "stat":         stat_key,
        "etr_mean":     round(etr_mean, 2),
        "cv_pct":       sim["cv_used"],
        "cv_source":    sim["cv_source"],
        "model":        sim["model"],
        "prop_line":    prop_line,
        "over_odds":    over_odds,
        "under_odds":   under_odds,
        "bookmaker":    bookmaker,
        "model_over":   sim["over_prob"],
        "model_under":  sim["under_prob"],
        "vf_book_over": round(vf_over,  4) if vf_over  is not None else None,
        "vf_book_under":round(vf_under, 4) if vf_under is not None else None,
        "over_edge":    over_edge,
        "under_edge":   under_edge,
        "best_side":    best_side,
        "best_edge":    best_edge,
        "edge_label":   edge_label,
        "actual":       actual_value,
        "hit_over":     hit_over,
        "hit_under":    hit_under,
    }


# ── Calibration Statistics ────────────────────────────────────────────────────

def compute_calibration(records: list[dict]) -> dict:
    """
    Aggregate backtest records into calibration metrics:
      - By edge tier: count, best-side hit rate, ROI simulation
      - By stat: calibration (avg model prob vs actual hit rate)
      - Overall Brier score proxy
    """
    # Filter to records with actual outcomes
    complete = [r for r in records if r["hit_over"] is not None and r["best_side"] is not None]

    # By edge tier
    tier_stats: dict[str, dict] = {
        label: {"bets": 0, "wins": 0, "total_edge": 0.0, "roi_units": 0.0}
        for label, _ in EDGE_TIERS
    }

    for r in complete:
        tier = r["edge_label"]
        if tier not in tier_stats:
            continue
        ts = tier_stats[tier]
        ts["bets"] += 1

        # Did the best side win?
        won = (r["best_side"] == "OVER" and r["hit_over"]) or \
              (r["best_side"] == "UNDER" and r["hit_under"])
        if won:
            ts["wins"] += 1

        # ROI simulation: 1-unit flat bet on best side
        if won:
            bet_odds = r["over_odds"] if r["best_side"] == "OVER" else r["under_odds"]
            if bet_odds is None:
                pass
            elif bet_odds >= 0:
                ts["roi_units"] += bet_odds / 100.0
            else:
                ts["roi_units"] += 100.0 / abs(bet_odds)
        else:
            ts["roi_units"] -= 1.0

        ts["total_edge"] += r["best_edge"] or 0.0

    tier_summary = {}
    for label, ts in tier_stats.items():
        if ts["bets"] == 0:
            continue
        tier_summary[label] = {
            "bets":       ts["bets"],
            "wins":       ts["wins"],
            "hit_rate":   round(ts["wins"] / ts["bets"], 4),
            "avg_edge":   round(ts["total_edge"] / ts["bets"], 2),
            "roi_units":  round(ts["roi_units"], 2),
            "roi_pct":    round(ts["roi_units"] / ts["bets"] * 100, 2),
        }

    # By stat — model probability calibration
    # Bin model probs into deciles and compare to actual hit rates
    stat_calibration: dict[str, dict] = {}
    all_with_outcomes = [r for r in records if r["hit_over"] is not None and
                         r["model_over"] is not None]

    for r in all_with_outcomes:
        stat = r["stat"]
        if stat not in stat_calibration:
            stat_calibration[stat] = {"samples": 0, "brier_sum": 0.0}
        sc = stat_calibration[stat]
        sc["samples"] += 1
        # Brier score for over side: (model_prob - actual_outcome)²
        sc["brier_sum"] += (r["model_over"] - float(r["hit_over"])) ** 2

    stat_summary = {}
    for stat, sc in stat_calibration.items():
        if sc["samples"] == 0:
            continue
        stat_summary[stat] = {
            "samples":     sc["samples"],
            "brier_score": round(sc["brier_sum"] / sc["samples"], 4),
        }

    # Coverage stats
    total_props   = len(records)
    with_actual   = len([r for r in records if r["actual"] is not None])
    with_etr      = len([r for r in records if r["etr_mean"] is not None])
    with_cv       = len([r for r in records if r["cv_source"] == "historical"])

    return {
        "total_props":   total_props,
        "with_actual":   with_actual,
        "with_etr":      with_etr,
        "with_cv":       with_cv,
        "by_edge_tier":  tier_summary,
        "by_stat":       stat_summary,
    }


# ── Dry-run Credit Estimator ──────────────────────────────────────────────────

def estimate_credits(start: date, end: date, avg_games_per_day: float = 7.0) -> None:
    """Print credit cost estimate without making any API calls."""
    days         = (end - start).days + 1
    markets      = len(MARKETS)
    regions      = 1
    cost_per_game = 10 * markets * regions
    est_games    = int(days * avg_games_per_day)
    est_credits  = est_games * cost_per_game

    print("\n" + "=" * 60)
    print("CREDIT ESTIMATE (DRY RUN)")
    print("=" * 60)
    print(f"  Date range:        {start} → {end} ({days} days)")
    print(f"  Markets:           {', '.join(MARKETS)}")
    print(f"  Regions:           us")
    print(f"  Credits per game:  10 × {markets} markets × {regions} region = {cost_per_game}")
    print(f"  Est. games:        {avg_games_per_day:.0f}/day × {days} days = {est_games}")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  ESTIMATED TOTAL:   ~{est_credits:,} credits")
    print(f"  % of 20k budget:   ~{est_credits/20000*100:.1f}%")
    print("=" * 60)
    print("\nRun without --dry-run to execute. Double-check dates first.\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build NBA prop backtest dataset from The Odds API + NBA Stats + ETR CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument("--start",            required=True,
                        help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end",              required=True,
                        help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--api-key",          default=os.environ.get("ODDS_API_KEY"),
                        help="The Odds API key (or set ODDS_API_KEY env var)")
    parser.add_argument("--etr-dir",          default=None,
                        help="Directory of ETR CSVs named YYYY-MM-DD.csv")
    parser.add_argument("--cv-snapshots-dir", default="cv_snapshots",
                        help="Directory of dated cv_data.json snapshots (default: cv_snapshots/)")
    parser.add_argument("--output",           default="backtest_results.json",
                        help="Output file path (default: backtest_results.json)")
    parser.add_argument("--bookmaker",        default=None,
                        help="Preferred bookmaker key (e.g. draftkings, fanduel). "
                             "Defaults to first available in priority list.")
    parser.add_argument("--cv-window",        default="last20",
                        choices=["season", "last20", "last10", "last5"],
                        help="CV window to use for simulation (default: last20)")
    parser.add_argument("--snapshot-hour",    type=int, default=SNAPSHOT_HOUR_UTC,
                        help=f"UTC hour for pre-game odds snapshot (default: {SNAPSHOT_HOUR_UTC} = ~6:30 PM ET)")
    parser.add_argument("--dry-run",          action="store_true",
                        help="Estimate credits and exit — no API calls")
    args = parser.parse_args()

    try:
        start_date = date.fromisoformat(args.start)
        end_date   = date.fromisoformat(args.end)
    except ValueError as exc:
        print(f"ERROR: Invalid date format: {exc}", file=sys.stderr)
        sys.exit(1)

    if start_date > end_date:
        print("ERROR: --start must be before or equal to --end", file=sys.stderr)
        sys.exit(1)

    # Always show estimate
    estimate_credits(start_date, end_date)

    if args.dry_run:
        return

    if not args.api_key:
        print("ERROR: --api-key is required (or set ODDS_API_KEY env var)", file=sys.stderr)
        sys.exit(1)

    etr_dir          = Path(args.etr_dir) if args.etr_dir else None
    cv_snapshots_dir = Path(args.cv_snapshots_dir)
    output_path      = Path(args.output)

    # ── Step 1: Fetch actual NBA results for the full date range ──────────────
    actual_results = fetch_actual_results(start_date, end_date)
    actual_players = set(actual_results.keys())

    # ── Step 2: Process each date ─────────────────────────────────────────────
    all_records: list[dict] = []
    dates = list(date_range(start_date, end_date))

    for i, game_date in enumerate(dates):
        date_str = game_date.isoformat()
        print(f"\n[{i+1}/{len(dates)}] Processing {date_str}")

        # Load CV snapshot for this date
        cv_data = load_cv_snapshot(cv_snapshots_dir, game_date)

        # Load ETR projections if available
        etr_projections: dict[str, dict[str, float]] = {}
        if etr_dir:
            etr_path = find_etr_csv_for_date(etr_dir, game_date)
            if etr_path:
                etr_projections = load_etr_csv(etr_path)
                print(f"  ETR: {len(etr_projections)} players loaded")
            else:
                print(f"  ETR: no CSV found for {date_str} in {etr_dir}")

        # Fetch events from The Odds API
        events = fetch_historical_events(game_date, args.api_key)
        if not events:
            print(f"  No events found, skipping date")
            continue

        # Fetch prop odds for each event
        for event in events:
            event_id  = event["id"]
            matchup   = f"{event.get('away_team', '?')} @ {event.get('home_team', '?')}"
            print(f"  → {matchup}")

            props = fetch_event_prop_odds(
                event_id, game_date, args.api_key,
                bookmaker=args.bookmaker,
            )
            print(f"    {len(props)} prop lines fetched")

            cv_players   = set(cv_data.get("players", {}).keys())
            etr_players  = set(etr_projections.keys())

            for prop in props:
                player_norm = prop["player"]

                # Resolve player name across data sources with fuzzy fallback
                cv_key  = player_norm if player_norm in cv_players \
                          else fuzzy_match_player(player_norm, cv_players)
                etr_key = player_norm if player_norm in etr_players \
                          else fuzzy_match_player(player_norm, etr_players)
                act_key = player_norm if player_norm in actual_players \
                          else fuzzy_match_player(player_norm, actual_players)

                # ETR projection mean for this stat
                etr_mean: float | None = None
                if etr_key and prop["stat"] in etr_projections.get(etr_key, {}):
                    etr_mean = etr_projections[etr_key][prop["stat"]]

                # Historical CV
                cv_pct: float | None = None
                if cv_key:
                    cv_pct = get_cv_for_player(
                        cv_data, cv_key, prop["stat"], args.cv_window
                    )

                # Actual game result
                actual_value: float | None = None
                if act_key and date_str in actual_results.get(act_key, {}):
                    actual_value = actual_results[act_key][date_str].get(prop["stat"])

                record = build_record(
                    game_date    = game_date,
                    player_norm  = player_norm,
                    stat_key     = prop["stat"],
                    prop_line    = prop["line"],
                    over_odds    = prop["over_odds"],
                    under_odds   = prop["under_odds"],
                    bookmaker    = prop["bookmaker"],
                    etr_mean     = etr_mean,
                    cv_pct       = cv_pct,
                    actual_value = actual_value,
                )
                if record:
                    all_records.append(record)

        print(f"  Records so far: {len(all_records)}")

    # ── Step 3: Compute calibration stats ────────────────────────────────────
    print(f"\nComputing calibration across {len(all_records)} records...")
    calibration = compute_calibration(all_records)

    # ── Step 4: Write output ──────────────────────────────────────────────────
    output = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_range":   {"start": args.start, "end": args.end},
        "parameters": {
            "n_sims":        N_SIMS,
            "cv_window":     args.cv_window,
            "snapshot_hour": args.snapshot_hour,
            "bookmaker_pref": args.bookmaker or "auto",
        },
        "calibration":  calibration,
        "records":      all_records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)
    print(f"  Total props:     {calibration['total_props']}")
    print(f"  With actual:     {calibration['with_actual']}")
    print(f"  With ETR mean:   {calibration['with_etr']}")
    print(f"  With hist. CV:   {calibration['with_cv']}")
    print()
    print("  Results by edge tier:")
    for tier_label, ts in calibration.get("by_edge_tier", {}).items():
        print(f"    {tier_label:<10}  {ts['bets']:>4} bets  "
              f"hit={ts['hit_rate']:.1%}  "
              f"ROI={ts['roi_pct']:+.1f}%  "
              f"avg_edge={ts['avg_edge']:+.1f}pp")
    print()
    print("  Brier scores by stat (lower = better calibrated):")
    for stat, sc in calibration.get("by_stat", {}).items():
        print(f"    {stat:<7}  n={sc['samples']:>4}  brier={sc['brier_score']:.4f}")
    print()
    print(f"  Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
