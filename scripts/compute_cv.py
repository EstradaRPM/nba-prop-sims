#!/usr/bin/env python3
"""
compute_cv.py — NBA player CV computation script

Pulls game logs from NBA Stats API and computes per-36 normalized CVs
for multiple time windows. Outputs cv_data.json for use by the simulator.

Usage:
  python scripts/compute_cv.py --test      # 5 test players only
  python scripts/compute_cv.py             # full league (~500 players, ~10-15 min)
  python scripts/compute_cv.py --output path/to/cv_data.json
"""

import json
import time
import argparse
import unicodedata
import sys
from datetime import datetime, timezone

from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.static import players, teams

# ── Config ──────────────────────────────────────────────────────────────────

SEASON = "2025-26"

# Stats to compute CV for
STAT_KEYS = ["pts", "reb", "ast", "stl", "blk", "threes", "pra", "pr", "pa", "ra", "sb"]

# API column → internal key
COL_MAP = {
    "PTS": "pts",
    "REB": "reb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "FG3M": "threes",
}

# Seconds to wait between player API requests — stay under rate limits
REQUEST_DELAY = 0.65

# 5 well-distributed test players (different teams/positions/usage patterns)
TEST_PLAYERS = [
    "LaMelo Ball",
    "Anthony Davis",
    "Stephen Curry",
    "Giannis Antetokounmpo",
    "Tyrese Haliburton",
]

# Known name mismatches: NBA Stats API full_name → normalized key written to JSON.
# Both sides (Python and JS) apply NFD unicode normalization independently;
# this table handles cases where the API and ETR spell names differently.
NAME_OVERRIDES: dict[str, str] = {
    # "Nikola Jokic": "Nikola Jokić",  # example — verify against ETR spellings
    # Add entries as mismatches are discovered in production
}


# ── Utilities ────────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """
    Normalize a player name for consistent cross-source matching.
    Strips whitespace and applies NFD unicode normalization.
    Both this script and the JS simulator apply the same normalization
    before using names as join keys.
    """
    return unicodedata.normalize("NFD", name.strip())


def parse_minutes(min_str) -> float:
    """
    Parse the NBA Stats API minutes field into a float.
    The API returns strings like '32:45' (minutes:seconds) or '32.0'.
    Returns 0.0 on any parse failure.
    """
    if min_str is None:
        return 0.0
    s = str(min_str).strip()
    if not s or s == "0":
        return 0.0
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def sample_std(values: list[float]) -> float:
    """Sample standard deviation (Bessel's correction, n-1 denominator)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((x - mean) ** 2 for x in values) / (n - 1)) ** 0.5


def compute_cv(values: list[float]) -> float | None:
    """
    Compute CV% = (std_dev / mean) * 100 for a list of per-36 rates.
    Returns None if fewer than 5 values or mean is zero.
    """
    if len(values) < 5:
        return None
    mean = sum(values) / len(values)
    if mean == 0.0:
        return None
    std = sample_std(values)
    return round((std / mean) * 100.0, 1)


# ── Situation Filter ─────────────────────────────────────────────────────────

def apply_situation_filter(games: list[dict]) -> list[dict]:
    """
    Filter games per the CV methodology spec:

    Exclude a game if EITHER condition holds:
      1. Minutes played < 10 (garbage time / DNP-adjacent)
      2. Minutes deviate > 25% from the player's own trailing seasonal mean
         (trailing mean = mean of ALL games seen so far, before this one)

    Games must be passed in chronological order (oldest first).
    The trailing mean is computed over all games seen so far regardless of
    whether they passed the filter — we want the true seasonal minutes baseline.

    Returns a list of games that passed both filters.
    """
    filtered = []
    all_mins_so_far: list[float] = []  # all games (unfiltered) for trailing mean

    for game in games:
        min_played = game["min"]

        # Condition 1: hard floor
        if min_played < 10.0:
            all_mins_so_far.append(min_played)
            continue

        # Condition 2: trailing mean deviation
        if all_mins_so_far:
            trailing_mean = sum(all_mins_so_far) / len(all_mins_so_far)
            if trailing_mean == 0:
                all_mins_so_far.append(min_played)
                continue
            deviation = abs(min_played - trailing_mean) / trailing_mean
            if deviation > 0.25:
                all_mins_so_far.append(min_played)
                continue

        # Game passed — add to both lists
        all_mins_so_far.append(min_played)
        filtered.append(game)

    return filtered


# ── Per-Player Computation ───────────────────────────────────────────────────

def compute_player_cv(player_id: int, player_name: str) -> dict | None:
    """
    Fetch game log for one player and compute per-36 CVs for all windows.

    Returns a dict matching the cv_data.json player schema, or None on failure.
    """
    try:
        log = PlayerGameLog(
            player_id=player_id,
            season=SEASON,
            season_type_all_star="Regular Season",
            timeout=60,
        )
        df = log.get_data_frames()[0]
    except Exception as exc:
        print(f"    ERROR fetching game log: {exc}", file=sys.stderr)
        return None

    if df.empty:
        print(f"    No games found for season {SEASON}")
        return None

    # Build game list in chronological order (API returns most-recent first)
    raw_games: list[dict] = []
    for _, row in df.iterrows():
        min_played = parse_minutes(row.get("MIN"))
        pts = float(row.get("PTS") or 0)
        reb = float(row.get("REB") or 0)
        ast = float(row.get("AST") or 0)
        # Extract team abbreviation from MATCHUP field ("CHA vs. SAC" or "CHA @ SAC")
        matchup = str(row.get("MATCHUP") or "")
        team_abbrev = matchup.split(" ")[0] if matchup else ""

        raw_games.append({
            "min":   min_played,
            "pts":   pts,
            "reb":   reb,
            "ast":   ast,
            "stl":   float(row.get("STL") or 0),
            "blk":   float(row.get("BLK") or 0),
            "threes": float(row.get("FG3M") or 0),
            "pra":   pts + reb + ast,
            "pr":    pts + reb,
            "pa":    pts + ast,
            "ra":    reb + ast,
            "sb":    float(row.get("STL") or 0) + float(row.get("BLK") or 0),
            "team":  team_abbrev,
            "date":  str(row.get("GAME_DATE") or ""),
        })
    # Reverse to chronological
    raw_games.reverse()

    total_games = len(raw_games)

    # Apply situation filter
    filtered = apply_situation_filter(raw_games)
    n_filtered = len(filtered)

    if n_filtered == 0:
        print(f"    0 qualifying games after filter (raw: {total_games})")
        return None

    # Convert filtered games to per-36 rates
    per36: list[dict] = []
    for g in filtered:
        if g["min"] <= 0:
            continue
        factor = 36.0 / g["min"]
        per36.append({stat: g[stat] * factor for stat in STAT_KEYS})

    if not per36:
        return None

    # Compute CV for each stat across each window
    # windows_filtered holds the raw filtered games per window (for minutes CV)
    windows_filtered = {
        "season": filtered,
        "last20": filtered[-20:],
        "last10": filtered[-10:],
        "last5":  filtered[-5:],
    }
    windows = {
        "season": per36,
        "last20": per36[-20:],
        "last10": per36[-10:],
        "last5":  per36[-5:],
    }

    cv_result: dict[str, dict] = {}
    for stat in STAT_KEYS:
        cv_result[stat] = {}
        for window_name, window_games in windows.items():
            rates = [g[stat] for g in window_games]
            cv_result[stat][window_name] = compute_cv(rates)

    # CV of minutes per window (raw filtered minutes, not per-36)
    # Used by the simulator to combine: CV_effective = sqrt(CV_per36² + CV_minutes²)
    cv_minutes: dict[str, float | None] = {}
    for window_name, window_games in windows_filtered.items():
        mins = [g["min"] for g in window_games]
        cv_minutes[window_name] = compute_cv(mins)

    # Mean minutes across last 20 qualifying games
    last20_mins = [g["min"] for g in filtered[-20:]]
    mean_min_last20 = round(sum(last20_mins) / len(last20_mins), 1) if last20_mins else None

    # Team from most recent qualifying game
    team = filtered[-1]["team"] if filtered else ""

    # Raw per-stat values per window (non-normalized, situation-filtered game totals)
    # Used by the JS simulator for empirical KDE hybrid sampling.
    # Stored as raw game totals (not per-36) because props are graded on raw game stats.
    def build_raw_windows(stat_key: str) -> tuple[dict, dict]:
        raw: dict[str, list[float] | None] = {}
        mean_raw: dict[str, float | None] = {}
        for window_name, window_games in windows_filtered.items():
            scores = [round(g[stat_key], 1) for g in window_games]
            if len(scores) >= 5:
                raw[window_name] = scores
                mean_raw[window_name] = round(sum(scores) / len(scores), 2)
            else:
                raw[window_name] = None
                mean_raw[window_name] = None
        return raw, mean_raw

    pts_raw, pts_mean_raw = build_raw_windows("pts")
    reb_raw, reb_mean_raw = build_raw_windows("reb")
    ast_raw, ast_mean_raw = build_raw_windows("ast")

    return {
        "nba_id": player_id,
        "team": team,
        "position": "",  # not available in PlayerGameLog; populated by ETR on JS side
        "games_available": n_filtered,
        "cv": cv_result,
        "cv_minutes": cv_minutes,
        "mean_minutes_last20": mean_min_last20,
        "pts_raw": pts_raw,
        "pts_mean_raw": pts_mean_raw,
        "reb_raw": reb_raw,
        "reb_mean_raw": reb_mean_raw,
        "ast_raw": ast_raw,
        "ast_mean_raw": ast_mean_raw,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute NBA player per-36 CVs and write cv_data.json"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Process only {len(TEST_PLAYERS)} test players instead of full league",
    )
    parser.add_argument(
        "--output",
        default="cv_data.json",
        help="Output file path (default: cv_data.json at repo root)",
    )
    args = parser.parse_args()

    # Build teams lookup (id → abbreviation)
    all_teams = teams.get_teams()
    teams_by_id = {t["id"]: t for t in all_teams}

    # Get all active players
    all_players = players.get_active_players()

    if args.test:
        test_normalized = {normalize_name(n) for n in TEST_PLAYERS}
        target_players = [
            p for p in all_players
            if normalize_name(p["full_name"]) in test_normalized
        ]
        mode_label = f"TEST ({len(TEST_PLAYERS)} players)"
        # Warn if any test player wasn't matched
        found = {normalize_name(p["full_name"]) for p in target_players}
        missing = test_normalized - found
        if missing:
            print(f"WARNING: Could not find active players: {missing}", file=sys.stderr)
    else:
        target_players = all_players
        mode_label = f"FULL LEAGUE ({len(all_players)} players)"

    print(f"Mode: {mode_label}")
    print(f"Season: {SEASON}")
    print(f"Output: {args.output}")
    print("-" * 60)

    output: dict = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": SEASON,
        "players": {},
    }

    success = 0
    errors = 0
    skipped = 0

    for i, player in enumerate(target_players):
        player_id: int = player["id"]
        raw_name: str = player["full_name"]

        # Apply name override table before writing to JSON key
        display_name = NAME_OVERRIDES.get(raw_name, raw_name)
        json_key = normalize_name(display_name)

        print(f"[{i + 1}/{len(target_players)}] {raw_name} (id={player_id})", end=" ... ", flush=True)

        result = compute_player_cv(player_id, raw_name)

        if result:
            output["players"][json_key] = result
            success += 1
            windows_summary = {
                stat: {
                    k: v for k, v in result["cv"][stat].items() if v is not None
                }
                for stat in ["pts", "reb", "ast"]
            }
            print(
                f"OK  games={result['games_available']}  "
                f"team={result['team']}  "
                f"pts_cv={result['cv']['pts'].get('last20', 'null')}"
            )
        else:
            errors += 1
            print("SKIP (no data)")
            skipped += 1

        # Rate limiting — avoid NBA Stats API 429s
        if i < len(target_players) - 1:
            time.sleep(REQUEST_DELAY)

    # Write JSON output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("-" * 60)
    print(f"Done: {success} players written, {errors} skipped → {args.output}")
    print(f"generated_at: {output['generated_at']}")

    # Quick schema validation on output
    if output["players"]:
        sample_name = next(iter(output["players"]))
        sample = output["players"][sample_name]
        assert "cv" in sample, "Schema error: missing 'cv' key"
        assert "cv_minutes" in sample, "Schema error: missing 'cv_minutes' key"
        assert all(stat in sample["cv"] for stat in STAT_KEYS), f"Schema error: missing stat keys in cv (expected {STAT_KEYS})"
        assert all(
            w in sample["cv"]["pts"] for w in ["season", "last20", "last10", "last5"]
        ), "Schema error: missing window keys in cv"
        assert all(
            w in sample["cv_minutes"] for w in ["season", "last20", "last10", "last5"]
        ), "Schema error: missing window keys in cv_minutes"
        assert "pts_raw" in sample, "Schema error: missing 'pts_raw' key"
        assert "pts_mean_raw" in sample, "Schema error: missing 'pts_mean_raw' key"
        assert all(
            w in sample["pts_raw"] for w in ["season", "last20", "last10", "last5"]
        ), "Schema error: missing window keys in pts_raw"
        assert "reb_raw" in sample, "Schema error: missing 'reb_raw' key"
        assert "reb_mean_raw" in sample, "Schema error: missing 'reb_mean_raw' key"
        assert "ast_raw" in sample, "Schema error: missing 'ast_raw' key"
        assert "ast_mean_raw" in sample, "Schema error: missing 'ast_mean_raw' key"
        print(f"Schema check PASSED (validated against '{sample_name}')")


if __name__ == "__main__":
    main()
