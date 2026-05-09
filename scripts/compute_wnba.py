#!/usr/bin/env python3
"""
compute_wnba.py — WNBA player points Beta-Binomial model

Pulls game logs from WNBA Stats API and computes posterior hit probabilities
per threshold for use by wnba.html.

Usage:
  python scripts/compute_wnba.py               # current season (2025)
  python scripts/compute_wnba.py --season 2025
  python scripts/compute_wnba.py --test        # 5 players only
  python scripts/compute_wnba.py --output path/to/wnba_data.json
"""

import argparse
import json
import sys
import time
import unicodedata
from datetime import datetime, timezone

import requests

# ── Constants ────────────────────────────────────────────────────────────────

THRESHOLDS = [10, 15, 18, 20, 25, 30]
DECAY_LAMBDA = 0.9

# League-average hit rates per threshold (rough empirical estimates)
LEAGUE_AVG = {10: 0.60, 15: 0.40, 18: 0.28, 20: 0.20, 25: 0.10, 30: 0.05}
LEAGUE_STRENGTH = 10  # pseudo-count for league-average prior

NAME_OVERRIDES = {
    "A'ja Wilson": "Aja Wilson",
    "Skylar Diggins-Smith": "Skylar Diggins",
}

# ── Roster ───────────────────────────────────────────────────────────────────

ROSTER = [
    # Established stars
    "A'ja Wilson",
    "Breanna Stewart",
    "Sabrina Ionescu",
    "Kelsey Plum",
    "Jewell Loyd",
    "Napheesa Collier",
    "Jonquel Jones",
    "Arike Ogunbowale",
    "Dearica Hamby",
    "Chelsea Gray",
    "Kayla McBride",
    "Skylar Diggins-Smith",
    "Satou Sabally",
    "Kelsey Mitchell",
    "Tina Charles",
    "Nneka Ogwumike",
    "DeWanna Bonner",
    "Marina Mabrey",
    "DiJonai Carrington",
    "Rhyne Howard",
    "Aliyah Boston",
    "Aerial Powers",
    # High-volume prop targets
    "Caitlin Clark",
    "Angel Reese",
    "Jackie Young",
    "Kahleah Copper",
    "Brittney Griner",
    "Natasha Cloud",
    "Chennedy Carter",
    "Courtney Williams",
    "Tiffany Hayes",
    "Destiny Slocum",
    "Ezi Magbegor",
    "Shakira Austin",
    "Sophie Cunningham",
    "Brittney Sykes",
    "Erica Wheeler",
    "Lexie Hull",
    "Brianna Turner",
    "Kia Nurse",
    # Rookies / upside
    "Paige Bueckers",
    "Azzi Fudd",
    "Rickea Jackson",
    "Kate Martin",
    "Kamilla Cardoso",
]


# ── NameNormalizer ────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """NFD decompose, strip combining marks, apply overrides, strip whitespace."""
    stripped = name.strip()
    if stripped in NAME_OVERRIDES:
        stripped = NAME_OVERRIDES[stripped]
    nfd = unicodedata.normalize("NFD", stripped)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


# ── PriorBuilder ─────────────────────────────────────────────────────────────

def build_prior(
    career_games: list[int],
    threshold: int,
    league_avg: dict | None = None,
) -> dict:
    """
    Given career point totals and a threshold, return (alpha, beta, confidence).

    career_games: list of integer point totals (unordered)
    threshold: integer like 20 (meaning 20+)
    league_avg: optional empirical hit rates per threshold; falls back to LEAGUE_AVG
    """
    n = len(career_games)
    hits = sum(1 for g in career_games if g >= threshold)

    avg = league_avg if league_avg is not None else LEAGUE_AVG
    league_p = avg.get(threshold, 0.15)
    league_alpha = league_p * LEAGUE_STRENGTH
    league_beta = (1 - league_p) * LEAGUE_STRENGTH

    if n >= 15:
        alpha = hits + 1
        beta = (n - hits) + 1
        confidence = "full"
    elif n >= 5:
        w = n / LEAGUE_STRENGTH
        alpha = w * hits + (1 - w) * league_alpha + 1
        beta = w * (n - hits) + (1 - w) * league_beta + 1
        confidence = "partial"
    else:
        alpha = league_alpha + 1
        beta = league_beta + 1
        confidence = "limited"

    return {"alpha": alpha, "beta": beta, "confidence": confidence}


# ── BetaBinomialEstimator ────────────────────────────────────────────────────

def estimate_posterior(
    current_games: list[int],
    threshold: int,
    prior_alpha: float,
    prior_beta: float,
) -> dict:
    """
    Apply decayed Beta-Binomial update over current-season games.

    current_games: chronological list of point totals (index 0 = oldest)
    threshold: integer like 20
    Returns posterior_p, n_current, alpha, beta.
    """
    n = len(current_games)
    alpha = prior_alpha
    beta = prior_beta

    # i=0 is most recent, so reverse to iterate newest-first
    for i, pts in enumerate(reversed(current_games)):
        w = DECAY_LAMBDA ** i
        if pts >= threshold:
            alpha += w
        else:
            beta += w

    posterior_p = alpha / (alpha + beta)
    return {"posterior_p": posterior_p, "n_current": n, "alpha": alpha, "beta": beta}


# ── WNBAGameLogFetcher ───────────────────────────────────────────────────────

class WNBAGameLogFetcher:
    BASE_URL = "https://stats.wnba.com/stats/playergamelog"
    PLAYER_LOOKUP_URL = "https://stats.wnba.com/stats/commonallplayers"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.wnba.com/",
        "Origin": "https://www.wnba.com",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "x-wnba-stats-origin": "stats",
        "x-wnba-stats-token": "true",
    }
    RATE_LIMIT = 0.65
    TIMEOUT = 30

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)

    def fetch_player_id_map(self, season: str) -> dict[str, int]:
        """Return {normalized_lower_name: player_id} for all WNBA players."""
        params = {"LeagueID": "10", "Season": season, "IsOnlyCurrentSeason": "0"}
        resp = self._session.get(self.PLAYER_LOOKUP_URL, params=params, timeout=self.TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        result_sets = data.get("resultSets", [])
        if not result_sets:
            return {}
        headers = result_sets[0]["headers"]
        rows = result_sets[0]["rowSet"]
        id_idx = headers.index("PERSON_ID")
        name_idx = headers.index("DISPLAY_FIRST_LAST")
        return {normalize_name(row[name_idx]).lower(): row[id_idx] for row in rows}

    def fetch(self, player_id: int, season: str) -> list[dict]:
        """Fetch game logs. Returns chronological list of {date, min, pts} dicts."""
        params = {
            "PlayerID": player_id,
            "Season": season,
            "SeasonType": "Regular Season",
            "LeagueID": "10",
        }
        resp = self._session.get(
            self.BASE_URL, params=params, timeout=self.TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()

        result_sets = data.get("resultSets", [])
        if not result_sets or not result_sets[0]["rowSet"]:
            return []

        headers = result_sets[0]["headers"]
        rows = result_sets[0]["rowSet"]

        date_idx = headers.index("GAME_DATE")
        min_idx = headers.index("MIN")
        pts_idx = headers.index("PTS")

        games = []
        for row in rows:
            games.append({
                "date": row[date_idx],
                "min": self._parse_minutes(row[min_idx]),
                "pts": int(row[pts_idx]) if row[pts_idx] is not None else 0,
            })

        # API returns newest-first; reverse to chronological order
        return list(reversed(games))

    @staticmethod
    def _parse_minutes(raw) -> float:
        if raw is None:
            return 0.0
        s = str(raw)
        if ":" in s:
            parts = s.split(":")
            return float(parts[0]) + float(parts[1]) / 60
        return float(s)


# ── Pipeline ─────────────────────────────────────────────────────────────────

def _compute_league_avg(all_pts: dict[str, list[int]]) -> dict[int, float]:
    """Compute empirical hit rate per threshold across all fetched player games."""
    flat = [pts for games in all_pts.values() for pts in games]
    if not flat:
        return LEAGUE_AVG
    n = len(flat)
    return {t: sum(1 for pts in flat if pts >= t) / n for t in THRESHOLDS}


def run_pipeline(season: str, test_mode: bool, output_path: str) -> None:
    fetcher = WNBAGameLogFetcher()
    roster = ROSTER[:5] if test_mode else ROSTER
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    print(f"Resolving player IDs for season {season}...")
    player_id_map = fetcher.fetch_player_id_map(season)
    time.sleep(fetcher.RATE_LIMIT)

    print(f"Fetching {len(roster)} players...")
    raw_pts: dict[str, list[int]] = {}
    resolved: list[str] = []
    for name in roster:
        key = normalize_name(name).lower()
        player_id = player_id_map.get(key)
        if player_id is None:
            print(f"  {name}: ID not found, skipping")
            continue
        games = fetcher.fetch(player_id, season)
        raw_pts[name] = [g["pts"] for g in games]
        resolved.append(name)
        print(f"  {name}: {len(games)} games")
        time.sleep(fetcher.RATE_LIMIT)

    league_avg = _compute_league_avg(raw_pts)

    output: dict = {"generated_at": now}
    for name in resolved:
        pts_list = raw_pts[name]
        normalized_key = normalize_name(name).lower()

        confidence = None
        thresholds: dict = {}
        for t in THRESHOLDS:
            prior = build_prior(pts_list, t, league_avg)
            if confidence is None:
                confidence = prior["confidence"]
            post = estimate_posterior(pts_list, t, prior["alpha"], prior["beta"])
            thresholds[str(t)] = {
                "p": round(post["posterior_p"], 6),
                "alpha": round(post["alpha"], 4),
                "beta": round(post["beta"], 4),
                "n_current": post["n_current"],
            }

        output[normalized_key] = {
            "thresholds": thresholds,
            "confidence": confidence,
            "generated_at": now,
        }

    validate_output(output, expected_players=len(resolved))

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Written to {output_path}")


def validate_output(data: dict, expected_players: int) -> None:
    """Assert output schema is valid; exits non-zero on failure."""
    player_keys = [k for k in data if k != "generated_at"]

    assert "generated_at" in data, "Missing root-level generated_at"
    assert len(player_keys) == expected_players, (
        f"Expected {expected_players} players, got {len(player_keys)}"
    )

    expected_thresholds = {str(t) for t in THRESHOLDS}
    for key in player_keys:
        player = data[key]
        assert "generated_at" in player, f"{key}: missing generated_at"
        assert "thresholds" in player, f"{key}: missing thresholds"
        actual = set(player["thresholds"].keys())
        assert actual == expected_thresholds, (
            f"{key}: expected thresholds {expected_thresholds}, got {actual}"
        )
        for t_key, entry in player["thresholds"].items():
            p = entry["p"]
            assert 0.0 <= p <= 1.0, f"{key} threshold {t_key}: p={p} outside [0, 1]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute WNBA Beta-Binomial posteriors")
    parser.add_argument("--season", default="2025", help="WNBA season year (default: 2025)")
    parser.add_argument("--test", action="store_true", help="Run against first 5 players only")
    parser.add_argument("--output", default="wnba_data.json", help="Output JSON path")
    args = parser.parse_args()

    run_pipeline(season=args.season, test_mode=args.test, output_path=args.output)


if __name__ == "__main__":
    main()
