#!/usr/bin/env python3
"""
compute_wnba.py — WNBA player points Beta-Binomial model

Pulls game logs from WNBA Stats API and computes posterior hit probabilities
per threshold for use by wnba.html.

Usage:
  python scripts/compute_wnba.py --season 2025   # seed prior from full season
  python scripts/compute_wnba.py                  # current season (2026)
  python scripts/compute_wnba.py --output path/to/wnba_data.json
"""

import unicodedata

# ── Constants ────────────────────────────────────────────────────────────────

THRESHOLDS = [10, 15, 18, 20, 25, 30]
DECAY_LAMBDA = 0.9

# League-average hit rates per threshold (rough empirical estimates)
LEAGUE_AVG = {10: 0.60, 15: 0.40, 18: 0.28, 20: 0.20, 25: 0.10, 30: 0.05}
LEAGUE_STRENGTH = 10  # pseudo-count for league-average prior

NAME_OVERRIDES = {
    "A'ja Wilson": "Aja Wilson",
}


# ── NameNormalizer ────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """NFD decompose, strip combining marks, apply overrides, strip whitespace."""
    stripped = name.strip()
    if stripped in NAME_OVERRIDES:
        stripped = NAME_OVERRIDES[stripped]
    nfd = unicodedata.normalize("NFD", stripped)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


# ── PriorBuilder ─────────────────────────────────────────────────────────────

def build_prior(career_games: list[int], threshold: int) -> dict:
    """
    Given career point totals and a threshold, return (alpha, beta, confidence).

    career_games: list of integer point totals (unordered)
    threshold: integer like 20 (meaning 20+)
    """
    n = len(career_games)
    hits = sum(1 for g in career_games if g >= threshold)

    league_p = LEAGUE_AVG.get(threshold, 0.15)
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
    Returns posterior_p and n_current.
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
    return {"posterior_p": posterior_p, "n_current": n}
