#!/usr/bin/env python3
"""
validate_distribution.py — Validates distribution choice for NBA player points props.

Tests whether truncated normal, log-normal, or gamma best fits empirical
NBA scoring distributions for volume scorers in the 15–22 PPG range.

Three independent tests:
  1. Empirical skewness vs. each distribution's imposed skewness
  2. Empirical median vs. each distribution's predicted median
  3. Near-mean prop line hit-rate calibration: model P(X > line) vs. empirical rate

Usage:
    python scripts/validate_distribution.py

Output: console table + validate_distribution_results.json
"""

import json
import math
import sys
import time
import unicodedata
from statistics import median as stat_median

from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.static import players as nba_players

# ── Config ────────────────────────────────────────────────────────────────────

SEASON = "2025-26"
REQUEST_DELAY = 0.65
TARGET_N = 20          # stop after collecting this many qualifying players
PPG_MIN = 15.0
PPG_MAX = 22.0
MIN_GAMES = 20         # require at least 20 filtered games for reliable stats

# Broad candidate list — actual season avg unknown at script-write time,
# so we fetch and filter. Over-provisioned intentionally.
CANDIDATE_PLAYERS = [
    "Dejounte Murray", "Darius Garland", "Zach LaVine", "Jordan Clarkson",
    "Mikal Bridges", "Brandon Ingram", "Khris Middleton", "Pascal Siakam",
    "OG Anunoby", "Cam Johnson", "Coby White", "Tyler Herro",
    "Tobias Harris", "Terry Rozier", "Jaylen Brown", "Tyrese Maxey",
    "De'Aaron Fox", "Jalen Brunson", "Scottie Barnes", "Franz Wagner",
    "Immanuel Quickley", "Alperen Sengun", "Evan Mobley", "Cade Cunningham",
    "Jimmy Butler", "Bam Adebayo", "Trae Young", "Donovan Mitchell",
    "Julius Randle", "Naz Reid", "Anfernee Simons", "Derrick White",
    "Keldon Johnson", "Cam Thomas", "Norman Powell", "Jalen Williams",
    "Josh Hart", "Spencer Dinwiddie", "Buddy Hield", "Luke Kennard",
    "Malik Monk", "Dillon Brooks", "Marcus Smart", "Desmond Bane",
    "Kyle Kuzma", "Kristaps Porzingis", "Brook Lopez", "Jaren Jackson Jr.",
    "Jabari Smith Jr.", "Keegan Murray", "Tre Mann", "Nikola Vucevic",
    "Andrew Wiggins", "Klay Thompson", "RJ Barrett", "Obi Toppin",
    "Jonathan Kuminga", "Jaden McDaniels", "Anthony Black", "Dyson Daniels",
]


# ── Pure-Python Statistics ────────────────────────────────────────────────────

def normal_cdf(z: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _gamma_series(a: float, x: float) -> float:
    """Series expansion of regularized lower incomplete gamma P(a, x). x < a+1."""
    ITMAX, EPS = 200, 3e-7
    ap = a
    total = delta = 1.0 / a
    for _ in range(ITMAX):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a: float, x: float) -> float:
    """Continued fraction of regularized upper incomplete gamma Q(a, x). x >= a+1."""
    ITMAX, EPS, FPMIN = 200, 3e-7, 1e-300
    b = x + 1.0 - a
    c = 1.0 / FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, ITMAX + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < FPMIN:
            d = FPMIN
        c = b + an / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gamma_inc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function P(a, x) = γ(a,x)/Γ(a)."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        return _gamma_series(a, x)
    return 1.0 - _gamma_cf(a, x)


def sample_skewness(values: list[float]) -> float | None:
    """Adjusted Fisher-Pearson skewness coefficient (same as Excel SKEW)."""
    n = len(values)
    if n < 3:
        return None
    mu = sum(values) / n
    sigma2 = sum((x - mu) ** 2 for x in values) / (n - 1)
    if sigma2 == 0.0:
        return None
    sigma = math.sqrt(sigma2)
    raw = sum((x - mu) ** 3 for x in values) / n / (sigma ** 3)
    return raw * math.sqrt(n * (n - 1)) / (n - 2)


def sample_std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = sum(values) / n
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (n - 1))


# ── Distribution Predictions ──────────────────────────────────────────────────

def tn_over(line: float, mu: float, sigma: float) -> float:
    """Truncated-normal P(X > line | X >= 0)."""
    p_over_line = 1.0 - normal_cdf((line - mu) / sigma)
    p_above_zero = 1.0 - normal_cdf(-mu / sigma)
    return p_over_line / p_above_zero if p_above_zero > 0 else 0.0


def tn_median(mu: float, sigma: float) -> float:
    """Median of TN(mu, sigma, 0, inf) via bisection on the CDF."""
    lo, hi = 0.0, mu + 10 * sigma
    norm_factor = 1.0 - normal_cdf(-mu / sigma)
    target = 0.5 * norm_factor  # 50th percentile of normalised dist
    for _ in range(60):
        mid = (lo + hi) / 2
        cdf_mid = 1.0 - normal_cdf((mid - mu) / sigma)  # P(X > mid) unnormalized
        p_below = norm_factor - cdf_mid
        if p_below < target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def tn_skewness(_mu: float, _sigma: float) -> float:
    """Truncated normal is very nearly symmetric for mu >> sigma; returns 0."""
    return 0.0  # exact value negligible when alpha < -2.5


def ln_params(mu: float, sigma: float) -> tuple[float, float]:
    """Log-normal (mu_L, sigma_L) from moment-matched mean and std dev."""
    cv = sigma / mu
    sigma_L = math.sqrt(math.log(1.0 + cv ** 2))
    mu_L = math.log(mu) - sigma_L ** 2 / 2.0
    return mu_L, sigma_L


def ln_over(line: float, mu: float, sigma: float) -> float:
    """Log-normal P(X > line)."""
    mu_L, sigma_L = ln_params(mu, sigma)
    z = (math.log(line) - mu_L) / sigma_L
    return 1.0 - normal_cdf(z)


def ln_median(mu: float, sigma: float) -> float:
    """Log-normal median = exp(mu_L)."""
    mu_L, _ = ln_params(mu, sigma)
    return math.exp(mu_L)


def ln_skewness(sigma: float, mu: float) -> float:
    """Log-normal skewness = (exp(sigma_L^2)+2)*sqrt(exp(sigma_L^2)-1)."""
    cv = sigma / mu
    sigma_L2 = math.log(1.0 + cv ** 2)
    e = math.exp(sigma_L2)
    return (e + 2.0) * math.sqrt(e - 1.0)


def gamma_params(mu: float, sigma: float) -> tuple[float, float]:
    """Gamma (alpha, beta) from moment-matched mean and std dev."""
    beta = sigma ** 2 / mu
    alpha = mu / beta
    return alpha, beta


def gamma_over(line: float, mu: float, sigma: float) -> float:
    """Gamma P(X > line)."""
    alpha, beta = gamma_params(mu, sigma)
    return 1.0 - gamma_inc(alpha, line / beta)


def gamma_median_approx(mu: float, sigma: float) -> float:
    """Gamma median via Wilson-Hilferty approximation."""
    alpha, beta = gamma_params(mu, sigma)
    return alpha * beta * (1.0 - 1.0 / (9.0 * alpha)) ** 3


def gamma_skewness(mu: float, sigma: float) -> float:
    """Gamma skewness = 2 / sqrt(alpha)."""
    alpha, _ = gamma_params(mu, sigma)
    return 2.0 / math.sqrt(alpha)


# ── NBA Data Fetch ────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFD", name.strip())


def parse_minutes(val) -> float:
    if val is None:
        return 0.0
    s = str(val).strip()
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


def situation_filter(games: list[dict]) -> list[dict]:
    """Same filter as compute_cv.py: exclude min<10 and >25% deviation from trailing mean."""
    filtered = []
    all_mins: list[float] = []
    for g in games:
        m = g["min"]
        if m < 10.0:
            all_mins.append(m)
            continue
        if all_mins:
            trail = sum(all_mins) / len(all_mins)
            if trail > 0 and abs(m - trail) / trail > 0.25:
                all_mins.append(m)
                continue
        all_mins.append(m)
        filtered.append(g)
    return filtered


def fetch_game_pts(player_id: int) -> list[float] | None:
    """
    Fetch situation-filtered raw points per game for a player.
    Returns list of raw game-by-game points (NOT per-36).
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
        print(f"    ERROR: {exc}", file=sys.stderr)
        return None

    if df.empty:
        return None

    raw = []
    for _, row in df.iterrows():
        raw.append({"min": parse_minutes(row.get("MIN")), "pts": float(row.get("PTS") or 0)})
    raw.reverse()  # chronological

    filtered = situation_filter(raw)
    if len(filtered) < MIN_GAMES:
        return None

    return [g["pts"] for g in filtered]


# ── Per-Player Analysis ───────────────────────────────────────────────────────

def analyze_player(name: str, pts: list[float]) -> dict:
    """
    Given a list of game-by-game raw points, compute all validation metrics.
    Prop lines tested: [mean-1.5, mean-0.5, mean+0.5, mean+1.5]
    """
    n = len(pts)
    mu = sum(pts) / n
    sigma = sample_std(pts)
    emp_median = stat_median(pts)
    emp_skew = sample_skewness(pts)

    # Predicted medians
    pred_med = {
        "tn":    tn_median(mu, sigma),
        "ln":    ln_median(mu, sigma),
        "gamma": gamma_median_approx(mu, sigma),
    }

    # Imposed skewness
    pred_skew = {
        "tn":    tn_skewness(mu, sigma),
        "ln":    ln_skewness(sigma, mu),
        "gamma": gamma_skewness(mu, sigma),
    }

    # Hit-rate calibration at 4 prop lines
    lines = [mu - 1.5, mu - 0.5, mu + 0.5, mu + 1.5]
    calibration = []
    for line in lines:
        emp_rate = sum(1 for p in pts if p > line) / n
        pred = {
            "tn":    tn_over(line, mu, sigma),
            "ln":    ln_over(line, mu, sigma),
            "gamma": gamma_over(line, mu, sigma),
        }
        calibration.append({
            "line_offset": round(line - mu, 1),
            "line":        round(line, 1),
            "empirical":   round(emp_rate, 4),
            "tn":          round(pred["tn"], 4),
            "ln":          round(pred["ln"], 4),
            "gamma":       round(pred["gamma"], 4),
            "err_tn":      round(pred["tn"] - emp_rate, 4),
            "err_ln":      round(pred["ln"] - emp_rate, 4),
            "err_gamma":   round(pred["gamma"] - emp_rate, 4),
        })

    return {
        "player":      name,
        "n_games":     n,
        "mean":        round(mu, 2),
        "std":         round(sigma, 2),
        "cv_pct":      round(sigma / mu * 100, 1),
        "emp_median":  round(emp_median, 2),
        "emp_skew":    round(emp_skew, 3) if emp_skew is not None else None,
        "pred_median": {k: round(v, 2) for k, v in pred_med.items()},
        "pred_skew":   {k: round(v, 3) for k, v in pred_skew.items()},
        "calibration": calibration,
    }


# ── Aggregate Scoring ─────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    """
    Compute aggregate statistics across all players:
    - Mean empirical skewness vs. each distribution's skewness
    - Mean absolute error (MAE) on empirical median
    - MAE on hit-rate calibration for each offset
    """
    emp_skews = [r["emp_skew"] for r in results if r["emp_skew"] is not None]
    pred_skews_ln    = [r["pred_skew"]["ln"]    for r in results]
    pred_skews_gamma = [r["pred_skew"]["gamma"] for r in results]

    mean_emp_skew   = sum(emp_skews) / len(emp_skews) if emp_skews else None
    mean_ln_skew    = sum(pred_skews_ln) / len(pred_skews_ln)
    mean_gamma_skew = sum(pred_skews_gamma) / len(pred_skews_gamma)

    # Median MAE
    med_err = {"tn": [], "ln": [], "gamma": []}
    for r in results:
        emp = r["emp_median"]
        for dist in med_err:
            med_err[dist].append(abs(r["pred_median"][dist] - emp))
    med_mae = {d: round(sum(v) / len(v), 3) for d, v in med_err.items()}

    # Calibration MAE by offset
    offsets = [-1.5, -0.5, 0.5, 1.5]
    cal_mae = {off: {"tn": [], "ln": [], "gamma": []} for off in offsets}
    for r in results:
        for c in r["calibration"]:
            off = c["line_offset"]
            for dist in ("tn", "ln", "gamma"):
                cal_mae[off][dist].append(abs(c[f"err_{dist}"]))

    cal_mae_summary = {}
    for off in offsets:
        cal_mae_summary[off] = {
            d: round(sum(v) / len(v), 4) for d, v in cal_mae[off].items()
        }

    # Overall MAE across all offsets
    all_errs = {"tn": [], "ln": [], "gamma": []}
    for r in results:
        for c in r["calibration"]:
            for dist in all_errs:
                all_errs[dist].append(abs(c[f"err_{dist}"]))
    overall_mae = {d: round(sum(v) / len(v), 4) for d, v in all_errs.items()}

    return {
        "n_players":           len(results),
        "skewness": {
            "empirical_mean":  round(mean_emp_skew, 3) if mean_emp_skew else None,
            "ln_imposed_mean": round(mean_ln_skew, 3),
            "gamma_imposed_mean": round(mean_gamma_skew, 3),
            "tn_imposed":      0.0,
        },
        "median_mae":          med_mae,
        "calibration_mae_by_offset": cal_mae_summary,
        "overall_calibration_mae":   overall_mae,
    }


# ── Pretty Print ──────────────────────────────────────────────────────────────

def print_results(results: list[dict], agg: dict) -> None:
    W = 110
    print("\n" + "=" * W)
    print("NBA POINTS DISTRIBUTION VALIDATION — Per-Player Results")
    print("=" * W)
    hdr = f"{'Player':<26} {'N':>4} {'Mean':>6} {'CV%':>5} {'EmpSkew':>8} {'LnSkew':>8} {'GamSkew':>8} | {'EmpMed':>7} {'TN-Med':>7} {'LN-Med':>7} {'GA-Med':>7}"
    print(hdr)
    print("-" * W)
    for r in results:
        ps = r["pred_skew"]
        pm = r["pred_median"]
        es = f"{r['emp_skew']:+.3f}" if r["emp_skew"] is not None else "  N/A "
        print(
            f"{r['player']:<26} {r['n_games']:>4} {r['mean']:>6.1f} {r['cv_pct']:>5.1f}"
            f" {es:>8} {ps['ln']:>8.3f} {ps['gamma']:>8.3f}"
            f" | {r['emp_median']:>7.1f} {pm['tn']:>7.2f} {pm['ln']:>7.2f} {pm['gamma']:>7.2f}"
        )

    print("\n" + "=" * W)
    print("CALIBRATION: Mean Absolute Error (model predicted% − empirical hit rate)")
    print("             Smaller = better. Tested at 4 prop lines relative to each player's mean.")
    print("=" * W)
    hdr2 = f"{'Player':<26} {'Line':>8} | {'Empirical':>10} {'TN pred':>8} {'LN pred':>8} {'GA pred':>8} | {'TN err':>7} {'LN err':>7} {'GA err':>7}"
    print(hdr2)
    print("-" * W)
    for r in results:
        first = True
        for c in r["calibration"]:
            name_col = r["player"] if first else ""
            first = False
            off_label = f"mean{c['line_offset']:+.1f}"
            print(
                f"{name_col:<26} {off_label:>8} | "
                f"{c['empirical']:>10.1%} {c['tn']:>8.1%} {c['ln']:>8.1%} {c['gamma']:>8.1%} | "
                f"{c['err_tn']:>+7.1%} {c['err_ln']:>+7.1%} {c['err_gamma']:>+7.1%}"
            )
        print()

    print("=" * W)
    print("AGGREGATE SUMMARY")
    print("=" * W)
    sk = agg["skewness"]
    print(f"\nSKEWNESS (n={agg['n_players']} players):")
    print(f"  Empirical mean skewness :  {sk['empirical_mean']}")
    print(f"  Log-normal imposed mean :  {sk['ln_imposed_mean']}   ← fixed by CV, no free parameter")
    print(f"  Gamma imposed mean      :  {sk['gamma_imposed_mean']}   ← fixed by CV, no free parameter")
    print(f"  Truncated normal        :  {sk['tn_imposed']}   ← symmetric")

    print(f"\nMEDIAN MAE (predicted vs. empirical, absolute points):")
    mm = agg["median_mae"]
    winner = min(mm, key=mm.get)
    for d, v in mm.items():
        tag = " ← BEST" if d == winner else ""
        print(f"  {d.upper():>5}: {v:.3f} pts{tag}")

    print(f"\nCALIBRATION MAE by prop line offset (percentage points):")
    best_overall = min(agg["overall_calibration_mae"], key=agg["overall_calibration_mae"].get)
    cal = agg["calibration_mae_by_offset"]
    print(f"  {'Offset':<10} {'TN':>8} {'LN':>8} {'GAMMA':>8}")
    for off in sorted(cal.keys()):
        row = cal[off]
        best = min(row, key=row.get)
        tags = {d: " *" if d == best else "  " for d in row}
        print(f"  {f'mean{off:+.1f}':<10} {row['tn']:>7.2%}{tags['tn']} {row['ln']:>7.2%}{tags['ln']} {row['gamma']:>7.2%}{tags['gamma']}")

    print(f"\n  OVERALL MAE (all offsets combined):")
    om = agg["overall_calibration_mae"]
    for d, v in om.items():
        tag = " ← BEST" if d == best_overall else ""
        print(f"    {d.upper():>5}: {v:.2%}{tag}")

    print("\n" + "=" * W)
    print("VERDICT")
    print("=" * W)
    emp_sk = sk["empirical_mean"]
    if emp_sk is not None:
        diffs = {
            "TN":    abs(0.0 - emp_sk),
            "LN":    abs(sk["ln_imposed_mean"] - emp_sk),
            "GAMMA": abs(sk["gamma_imposed_mean"] - emp_sk),
        }
        skew_winner = min(diffs, key=diffs.get)
        print(f"\n  Skewness closest to empirical : {skew_winner}")
        print(f"    TN  |0.0 − {emp_sk:.3f}|   = {diffs['TN']:.3f}")
        print(f"    LN  |{sk['ln_imposed_mean']:.3f} − {emp_sk:.3f}|  = {diffs['LN']:.3f}")
        print(f"    GAMMA |{sk['gamma_imposed_mean']:.3f} − {emp_sk:.3f}|  = {diffs['GAMMA']:.3f}")

    print(f"\n  Calibration winner (lowest MAE across all lines): {best_overall.upper()}")
    print(f"  Median prediction winner: {winner.upper()}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    all_active = nba_players.get_active_players()
    name_to_id = {normalize_name(p["full_name"]): p["id"] for p in all_active}

    results: list[dict] = []
    attempted = 0

    print(f"Targeting {TARGET_N} players in {PPG_MIN}–{PPG_MAX} PPG range")
    print(f"Season: {SEASON}  |  Min qualifying games: {MIN_GAMES}")
    print("-" * 60)

    for raw_name in CANDIDATE_PLAYERS:
        if len(results) >= TARGET_N:
            break

        norm = normalize_name(raw_name)
        pid = name_to_id.get(norm)
        if pid is None:
            print(f"  SKIP (not found in active roster): {raw_name}")
            continue

        attempted += 1
        print(f"  [{attempted}] {raw_name} ... ", end="", flush=True)

        pts = fetch_game_pts(pid)

        if pts is None:
            print(f"SKIP (<{MIN_GAMES} qualifying games)")
            time.sleep(REQUEST_DELAY)
            continue

        mu = sum(pts) / len(pts)
        if not (PPG_MIN <= mu <= PPG_MAX):
            print(f"SKIP (mean={mu:.1f} outside {PPG_MIN}–{PPG_MAX})")
            time.sleep(REQUEST_DELAY)
            continue

        result = analyze_player(raw_name, pts)
        results.append(result)
        print(f"OK  mean={result['mean']}  cv={result['cv_pct']}%  n={result['n_games']}  skew={result['emp_skew']}")

        if len(results) < TARGET_N:
            time.sleep(REQUEST_DELAY)

    if not results:
        print("ERROR: No qualifying players found. Check API connectivity.", file=sys.stderr)
        sys.exit(1)

    print(f"\nCollected {len(results)} qualifying players.")

    agg = aggregate(results)
    print_results(results, agg)

    # Write JSON output
    output = {"players": results, "aggregate": agg}
    out_path = "validate_distribution_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results written to: {out_path}")


if __name__ == "__main__":
    main()
