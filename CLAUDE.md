# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Hard Constraints — Engineering Principles

Check all work against these before proceeding. They are not style preferences.

**Avoid complexity.** Complexity is obscurity (important info is hidden) or change amplification (one change requires many edits). Before adding anything — a parameter, abstraction, file, or concept — ask: does this make the system easier or harder to understand and modify? If harder, don't add it.

**Always take small, deliberate steps.** Each step does exactly one thing, leaves the system working, and is verifiable before the next step begins. If you need a list to describe the step, it's too big.

**The rate of feedback is your speed limit.** Before starting any step, know: how will I verify this worked? If the answer is "I'll check it later" or "it's hard to test," stop and find a faster feedback path first. Slow feedback forces guessing. Guessing creates bugs. Bugs create complexity.

**Never take on a task that's too big.** A task is too big when you can't hold the full change in your head, can't describe a clean intermediate state, or the feedback loop spans the entire change. Decompose first. Find the smallest change that is independently useful and verifiable. If you can't decompose it, you don't understand it well enough — understanding it is the first step.

**The best modules are deep.** A deep module has a simple interface and a lot of functionality behind it. A shallow module has an interface nearly as complex as its implementation — it leaks internals and adds complexity. Push complexity inward. Make the interface the smallest surface that still gives callers what they need. Deletion test: if you deleted this module, would its complexity disappear (shallow, not earning its keep) or reappear across N callers (deep, earning its keep)? Only keep modules that pass.

---

## Comment Rule

Default to no comments. Only add one when the WHY is non-obvious: a hidden constraint, a subtle invariant, a workaround for a specific bug. Never explain what the code does — well-named identifiers do that. Never reference the current task, fix, or caller — those belong in the PR description and rot as the codebase evolves.

---

## JS Style Rules

- `const`/`let` only — never `var`
- Arrow functions for callbacks and short utilities
- Template literals for string interpolation
- `async`/`await` for all async — no raw `.then()` chains
- No external libraries — scripts must be self-contained
- Wrap script body in an IIFE: `(function () { 'use strict'; ... })()`

---

## Project: NBA Prop Simulator

Single-file, browser-based static SPA. Monte Carlo simulation of player stat distributions → over/under probabilities → edge vs. book odds → Kelly stake sizing.

**No build step. No bundler. No backend.** All dependencies load from CDN. Do not introduce a build system or separate JS/CSS files.

Everything lives in `index.html`. `index-OLD.html` is a historical reference — do not modify it.

---

## Critical Domain Constraint — ETR Projections

ETR (Establish The Run) projections are **already** game-context adjusted (opponent, pace, total, spread, injuries, lineup). **Never suggest adjusting the ETR projection mean for any contextual factor.** Doing so double-counts adjustments ETR already made and corrupts the model input. The simulator models the distribution *around* the projection — it does not second-guess the projection itself.

---

## Architecture

The entire app is a single `<script type="text/babel">` block in `index.html`, organized into sections marked `// ──`.

**Simulation routing** — `runSimulation(stats, numSims)` routes each stat key to the correct sampler:
- `pts`, `reb`, `ast` → KDE (preferred: unfiltered `*_raw_all` path = "KDE+") or log-normal fallback
- `stl`, `blk`, `threes` → NegBin when CV data active and overdispersed; else Poisson
- All output is `Float64Array`

**THREES NegBin floor** — THREES always enforces minimum overdispersion `r ≤ 2.0` regardless of CV data. CV floor = `sqrt(1/μ + 0.5) × 100`. Pure Poisson for THREES systematically underestimates zero-make games; this floor is non-negotiable.

**KDE+ vs KDE** — when `pts_raw_all` / `reb_raw_all` / `ast_raw_all` are available in `cv_data.json`, the KDE path uses unfiltered raw scores and `computeEmpiricalStd` for `targetStd`. This captures the rate-minutes covariance (stars play more *and* score more in close games) that the Pythagorean `sqrt(cvPer36² + cvMin²)` ignores. Badge shows "KDE+" (green) vs "KDE" (cyan).

**CV Auto-Blend** — `handleLoadCv` blends across all four windows:
```
last5: weight=15 (n=5, recency=3×)
last10: weight=20 (n=10, recency=2×)
last20: weight=30 (n=20, recency=1.5×)
season: weight=N_games (recency=1×)
```
`blendedCV = Σ(weight_i × CV_i) / Σ(weight_i)` — null windows excluded. Effective CV: `sqrt(blendedCvPer36² + blendedCvMin²)`. KDE raw data uses largest available window since `simulateStatKDE` has built-in `LAMBDA=0.9` recency decay.

**Combo props** — direct log-normal simulation using empirical combo CVs from `cv_data.json` (preferred) captures within-game correlation. Falls back to element-wise component summing when combo CV is absent.

**Edge calculation** — always use `vigFreeProb(overOdds, underOdds)` when both sides are present. Edge = model probability − vig-free book implied probability. Display ¼ Kelly as primary stake; full Kelly as secondary reference only.

**Distribution model per stat:**

| Stat | Model | Why |
|---|---|---|
| PTS | KDE+ / KDE / LogNormal | Empirical shape; log-normal materially better on high alt lines |
| REB, AST | KDE+ / KDE / LogNormal | Right-skewed counts; unfiltered empirical std captures covariance |
| STL, BLK | NegBin / Poisson | Discrete; observed CV far exceeds Poisson theory (+20pp STL, +63pp BLK) |
| THREES | NegBin (r≤2 floor) | Shot selection produces excess zeros not captured by Poisson |
| PRA/PR/PA/RA/SB combos | LogNormal (direct CV) | Empirical combo CV captures co-movement with minutes |

---

## `cv_data.json`

Generated nightly by `.github/workflows/update_cv.yml` → **never edit manually.**

Key schema facts:
- JSON key = player full name after NFD unicode normalization (`unicodedata.normalize("NFD", name.strip())` in Python; equivalent in JS). Both sides must apply this or the join fails.
- `null` CV values = fewer than 5 qualifying games or stat mean was 0 — not a missing-data bug.
- `pts_raw_all` / `reb_raw_all` / `ast_raw_all` = **unfiltered** (all games `MIN ≥ 5`). Preferred KDE training data.
- `pts_raw` / `reb_raw` / `ast_raw` = situation-filtered. Legacy fallback only.
- `position` is always `""` — NBA Stats API `PlayerGameLog` does not return position.
- `NAME_OVERRIDES` dict in `compute_cv.py` handles API-vs-ETR name mismatches. Add entries as discovered.

**CV methodology (do not deviate):**
1. Situation filter: exclude `MIN < 10` OR `|game_MIN − trailing_mean_MIN| / trailing_mean_MIN > 0.25`. Trailing mean uses all unfiltered games seen so far.
2. Per-36 normalize: `per36 = (stat / minutes) × 36`.
3. CV% = `(stddev(per36) / mean(per36)) × 100`. Requires ≥ 5 filtered games; returns `null` otherwise.

---

## Running the CV Script

```bash
pip install nba_api          # one-time

# Always validate before full run
python scripts/compute_cv.py --test
python scripts/compute_cv.py --test --output cv_data_test.json

# Full league run (~500 players, 10-15 min, rate-limited ~0.65s/player)
python scripts/compute_cv.py   # writes cv_data.json at repo root
```

No test suite or linter. Manual browser testing for the simulator. The script self-validates output schema via `assert` checks at the end of `main()`.

---

## UI Conventions

- **Inline styles only** — `style={{}}` props. No CSS classes, no stylesheet.
- `'JetBrains Mono', monospace` for all numerical values; `'Outfit', sans-serif` for labels.
- Dark theme: background `#0b0f1a`/`#111827`, cards `rgba(15,23,42,0.6)`, primary text `#f8fafc`.
- Simulations use `Float64Array` for memory efficiency. `DistBar` caps at 5,000 samples for render performance.
- Calibration journal persisted to `localStorage` key `nbaSimCalibration`.
