# CLAUDE.md

## Hard Constraints — Engineering Principles

**Avoid complexity.** Complexity is obscurity (important info is hidden) or change amplification (one change requires many edits). Before adding anything — a parameter, abstraction, file, or concept — ask: does this make the system easier or harder to understand and modify? If harder, don't add it.

**Always take small, deliberate steps.** Each step does exactly one thing, leaves the system working, and is verifiable before the next step begins. If you need a list to describe the step, it's too big.

**The rate of feedback is your speed limit.** Before starting any step, know: how will I verify this worked? If the answer is "I'll check it later" or "it's hard to test," stop and find a faster feedback path first.

**Never take on a task that's too big.** Decompose first. Find the smallest change that is independently useful and verifiable.

**The best modules are deep.** Simple interface, lots of functionality behind it. Push complexity inward. Deletion test: if you deleted this module, would its complexity disappear (shallow) or reappear across N callers (deep)? Only keep modules that pass.

---

## Comment Rule

Default to no comments. Only add one when the WHY is non-obvious: a hidden constraint, a subtle invariant, a workaround for a specific bug. Never explain what the code does. Never reference the current task or caller.

---

## JS Style Rules

- `const`/`let` only — never `var`
- Arrow functions for callbacks and short utilities
- Template literals for string interpolation
- `async`/`await` for all async — no raw `.then()` chains
- No external libraries — scripts must be self-contained
- Wrap script body in an IIFE: `(function () { 'use strict'; ... })()`

---

## NBA Prop Simulator

Single-file browser SPA (`index.html`). No build step, no bundler, no backend — all deps from CDN. `index-OLD.html` is read-only historical reference.

**ETR projections** — already game-context adjusted (opponent, pace, total, spread, injuries, lineup). Never adjust the mean for any contextual factor — doing so double-counts.

**Simulation routing** — `runSimulation(stats, numSims)`:
- `pts`, `reb`, `ast` → KDE+ (unfiltered `*_raw_all` + `computeEmpiricalStd`) or log-normal fallback
- `stl`, `blk`, `threes` → NegBin when overdispersed; else Poisson
- THREES: always enforce `r ≤ 2.0` regardless of CV data (excess zeros not captured by pure Poisson)
- All output is `Float64Array`

**Edge calculation** — `vigFreeProb(overOdds, underOdds)` when both sides present. Display ¼ Kelly primary; full Kelly secondary reference only.

**Distribution models:**

| Stat | Model |
|---|---|
| PTS/REB/AST | KDE+ / KDE / LogNormal |
| STL/BLK | NegBin / Poisson |
| THREES | NegBin (r≤2 floor) |
| PRA/PR/PA/RA/SB combos | LogNormal (direct CV) |

**`cv_data.json`** — never edit manually (generated nightly by GitHub Actions).
- Keys: NFD unicode-normalized full names. Both sides must normalize or the join silently fails.
- `null` CV = fewer than 5 qualifying games or zero mean — not a missing-data bug.
- `*_raw_all` = unfiltered (MIN≥5). Preferred KDE training data. `*_raw` = situation-filtered, legacy fallback only.
- CV methodology (do not deviate): (1) exclude `MIN<10` or `|game_MIN − trailing_mean| / trailing_mean > 0.25`; (2) per-36 normalize; (3) CV% = stddev/mean × 100, requires ≥5 filtered games.

**UI** — inline styles only (`style={{}}`). JetBrains Mono for numbers, Outfit for labels. Dark theme: bg `#0b0f1a`/`#111827`, cards `rgba(15,23,42,0.6)`, text `#f8fafc`. Calibration journal in `localStorage` key `nbaSimCalibration`.

---

## WNBA Prop Tool

Single-file browser SPA (`wnba.html`). `scripts/compute_wnba.py` → `wnba_data.json`. Same single-file/CDN/no-build pattern as NBA tool. Full spec in memory: `project_wnba_tool.md`.

**Hard constraints:**
- No Kelly sizing — fixed unit staking only.
- No contextual modifiers (pace, opponent, usage) — dropped as uncalibrated.
- Model: Beta-Binomial posterior per player per threshold with λ=0.9 recency decay.
- Odds API thresholds are half-point (19.5); model bins are whole numbers (20+). Map: `bin = point + 0.5`.
- WNBA Stats API: season format `"2025"` (4-digit), headers `Referer: https://www.wnba.com` and `Origin: https://www.wnba.com`.
- Name normalization: NFD strip + `NAME_OVERRIDES` (e.g. `"A'ja Wilson"` → `"Aja Wilson"`).
