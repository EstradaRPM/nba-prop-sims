# NBA Prop Simulator — Roadmap

Items queued for future sprints, in rough priority order. None of these are in scope until explicitly scheduled.

---

## Queue

### 1. Vig Removal for True Edge Calculation
**Problem**: `americanToProb()` returns raw implied probability including vig. Edge = model prob − vig-inflated book prob. This systematically understates true edge (e.g., at -110/-110, raw implied = 52.4% each; vig-free = 50.0%). A player with a model prob of 54% has a true edge of 4pp, but the current display shows 1.6pp.

**Approach**: Add `vigFreeProb(overOdds, underOdds)` that normalizes both sides to sum to 1.0. Show vig-free edge as the primary edge signal; retain raw edge as secondary reference. Update `getEdgeLabel` thresholds accordingly (vig-free thresholds should be slightly lower since the signal is cleaner).

**Impact**: High — changes all edge numbers. Needs careful threshold recalibration before shipping.

---

### 2. Projection Uncertainty Layer
**Problem**: ETR projections have their own error (typically ±2–3 pts for PTS, ±1 for REB/AST). Currently the model treats the ETR mean as ground truth. The true predictive distribution should reflect both projection error and performance variance.

**Approach**: Hierarchical sampling — before each KDE/parametric trial, draw a mean from `N(etrMean, σ_proj)` where `σ_proj` is estimated from ETR RMSE data. This widens the output distribution appropriately and reduces spurious "STRONG" edges on borderline props.

**Impact**: Medium — modest change to individual player results; increases model honesty.

---

### 3. Correlated Parlay EV Calculator
**Problem**: The correlated signal banner flags when PTS/REB/AST all show edge in the same direction but doesn't compute parlay EV.

**Approach**: Extend the banner to compute two-leg and three-leg parlay true probability using the existing `Float64Array` simulation results (count trials where all legs hit simultaneously). Compare to parlay book odds if available. Uses the same simulation outputs — zero additional sampling cost.

**Impact**: Medium — adds direct parlay EV output for correlated plays.

---

### 4. CLV Pipeline (Closing Line Value Tracking)
**Problem**: No ability to track whether model prices beat the close, which is the gold-standard proxy for long-term edge.

**Approach**:
1. The Odds API exposes historical odds via `/v4/historical/sports/{sport}/events/{eventId}/odds`. Closing line = snapshot closest to game time.
2. Add a "Log Bet" button in the batch results row that saves the bet to `localStorage` (player, stat, side, model odds, book odds, timestamp, game date).
3. Add a "CLV Review" panel that fetches historical closing odds for logged bets and computes: CLV% = (model_implied − closing_implied) / closing_implied.
4. Export CLV log to CSV for Google Sheets integration.

**Complexity**: High — requires event ID resolution, historical API calls (separate quota), and persistent state management.

---

### 5. Batch Portfolio Kelly
**Problem**: Current batch EV hunt shows isolated ¼Kelly per prop, ignoring that simultaneous bets compete for the same bankroll. True simultaneous Kelly allocates bankroll proportional to edge magnitude across all positive-EV plays.

**Approach**: After batch simulation, collect all positive-EV rows, compute simultaneous Kelly weights: `w_i = kelly_i / Σ kelly_j`. Display adjusted stake per bet. Needs bankroll input (already in state).

**Impact**: Medium — changes Kelly display in batch only. Individual player modal already has portfolio Kelly.

---

### 6. Minutes-Based Line Adjustment
**Problem**: When a player's implied minutes (from Vegas O/U, pace, or injury reports) differ from ETR's projection, the mean input should be scaled accordingly.

**Approach**: Use `mean_minutes_last20` from CV database and the game O/U pace implied minutes. If pace implies 29 min but CV database shows player averages 24 min, offer a one-click adjustment: scale all projections by (implied_min / historical_mean_min).

**Impact**: Low-medium — useful for specific situations (pace mismatch games, lineup changes).

---

## Notes

- Items are independent and can be scheduled in any order.
- Items 1 (vig removal) and 3 (parlay EV) are the highest mathematical value adds.
- Item 4 (CLV pipeline) is the highest operational value for long-term edge tracking.
- Always run `--test` before `--full` on `compute_cv.py` after any schema changes.
