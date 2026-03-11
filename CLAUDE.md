# CLAUDE.md — NBA Prop Simulator

## Project Overview

This is a **single-file, browser-based NBA prop betting simulator** built as a static SPA. It uses Monte Carlo simulation to model player stat distributions, calculate over/under probabilities, compute edge vs. book odds, and provide Kelly Criterion stake sizing. The entire application lives in `index.html` — there is no build step, no bundler, and no backend.

---

## Repository Structure

```
nba-prop-sims/
├── index.html                  # Active application (single source of truth)
├── index-OLD.html              # Previous version — kept for reference only
├── nba_stddev_library.xlsx     # Std dev reference data (not loaded at runtime)
└── NBA Daily Injury Notes.xlsx # Injury reference data (not loaded at runtime)
```

The `.xlsx` files are reference spreadsheets used by the operator offline — they are **not** parsed or consumed by the application at runtime.

---

## Tech Stack

All dependencies load from CDN. No `npm install` or build process is needed.

| Dependency | Version | Purpose |
|---|---|---|
| React | 18.2.0 | UI rendering |
| ReactDOM | 18.2.0 | DOM mounting |
| Babel Standalone | 7.23.9 | JSX transpilation in-browser |
| Google Fonts (Outfit, JetBrains Mono) | — | Typography |

There is no `package.json`, `node_modules`, or lockfile. Do **not** introduce a build system unless the project explicitly requires one.

---

## Application Architecture

The entire app is a single `<script type="text/babel">` block in `index.html`. Code is organized into clearly commented sections (marked `// ──`):

### 1. Monte Carlo Engine (`index.html:20–67`)
Core simulation logic. No external dependencies.

- **`randn()`** — Box-Muller transform for normally distributed random numbers
- **`simulateStat(mean, stdDev, n)`** — Fills a `Float64Array(n)` with truncated-normal samples (`Math.max(0, ...)` ensures non-negative)
- **`runSimulation(stats, numSims)`** — Iterates over all active stats; returns a map of `{ key → Float64Array }`
- **`calcProbability(simValues, propLine)`** — Returns `{ over, under, push }` fractions
- **`calcCombo(simResults, keys)`** — Sums multiple stat arrays element-wise for combo props

### 2. Odds Utilities (`index.html:69–127`)

- **`probToAmerican(prob)`** — Converts model probability to American odds string
- **`americanToProb(odds)`** — Converts American odds to implied probability (no vig removal)
- **`americanToDecimal(odds)`** — Converts American odds to decimal format for Kelly
- **`calcKelly(modelProb, americanOdds)`** — Returns full Kelly fraction; returns `0` if no edge
- **`fmtOdds(o)`** — Formats American odds with `+` prefix for positives
- **`getEdgeColor(edge)`** — Maps edge percentage to hex color
- **`getEdgeLabel(edge)`** — Maps edge to `STRONG / SOLID / MARGINAL / NO EDGE`
- **`getProbColor(prob)`** — Maps raw probability to hex color

### 3. Std Dev Estimation & Config (`index.html:129–154`)

**`STD_RATIOS`** — Hardcoded CV (coefficient of variation) per stat type:

| Stat | Key | STD_RATIO |
|---|---|---|
| Points | `pts` | 0.30 |
| Rebounds | `reb` | 0.42 |
| Assists | `ast` | 0.40 |
| Steals | `stl` | 0.65 |
| Blocks | `blk` | 0.75 |
| 3-Pointers | `threes` | 0.52 |

`estimateStdDev(statKey, mean)` = `round(mean * STD_RATIOS[statKey], 1)`

**`STAT_CONFIG`** — Array of `{ key, label, defaultMean }` for the 6 individual stats.

**`COMBO_CONFIG`** — Array of `{ key, label, parts[] }` for the 5 combo props:
- `pra` = PTS + REB + AST
- `pr` = PTS + REB
- `pa` = PTS + AST
- `ra` = REB + AST
- `sb` = STL + BLK

### 4. UI Components (`index.html:156–316`)

All components use **inline styles** (no CSS classes, no styled-components).

- **`DistBar`** — 40-bucket histogram rendered as `<div>` bars. Bars left of the prop line are red; bars right are green. Renders up to 5,000 sample points. White vertical line marks the prop line.
- **`EdgeBox`** — Displays book implied probability, edge percentage, edge label, and ¼Kelly stake. Only renders if book odds are provided and edge > 0.
- **`ResultRow`** — Full result card per stat: shows the distribution histogram, OVER/UNDER probability boxes, `EdgeBox` for each side, and a best-side badge if edge ≥ 2%.

### 5. CSV Export (`index.html:318–349`)

`buildCSV(playerName, numSims, statResults, comboResults, timestamp, bankroll)` generates a CSV string with columns:
`Timestamp, Date, Player, Stat, Projection, Std Dev, SD Method, CV %, Prop Line, Book Over Odds, Book Under Odds, Model Over %, Model Under %, Fair Over Odds, Fair Under Odds, Over Edge %, Under Edge %, Best Side, Best Edge %, Edge Rating, Quarter Kelly %, Stake $, Simulations`

### 6. Main App Component (`index.html:354–954`)

`NBASimulator` is the single root React component. Key state:

| State variable | Type | Purpose |
|---|---|---|
| `playerName` | string | Player identifier (display only) |
| `numSims` | number | Simulation count (5,000–100,000) |
| `stats` | object | Per-stat inputs: `{ [key]: { mean, stdDev, propLine, overOdds, underOdds, mode, cvPct } }` |
| `comboLines` | object | Per-combo inputs: `{ [key]: { propLine, overOdds, underOdds } }` |
| `results` | object \| null | Simulation output after "RUN SIMULATION" |
| `bankroll` | string | Optional dollar bankroll for Kelly stake display |
| `simTime` | number | Last simulation duration in milliseconds |

**Std Dev modes per stat** (controlled by `mode` field):
- `"auto"` — Computed as `estimateStdDev(key, mean)` using `STD_RATIOS`
- `"manual"` — User-entered value
- `"cv"` — User-entered CV %; std dev = `mean * (cvPct / 100)`

**Correlated Signal Detection** (`index.html:856–904`): After results are computed, if all three core stats (PTS, REB, AST) show ≥ 2% edge in the same direction (all over or all under), a banner is shown suggesting PRA combo or same-game parlay consideration.

---

## Edge Scale

| Label | Edge Threshold | Color |
|---|---|---|
| STRONG | ≥ 7% | `#22c55e` |
| SOLID | 4–7% | `#4ade80` |
| MARGINAL | 2–4% | `#86efac` |
| NO EDGE | < 2% | `#ef4444` |

Edge = model probability − book implied probability (not vig-adjusted).

---

## Development Workflow

### Running the App
Open `index.html` directly in a browser. No server required.

```bash
open index.html          # macOS
xdg-open index.html      # Linux
```

### Editing
All changes go in `index.html`. The `index-OLD.html` file is a historical reference — do **not** modify it.

### No Tests
There is no test suite. Manual browser testing is the only validation method.

### No Linter / Formatter
There is no ESLint, Prettier, or other tooling configured.

---

## Key Conventions

1. **Single-file architecture** — Keep everything in `index.html`. Do not introduce separate JS/CSS files or a build pipeline unless explicitly requested.
2. **Inline styles only** — All styling uses inline `style={{}}` props. Do not add CSS classes or a stylesheet.
3. **Naming**:
   - `camelCase` for functions and variables
   - `PascalCase` for React components
   - `UPPER_SNAKE_CASE` for constants (`STD_RATIOS`, `STAT_CONFIG`, `COMBO_CONFIG`)
   - Stat keys: `pts`, `reb`, `ast`, `stl`, `blk`, `threes` (lowercase)
4. **Fonts**: Use `'JetBrains Mono', monospace` for all numerical/technical values; `'Outfit', sans-serif` for labels and headings.
5. **Color palette** (dark theme):
   - Background: `#0b0f1a` → `#111827`
   - Card backgrounds: `rgba(15,23,42,0.6)`
   - Muted text: `#475569`, `#64748b`, `#94a3b8`
   - Primary text: `#f8fafc`, `#e2e8f0`
   - Green (positive): `#22c55e`, `#4ade80`, `#86efac`
   - Red (negative): `#ef4444`, `#fca5a5`
6. **Performance**: Simulations use `Float64Array` for memory efficiency. `DistBar` caps rendering at 5,000 of the simulation samples for display performance.
7. **Kelly sizing**: Always display ¼ Kelly (quarter Kelly), never full Kelly, as the primary stake recommendation. Full Kelly is shown as a secondary reference.
8. **No vig removal**: `americanToProb()` returns raw implied probability including vig. Edge calculations do not remove the vig from book odds — this is intentional.

---

## Git Branch

Active development branch: `claude/claude-md-mmmj6bvckghzkqfb-JpeTp`

Commit messages should be descriptive (e.g., `Add correlation banner for PRA signals`, not `Update index.html`).
