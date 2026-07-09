#!/usr/bin/env bash
#
# setup_github.sh
# Updates the 12 open issues with expanded bodies, creates labels + milestones,
# and opens a PR that updates the README. Run from inside your cloned repo.
#
# Prereqs:
#   - GitHub CLI installed:  https://cli.github.com
#   - Authenticated:         gh auth login
#   - Run from the repo root (git clone ... && cd futures_trend_following)
#
# Usage:
#   chmod +x setup_github.sh
#   ./setup_github.sh
#
set -euo pipefail

REPO="AJYS-Arc/futures_trend_following"

echo "==> Checking prerequisites"
command -v gh  >/dev/null || { echo "Install GitHub CLI: https://cli.github.com"; exit 1; }
command -v git >/dev/null || { echo "git is required"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Run: gh auth login"; exit 1; }

# --------------------------------------------------------------------------
# 1. Labels (idempotent: create, or update if it already exists)
# --------------------------------------------------------------------------
echo "==> Creating / updating labels"
mklabel() {
  gh label create "$1" --repo "$REPO" --color "$2" --description "$3" 2>/dev/null \
    || gh label edit "$1" --repo "$REPO" --color "$2" --description "$3" 2>/dev/null \
    || true
}
mklabel data     0366d6 "Data pipeline"
mklabel setup    5319e7 "Project setup"
mklabel signals  0e8a16 "Signal calculation"
mklabel strategy d93f0b "Strategy construction"
mklabel backtest fbca04 "Backtesting"
mklabel analysis 1d76db "Analysis & metrics"
mklabel docs     c5def5 "Documentation"

# --------------------------------------------------------------------------
# 2. Milestones (idempotent)
# --------------------------------------------------------------------------
echo "==> Creating milestones"
mkmilestone() { gh api "repos/$REPO/milestones" -f title="$1" >/dev/null 2>&1 || true; }
M1="Phase 1 - Data Foundation"
M2="Phase 2 - Signals & Strategies"
M3="Phase 3 - Backtest, Metrics & Delivery"
mkmilestone "$M1"
mkmilestone "$M2"
mkmilestone "$M3"

# --------------------------------------------------------------------------
# 3. Write expanded issue bodies to temp files
# --------------------------------------------------------------------------
echo "==> Preparing issue bodies"
TMP="$(mktemp -d)"

cat > "$TMP/2.md" <<'EOF'
Select the specific CME contracts we will trade and assign each to a sector. This is the foundation every later step depends on.

**Steps**
1. Create a universe config at `config/universe.yaml` (or `src/universe.py`).
2. Pick ~2-4 liquid contracts per sector (suggested roots, all available on Databento `GLBX.MDP3`):
   - **Equity indices:** `ES`, `NQ`, `YM`, `RTY`
   - **Interest rates:** `ZT` (2Y), `ZF` (5Y), `ZN` (10Y), `ZB` (30Y)
   - **Currencies (FX):** `6E` (EUR), `6J` (JPY), `6B` (GBP), `6C` (CAD)
   - **Energy:** `CL` (WTI), `NG` (nat gas), `HO` (heating oil), `RB` (gasoline)
   - **Metals:** `GC` (gold), `SI` (silver), `HG` (copper)
   - **Agriculture:** `ZC` (corn), `ZS` (soybeans), `ZW` (wheat), `LE` (live cattle)
3. For each root record: `symbol`, `sector`, `exchange`, `multiplier`, `tick_size`, `currency`.
4. Confirm each root is available under the course's Databento CME license before committing.
5. Save the config with a clear `sector` field so downstream code can group by sector.
6. Add a short "Futures Universe" table to the README/docs.

**Acceptance criteria:** A committed config lists every symbol -> sector + contract specs, and availability is verified.
**Depends on:** none
EOF

cat > "$TMP/3.md" <<'EOF'
Stand up the project environment and a working authenticated Databento client.

**Steps**
1. Add project scaffolding if not present: `src/`, `data/` (git-ignored), `results/`, `tests/`.
2. Create `requirements.txt` with: `databento`, `pandas`, `numpy`, `pyarrow`, `python-dotenv`, `matplotlib`.
3. Obtain a Databento API key. Copy `.env.example` -> `.env` and set `DATABENTO_API_KEY=...`. Confirm `.env` is in `.gitignore`.
4. Create `src/data/client.py` that loads the key with `python-dotenv` and instantiates `databento.Historical(key)`.
5. Write a smoke test: fetch one day of `ES` daily OHLCV to confirm auth works.
6. Document the dataset (`GLBX.MDP3`) and schemas: `ohlcv-1d` for prices, `definition` for contract metadata.

**Acceptance criteria:** `python -m src.data.client` prints a small dataframe using the key from `.env`; key is not committed.
**Depends on:** none (pairs with #2)
EOF

cat > "$TMP/4.md" <<'EOF'
Pull and cache clean OHLCV data for the whole universe.

**Steps**
1. Create `src/data/download.py`.
2. Decide symbology: Databento **continuous** front-month (e.g. `ES.c.0`, `stype_in="continuous"`) or **raw** individual contracts (`stype_in="raw_symbol"`).
3. For each root, call `client.timeseries.get_range(dataset="GLBX.MDP3", symbols=[...], schema="ohlcv-1d", start=..., end=...)`.
4. Also pull the `definition` schema to capture expiration dates and metadata.
5. Convert to a DataFrame (`.to_df()`), keep `ts_event/open/high/low/close/volume`, sort by time, drop duplicates.
6. **Gotcha:** Databento raw prices are fixed-point (scaled by 1e-9). Use the client's dataframe price conversion so prices are in real units.
7. Cache each symbol to `data/raw/<symbol>.parquet`; skip re-download if the file already exists.

**Acceptance criteria:** A cached parquet per symbol with clean, correctly-scaled daily OHLCV in a common timezone.
**Depends on:** #2, #3
EOF

cat > "$TMP/5.md" <<'EOF'
Turn expiring contracts into continuous, tradable return series without spurious jumps at rolls.

**Steps**
1. Create `src/data/roll.py`.
2. Choose a roll rule: (a) Databento pre-rolled continuous front-month; (b) volume/open-interest crossover; (c) calendar rule (roll N business days before expiry). Document the choice.
3. If rolling manually: order each root's contracts by expiry, compute the roll date, and stitch the front-month series.
4. Compute returns from the **held** contract each day, then chain them - do NOT take a raw percentage change across the roll price gap. Equivalent: back-adjust (ratio/Panama) to remove roll jumps.
5. Prefer building a continuous **return** series ("hold then roll") for signals rather than a back-adjusted price level.
6. Validate: plot returns around 2-3 known roll dates and confirm there is no artificial spike.

**Acceptance criteria:** A continuous, roll-consistent daily return series for every root, validated at known roll dates.
**Depends on:** #4
EOF

cat > "$TMP/6.md" <<'EOF'
Compute the trend signal used by both strategies.

**Steps**
1. Create `src/signals/trend.py`.
2. Build an aligned panel: `dates x symbols` of daily returns (from #5) on a common calendar.
3. Trailing return over lookback `L` (start with `L = 252` trading days; also try 60 and 120).
4. Trailing volatility: rolling std of daily returns x sqrt(252), or an EWMA vol.
5. **Trend Score = trailing return / trailing volatility.** Parameterize `L`; optionally average across multiple lookbacks.
6. Enforce a minimum warm-up (`min_periods`) and leave `NaN` until enough history exists.
7. Add a unit test: a synthetic rising series -> positive score; a flat series -> ~0.

**Acceptance criteria:** A `dates x symbols` trend-score matrix is produced and the unit test passes.
**Depends on:** #5
EOF

cat > "$TMP/7.md" <<'EOF'
Rank within each sector, go long the strongest trends and short the weakest, keeping sector exposure balanced.

**Steps**
1. Create `src/strategies/sector_neutral.py`.
2. On each rebalance date, group symbols by sector.
3. Within each sector, rank symbols by trend score.
4. Go long the top-k and short the bottom-k (or top/bottom tercile). Weights: equal-weight or inverse-volatility (`w ~ 1/vol`).
5. Balance each sector so long $ = short $ (dollar-neutral), and weight sectors equally.
6. Output a target-weight matrix `dates x symbols`; net exposure per sector should be ~0.
7. Edge case: if a sector has too few names, reduce `k` or skip that sector for that date.

**Acceptance criteria:** A balanced target-weight matrix where per-sector net exposure ~0.
**Depends on:** #6
EOF

cat > "$TMP/8.md" <<'EOF'
Evaluate each contract independently: long positive trends, short negative trends.

**Steps**
1. Create `src/strategies/absolute.py`.
2. Position sign = `sign(trend_score)`: long if > 0, short if < 0. Optionally add a small dead-zone near 0.
3. Size positions equal-weight or inverse-volatility, then normalize gross exposure to a target (e.g. `sum(|w|) = 1`).
4. Optional: apply portfolio volatility targeting so this book's risk matches the sector-neutral book (fair comparison).
5. Output a signed target-weight matrix `dates x symbols`; sector exposure is allowed to drift over time.

**Acceptance criteria:** A signed weight matrix with gross exposure / vol comparable to the sector-neutral strategy.
**Depends on:** #6
EOF

cat > "$TMP/9.md" <<'EOF'
Apply target weights over time and produce portfolio returns for both strategies through one shared engine.

**Steps**
1. Create `src/backtest/engine.py`.
2. Inputs: a target-weight matrix, the returns panel, and a rebalance frequency.
3. **Lag weights by one period** to avoid look-ahead: today's return uses yesterday's signal (`w.shift(1)`).
4. Portfolio return each day: `r_p,t = sum_i w_{i,t-1} * r_{i,t}`.
5. Turnover each rebalance: `sum_i |w_{i,t} - w_{i,t-1}|`. Optional transaction cost = turnover x cost_bps.
6. Run **both** strategies through the same engine and return daily return series + weight history.

**Acceptance criteria:** A daily portfolio-return series for each strategy over the full period, turnover tracked, no look-ahead bias.
**Depends on:** #7, #8
EOF

cat > "$TMP/10.md" <<'EOF'
Compute the evaluation metrics named in the README for both strategies.

**Steps**
1. Create `src/analysis/metrics.py`.
2. **Cumulative return / equity curve:** `cumprod(1 + r) - 1`.
3. **Annualized return:** `(1 + total)^(252/N) - 1`.
4. **Annualized volatility:** `std(daily) * sqrt(252)`.
5. **Sharpe ratio:** `ann_return / ann_vol` (state the risk-free assumption).
6. **Max drawdown:** `min(equity / cummax(equity) - 1)`.
7. **Turnover:** average annualized turnover from the engine.
8. **Sector exposure over time:** weights x sector map, summed per sector per date.
9. Assemble a side-by-side comparison table and save to `results/metrics.csv`.

**Acceptance criteria:** A comparison table with all metrics is printed and saved for both strategies.
**Depends on:** #9
EOF

cat > "$TMP/11.md" <<'EOF'
Produce the charts for the final report.

**Steps**
1. Create `src/analysis/plots.py` (matplotlib).
2. **Equity curves:** both strategies on one axis (log scale optional).
3. **Drawdown / underwater plot:** `equity / cummax(equity) - 1` over time.
4. **Sector exposure:** stacked-area chart of net exposure per sector over time.
5. Optional: rolling Sharpe or rolling volatility.
6. Label axes, titles, and legends; save PNGs to `results/` at a consistent dpi.

**Acceptance criteria:** Running the plotting script regenerates all figures into `results/`.
**Depends on:** #10
EOF

cat > "$TMP/12.md" <<'EOF'
Write the conclusion that answers the project's research question.

**Steps**
1. Create `docs/analysis.md` (or a notebook).
2. Present the metrics table and state which strategy performed better on a **risk-adjusted** basis (Sharpe, max drawdown).
3. Interpret *why*: sector-neutral diversification vs absolute directional exposure, turnover trade-offs, behavior across regimes.
4. Note limitations: transaction-cost assumptions, roll methodology, sample period, contract selection.
5. Embed the key charts from #11.

**Acceptance criteria:** A written summary answering "which strategy is better and why," supported by metrics and charts.
**Depends on:** #10, #11
EOF

cat > "$TMP/13.md" <<'EOF'
Make the repo reproducible end-to-end from a clean clone.

**Steps**
1. Replace the aspirational "How to Run" with the **exact** commands (env setup, `.env`, download, prepare, backtest, report).
2. Confirm the README includes: repository structure, setup/installation, Databento configuration, and the project roadmap.
3. Embed the final comparison table (from #10) and key charts (from #11).
4. Verify a fresh `git clone` runs top-to-bottom with only the documented steps.

**Acceptance criteria:** A new user can reproduce the results using only the README.
**Depends on:** #10, #11, #12
EOF

# --------------------------------------------------------------------------
# 4. Update each issue: body + label + milestone
#    Format: issue_number  label  milestone
# --------------------------------------------------------------------------
echo "==> Updating issues"
update_issue() {
  local num="$1" label="$2" milestone="$3"
  echo "    - issue #$num"
  gh issue edit "$num" --repo "$REPO" \
    --body-file "$TMP/$num.md" \
    --add-label "$label" \
    --milestone "$milestone"
}

update_issue 2  data     "$M1"
update_issue 3  data     "$M1"
update_issue 4  data     "$M1"
update_issue 5  data     "$M1"
update_issue 6  signals  "$M2"
update_issue 7  strategy "$M2"
update_issue 8  strategy "$M2"
update_issue 9  backtest "$M3"
update_issue 10 analysis "$M3"
update_issue 11 analysis "$M3"
update_issue 12 analysis "$M3"
update_issue 13 docs     "$M3"

# Add the 'setup' label to the environment/access issue too
gh issue edit 3 --repo "$REPO" --add-label setup >/dev/null 2>&1 || true

echo "==> Issues updated."

# --------------------------------------------------------------------------
# 5. Update the README and open a PR
#    (Comment this whole block out if you only want the issue edits.)
# --------------------------------------------------------------------------
echo "==> Updating README on a new branch"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not inside a git repo - skipping README step."; exit 0; }

git checkout -B readme-update

cat > README.md <<'READMEEOF'
# Futures Trend-Following Strategy Comparison

> Build, backtest, and compare **sector-neutral** vs **absolute** trend-following
> strategies across a diversified universe of CME futures, using Databento market data.

**Status:** In development - See the [Project Roadmap](#project-roadmap) for open work items.

---

## Project Group
- **Tech Lead**: Aiden Shin
- **Communication Lead**: Enrique Zambrano Valero
- **Design Leads**: Bangjie Xu & Huarun Dai

---

## Project Objective

The objective of this project is to **build, backtest, and compare** systematic trend-following strategies across a diversified universe of **CME futures markets using Databento market data**, spanning equity indices, interest rates, currencies, energy, metals, and agriculture.

We compare two portfolio construction approaches:

1. **Sector-neutral trend-following:** rank contracts by trend strength within each sector; go long the strongest and short the weakest while keeping sector exposures balanced.
2. **Absolute trend-following:** evaluate each contract independently; go long positive-trend contracts and short negative-trend contracts, letting sector exposures vary.

The research question is whether sector-neutral relative trend-following or absolute trend-following produces stronger risk-adjusted portfolio performance.

---

## Repository Structure

```
futures_trend_following/
|- data/                # Cached Databento data (git-ignored)
|- src/
|  |- data/             # Databento ingestion, roll handling, return prep
|  |- signals/          # Trend score calculation
|  |- strategies/       # Sector-neutral & absolute construction
|  |- backtest/         # Backtesting engine
|  |- analysis/         # Metrics & plotting
|- notebooks/           # Exploratory analysis
|- results/             # Output tables and charts
|- tests/               # Unit tests
|- requirements.txt
|- .env.example
|- README.md
```

---

## Setup & Installation

1. Clone the repository
   ```bash
   git clone https://github.com/AJYS-Arc/futures_trend_following.git
   cd futures_trend_following
   ```
2. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   .venv\Scripts\activate         # Windows
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

---

## Databento Configuration

This project reads CME futures data from Databento's `GLBX.MDP3` dataset and requires a Databento API key.

1. Copy the example env file: `cp .env.example .env`
2. Add your key to `.env`: `DATABENTO_API_KEY=your_key_here`

The `.env` file is git-ignored and must never be committed.

---

## Methodology

1. Define the futures universe and assign each contract to a sector.
2. Prepare futures returns, accounting for contract expiration and rolling.
3. Calculate a trend score per market: **trailing return / trailing volatility**.
4. Construct the sector-neutral strategy (rank within sector, long strongest / short weakest, balanced).
5. Construct the absolute strategy (long positive trend / short negative trend, per contract).
6. Backtest both strategies through one shared engine.
7. Compare using return, volatility, Sharpe, max drawdown, turnover, and sector exposure.

---

## Project Roadmap

Work items are tracked as GitHub [Issues](https://github.com/AJYS-Arc/futures_trend_following/issues), in three phases:

**Phase 1 - Data Foundation**
- #2 Define CME futures universe
- #3 Set up Databento data access
- #4 Download and prepare futures price data
- #5 Implement contract roll methodology

**Phase 2 - Signals & Strategies**
- #6 Calculate trend scores
- #7 Implement sector-neutral strategy
- #8 Implement absolute trend-following strategy

**Phase 3 - Backtest, Metrics & Delivery**
- #9 Build backtesting engine
- #10 Calculate performance metrics
- #11 Generate visualization
- #12 Write final analysis
- #13 Update README with final reproduction steps

---

## How to Run the Project

The commands below reflect the intended structure; scripts become available as their roadmap issues are completed.

1. Set up the environment (see Setup & Installation).
2. Configure Databento (see Databento Configuration).
3. Download / cache the data: `python -m src.data.download`
4. Prepare returns (rolling + continuous series): `python -m src.data.prepare`
5. Run the backtests (both strategies): `python -m src.backtest.run`
6. Generate performance tables and charts: `python -m src.analysis.report`

Outputs are written to `results/`.

---

## Contributing

1. Pick an open issue from the Project Roadmap and assign it to yourself.
2. Create a feature branch: `git checkout -b <short-description>`.
3. Commit your work and open a pull request against `main`.
4. Request a review from the Tech Lead before merging.
READMEEOF

git add README.md
git commit -m "Update README: structure, setup, roadmap linked to issues, run steps"
git push -u origin readme-update

echo "==> Opening pull request"
gh pr create --repo "$REPO" --base main --head readme-update \
  --title "Update README: structure, setup, roadmap & run steps" \
  --body "Adds repository structure, setup/installation, Databento configuration, a roadmap linked to the open issues, and concrete run steps."

echo "==> Done. Issues expanded and README PR opened."
echo "    (To commit the README straight to main instead of a PR, run:"
echo "     git checkout main && git merge readme-update && git push )"
