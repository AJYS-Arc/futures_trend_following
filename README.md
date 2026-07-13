# Futures Trend-Following Strategy Comparison

> Build, backtest, and compare **sector-neutral relative trend-following** and **absolute trend-following** 
> strategies across a diversified universe of CME futures using Databento market data.

**Status:** Project planning and initial development. See the [Project Roadmap](#project-roadmap) for open work items.

---

## Project Group
- **Tech Lead**: Aiden Shin
- **Communication Lead**: Enrique Zambrano Valero
- **Design Leads**: Bangjie Xu & Huarun Dai

---

## Project Objective

The objective of this project is to build a reproducible Python framework that compares two systematic trend-following approaches across CME futures markets in equity indices, interest rates, currencies, energy, metals, and agriculture.

The two strategies are:

1. **Sector-neutral relative trend-following:** Within each sector, rank contracts by trend strength, go long the strongest-trending contracts, and short the weakest-trending contracts while maintaining balanced long and short exposure.

2. **Absolute trend-following:** Within each sector, evaluate each contract independently, go long contracts with positive trend signals, and short contracts with negative trend signals. The resulting sector portfolios are then combined at the portfolio level.

The primary research question is whether relative ranking within sectors or absolute trend direction produces stronger portfolio-level performance after controlling for volatility, sector concentration, turnover, drawdowns, and transaction costs.

---

## Project Description

Trend-following strategies can be implemented in different ways. A relative approach identifies the strongest and weakest trends within a peer group, while an absolute approach trades each market based on whether its own trend is positive or negative.

This project compares those approaches within a common portfolio framework. Both strategies will use the same futures universe, data-processing rules, rebalance schedule, transaction-cost assumptions, volatility target, and backtesting engine. Keeping these assumptions consistent allows the analysis to focus on the difference between relative and absolute trend signals rather than differences in implementation.

---

## Data

The project will use CME futures market data available through Databento, primarily from the `GLBX.MDP3` dataset.

The intended futures universe will include liquid contracts across the following sectors, subject to course-license availability and data quality:

- **Equity indices:** E-mini or Micro E-mini index futures
- **Interest rates:** Treasury or short-term interest-rate futures
- **Currencies:** Major FX futures
- **Energy:** Crude oil, natural gas, or refined-product futures
- **Metals:** Gold, silver, or copper futures
- **Agriculture:** Grain, oilseed, or livestock futures

The project is expected to use OHLCV records together with instrument-definition data for symbols, expirations, multipliers, tick sizes, and other contract metadata.

Because individual futures contracts expire, the analysis will explicitly address contract selection and rolling. The initial implementation will prioritize a transparent continuous-return construction that avoids artificial return jumps at roll dates. The final roll rule, return treatment, and validation checks will be finalized and documented in the corresponding GitHub Issue.

The analysis will use futures returns rather than spot returns because futures performance is affected by contract expiration, rolling, carry, contango, backwardation, and changes in the term structure.

---

## Methodology

The analysis will follow this workflow:

1. **Define the futures universe:** Select liquid CME futures contracts available through Databento and classify each contract by sector.
2. **Prepare futures return data:** Load and clean the data, apply the selected roll methodology, and create aligned return series.
3. **Calculate trend signals:** Compute the baseline trend score and selected alternative trend measures for each contract.
4. **Construct the sector-neutral strategy:** Within each sector, rank contracts by trend score, go long the strongest contracts, and short the weakest contracts while maintaining balanced exposure.
5. **Construct the absolute strategy:** Within each sector, go long contracts with positive trend signals and short contracts with negative trend signals.
6. **Combine sector portfolios:** Aggregate the sector portfolios using either equal-risk contribution or equal target gross exposure, based on the final implementation decision documented in the roadmap.
7. **Scale portfolio risk:** Scale each completed portfolio to the same ex-ante volatility target, initially proposed as 10% annualized volatility.
8. **Backtest consistently:** Apply the same rebalance frequency, signal lag, transaction-cost assumptions, and data-availability rules to both strategies.
9. **Compare results:** Evaluate return, volatility, Sharpe ratio, drawdown, turnover, sector exposure, and robustness across parameter choices.

---

## Trend Score

The baseline signal will be:

**Trend Score = Trailing Futures Return / Trailing Futures Volatility**

Trailing return captures the direction and strength of the recent price trend. Dividing by trailing volatility normalizes the signal so that contracts with materially different risk levels can be compared on a more consistent basis.

The initial analysis will consider multiple lookback windows, including approximately 60, 120, and 252 trading days. Volatility may be estimated using a rolling standard deviation or an exponentially weighted measure, with the final choice documented before the backtest results are evaluated.

To reduce the risk of overfitting, the project will not select a specification solely because it produces the highest full-sample Sharpe ratio. Instead, the team will:

- define a baseline specification in advance;
- compare nearby parameter values for stability;
- evaluate results using a time-based train/test split or walk-forward framework;
- favor specifications that remain economically intuitive and reasonably robust across market periods.

Alternative trend measures may include:

- price minus moving average;
- short-term versus long-term moving-average crossover;
- breakout-based signals;
- regression slope or normalized price trend;
- composite signals that combine multiple horizons.

The final analysis may compare the baseline score against one or more alternatives to determine whether the main conclusions are robust to signal definition.

---

## Portfolio Construction

### Sector-Neutral Relative Trend-Following

Within each sector, contracts will be ranked by trend score. The strategy will go long the strongest-trending contracts and short the weakest-trending contracts. Long and short exposure will be balanced so that the result reflects relative trend strength rather than a broad directional sector view.

### Absolute Trend-Following

Within each sector, contracts with positive trend signals will receive long positions and contracts with negative trend signals will receive short positions. This allows some variation in sector direction while still preventing the full portfolio from being dominated by a single sector.

### Common Portfolio Framework

For both strategies:

- sector portfolios will be combined using a common allocation rule;
- equal-risk contribution may be estimated using a rolling covariance matrix;
- an equal target gross-exposure approach may be used as a simpler benchmark;
- the completed portfolio will be scaled to a common volatility target;
- identical rebalancing and transaction-cost assumptions will be applied;
- positions will be lagged so that signals formed using information through time `t` are not applied before time `t+1`.

The final choices for covariance estimation, rebalance frequency, target volatility, and transaction costs will be documented in the relevant GitHub Issues before implementation is finalized.

---

## Performance Evaluation

The strategies will be compared using both return-based and risk-based measures:

- **Cumulative return:** Total compounded growth over the backtest period
- **Annualized return:** Average yearly compounded return
- **Annualized volatility:** Annualized variability of portfolio returns
- **Sharpe ratio:** Return earned per unit of volatility
- **Maximum drawdown:** Largest peak-to-trough decline
- **Turnover:** Trading activity required to maintain the strategy
- **Sector exposure:** Portfolio allocation across sectors over time
- **Transaction-cost impact:** Difference between gross and net performance
- **Parameter robustness:** Stability of results across reasonable signal and lookback choices

The purpose is not only to identify the strategy with the highest return, but also to determine which approach delivers more stable, diversified, and risk-efficient performance.

---

## Expected Outputs

The completed project is expected to produce:

1. A defined CME futures universe with sector classifications and contract metadata
2. A documented Databento data-ingestion process
3. Continuous futures return series with validated roll handling
4. Baseline and alternative trend-signal outputs
5. Sector-neutral strategy weights and returns
6. Absolute trend-following strategy weights and returns
7. A shared backtest with consistent assumptions across both strategies
8. Performance comparison tables
9. Equity-curve, drawdown, turnover, and sector-exposure charts
10. A written conclusion explaining which strategy performed better, under what conditions, and with what limitations

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

This project uses Databento's `GLBX.MDP3` dataset and requires a Databento API key.

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Add the API key to `.env`:

   ```text
   DATABENTO_API_KEY=your_key_here
   ```

The `.env` file is git-ignored and must not be committed.

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
