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
