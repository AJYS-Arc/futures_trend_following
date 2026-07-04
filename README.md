# Futures Trend-Following Strategy Comparison

---

## Project Group
- **Tech Lead**: Aiden Shin
- **Communication Lead**: Enrique Zambrano Valero
- **Design Leads**: Bangjie Xu & Huarun Dai

---

## Project Objective

The objective of this project is to **build, backtest, and compare** systematic trend-following strategies across a diversified universe of **CME futures markets using Databento market data**. The project will include futures contracts across multiple sectors, such as **equity indices, interest rates, currencies, energy, metals, and agriculture**, subject to availability under the course's Databento CME data license.

We will compare two different portfolio construction approaches:

1. **Sector-neutral trend-following:** Futures contracts are ranked by trend strength within each sector. The strategy goes long the contracts with the strongest trend scores and short the contracts with the weakest trend scores, while keeping sector exposures balanced.

2. **Absolute trend-following:** Each futures contract is evaluated independently. The strategy goes long contracts with positive trend signals and short contracts with negative trend signals, allowing sector exposures to vary depending on where trends appear in the market.

The main research question is whether **sector-neutral relative trend-following** or **absolute trend-following** produces stronger portfolio-level performance after accounting for risk, volatility, drawdowns, and trading activity.

The final analysis will compare the strategies using performance and risk metrics such as **cumulative return, annualized return, annualized volatility, Sharpe ratio, maximum drawdown, turnover, and sector exposure**.

---
## Project Description

This project will develop a Python-based backtesting framework for comparing two futures trend-following strategies. The analysis will use historical futures price data to calculate trend signals, construct portfolios, and evaluate strategy performance over time.

The project is designed to answer a practical portfolio construction question: whether trend-following works better when signals are applied within each sector in a balanced way, or when each market is traded independently based on its own absolute trend.

---

## Data

This project will use **CME futures market data provided through Databento**, as required by the course project guidelines. The primary dataset is expected to be Databento's CME Globex dataset, `GLBX.MDP3`.

The project will focus on futures contracts available through the course's CME data license. The intended futures universe may include contracts across several CME product groups, such as:

- **Equity index futures:** E-mini or Micro E-mini equity index futures
- **Interest rate futures:** Treasury futures or short-term interest rate futures
- **Currency futures:** Major FX futures
- **Energy futures:** Crude oil, natural gas, or refined product futures
- **Metals futures:** Gold, silver, or copper futures
- **Agricultural futures:** Grain, oilseed, or livestock futures

The main price data will likely come from Databento OHLCV data, such as daily or intraday open, high, low, close, and volume records. The project may also use Databento instrument definition data to identify contract symbols, expiration dates, and other contract metadata.

Because individual futures contracts expire, the project will account for contract selection and rolling. Where appropriate, the analysis may use Databento continuous contract symbology or construct continuous futures return series from individual contracts. Since continuous futures prices may not be back-adjusted, the project will handle roll dates carefully when calculating returns.

The analysis will focus on **tradable futures returns**, not underlying spot returns, because the strategies are designed to trade futures contracts. This distinction matters because futures returns can differ from spot returns due to contract expiration, rolling, carry, contango, backwardation, and changes in the futures curve.

If additional non-Databento data is used, the README will provide clear instructions for obtaining that data. If that additional data is unavailable to other users, the project will include instructions for running the analysis using only the required Databento CME data.

---

## Methodology

The project will use Databento CME futures data to build a repeatable Python analysis for comparing two trend-following strategies.

At a high level, the analysis will follow this workflow:

1. **Define the futures universe:** Select CME futures contracts available through Databento and assign each contract to a sector, such as equity indices, interest rates, currencies, energy, metals, or agriculture.

2. **Prepare futures return data:** Load the relevant CME futures data, calculate returns, and account for contract expiration and rolling where necessary.

3. **Calculate trend signals:** For each futures market, calculate a trend score using historical futures returns and volatility.

4. **Construct the sector-neutral strategy:** Within each sector, rank contracts by trend score, go long the strongest-trending contracts, and short the weakest-trending contracts while keeping sector exposures balanced.

5. **Construct the absolute trend-following strategy:** Evaluate each contract independently, going long contracts with positive trend signals and short contracts with negative trend signals.

6. **Backtest both strategies:** Apply the portfolio construction rules over time and calculate historical portfolio returns.

7. **Compare results:** Evaluate both strategies using return, volatility, Sharpe ratio, maximum drawdown, turnover, and sector exposure metrics.

The methodology is designed to produce a runnable analysis that can be reproduced by users following the instructions in this repository.

---

## Trend Score

The trend score is the signal used to measure the strength and direction of the price trend for each futures market.

The project will calculate trend scores using historical futures returns from Databento CME futures data. A simple starting definition is:

**Trend Score = Trailing Futures Return / Trailing Futures Volatility**

For example, the project may use a trailing futures return divided by trailing volatility, with the exact lookback window selected based on data availability and strategy design. A higher trend score indicates a stronger positive trend, while a lower or negative trend score indicates a weaker or negative trend.

This signal will be used differently in the two strategies:

1. **Sector-neutral strategy:** Contracts will be ranked by trend score within each sector. The strategy will go long the highest-ranked contracts and short the lowest-ranked contracts.

2. **Absolute trend-following strategy:** Each contract will be evaluated independently. The strategy will go long contracts with positive trend scores and short contracts with negative trend scores.

The exact lookback window and volatility calculation may be adjusted during implementation based on data availability and backtesting results.

---

## Portfolio Construction

The project will compare two portfolio construction approaches: a **sector-neutral trend-following strategy** and an **absolute trend-following strategy**.

For the **sector-neutral strategy**, futures contracts will first be grouped by sector. Within each sector, contracts will be ranked by trend score. The strategy will go long the contracts with the strongest trend scores and short the contracts with the weakest trend scores. Long and short exposure will be balanced within each sector so that the portfolio is not dominated by one sector.

For the **absolute trend-following strategy**, each futures contract will be evaluated independently. Contracts with positive trend scores will receive long positions, while contracts with negative trend scores will receive short positions. This approach allows sector exposures to change over time depending on where positive and negative trends appear.

Position sizes may be based on simple equal weighting or risk-adjusted weighting. A risk-adjusted approach, such as inverse volatility weighting, may be used to reduce the impact of highly volatile futures contracts on total portfolio returns.

At the portfolio level, the project may also apply sector-level equal weighting or portfolio volatility targeting so that performance comparisons between the two strategies are more consistent and risk-aware.

---

## Performance Evaluation

The two strategies will be evaluated at the portfolio level using both return-based and risk-based metrics.

The main performance metrics may include:

- **Cumulative return:** Total growth of the strategy over the backtest period
- **Annualized return:** Average yearly return of the strategy
- **Annualized volatility:** Annualized risk based on portfolio return variability
- **Sharpe ratio:** Risk-adjusted return relative to volatility
- **Maximum drawdown:** Largest peak-to-trough portfolio decline
- **Turnover:** Trading activity required to maintain the strategy
- **Sector exposure:** Allocation across futures sectors over time

The goal of the performance evaluation is not only to determine which strategy had higher returns, but also to understand which strategy produced better risk-adjusted performance, lower drawdowns, and more stable portfolio behavior.

---

## Expected Output

The final project will produce a runnable Python analysis that compares sector-neutral and absolute trend-following strategies using CME futures data from Databento.

The expected outputs include:

1. A defined CME futures universe with sector classifications
2. Databento CME futures price and return data preparation
3. Trend score calculations for each futures market
4. A sector-neutral trend-following backtest
5. An absolute trend-following backtest
6. Performance comparison tables
7. Portfolio return and drawdown charts
8. A written summary explaining which strategy performed better and why

The final repository should allow users to reproduce the analysis using the instructions provided in the README.

---

## How to Run the Project

This project will be run using Python. The final repository will include code and instructions that allow users to reproduce the analysis.

Expected steps:

1. Clone the GitHub repository.
2. Install the required Python packages.
3. Configure Databento access using the user's Databento API key.
4. Download or load the required CME futures data.
5. Run the data preparation scripts.
6. Run the strategy backtests.
7. Generate performance tables and charts.

More detailed commands will be added as the project code is developed. This section is aspirational at the setup stage and will be updated once the final project structure is complete.