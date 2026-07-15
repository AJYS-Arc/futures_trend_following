# Trend signal methodology

## Definition

For each continuous futures market, the trend score is:

`trailing compounded return / annualized trailing volatility`

Daily continuous returns are first reshaped into a common dates-by-roots panel.
Missing market observations remain missing and are not converted to zero
returns.

## Predefined baseline

The 252-trading-day lookback is the predefined baseline. It represents
approximately one trading year and was selected before reviewing backtest
results.

The implementation also produces 60- and 120-day specifications for
sensitivity analysis. Those windows represent shorter trend horizons, but they
must not replace the baseline solely because they produce a higher
full-sample Sharpe ratio.

## Warm-up

By default, a trend score remains `NaN` until the complete lookback window is
available. Thus:

- the 60-day score begins after 60 valid observations;
- the 120-day score begins after 120 valid observations;
- the 252-day score begins after 252 valid observations.

## Out-of-sample comparison

`walk_forward_sensitivity` divides the return history into chronological
training and testing periods. It calculates each signal using only information
available through that date and evaluates the signal only during the subsequent
test period.

The initial built-in evaluation metric is next-day directional accuracy. This
is a lightweight signal-level diagnostic. Once the strategy and backtesting
interfaces are finalized, the evaluator argument should be used to run each
lookback through the common backtest engine and compare agreed out-of-sample
metrics such as Sharpe ratio, drawdown, turnover, and stability across folds.

## Parameter-selection rule

The 252-day lookback remains the project baseline unless shorter windows show
consistent out-of-sample improvement across multiple chronological folds and
the improvement remains economically meaningful after transaction costs.
The final decision should consider stability and turnover, not just the
highest average test-period performance.
