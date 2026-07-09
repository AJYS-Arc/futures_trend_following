"""Issue #9 — Backtesting engine.

Applies target weights to market returns to produce a portfolio return series.
Weights are lagged by one day to avoid look-ahead bias (today's P&L uses
yesterday's signal). Turnover and simple transaction costs are tracked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest(weights: pd.DataFrame, returns: pd.DataFrame,
             rebalance: str = "W", cost_bps: float = 1.0) -> dict:
    """Run a backtest and return portfolio returns + diagnostics.

    Parameters
    ----------
    weights : target weights (dates x symbols)
    returns : daily simple returns (dates x symbols)
    rebalance : pandas offset alias for rebalance frequency ('D', 'W', 'M')
    cost_bps : transaction cost in basis points applied to turnover

    Returns a dict with: 'returns' (daily portfolio returns, net of cost),
    'gross_returns', 'turnover', and 'weights' (held weights).
    """
    # Align to the return calendar, then hold weights between rebalance dates.
    w = weights.reindex(returns.index).ffill()
    if rebalance:
        # Sample the target at each rebalance date and carry it forward.
        held = (
            w.resample(rebalance).last()
            .reindex(returns.index, method="ffill")
        )
    else:
        held = w
    held = held.fillna(0.0)

    # Lag one day: yesterday's target drives today's return (no look-ahead).
    lagged = held.shift(1).fillna(0.0)
    gross_ret = (lagged * returns).sum(axis=1)

    # Turnover = sum of absolute weight changes at each rebalance.
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps / 1e4)
    net_ret = gross_ret - cost

    return {
        "returns": net_ret,
        "gross_returns": gross_ret,
        "turnover": turnover,
        "weights": held,
    }


if __name__ == "__main__":
    from src.data.prepare import prepare_panel
    from src.signals.trend import trend_score
    from src.strategies.absolute import absolute_weights

    returns, vol = prepare_panel(start="2018-01-01", end="2022-12-31")
    scores = trend_score(returns, 126)
    w = absolute_weights(scores, vol=vol)
    res = backtest(w, returns)
    print("Backtest days:", len(res["returns"]))
    print("Mean daily return:", round(res["returns"].mean(), 6))
    print("Avg turnover/reb:", round(res["turnover"][res["turnover"] > 0].mean(), 4))
