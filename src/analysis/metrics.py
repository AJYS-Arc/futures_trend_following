"""Issue #10 — Performance & risk metrics.

Computes the return- and risk-based metrics named in the README and assembles a
side-by-side comparison table for the two strategies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def cumulative_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series) -> float:
    n = len(returns)
    if n == 0:
        return np.nan
    total = (1 + returns).prod()
    return float(total ** (TRADING_DAYS / n) - 1)


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    vol = annualized_vol(returns)
    if vol == 0:
        return np.nan
    return float((annualized_return(returns) - rf) / vol)


def max_drawdown(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    dd = eq / eq.cummax() - 1
    return float(dd.min())


def annualized_turnover(turnover: pd.Series) -> float:
    """Average per-period turnover scaled to an annual figure."""
    active = turnover[turnover > 0]
    if active.empty:
        return 0.0
    # infer rebalances per year from spacing of nonzero turnover
    per_year = TRADING_DAYS / max(1, (len(turnover) / max(1, len(active))))
    return float(active.mean() * per_year)


def sector_exposure(weights: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """Net exposure per sector over time (dates x sectors)."""
    aligned = sectors.reindex(weights.columns)
    return weights.T.groupby(aligned).sum().T


def summarize(result: dict, name: str) -> dict:
    r = result["returns"]
    return {
        "Strategy": name,
        "Cumulative Return": cumulative_return(r),
        "Annualized Return": annualized_return(r),
        "Annualized Vol": annualized_vol(r),
        "Sharpe Ratio": sharpe_ratio(r),
        "Max Drawdown": max_drawdown(r),
        "Annualized Turnover": annualized_turnover(result["turnover"]),
    }


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """results: {name: backtest_result_dict} -> tidy comparison DataFrame."""
    rows = [summarize(res, name) for name, res in results.items()]
    df = pd.DataFrame(rows).set_index("Strategy")
    return df


if __name__ == "__main__":
    from src.data.prepare import prepare_panel
    from src.signals.trend import trend_score
    from src.strategies.absolute import absolute_weights
    from src.backtest.engine import backtest

    returns, vol = prepare_panel(start="2016-01-01", end="2024-12-31")
    scores = trend_score(returns, 126)
    res = backtest(absolute_weights(scores, vol=vol), returns)
    print(comparison_table({"Absolute": res}).round(3).T)
