"""Issue #6 — Calculate trend scores.

Trend Score = trailing return / trailing volatility, computed per market.
A higher score means a stronger positive trend; negative means a downtrend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trend_score(returns: pd.DataFrame, lookback: int = 252,
                vol_window: int | None = None) -> pd.DataFrame:
    """Volatility-scaled trailing-return trend score.

    Parameters
    ----------
    returns : dates x symbols daily simple returns
    lookback : window (trading days) for the trailing return
    vol_window : window for volatility (defaults to ``lookback``)
    """
    vol_window = vol_window or lookback
    # Trailing cumulative return over the lookback window.
    trailing_ret = (1 + returns).rolling(lookback, min_periods=lookback // 2).apply(
        np.prod, raw=True
    ) - 1
    trailing_vol = returns.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(252)
    score = trailing_ret / trailing_vol.replace(0, np.nan)
    return score


def multi_lookback_score(returns: pd.DataFrame,
                         lookbacks: tuple[int, ...] = (60, 120, 252)) -> pd.DataFrame:
    """Average the trend score across several lookbacks (a common robustifier)."""
    scores = [trend_score(returns, lb) for lb in lookbacks]
    stacked = pd.concat(scores).groupby(level=0).mean()
    return stacked


if __name__ == "__main__":
    from src.data.prepare import prepare_panel

    returns, _ = prepare_panel(start="2018-01-01", end="2022-12-31")
    score = trend_score(returns, lookback=126)
    print("Trend score shape:", score.shape)
    print(score.tail(2).iloc[:, :6].round(2))
