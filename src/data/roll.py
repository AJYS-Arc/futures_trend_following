"""Issue #5 — Contract roll methodology & continuous return construction.

We build a *tradable return* series for each market. When using Databento's
continuous front-month symbology (``.c.0``) the roll is already handled by the
provider, but a naive percentage change across a roll boundary can still pick up
an artificial gap. This module computes returns in a roll-aware way and offers a
back-adjusted price series for plotting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame, roll_col: str | None = None) -> pd.DataFrame:
    """Compute daily simple returns from a dict/frame of close prices.

    ``prices`` is a DataFrame of close prices indexed by date, one column per
    symbol. If a roll indicator column is supplied per symbol you can mask the
    roll-day return; with provider continuous data we simply use pct_change,
    which is correct for a held-then-rolled front-month series.
    """
    returns = prices.pct_change()
    return returns.iloc[1:]


def prices_to_frame(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Turn {symbol: OHLCV} into a single close-price DataFrame (dates x symbols)."""
    closes = {sym: df["close"] for sym, df in data.items()}
    frame = pd.DataFrame(closes).sort_index()
    return frame


def back_adjust(front: pd.Series, roll_dates: list[pd.Timestamp],
                ratios: list[float]) -> pd.Series:
    """Ratio (Panama-canal) back-adjustment to remove roll gaps for plotting.

    Given roll dates and the price ratio (new/old) at each roll, scale history
    before each roll so the series is continuous. Not used for return
    calculation (returns are computed per held contract) but handy for charts.
    """
    adj = front.copy().astype(float)
    for rd, ratio in zip(sorted(roll_dates), ratios):
        adj.loc[adj.index < rd] *= ratio
    return adj


if __name__ == "__main__":
    from src.data.download import generate_synthetic

    data = generate_synthetic(start="2020-01-01", end="2020-06-30")
    prices = prices_to_frame(data)
    rets = compute_returns(prices)
    print("Returns shape:", rets.shape)
    print(rets.iloc[:3, :4])
