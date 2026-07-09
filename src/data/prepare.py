"""Data preparation — build the aligned returns + volatility panel.

Bridges the raw-data layer (real Databento or synthetic) and the signal layer.
If Databento data isn't available, falls back to the synthetic generator so the
pipeline always runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.download import download_universe, generate_synthetic
from src.data.roll import prices_to_frame, compute_returns


def load_prices(start: str, end: str, use_synthetic: bool = True) -> pd.DataFrame:
    """Return a close-price DataFrame (dates x symbols)."""
    if use_synthetic:
        data = generate_synthetic(start, end)
    else:
        try:
            data = download_universe(start, end)
        except Exception as e:  # noqa: BLE001
            print(f"[prepare] Databento unavailable ({e}); using synthetic data.")
            data = generate_synthetic(start, end)
    return prices_to_frame(data)


def prepare_panel(start: str = "2015-01-01", end: str = "2024-12-31",
                  vol_window: int = 60, use_synthetic: bool = True):
    """Return (returns, volatility) aligned DataFrames.

    - returns:    daily simple returns, dates x symbols
    - volatility: trailing annualised volatility (rolling std x sqrt(252))
    """
    prices = load_prices(start, end, use_synthetic=use_synthetic)
    prices = prices.dropna(how="all").ffill()
    returns = compute_returns(prices)
    volatility = returns.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(252)
    return returns, volatility


if __name__ == "__main__":
    r, v = prepare_panel(start="2018-01-01", end="2020-12-31")
    print("Returns:", r.shape, "| Volatility:", v.shape)
    print(r.tail(3).iloc[:, :5])
