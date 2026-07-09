"""Issue #4 — Download and prepare futures price data.

For each symbol in the universe, pull continuous front-month daily OHLCV from
Databento and cache it to ``data/raw/<symbol>.parquet`` (skip re-download if
present). Also includes a synthetic price generator so the whole pipeline is
runnable without a Databento key — useful for development and grading.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.universe import load_universe

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Real data path (Databento)
# ---------------------------------------------------------------------------
def download_symbol(symbol: str, start: str, end: str, force: bool = False) -> pd.DataFrame:
    """Download + cache one symbol's continuous front-month daily OHLCV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{symbol}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    from src.data.client import fetch_ohlcv

    df = fetch_ohlcv([f"{symbol}.c.0"], start, end)
    df = _tidy(df, symbol)
    df.to_parquet(path)
    return df


def download_universe(start: str, end: str, force: bool = False) -> dict[str, pd.DataFrame]:
    """Download every symbol in the universe. Returns {symbol: OHLCV DataFrame}."""
    universe = load_universe()
    out = {}
    for symbol in universe.index:
        out[symbol] = download_symbol(symbol, start, end, force=force)
    return out


def _tidy(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalise a Databento OHLCV frame to a clean date-indexed table."""
    df = df.reset_index()
    ts_col = "ts_event" if "ts_event" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[ts_col]).dt.tz_localize(None).dt.normalize()
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[["date", *cols]].dropna(subset=["close"]).drop_duplicates("date")
    return df.set_index("date").sort_index()


# ---------------------------------------------------------------------------
# Synthetic data path (no API key required)
# ---------------------------------------------------------------------------
def generate_synthetic(start: str = "2015-01-01", end: str = "2024-12-31",
                       seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate plausible trending futures prices for the whole universe.

    Each market is a geometric random walk with a slowly time-varying drift, so
    that trend-following signals have something real to latch onto. Markets in
    the same sector share a common factor to create realistic correlations.
    """
    rng = np.random.default_rng(seed)
    universe = load_universe()
    dates = pd.bdate_range(start, end)
    n = len(dates)

    # Sector-level common factors (regime-switching drift + shared shocks)
    sectors = universe["sector"].unique()
    sector_factor = {}
    for sec in sectors:
        drift = _regime_drift(n, rng)
        shocks = rng.normal(0, 0.008, n)
        sector_factor[sec] = drift + shocks

    out = {}
    for symbol, row in universe.iterrows():
        sec = row["sector"]
        idio = rng.normal(0, 0.010, n)
        own_drift = _regime_drift(n, rng) * 0.5
        daily_ret = 0.6 * sector_factor[sec] + idio + own_drift
        price = 100 * np.exp(np.cumsum(daily_ret))
        close = pd.Series(price, index=dates)
        df = pd.DataFrame({
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * (1 + np.abs(rng.normal(0, 0.003, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.003, n))),
            "close": close,
            "volume": rng.integers(1_000, 100_000, n),
        }, index=dates)
        df.index.name = "date"
        out[symbol] = df
    return out


def _regime_drift(n: int, rng: np.random.Generator) -> np.ndarray:
    """Piecewise-constant drift that flips sign occasionally (trends + reversals)."""
    drift = np.zeros(n)
    i = 0
    while i < n:
        length = rng.integers(40, 200)
        level = rng.normal(0, 0.0006)
        drift[i:i + length] = level
        i += length
    return drift


if __name__ == "__main__":
    data = generate_synthetic()
    print(f"Generated synthetic data for {len(data)} symbols.")
    sample = next(iter(data))
    print(f"\n{sample} tail:")
    print(data[sample].tail())
