"""End-to-end pipeline: rolled data -> signals -> strategies -> backtest -> results.

Run with:  python -m src.backtest.run
Use --synthetic to run a generated-data smoke test instead of the saved rolling output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import backtest
from src.signals.trend import build_return_panel, trailing_volatility, trend_score
from src.strategies.absolute import absolute_weights
from src.strategies.sector_neutral import sector_neutral_weights
from src.universe import load_universe, sector_map


RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
ROLLING_PATH = (Path(__file__).resolve().parent.parent
                / "rolling_data" / "continuous_oi_returns.parquet")


def _monthly_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Keep the final observed target from each month and carry it forward."""
    rebalance_dates = weights.groupby(weights.index.to_period("M")).tail(1).index
    return weights.loc[rebalance_dates].reindex(weights.index).ffill().fillna(0.0)


def _load_real_inputs(start: str, end: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the rolling output and derive its matching sector map."""
    rolled = pd.read_parquet(ROLLING_PATH)
    rolled["trade_date"] = pd.to_datetime(rolled["trade_date"])

    rolled = rolled[(rolled["trade_date"] >= pd.Timestamp(start))
                    & (rolled["trade_date"] <= pd.Timestamp(end))].copy()

    if rolled.empty:
        raise ValueError("No rolling data remains inside the requested date range.")

    returns = build_return_panel(rolled)

    sectors = (rolled[["root", "category"]].dropna().drop_duplicates("root")
               .set_index("root")["category"].reindex(returns.columns))

    return returns, sectors


def _load_synthetic_inputs(start: str, end: str,
                           seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """Create a reproducible return matrix for smoke tests."""
    sectors = sector_map(load_universe())
    dates = pd.bdate_range(start, end)

    rng = np.random.default_rng(seed)
    market_trends = rng.normal(0.0, 0.00035, len(sectors))
    noise = rng.normal(0.0, 0.01, (len(dates), len(sectors)))

    returns = pd.DataFrame(noise + market_trends, index = dates,
                           columns = sectors.index)

    return returns, sectors


def run(start: str = "2015-01-01", end: str = "2026-06-30",
        lookback: int = 252, use_synthetic: bool = False) -> dict:
    """Run both strategies through the shared backtest engine."""
    if use_synthetic:
        returns, sectors = _load_synthetic_inputs(start, end)
    else:
        returns, sectors = _load_real_inputs(start, end)

    scores = trend_score(returns, lookback = lookback)
    vol = trailing_volatility(returns, vol_window = 126, min_periods = 126)

    sector_neutral = sector_neutral_weights(
        scores = scores, sectors = sectors, vol = vol, returns = returns,
        rebalance_frequency = "monthly", allocation_method = "erc",
        target_vol = 0.10
    )

    absolute = absolute_weights(scores = scores, vol = vol)
    absolute = _monthly_weights(absolute)

    results = {
        "Sector-Neutral": backtest(sector_neutral, returns, rebalance = None),
        "Absolute": backtest(absolute, returns, rebalance = None),
    }

    RESULTS_DIR.mkdir(parents = True, exist_ok = True)

    strategy_returns = pd.DataFrame({
        name: result["returns"] for name, result in results.items()
    })

    strategy_returns.to_parquet(
        RESULTS_DIR / "strategy_returns.parquet"
    )

    return {
        "results": results,
        "sectors": sectors,
        "returns": returns,
        "scores": scores,
        "vol": vol,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--synthetic", action = "store_true",
        help = "use generated data instead of the saved rolling output"
    )
    parser.add_argument("--start", default = "2015-01-01")
    parser.add_argument("--end", default = "2026-06-30")
    parser.add_argument("--lookback", type = int, default = 252)

    args = parser.parse_args()

    output = run(start = args.start, end = args.end, lookback = args.lookback,
                 use_synthetic = args.synthetic)

    print("Backtest complete. Strategies:", list(output["results"]))
    print("Saved returns to results/strategy_returns.parquet")
    print("Now run: python -m src.analysis.report")