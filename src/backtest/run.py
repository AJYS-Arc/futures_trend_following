"""End-to-end pipeline: data -> signals -> strategies -> backtest -> results.

Run with:  python -m src.backtest.run
Uses synthetic data by default; pass --real to use Databento (needs an API key).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.universe import load_universe, sector_map
from src.data.prepare import prepare_panel
from src.signals.trend import trend_score
from src.strategies.sector_neutral import sector_neutral_weights
from src.strategies.absolute import absolute_weights
from src.backtest.engine import backtest

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def run(start="2015-01-01", end="2024-12-31", lookback=126,
        use_synthetic=True) -> dict:
    universe = load_universe()
    sectors = sector_map(universe)

    returns, vol = prepare_panel(start, end, use_synthetic=use_synthetic)
    scores = trend_score(returns, lookback=lookback)

    sn_w = sector_neutral_weights(scores, sectors, vol=vol)
    ab_w = absolute_weights(scores, vol=vol)

    results = {
        "Sector-Neutral": backtest(sn_w, returns),
        "Absolute": backtest(ab_w, returns),
    }

    # Persist portfolio return series for the report step.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rets = pd.DataFrame({name: res["returns"] for name, res in results.items()})
    rets.to_parquet(RESULTS_DIR / "strategy_returns.parquet")
    return {"results": results, "sectors": sectors, "returns": returns}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true", help="use Databento instead of synthetic data")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--lookback", type=int, default=126)
    args = p.parse_args()

    out = run(args.start, args.end, args.lookback, use_synthetic=not args.real)
    print("Backtest complete. Strategies:", list(out["results"]))
    print("Saved returns to results/strategy_returns.parquet")
    print("Now run:  python -m src.analysis.report")
