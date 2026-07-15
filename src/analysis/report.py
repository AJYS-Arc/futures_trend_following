"""Generate the performance comparison table and charts.

Run with:  python -m src.analysis.report
(Runs the backtest first if results aren't cached.)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.backtest.run import run
from src.analysis.metrics import comparison_table
from src.analysis.plots import (
    plot_equity_curves, plot_drawdowns, plot_sector_exposure,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def main(use_synthetic: bool = False):
    out = run(use_synthetic = use_synthetic)
    results, sectors = out["results"], out["sectors"]

    table = comparison_table(results)
    table.to_csv(RESULTS_DIR / "metrics.csv")

    print("\n===== Strategy Comparison =====")
    with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
        print(table.T)

    plot_equity_curves(results)
    plot_drawdowns(results)
    for name, res in results.items():
        plot_sector_exposure(res, sectors, name)

    print(f"\nSaved metrics.csv and charts to {RESULTS_DIR}/")
    return table


if __name__ == "__main__":
    main()
