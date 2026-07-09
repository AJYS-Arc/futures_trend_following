"""Issue #11 — Visualization: equity curves, drawdowns, sector exposure."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.metrics import equity_curve, sector_exposure

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def plot_equity_curves(results: dict[str, dict], path: Path | None = None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        equity_curve(res["returns"]).plot(ax=ax, label=name, lw=1.6)
    ax.set_title("Cumulative Return — Sector-Neutral vs Absolute")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path or RESULTS_DIR / "equity_curves.png")


def plot_drawdowns(results: dict[str, dict], path: Path | None = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, res in results.items():
        eq = equity_curve(res["returns"])
        dd = eq / eq.cummax() - 1
        dd.plot(ax=ax, label=name, lw=1.2)
    ax.set_title("Drawdown")
    ax.set_ylabel("Peak-to-trough")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path or RESULTS_DIR / "drawdowns.png")


def plot_sector_exposure(result: dict, sectors, name: str, path: Path | None = None):
    exp = sector_exposure(result["weights"], sectors)
    fig, ax = plt.subplots(figsize=(10, 4))
    exp.plot.area(ax=ax, linewidth=0, stacked=False, alpha=0.6)
    ax.set_title(f"Sector Exposure Over Time — {name}")
    ax.set_ylabel("Net exposure")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    return _save(fig, path or RESULTS_DIR / f"sector_exposure_{name.lower().replace(' ', '_')}.png")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
