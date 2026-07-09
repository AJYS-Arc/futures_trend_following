"""Issue #2 — Define the CME futures universe and sector classification.

Loads ``config/universe.yaml`` into a tidy DataFrame that the rest of the
pipeline uses to know which symbols exist and which sector each belongs to.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "universe.yaml"


def load_universe(config_path: Path | str = CONFIG_PATH) -> pd.DataFrame:
    """Return a DataFrame with one row per contract.

    Columns: symbol, sector, name, exchange, multiplier, tick_size, currency.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    rows = []
    for sector, contracts in raw.items():
        for c in contracts:
            rows.append({"sector": sector, **c})

    df = pd.DataFrame(rows).set_index("symbol")
    return df[["sector", "name", "exchange", "multiplier", "tick_size", "currency"]]


def symbols_by_sector(universe: pd.DataFrame) -> dict[str, list[str]]:
    """Map each sector -> list of symbols."""
    return {s: g.index.tolist() for s, g in universe.groupby("sector")}


def sector_map(universe: pd.DataFrame) -> pd.Series:
    """Series indexed by symbol -> sector (handy for groupby on weights)."""
    return universe["sector"]


if __name__ == "__main__":
    u = load_universe()
    print(u)
    print("\nSectors:")
    for sec, syms in symbols_by_sector(u).items():
        print(f"  {sec:12s}: {', '.join(syms)}")
