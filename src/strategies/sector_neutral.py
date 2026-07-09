"""Issue #7 — Sector-neutral trend-following strategy.

Within each sector, rank contracts by trend score, go long the strongest and
short the weakest, and balance long vs short so each sector is dollar-neutral.
Sectors are weighted equally so no single sector dominates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _weights_one_date(scores: pd.Series, sectors: pd.Series,
                      top_frac: float, vol: pd.Series | None) -> pd.Series:
    """Compute target weights for a single date's cross-section of scores."""
    w = pd.Series(0.0, index=scores.index)
    valid = scores.dropna()
    n_sectors_used = 0
    sector_weights = {}

    for sec, syms in sectors.groupby(sectors):
        members = [s for s in syms.index if s in valid.index]
        if len(members) < 2:
            continue
        s = valid[members].sort_values(ascending=False)
        k = max(1, int(round(len(s) * top_frac)))
        longs, shorts = s.index[:k], s.index[-k:]

        lw = _leg_weights(longs, vol)
        sw = _leg_weights(shorts, vol)
        sec_w = pd.Series(0.0, index=members)
        sec_w[longs] += lw
        sec_w[shorts] -= sw
        sector_weights[sec] = sec_w
        n_sectors_used += 1

    if n_sectors_used == 0:
        return w
    # Equal weight across sectors, so gross exposure sums to 1.
    for sec_w in sector_weights.values():
        w[sec_w.index] += sec_w / n_sectors_used
    return w


def _leg_weights(symbols, vol: pd.Series | None) -> pd.Series:
    """Equal or inverse-vol weights within a leg, normalised to sum to 0.5."""
    if vol is not None:
        iv = 1.0 / vol[symbols].replace(0, np.nan)
        iv = iv.fillna(iv.mean())
        w = iv / iv.sum()
    else:
        w = pd.Series(1.0 / len(symbols), index=symbols)
    return w * 0.5  # each leg is half the sector's gross


def sector_neutral_weights(scores: pd.DataFrame, sectors: pd.Series,
                           top_frac: float = 0.34,
                           vol: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the full dates x symbols target-weight matrix."""
    rows = {}
    for dt, row in scores.iterrows():
        v = vol.loc[dt] if vol is not None and dt in vol.index else None
        rows[dt] = _weights_one_date(row, sectors, top_frac, v)
    return pd.DataFrame(rows).T.reindex(columns=scores.columns).fillna(0.0)


if __name__ == "__main__":
    from src.data.prepare import prepare_panel
    from src.signals.trend import trend_score
    from src.universe import load_universe, sector_map

    returns, vol = prepare_panel(start="2018-01-01", end="2022-12-31")
    scores = trend_score(returns, 126)
    sectors = sector_map(load_universe())
    w = sector_neutral_weights(scores, sectors, vol=vol)
    last = w.iloc[-1]
    print("Non-zero positions on last date:", (last != 0).sum())
    print("Net exposure (should be ~0):", round(last.sum(), 4))
    print("Gross exposure:", round(last.abs().sum(), 4))
