"""Issue #8 — Absolute trend-following strategy.

Each contract is evaluated independently: long if its trend score is positive,
short if negative. Positions are inverse-volatility sized and the book's gross
exposure is normalised so it is comparable to the sector-neutral strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def absolute_weights(scores: pd.DataFrame, vol: pd.DataFrame | None = None,
                     dead_zone: float = 0.0, gross: float = 1.0) -> pd.DataFrame:
    """Build the dates x symbols target-weight matrix.

    Parameters
    ----------
    scores : trend scores (dates x symbols)
    vol : trailing volatility for inverse-vol sizing (optional)
    dead_zone : ignore signals with |score| below this (reduces churn)
    gross : target gross exposure (sum of |weights|) per date
    """
    sign = scores.copy()
    sign[sign.abs() < dead_zone] = 0.0
    sign = np.sign(sign)

    if vol is not None:
        size = sign / vol.reindex_like(sign).replace(0, np.nan)
    else:
        size = sign.astype(float)

    size = size.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Normalise each row so gross exposure == `gross`.
    row_gross = size.abs().sum(axis=1).replace(0, np.nan)
    weights = size.div(row_gross, axis=0).mul(gross).fillna(0.0)
    return weights


if __name__ == "__main__":
    from src.data.prepare import prepare_panel
    from src.signals.trend import trend_score

    returns, vol = prepare_panel(start="2018-01-01", end="2022-12-31")
    scores = trend_score(returns, 126)
    w = absolute_weights(scores, vol=vol)
    last = w.iloc[-1]
    print("Non-zero positions on last date:", (last != 0).sum())
    print("Net exposure (may drift):", round(last.sum(), 4))
    print("Gross exposure:", round(last.abs().sum(), 4))
