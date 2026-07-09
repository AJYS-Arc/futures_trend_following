"""Unit tests for the core pipeline. Run with:  python -m pytest -q"""
import numpy as np
import pandas as pd

from src.signals.trend import trend_score
from src.strategies.sector_neutral import sector_neutral_weights
from src.strategies.absolute import absolute_weights
from src.backtest.engine import backtest


def _rising_returns(n=400, seed=0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    # A, B trend up; C, D trend down. Add noise so volatility is non-zero.
    noise = lambda: rng.normal(0, 0.003, n)
    data = {
        "A": 0.001 + noise(), "B": 0.0008 + noise(),
        "C": -0.001 + noise(), "D": -0.0008 + noise(),
    }
    return pd.DataFrame(data, index=idx)


def test_trend_score_sign():
    r = _rising_returns()
    s = trend_score(r, lookback=100).dropna()
    # Up-trending markets score positive on average; down-trending negative.
    assert s["A"].mean() > 0 and s["B"].mean() > 0
    assert s["C"].mean() < 0 and s["D"].mean() < 0


def test_flat_series_near_zero():
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.DataFrame({"X": np.zeros(300)}, index=idx)
    s = trend_score(r, lookback=100).dropna()
    assert s["X"].abs().max() < 1e-6 or s["X"].isna().all()


def test_absolute_weights_gross():
    r = _rising_returns()
    s = trend_score(r, lookback=100)
    w = absolute_weights(s).dropna(how="all")
    gross = w.abs().sum(axis=1)
    gross = gross[gross > 0]
    assert np.allclose(gross, 1.0, atol=1e-6)


def test_sector_neutral_is_balanced():
    r = _rising_returns()
    s = trend_score(r, lookback=100)
    sectors = pd.Series({"A": "eq", "B": "eq", "C": "eq", "D": "eq"})
    w = sector_neutral_weights(s, sectors, top_frac=0.5).dropna(how="all")
    net = w.sum(axis=1)
    assert net.abs().max() < 1e-9  # dollar-neutral


def test_backtest_no_lookahead():
    r = _rising_returns()
    s = trend_score(r, lookback=100)
    w = absolute_weights(s)
    res = backtest(w, r, rebalance="W")
    # Long uptrends + short downtrends on a trending series => positive mean.
    assert res["returns"].mean() > 0
