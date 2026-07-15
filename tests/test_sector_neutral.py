"""Unit tests for the sector-neutral relative trend-following strategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.sector_neutral import sector_neutral_weights


def _dates(periods: int = 80) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods = periods)


def _scores(index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    if index is None:
        index = _dates()

    values = {
        "A": 4.0,
        "B": 3.0,
        "C": 2.0,
        "D": 1.0,
        "E": 4.0,
        "F": 3.0,
        "G": 2.0,
        "H": 1.0,
    }
    return pd.DataFrame({symbol: np.full(len(index), value)
                         for symbol, value in values.items()}, index = index)


def _sectors() -> pd.Series:
    return pd.Series({
        "A": "energy", "B": "energy", "C": "energy", "D": "energy",
        "E": "metals", "F": "metals", "G": "metals", "H": "metals",
    })


def test_strongest_contracts_are_long_and_weakest_are_short():
    scores = _scores(_dates(1))
    weights = sector_neutral_weights(scores = scores, sectors = _sectors(), top_frac = 0.25,
                                     rebalance_frequency = "daily", allocation_method = "equal",
                                     target_vol = None)

    last = weights.iloc[-1]
    assert last["A"] > 0 and last["E"] > 0
    assert last["D"] < 0 and last["H"] < 0
    assert last[["B", "C", "F", "G"]].eq(0.0).all()


def test_each_sector_is_dollar_neutral():
    scores = _scores(_dates(5))
    sectors = _sectors()
    weights = sector_neutral_weights(scores = scores, sectors = sectors, top_frac = 0.50,
                                     rebalance_frequency = "daily", allocation_method = "equal",
                                     target_vol = None)

    for sector in sectors.unique():
        members = sectors.index[sectors == sector]
        assert np.allclose(weights[members].sum(axis = 1), 0.0, atol = 1e-12)


def test_long_and_short_selections_do_not_overlap_in_small_sector():
    index = _dates(1)
    scores = pd.DataFrame({"A": [3.0], "B": [2.0], "C": [1.0]}, index = index)
    sectors = pd.Series({"A": "energy", "B": "energy", "C": "energy"})

    weights = sector_neutral_weights(scores = scores, sectors = sectors, top_frac = 0.50,
                                     rebalance_frequency = "daily", allocation_method = "equal",
                                     target_vol = None)

    last = weights.iloc[-1]
    assert last["A"] > 0
    assert last["B"] == pytest.approx(0.0)
    assert last["C"] < 0


def test_inverse_volatility_sizing_gives_more_weight_to_lower_volatility():
    index = _dates(1)
    scores = pd.DataFrame({"A": [4.0], "B": [3.0], "C": [2.0], "D": [1.0]}, index = index)
    sectors = pd.Series({"A": "energy", "B": "energy", "C": "energy", "D": "energy"})
    vol = pd.DataFrame({"A": [0.10], "B": [0.20], "C": [0.10], "D": [0.20]}, index = index)

    weights = sector_neutral_weights(scores = scores, sectors = sectors, top_frac = 0.50, vol = vol,
                                     rebalance_frequency = "daily", allocation_method = "equal",
                                     target_vol = None)

    last = weights.iloc[-1]
    assert last["A"] > last["B"] > 0
    assert abs(last["C"]) > abs(last["D"]) > 0
    assert last.sum() == pytest.approx(0.0)


def test_missing_and_zero_volatility_fall_back_to_equal_weights():
    index = _dates(1)
    scores = pd.DataFrame({"A": [4.0], "B": [3.0], "C": [2.0], "D": [1.0]}, index = index)
    sectors = pd.Series({"A": "energy", "B": "energy", "C": "energy", "D": "energy"})
    vol = pd.DataFrame({"A": [0.0], "B": [np.nan], "C": [0.0], "D": [np.nan]}, index = index)

    weights = sector_neutral_weights(scores = scores, sectors = sectors, top_frac = 0.50, vol = vol,
                                     rebalance_frequency = "daily", allocation_method = "equal",
                                     target_vol = None)

    expected = pd.Series({"A": 0.25, "B": 0.25, "C": -0.25, "D": -0.25})
    pd.testing.assert_series_equal(weights.iloc[-1], expected, check_names = False)


def test_sector_with_fewer_than_two_valid_contracts_is_skipped():
    index = _dates(1)
    scores = pd.DataFrame({"A": [2.0], "B": [1.0], "C": [3.0]}, index = index)
    sectors = pd.Series({"A": "energy", "B": "energy", "C": "metals"})

    weights = sector_neutral_weights(scores = scores, sectors = sectors, top_frac = 0.50,
                                     rebalance_frequency = "daily", allocation_method = "equal",
                                     target_vol = None)

    last = weights.iloc[-1]
    assert last["A"] > 0 and last["B"] < 0
    assert last["C"] == pytest.approx(0.0)


def test_monthly_weights_are_held_until_the_next_rebalance():
    index = pd.bdate_range("2024-01-02", "2024-03-05")
    scores = pd.DataFrame(index = index, columns = ["A", "B", "C", "D"], dtype = float)
    scores.loc[:, :] = [4.0, 3.0, 2.0, 1.0]
    scores.loc[index.to_period("M") >= pd.Period("2024-02"), :] = [1.0, 2.0, 3.0, 4.0]
    sectors = pd.Series({"A": "energy", "B": "energy", "C": "energy", "D": "energy"})

    weights = sector_neutral_weights(scores = scores, sectors = sectors, top_frac = 0.25,
                                     rebalance_frequency = "monthly", allocation_method = "equal",
                                     target_vol = None)

    assert weights.loc[:"2024-01-30"].eq(0.0).all().all()
    pd.testing.assert_series_equal(weights.loc["2024-01-31"], weights.loc["2024-02-28"], check_names = False)
    assert weights.loc["2024-01-31", "A"] > 0
    assert weights.loc["2024-02-29", "A"] < 0
    pd.testing.assert_series_equal(weights.loc["2024-02-29"], weights.loc["2024-03-05"], check_names = False)


def test_erc_falls_back_to_equal_sector_allocation_with_insufficient_history():
    index = _dates(20)
    scores = _scores(index)
    returns = pd.DataFrame(0.001, index = index, columns = scores.columns)

    erc_weights = sector_neutral_weights(scores = scores, sectors = _sectors(), top_frac = 0.25,
                                         returns = returns, rebalance_frequency = "daily",
                                         allocation_method = "erc", min_cov_observations = 60,
                                         target_vol = None)
    equal_weights = sector_neutral_weights(scores = scores, sectors = _sectors(), top_frac = 0.25,
                                           returns = returns, rebalance_frequency = "daily",
                                           allocation_method = "equal", target_vol = None)

    pd.testing.assert_frame_equal(erc_weights, equal_weights)


def test_output_matches_score_shape_and_contains_only_finite_values():
    scores = _scores(_dates(10))
    scores.loc[scores.index[3], "A"] = np.nan

    weights = sector_neutral_weights(scores = scores, sectors = _sectors(), top_frac = 0.25,
                                     rebalance_frequency = "daily", allocation_method = "equal",
                                     target_vol = None)

    assert weights.shape == scores.shape
    assert weights.index.equals(scores.index)
    assert weights.columns.equals(scores.columns)
    assert np.isfinite(weights.to_numpy()).all()


def test_volatility_targeting_respects_maximum_gross_exposure():
    index = _dates(80)
    scores = pd.DataFrame({"A": 1.0, "B": -1.0}, index = index)
    sectors = pd.Series({"A": "energy", "B": "energy"})
    rng = np.random.default_rng(7)
    returns = pd.DataFrame({
        "A": rng.normal(0.0, 1e-5, len(index)),
        "B": rng.normal(0.0, 1e-5, len(index)),
    }, index = index)

    weights = sector_neutral_weights(scores = scores, sectors = sectors, top_frac = 0.50,
                                     returns = returns, rebalance_frequency = "daily",
                                     allocation_method = "equal", target_vol = 0.10,
                                     min_vol_observations = 20, max_gross = 1.20)

    final_gross = weights.iloc[-1].abs().sum()
    assert final_gross <= 1.20 + 1e-12
    assert final_gross == pytest.approx(1.20)