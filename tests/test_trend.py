"""Unit tests for Issue #6 trend-score calculations."""

import numpy as np
import pandas as pd
import pytest

from src.signals.trend import (
    DEFAULT_LOOKBACK,
    build_return_panel,
    make_walk_forward_folds,
    multi_lookback_score,
    summarize_sensitivity,
    trailing_volatility,
    trend_score,
    trend_scores_by_lookback,
    walk_forward_sensitivity,
)


def _business_index(periods: int = 400) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=periods)


def _noisy_returns(
    mean: float,
    periods: int = 400,
    seed: int = 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = mean + rng.normal(0.0, 0.002, periods)
    return pd.DataFrame(
        {"TEST": values},
        index=_business_index(periods),
    )


def test_rising_series_has_positive_score():
    returns = _noisy_returns(mean=0.001, seed=10)
    scores = trend_score(
        returns,
        lookback=60,
        min_periods=60,
    )

    assert scores["TEST"].dropna().iloc[-1] > 0.0


def test_declining_series_has_negative_score():
    returns = _noisy_returns(mean=-0.001, seed=20)
    scores = trend_score(
        returns,
        lookback=60,
        min_periods=60,
    )

    assert scores["TEST"].dropna().iloc[-1] < 0.0


def test_flat_series_has_zero_score():
    returns = pd.DataFrame(
        {"TEST": np.zeros(100)},
        index=_business_index(100),
    )
    scores = trend_score(
        returns,
        lookback=60,
        min_periods=60,
    )

    assert scores["TEST"].dropna().iloc[-1] == pytest.approx(0.0)


def test_insufficient_history_stays_nan():
    returns = _noisy_returns(mean=0.001, periods=59)
    scores = trend_score(
        returns,
        lookback=60,
        min_periods=60,
    )

    assert scores["TEST"].isna().all()


def test_first_score_appears_after_full_warmup():
    returns = _noisy_returns(mean=0.001, periods=100)
    scores = trend_score(
        returns,
        lookback=60,
        min_periods=60,
    )

    assert scores["TEST"].iloc[:59].isna().all()
    assert pd.notna(scores["TEST"].iloc[59])


def test_output_matches_input_shape_and_labels():
    index = _business_index(100)
    returns = pd.DataFrame(
        {
            "CL": np.linspace(-0.001, 0.002, 100),
            "GC": np.linspace(0.002, -0.001, 100),
        },
        index=index,
    )

    scores = trend_score(
        returns,
        lookback=20,
        min_periods=20,
    )

    assert scores.shape == returns.shape
    assert scores.index.equals(returns.index)
    assert scores.columns.equals(returns.columns)


def test_build_return_panel_aligns_roots_on_common_calendar():
    rolled = pd.DataFrame(
        {
            "trade_date": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
            ],
            "root": ["CL", "CL", "GC", "GC"],
            "continuous_return": [0.01, 0.02, -0.01, 0.03],
        }
    )

    panel = build_return_panel(rolled)

    assert list(panel.columns) == ["CL", "GC"]
    assert list(panel.index) == list(
        pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04"]
        )
    )
    assert panel.loc["2024-01-02", "CL"] == pytest.approx(0.01)
    assert pd.isna(panel.loc["2024-01-02", "GC"])
    assert panel.loc["2024-01-03", "GC"] == pytest.approx(-0.01)


def test_build_return_panel_rejects_duplicate_date_root_rows():
    rolled = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02"],
            "root": ["CL", "CL"],
            "continuous_return": [0.01, 0.02],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_return_panel(rolled)


def test_three_required_lookbacks_are_produced():
    returns = _noisy_returns(mean=0.001, periods=300)
    outputs = trend_scores_by_lookback(
        returns,
        lookbacks=(60, 120, 252),
    )

    assert set(outputs) == {60, 120, 252}

    for matrix in outputs.values():
        assert matrix.index.equals(returns.index)
        assert matrix.columns.equals(returns.columns)


def test_252_day_baseline_requires_252_observations():
    returns = _noisy_returns(mean=0.001, periods=300)
    scores = trend_score(returns)

    assert scores["TEST"].iloc[:251].isna().all()
    assert pd.notna(scores["TEST"].iloc[251])
    assert DEFAULT_LOOKBACK == 252


def test_multi_lookback_requires_all_scores_by_default():
    returns = _noisy_returns(mean=0.001, periods=300)
    composite = multi_lookback_score(
        returns,
        lookbacks=(60, 120, 252),
    )

    assert composite["TEST"].iloc[:251].isna().all()
    assert pd.notna(composite["TEST"].iloc[251])


def test_trailing_volatility_is_annualized():
    index = _business_index(20)
    daily = pd.DataFrame(
        {"TEST": [0.01, -0.01] * 10},
        index=index,
    )

    volatility = trailing_volatility(
        daily,
        vol_window=20,
        min_periods=20,
    )

    expected = daily["TEST"].std(ddof=1) * np.sqrt(252)
    assert volatility["TEST"].iloc[-1] == pytest.approx(expected)


def test_walk_forward_folds_do_not_overlap_train_and_test():
    index = _business_index(300)
    folds = make_walk_forward_folds(
        index,
        min_train_size=120,
        test_size=40,
        step_size=40,
    )

    assert len(folds) > 0

    for fold in folds:
        assert fold.train_start <= fold.train_end
        assert fold.train_end < fold.test_start
        assert fold.test_start <= fold.test_end


def test_walk_forward_sensitivity_includes_each_lookback():
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        {
            "A": rng.normal(0.0005, 0.01, 400),
            "B": rng.normal(-0.0002, 0.01, 400),
        },
        index=_business_index(400),
    )

    results = walk_forward_sensitivity(
        returns,
        lookbacks=(60, 120, 252),
        min_train_size=252,
        test_size=50,
        step_size=50,
    )

    assert not results.empty
    assert set(results["lookback"]) == {60, 120, 252}
    assert set(results["metric"]) == {"direction_accuracy"}
    assert results.loc[
        results["lookback"] == 252,
        "baseline",
    ].all()


def test_sensitivity_summary_marks_252_as_baseline():
    results = pd.DataFrame(
        {
            "lookback": [60, 60, 120, 120, 252, 252],
            "value": [0.51, 0.52, 0.53, 0.50, 0.55, 0.54],
        }
    )

    summary = summarize_sensitivity(results)
    baseline_row = summary.loc[summary["lookback"] == 252].iloc[0]

    assert bool(baseline_row["baseline"]) is True
    assert baseline_row["folds"] == 2


def test_invalid_min_periods_is_rejected():
    returns = _noisy_returns(mean=0.001, periods=100)

    with pytest.raises(ValueError, match="cannot exceed"):
        trend_score(
            returns,
            lookback=60,
            min_periods=61,
        )
