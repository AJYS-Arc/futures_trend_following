"""Trend-score construction and lookback sensitivity analysis.

Issue #6 defines the trend signal used by the project's strategies.

Baseline specification
----------------------
The predefined baseline is a 252-trading-day lookback. Alternative 60- and
120-day windows are included for sensitivity analysis. The final lookback
should be selected using time-based out-of-sample or walk-forward evidence,
not by choosing the highest full-sample Sharpe ratio.

Core definition
---------------
Trend score = trailing compounded return / annualized trailing volatility.

The module also converts the long-form output from the contract-rolling
pipeline into a common dates x roots return matrix.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
DEFAULT_LOOKBACK = 252
DEFAULT_LOOKBACKS = (60, 120, 252)

REQUIRED_ROLLING_COLUMNS = {
    "trade_date",
    "root",
    "continuous_return",
}


@dataclass(frozen=True)
class WalkForwardFold:
    """One chronological train/test split."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _validate_positive_integer(value: int, name: str) -> int:
    """Validate and normalize a positive integer argument."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return int(value)


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate a dates x symbols daily-return matrix."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.columns.has_duplicates:
        raise ValueError("returns contains duplicate symbol columns.")

    result = returns.copy()

    try:
        result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    except Exception as exc:
        raise ValueError("returns index must be convertible to dates.") from exc

    if result.index.has_duplicates:
        raise ValueError("returns contains duplicate dates.")

    result = result.sort_index()

    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    finite_or_missing = np.isfinite(result.to_numpy(dtype=float)) | result.isna().to_numpy()
    if not finite_or_missing.all():
        raise ValueError("returns contains infinite values.")

    return result.astype(float)

def _rolling_valid_observations(returns: pd.DataFrame, window: int, min_periods: int,
                                calculation: str, annualization: int = TRADING_DAYS_PER_YEAR,
                                ddof: int = 1) -> pd.DataFrame:
    """Calculate rolling statistics using each market's valid observations.

    Missing dates are not converted to zero returns and do not count toward the
    lookback window. Results are written only on dates when that market has an
    observed return.
    """
    output = pd.DataFrame(index = returns.index, columns = returns.columns, dtype = float)

    for symbol in returns.columns:
        valid = returns[symbol].dropna()

        if calculation == "return":
            values = ((1.0 + valid).rolling(window = window, min_periods = min_periods)
                      .apply(np.prod, raw = True) - 1.0)
        elif calculation == "volatility":
            values = (valid.rolling(window = window, min_periods = min_periods).std(ddof = ddof)
                      * np.sqrt(annualization))
        else:
            raise ValueError("calculation must be either 'return' or 'volatility'.")

        output.loc[values.index, symbol] = values

    return output

def build_return_panel(
    rolled_data: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    symbol_col: str = "root",
    return_col: str = "continuous_return",
    calendar: str = "observed",
) -> pd.DataFrame:
    """Convert long-form rolled returns into a dates x roots matrix.

    Parameters
    ----------
    rolled_data
        Long-form output from ``build_oi_continuous_series``.
    date_col
        Name of the trading-date column.
    symbol_col
        Name of the continuous-market identifier. ``root`` is preferred over
        ``held_symbol`` because held contracts change through time.
    return_col
        Name of the daily continuous-return column.
    calendar
        ``"observed"`` uses the union of all dates present in the input.
        ``"business"`` expands the index to every Monday-Friday date between
        the first and last observations.

    Returns
    -------
    pandas.DataFrame
        Sorted dates x roots daily simple-return matrix. Missing observations
        remain NaN; they are not treated as zero returns.
    """
    if not isinstance(rolled_data, pd.DataFrame):
        raise TypeError("rolled_data must be a pandas DataFrame.")

    required = {date_col, symbol_col, return_col}
    missing = required.difference(rolled_data.columns)
    if missing:
        raise KeyError(
            f"rolled_data is missing required columns: {sorted(missing)}"
        )

    if calendar not in {"observed", "business"}:
        raise ValueError("calendar must be either 'observed' or 'business'.")

    data = rolled_data[[date_col, symbol_col, return_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[return_col] = pd.to_numeric(data[return_col], errors="coerce")

    invalid_metadata = data[date_col].isna() | data[symbol_col].isna()
    data = data.loc[~invalid_metadata].copy()

    if data.empty:
        return pd.DataFrame(dtype=float)

    duplicate_keys = data.duplicated([date_col, symbol_col], keep=False)
    if duplicate_keys.any():
        examples = (
            data.loc[duplicate_keys, [date_col, symbol_col]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "rolled_data contains duplicate date/root observations. "
            f"Examples: {examples}"
        )

    panel = data.pivot(
        index=date_col,
        columns=symbol_col,
        values=return_col,
    )
    panel.index.name = "trade_date"
    panel.columns.name = "root"
    panel = panel.sort_index().sort_index(axis=1).astype(float)

    if calendar == "business" and not panel.empty:
        full_calendar = pd.bdate_range(panel.index.min(), panel.index.max())
        panel = panel.reindex(full_calendar)
        panel.index.name = "trade_date"

    return panel


def trailing_compounded_return(
    returns: pd.DataFrame,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Calculate trailing compounded simple return."""
    clean = _validate_returns(returns)
    lookback = _validate_positive_integer(lookback, "lookback")

    if min_periods is None:
        min_periods = lookback
    min_periods = _validate_positive_integer(min_periods, "min_periods")

    if min_periods > lookback:
        raise ValueError("min_periods cannot exceed lookback.")

    # return (
    #     (1.0 + clean)
    #     .rolling(window=lookback, min_periods=min_periods)
    #     .apply(np.prod, raw=True)
    #     - 1.0
    # )
    return _rolling_valid_observations(returns=clean, window=lookback,
                                       min_periods=min_periods, calculation="return")


def trailing_volatility(
    returns: pd.DataFrame,
    *,
    vol_window: int = DEFAULT_LOOKBACK,
    min_periods: int | None = None,
    annualization: int = TRADING_DAYS_PER_YEAR,
    ddof: int = 1,
) -> pd.DataFrame:
    """Calculate annualized rolling standard deviation of daily returns."""
    clean = _validate_returns(returns)
    vol_window = _validate_positive_integer(vol_window, "vol_window")
    annualization = _validate_positive_integer(annualization, "annualization")

    if min_periods is None:
        min_periods = vol_window
    min_periods = _validate_positive_integer(min_periods, "min_periods")

    if min_periods > vol_window:
        raise ValueError("min_periods cannot exceed vol_window.")

    return _rolling_valid_observations(returns=clean, window = vol_window,
                                       min_periods = min_periods, calculation="volatility",
                                       annualization=annualization, ddof=ddof)


def trend_score(
    returns: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    vol_window: int | None = None,
    *,
    min_periods: int | None = None,
    annualization: int = TRADING_DAYS_PER_YEAR,
    zero_volatility_score: float | None = 0.0,
) -> pd.DataFrame:
    """Calculate a volatility-scaled trailing-return trend score.

    Parameters
    ----------
    returns
        Dates x symbols matrix of daily simple returns.
    lookback
        Window used for trailing compounded return.
    vol_window
        Window used for trailing volatility. Defaults to ``lookback``.
    min_periods
        Required observations before a score is emitted. Defaults to the
        larger of ``lookback`` and ``vol_window``, enforcing a full warm-up.
    annualization
        Trading periods per year used to annualize volatility.
    zero_volatility_score
        Value assigned when trailing volatility is zero and trailing return is
        also approximately zero. The default is 0.0, which gives a flat,
        constant-price series a neutral trend score. Set to ``None`` to leave
        all zero-volatility observations as NaN.

    Returns
    -------
    pandas.DataFrame
        Trend-score matrix with the same dates and columns as ``returns``.
    """
    clean = _validate_returns(returns)
    lookback = _validate_positive_integer(lookback, "lookback")

    if vol_window is None:
        vol_window = lookback
    vol_window = _validate_positive_integer(vol_window, "vol_window")

    if min_periods is None:
        min_periods = max(lookback, vol_window)
    min_periods = _validate_positive_integer(min_periods, "min_periods")

    if min_periods > lookback:
        raise ValueError(
            "min_periods cannot exceed lookback for trailing returns."
        )
    if min_periods > vol_window:
        raise ValueError(
            "min_periods cannot exceed vol_window for trailing volatility."
        )

    trailing_return = trailing_compounded_return(
        clean,
        lookback=lookback,
        min_periods=min_periods,
    )
    volatility = trailing_volatility(
        clean,
        vol_window=vol_window,
        min_periods=min_periods,
        annualization=annualization,
    )

    score = trailing_return.div(volatility.replace(0.0, np.nan))

    if zero_volatility_score is not None:
        flat = volatility.eq(0.0) & trailing_return.abs().le(1e-14)
        score = score.mask(flat, float(zero_volatility_score))

    return score.replace([np.inf, -np.inf], np.nan)


def trend_scores_by_lookback(
    returns: pd.DataFrame,
    *,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    min_periods: int | Mapping[int, int] | None = None,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> dict[int, pd.DataFrame]:
    """Calculate separate trend-score matrices for multiple lookbacks."""
    normalized = tuple(
        _validate_positive_integer(value, "lookback")
        for value in lookbacks
    )

    if not normalized:
        raise ValueError("lookbacks cannot be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("lookbacks cannot contain duplicate values.")

    outputs: dict[int, pd.DataFrame] = {}

    for lookback in normalized:
        if isinstance(min_periods, Mapping):
            if lookback not in min_periods:
                raise KeyError(
                    f"No min_periods value provided for lookback {lookback}."
                )
            window_min_periods = min_periods[lookback]
        else:
            window_min_periods = min_periods

        outputs[lookback] = trend_score(
            returns,
            lookback=lookback,
            min_periods=window_min_periods,
            annualization=annualization,
        )

    return outputs


def multi_lookback_score(
    returns: pd.DataFrame,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    *,
    min_periods: int | Mapping[int, int] | None = None,
    weights: Mapping[int, float] | None = None,
    require_all: bool = True,
) -> pd.DataFrame:
    """Average trend scores across multiple lookbacks.

    This is an optional robust composite, not the predefined baseline. The
    predefined baseline remains the standalone 252-day score.
    """
    scores = trend_scores_by_lookback(
        returns,
        lookbacks=lookbacks,
        min_periods=min_periods,
    )

    if weights is None:
        normalized_weights = {
            lookback: 1.0 / len(scores)
            for lookback in scores
        }
    else:
        missing = set(scores).difference(weights)
        extra = set(weights).difference(scores)

        if missing or extra:
            raise ValueError(
                "weights keys must exactly match lookbacks. "
                f"Missing: {sorted(missing)}; extra: {sorted(extra)}"
            )

        raw_weights = {
            lookback: float(weights[lookback])
            for lookback in scores
        }

        if not all(np.isfinite(value) and value >= 0.0
                   for value in raw_weights.values()):
            raise ValueError("weights must be finite and non-negative.")

        total_weight = sum(raw_weights.values())
        if total_weight <= 0.0:
            raise ValueError("At least one weight must be positive.")

        normalized_weights = {
            lookback: value / total_weight
            for lookback, value in raw_weights.items()
        }

    weighted = [
        matrix * normalized_weights[lookback]
        for lookback, matrix in scores.items()
    ]

    composite = sum(weighted)

    if require_all:
        all_available = pd.concat(
            [matrix.notna() for matrix in scores.values()],
            axis=0,
        ).groupby(level=0).all()
        composite = composite.where(all_available)

    return composite


def make_walk_forward_folds(
    index: Iterable[pd.Timestamp],
    *,
    min_train_size: int,
    test_size: int,
    step_size: int | None = None,
    expanding: bool = True,
) -> list[WalkForwardFold]:
    """Create chronological train/test folds with no future leakage."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(index))).sort_values().unique()

    min_train_size = _validate_positive_integer(
        min_train_size,
        "min_train_size",
    )
    test_size = _validate_positive_integer(test_size, "test_size")

    if step_size is None:
        step_size = test_size
    step_size = _validate_positive_integer(step_size, "step_size")

    if len(dates) < min_train_size + test_size:
        return []

    folds: list[WalkForwardFold] = []
    train_end_position = min_train_size
    fold_number = 1

    while train_end_position + test_size <= len(dates):
        test_end_position = train_end_position + test_size

        if expanding:
            train_start_position = 0
        else:
            train_start_position = train_end_position - min_train_size

        folds.append(
            WalkForwardFold(
                fold=fold_number,
                train_start=pd.Timestamp(dates[train_start_position]),
                train_end=pd.Timestamp(dates[train_end_position - 1]),
                test_start=pd.Timestamp(dates[train_end_position]),
                test_end=pd.Timestamp(dates[test_end_position - 1]),
            )
        )

        fold_number += 1
        train_end_position += step_size

    return folds


def signal_direction_accuracy(
    scores: pd.DataFrame,
    future_returns: pd.DataFrame,
) -> float:
    """Measure whether signal direction matches next-period return direction.

    This is a lightweight placeholder evaluator for Issue #6. It can be
    replaced or supplemented by the project backtester once strategy Issues
    #7-#9 are finalized.
    """
    aligned_scores, aligned_returns = scores.align(
        future_returns,
        join="inner",
        axis=0,
    )
    aligned_scores, aligned_returns = aligned_scores.align(
        aligned_returns,
        join="inner",
        axis=1,
    )

    valid = aligned_scores.notna() & aligned_returns.notna()
    matches = (
        np.sign(aligned_scores)
        == np.sign(aligned_returns)
    ).where(valid)

    values = matches.stack(future_stack=True).dropna()

    if values.empty:
        return np.nan

    return float(values.astype(float).mean())


def walk_forward_sensitivity(
    returns: pd.DataFrame,
    *,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    min_train_size: int = 504,
    test_size: int = 126,
    step_size: int | None = None,
    min_periods: int | Mapping[int, int] | None = None,
    expanding: bool = True,
    evaluator: Callable[[pd.DataFrame, pd.DataFrame], float] | None = None,
) -> pd.DataFrame:
    """Evaluate lookbacks using chronological out-of-sample folds.

    Scores are computed using history available through each date. Only score
    dates falling in the fold's test period are evaluated. The default metric
    is next-day directional accuracy.

    ``evaluator`` is the integration placeholder. It must accept:

        evaluator(test_scores, test_future_returns) -> float

    A later strategy-level implementation may instead pass an evaluator that
    constructs weights, calls ``src.backtest.engine.backtest``, and returns an
    out-of-sample Sharpe ratio or another agreed metric.
    """
    clean = _validate_returns(returns)
    normalized_lookbacks = tuple(
        _validate_positive_integer(value, "lookback")
        for value in lookbacks
    )

    folds = make_walk_forward_folds(
        clean.index,
        min_train_size=min_train_size,
        test_size=test_size,
        step_size=step_size,
        expanding=expanding,
    )

    if evaluator is None:
        evaluator = signal_direction_accuracy
        metric_name = "direction_accuracy"
    else:
        metric_name = getattr(evaluator, "__name__", "custom_metric")

    all_scores = trend_scores_by_lookback(
        clean,
        lookbacks=normalized_lookbacks,
        min_periods=min_periods,
    )
    next_day_returns = clean.shift(-1)

    records: list[dict] = []

    for fold in folds:
        test_mask = (
            (clean.index >= fold.test_start)
            & (clean.index <= fold.test_end)
        )

        for lookback in normalized_lookbacks:
            test_scores = all_scores[lookback].loc[test_mask]
            test_future_returns = next_day_returns.loc[test_mask]
            value = evaluator(test_scores, test_future_returns)

            records.append(
                {
                    "fold": fold.fold,
                    "lookback": lookback,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "metric": metric_name,
                    "value": float(value) if pd.notna(value) else np.nan,
                    "baseline": lookback == DEFAULT_LOOKBACK,
                }
            )

    return pd.DataFrame.from_records(records)


def summarize_sensitivity(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize walk-forward results by lookback."""
    required = {"lookback", "value"}
    missing = required.difference(results.columns)

    if missing:
        raise KeyError(
            f"results is missing required columns: {sorted(missing)}"
        )

    if results.empty:
        return pd.DataFrame(
            columns=[
                "lookback",
                "mean_oos_value",
                "median_oos_value",
                "std_oos_value",
                "folds",
                "baseline",
            ]
        )

    summary = (
        results.groupby("lookback", as_index=False)
        .agg(
            mean_oos_value=("value", "mean"),
            median_oos_value=("value", "median"),
            std_oos_value=("value", "std"),
            folds=("value", "count"),
        )
        .sort_values("lookback")
        .reset_index(drop=True)
    )
    summary["baseline"] = summary["lookback"].eq(DEFAULT_LOOKBACK)
    return summary


def load_rolled_return_panel(
    path: str = "src/rolling_data/continuous_oi_returns.parquet",
    *,
    calendar: str = "observed",
) -> pd.DataFrame:
    """Load Huarun's rolling output and construct the aligned return panel."""
    rolled = pd.read_parquet(path)
    return build_return_panel(rolled, calendar=calendar)


def main() -> None:
    """Run the baseline and sensitivity calculations on the rolling output."""
    returns = load_rolled_return_panel()

    scores = trend_scores_by_lookback(
        returns,
        lookbacks=DEFAULT_LOOKBACKS,
    )

    for lookback, matrix in scores.items():
        output_path = (
            f"src/signals/trend_scores_{lookback}d.parquet"
        )
        matrix.to_parquet(output_path)
        print(
            f"Saved {lookback}-day scores: "
            f"{matrix.shape[0]:,} dates x {matrix.shape[1]:,} roots "
            f"to {output_path}"
        )

    sensitivity = walk_forward_sensitivity(
        returns,
        lookbacks=DEFAULT_LOOKBACKS,
        min_train_size=504,
        test_size=126,
    )
    sensitivity.to_csv(
        "src/signals/trend_walk_forward_results.csv",
        index=False,
    )

    summary = summarize_sensitivity(sensitivity)
    summary.to_csv(
        "src/signals/trend_walk_forward_summary.csv",
        index=False,
    )

    print("\nWalk-forward summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
