"""Sector-neutral relative trend-following strategy.

Contracts are ranked within each sector. The strategy goes long the strongest
trends and short the weakest trends, keeps every sector dollar-neutral, combines
sector sleeves using equal weights or equal-risk contribution, and optionally
scales the portfolio to a common annualized volatility target.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


TRADING_DAYS = 252
DEFAULT_COV_LOOKBACK = 126
DEFAULT_TARGET_VOL = 0.10


def _validate_inputs(scores: pd.DataFrame, sectors: pd.Series, top_frac: float,
                     vol: pd.DataFrame | None, returns: pd.DataFrame | None) -> None:
    """Validate the strategy inputs."""
    if not isinstance(scores, pd.DataFrame):
        raise TypeError("scores must be a pandas DataFrame.")
    if scores.empty:
        raise ValueError("scores cannot be empty.")
    if not isinstance(scores.index, pd.DatetimeIndex):
        raise TypeError("scores must use a pandas DatetimeIndex.")
    if scores.index.has_duplicates or scores.columns.has_duplicates:
        raise ValueError("scores cannot contain duplicate dates or symbols.")
    if not isinstance(sectors, pd.Series):
        raise TypeError("sectors must be a pandas Series.")
    if not 0 < top_frac <= 0.50:
        raise ValueError("top_frac must be greater than 0 and no more than 0.50.")

    for name, frame in (("vol", vol), ("returns", returns)):
        if frame is None:
            continue
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame or None.")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{name} must use a pandas DatetimeIndex.")
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError(f"{name} cannot contain duplicate dates or symbols.")


def _leg_weights(symbols: Sequence[str], vol: pd.Series | None) -> pd.Series:
    """Return positive equal or inverse-volatility weights that sum to one."""
    symbols = pd.Index(symbols)
    equal = pd.Series(1.0 / len(symbols), index = symbols, dtype = float)

    if vol is None:
        return equal

    selected_vol = pd.to_numeric(vol.reindex(symbols), errors = "coerce")
    selected_vol = selected_vol.replace([np.inf, -np.inf], np.nan).where(lambda x: x > 0)

    if selected_vol.notna().sum() == 0:
        return equal

    selected_vol = selected_vol.fillna(selected_vol.median())
    inverse_vol = 1.0 / selected_vol
    total = inverse_vol.sum()

    if not np.isfinite(total) or total <= 0:
        return equal

    return inverse_vol / total


def _sector_sleeves_one_date(scores: pd.Series, sectors: pd.Series, top_frac: float,
                             vol: pd.Series | None) -> dict[str, pd.Series]:
    """Build one dollar-neutral long-short sleeve for each usable sector."""
    scores = pd.to_numeric(scores, errors = "coerce").replace([np.inf, -np.inf], np.nan)
    sectors = sectors.reindex(scores.index)
    sleeves = {}

    for sector in sectors.dropna().unique():
        members = sectors.index[sectors == sector]
        ranked = scores.reindex(members).dropna().sort_values(ascending = False, kind = "mergesort")

        if len(ranked) < 2:
            continue

        number_selected = max(1, int(round(len(ranked) * top_frac)))
        number_selected = min(number_selected, len(ranked) // 2)
        long_symbols = ranked.index[:number_selected]
        short_symbols = ranked.index[-number_selected:]

        sleeve = pd.Series(0.0, index = scores.index, dtype = float)
        sleeve.loc[long_symbols] = 0.5 * _leg_weights(long_symbols, vol)
        sleeve.loc[short_symbols] = -0.5 * _leg_weights(short_symbols, vol)
        sleeves[str(sector)] = sleeve

    return sleeves


def _portfolio_return_history(returns: pd.DataFrame, weights: pd.Series,
                              min_coverage: float = 0.80) -> pd.Series:
    """Calculate historical returns for a fixed set of portfolio weights."""
    active = weights[weights.abs() > 0]
    if active.empty:
        return pd.Series(index = returns.index, dtype = float)

    aligned = returns.reindex(columns = active.index)
    weighted = aligned.mul(active, axis = 1).sum(axis = 1, min_count = 1)
    available_gross = aligned.notna().mul(active.abs(), axis = 1).sum(axis = 1)
    total_gross = active.abs().sum()
    coverage = available_gross / total_gross

    return (weighted / coverage.replace(0, np.nan)).where(coverage >= min_coverage)


def _equal_allocations(sector_names: Sequence[str]) -> pd.Series:
    """Allocate capital equally across the available sectors."""
    sector_names = list(sector_names)
    return pd.Series(1.0 / len(sector_names), index = sector_names, dtype = float)


def _erc_weights(covariance: pd.DataFrame, tolerance: float = 1e-10,
                 max_iterations: int = 1000) -> pd.Series:
    """Calculate long-only equal-risk-contribution weights without SciPy."""
    matrix = covariance.to_numpy(dtype = float)
    matrix = 0.5 * (matrix + matrix.T)
    diagonal = np.diag(matrix)

    if not np.isfinite(matrix).all() or (diagonal <= 0).any():
        raise ValueError("Covariance matrix is not usable.")

    # Small diagonal shrinkage and jitter improve numerical stability.
    matrix = 0.95 * matrix + 0.05 * np.diag(diagonal)
    matrix += np.eye(len(matrix)) * max(float(diagonal.mean()) * 1e-10, 1e-12)

    if not np.isfinite(np.linalg.cond(matrix)) or np.linalg.cond(matrix) > 1e12:
        raise ValueError("Covariance matrix is unstable.")

    number_of_sectors = len(matrix)
    risk_budgets = np.full(number_of_sectors, 1.0 / number_of_sectors)
    x = 1.0 / np.sqrt(np.diag(matrix))

    for _ in range(max_iterations):
        previous = x.copy()
        for i in range(number_of_sectors):
            other_covariance = matrix[i] @ x - matrix[i, i] * x[i]
            discriminant = other_covariance ** 2 + 4.0 * matrix[i, i] * risk_budgets[i]
            x[i] = (-other_covariance + np.sqrt(discriminant)) / (2.0 * matrix[i, i])

        if np.max(np.abs(x - previous)) <= tolerance * max(1.0, np.max(np.abs(previous))):
            break

    weights = x / x.sum()
    if not np.isfinite(weights).all():
        raise ValueError("ERC produced non-finite weights.")

    return pd.Series(weights, index = covariance.index, dtype = float)


def _sector_allocations(sleeves: dict[str, pd.Series], historical_returns: pd.DataFrame | None,
                        allocation_method: str, min_cov_observations: int,
                        min_return_coverage: float) -> pd.Series:
    """Return equal-sector or ERC allocations, with equal weights as fallback."""
    equal = _equal_allocations(sleeves.keys())
    method = allocation_method.lower().strip()

    if method == "equal" or historical_returns is None or len(sleeves) == 1:
        return equal
    if method != "erc":
        raise ValueError("allocation_method must be either 'equal' or 'erc'.")

    sector_returns = pd.DataFrame({
        sector: _portfolio_return_history(historical_returns, sleeve, min_coverage = min_return_coverage)
        for sector, sleeve in sleeves.items()
    }).dropna()

    if len(sector_returns) < min_cov_observations:
        return equal

    covariance = sector_returns.cov()
    try:
        return _erc_weights(covariance).reindex(equal.index)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return equal


def _combine_sleeves(sleeves: dict[str, pd.Series], allocations: pd.Series,
                     symbols: pd.Index) -> pd.Series:
    """Combine sector sleeves into one portfolio."""
    weights = pd.Series(0.0, index = symbols, dtype = float)
    for sector, sleeve in sleeves.items():
        weights = weights.add(sleeve * allocations.loc[sector], fill_value = 0.0)
    return weights.reindex(symbols).fillna(0.0)


def _target_volatility(weights: pd.Series, historical_returns: pd.DataFrame | None,
                       target_vol: float | None, annualization: int,
                       min_vol_observations: int, min_return_coverage: float,
                       max_gross: float | None) -> pd.Series:
    """Scale the portfolio toward the requested annualized volatility target."""
    scaled = weights.copy()

    if target_vol is not None and historical_returns is not None:
        portfolio_returns = _portfolio_return_history(
            historical_returns, scaled, min_coverage = min_return_coverage
        ).dropna()

        if len(portfolio_returns) >= min_vol_observations:
            estimated_vol = portfolio_returns.std(ddof = 1) * np.sqrt(annualization)
            if np.isfinite(estimated_vol) and estimated_vol > 0:
                scaled *= target_vol / estimated_vol

    if max_gross is not None:
        gross = scaled.abs().sum()
        if gross > max_gross:
            scaled *= max_gross / gross

    return scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Return the last observed date in each requested rebalance period."""
    frequency = frequency.lower().strip()
    if frequency in {"daily", "d"}:
        return index
    if frequency in {"weekly", "w"}:
        periods = index.to_period("W-FRI")
    elif frequency in {"monthly", "m"}:
        periods = index.to_period("M")
    else:
        raise ValueError("rebalance_frequency must be 'daily', 'weekly', or 'monthly'.")

    dates = pd.Series(index, index = index)
    return pd.DatetimeIndex(dates.groupby(periods).last().to_numpy())


def sector_neutral_weights(scores: pd.DataFrame, sectors: pd.Series, top_frac: float = 0.34,
                           vol: pd.DataFrame | None = None, returns: pd.DataFrame | None = None,
                           rebalance_frequency: str = "monthly", allocation_method: str = "erc",
                           cov_lookback: int = DEFAULT_COV_LOOKBACK, min_cov_observations: int = 60,
                           target_vol: float | None = DEFAULT_TARGET_VOL, annualization: int = TRADING_DAYS,
                           min_vol_observations: int = 60, min_return_coverage: float = 0.80,
                           max_gross: float | None = 3.0) -> pd.DataFrame:
    """Build a dates-by-symbol sector-neutral target-weight matrix.

    Unmapped symbols receive zero weight. If returns are unavailable, ERC falls
    back to equal-sector allocation and volatility targeting is skipped.
    """
    _validate_inputs(scores, sectors, top_frac, vol, returns)

    if cov_lookback < 2 or min_cov_observations < 2 or min_cov_observations > cov_lookback:
        raise ValueError("cov_lookback and min_cov_observations are inconsistent.")
    if target_vol is not None and target_vol <= 0:
        raise ValueError("target_vol must be positive or None.")
    if annualization <= 0 or min_vol_observations < 2:
        raise ValueError("annualization and min_vol_observations must be positive.")
    if not 0 < min_return_coverage <= 1:
        raise ValueError("min_return_coverage must be greater than 0 and no more than 1.")
    if max_gross is not None and max_gross <= 0:
        raise ValueError("max_gross must be positive or None.")

    scores = scores.sort_index().copy().ffill(limit=5)
    symbols = scores.columns
    sectors = sectors.reindex(symbols)

    if sectors.notna().sum() == 0:
        raise ValueError("No score columns have a corresponding sector mapping.")

    if vol is not None:
        vol = vol.sort_index().reindex(index=scores.index, columns=symbols).ffill(limit=5)

    if returns is not None:
        returns = returns.sort_index().reindex(columns = symbols)

    target_rows = {}
    for date in _rebalance_dates(scores.index, rebalance_frequency):
        vol_row = vol.loc[date] if vol is not None else None
        sleeves = _sector_sleeves_one_date(scores.loc[date], sectors, top_frac, vol_row)

        if not sleeves:
            target_rows[date] = pd.Series(0.0, index = symbols, dtype = float)
            continue

        history = returns.loc[:date].tail(cov_lookback) if returns is not None else None
        allocations = _sector_allocations(sleeves, history, allocation_method,
                                          min_cov_observations, min_return_coverage)
        weights = _combine_sleeves(sleeves, allocations, symbols)
        target_rows[date] = _target_volatility(weights, history, target_vol, annualization,
                                               min_vol_observations, min_return_coverage, max_gross)

    targets = pd.DataFrame.from_dict(target_rows, orient = "index").reindex(columns = symbols)
    weights = targets.reindex(scores.index).ffill().fillna(0.0)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.mask(weights.abs() < 1e-14, 0.0)


if __name__ == "__main__":
    from src.signals.trend import load_rolled_return_panel, trailing_volatility, trend_score
    from src.universe import load_universe, sector_map

    rolled_returns = load_rolled_return_panel()
    trend_scores = trend_score(rolled_returns, lookback = 252)
    volatility = trailing_volatility(rolled_returns, vol_window = 126)
    sectors = sector_map(load_universe())

    weights = sector_neutral_weights(scores = trend_scores, sectors = sectors, vol = volatility,
                                     returns = rolled_returns, rebalance_frequency = "monthly",
                                     allocation_method = "erc", target_vol = 0.10)

    last = weights.iloc[-1]
    print("Non-zero positions on last date:", int((last != 0).sum()))
    print("Net exposure (should be ~0):", round(float(last.sum()), 6))
    print("Gross exposure:", round(float(last.abs().sum()), 4))