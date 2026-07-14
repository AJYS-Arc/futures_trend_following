# rolling_function.py
# This cover issue 4 partially, pending a more complete solution if methodology changes. 
#

# ---------------------------------------------------------------------------
# MISSING DATA — read before wiring this in
# ---------------------------------------------------------------------------
# Volume is missing from the data source so we implement a rolling method that uses open interest only. T
# Fields like First Service Date, Delievery Method need to be confirmed with the data source. The rolling method is based on the following rules:

# ---------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

BDay = pd.tseries.offsets.BusinessDay

REQUIRED_COLUMNS = {
    "trade_date",
    "root",
    "raw_symbol",
    "expiration",
    "settlement_price",
    "open_interest",
}

# Default roll deadline. A dictionary can be passed for root-specific values.
DEFAULT_HARD_ROLL_BDAYS = 5

# Can be later populated with root specifc value
HardRollInput = Union[int, Mapping[str, int]]


def _resolve_root_setting(
    value: HardRollInput,
    root: str,
    *,
    setting_name: str,
) -> int:
    """Get the configured integer value for a root."""
    if isinstance(value, Mapping):
        if root not in value:
            raise KeyError(
                f"No {setting_name} configured for root {root!r}. "
                f"Available roots: {sorted(value)}"
            )
        resolved = value[root]
    else:
        resolved = value

    if isinstance(resolved, bool) or not isinstance(resolved, (int, np.integer)):
        raise TypeError(f"{setting_name} for {root!r} must be an integer.")
    if resolved < 0:
        raise ValueError(f"{setting_name} for {root!r} cannot be negative.")
    return int(resolved)


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the downloaded contract data."""
    missing = REQUIRED_COLUMNS.difference(panel.columns)
    if missing:
        raise KeyError(f"Panel is missing required columns: {sorted(missing)}")
    df = panel.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.tz_localize(None)
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.tz_localize(None)
    df["settlement_price"] = pd.to_numeric(df["settlement_price"], errors="coerce")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")

    key_cols = ["trade_date", "root", "raw_symbol", "expiration"]
    invalid = df[key_cols].isna().any(axis=1)

    if invalid.any():
        print(f"Dropping {invalid.sum():,} rows with missing contract metadata.")
        df = df.loc[~invalid].copy()

    if df.empty:
        raise ValueError("No valid contract rows remain after cleaning.")
    # Keep one row per contract and trade date.
    sort_cols = ["root", "trade_date", "expiration", "raw_symbol"]
    df = df.sort_values(sort_cols)
    df = df.drop_duplicates(
        subset=["root", "trade_date", "raw_symbol"], keep="last"
    )

    return df.reset_index(drop=True)


def _select_front_contract(
    day: pd.DataFrame,
    trade_date: pd.Timestamp,
    deadlines: Mapping[str, pd.Timestamp],
) -> Optional[str]:
    """Return the nearest valid contract."""
    candidates = day[
        day["settlement_price"].notna() & (day["expiration"] >= trade_date)
    ].sort_values("expiration")
    if candidates.empty:
        return None

    safe = candidates[
        candidates["raw_symbol"].map(deadlines).gt(trade_date)
    ]
    selected = safe.iloc[0] if not safe.empty else candidates.iloc[0]
    return str(selected["raw_symbol"])


def _select_next_contract(
    day: pd.DataFrame,
    current_expiration: pd.Timestamp,
    trade_date: pd.Timestamp,
    deadlines: Mapping[str, pd.Timestamp],
) -> Optional[str]:
    """Return the nearest valid contract after the current one."""
    candidates = day[
        day["settlement_price"].notna()
        & (day["expiration"] > current_expiration)
        & (day["expiration"] >= trade_date)
    ].sort_values("expiration")
    if candidates.empty:
        return None

    # Use a contract that is still before its roll deadline when possible.
    safe = candidates[
        candidates["raw_symbol"].map(deadlines).gt(trade_date)
    ]
    selected = safe.iloc[0] if not safe.empty else candidates.iloc[0]
    return str(selected["raw_symbol"])


def _build_one_root(
    root_panel: pd.DataFrame,
    *,
    hard_roll_bdays: int,
    initial_index: float,
    settlement_type: str,
) -> pd.DataFrame:
    """Build the continuous series for one futures root."""
    root = str(root_panel["root"].iloc[0])
    root_panel = root_panel.sort_values(
        ["trade_date", "expiration", "raw_symbol"]
    ).reset_index(drop=True)

    contract_meta = (
        root_panel[["raw_symbol", "expiration"]]
        .drop_duplicates("raw_symbol")
        .set_index("raw_symbol")
    )
    expirations = contract_meta["expiration"].to_dict()
    deadlines = {
        symbol: pd.Timestamp(expiration) - BDay(hard_roll_bdays)
        for symbol, expiration in expirations.items()
    }

    category = (
        root_panel["category"].dropna().iloc[0]
        if "category" in root_panel.columns and root_panel["category"].notna().any()
        else None
    )

    records: list[dict] = []
    held_symbol: Optional[str] = None
    previous_day_prices: dict[str, float] = {}
    previous_trade_date: Optional[pd.Timestamp] = None
    continuous_index = float(initial_index)

    for trade_date, raw_day in root_panel.groupby("trade_date", sort=True):
        trade_date = pd.Timestamp(trade_date)
        day = (
            raw_day.sort_values(["expiration", "raw_symbol"])
            .drop_duplicates("raw_symbol", keep="last")
            .set_index("raw_symbol", drop=False)
        )

        initialized_today = False
        if held_symbol is None:
            held_symbol = _select_front_contract(day, trade_date, deadlines)
            initialized_today = True

        position_symbol = held_symbol
        position_expiration = (
            expirations.get(position_symbol) if position_symbol is not None else pd.NaT
        )
        position_deadline = (
            deadlines.get(position_symbol) if position_symbol is not None else pd.NaT
        )

        current_row_available = (
            position_symbol is not None and position_symbol in day.index
        )
        held_price = (
            day.at[position_symbol, "settlement_price"]
            if current_row_available
            else np.nan
        )
        held_oi = (
            day.at[position_symbol, "open_interest"]
            if current_row_available
            else np.nan
        )
        prior_held_price = (
            previous_day_prices.get(position_symbol, np.nan)
            if position_symbol is not None
            else np.nan
        )

        # Calculate the return using the same contract on both dates.
        if previous_trade_date is None or initialized_today:
            daily_return = np.nan
        elif (
            pd.notna(held_price)
            and pd.notna(prior_held_price)
            and float(prior_held_price) != 0.0
        ):
            daily_return = float(held_price) / float(prior_held_price) - 1.0
        else:
            daily_return = np.nan

        missing_return = previous_trade_date is not None and pd.isna(daily_return)
        if pd.notna(daily_return):
            continuous_index *= 1.0 + float(daily_return)

        next_symbol = None
        next_expiration = pd.NaT
        next_price = np.nan
        next_oi = np.nan
        if position_symbol is not None and pd.notna(position_expiration):
            next_symbol = _select_next_contract(
                day,
                pd.Timestamp(position_expiration),
                trade_date,
                deadlines,
            )
            if next_symbol is not None:
                next_expiration = expirations[next_symbol]
                next_price = day.at[next_symbol, "settlement_price"]
                next_oi = day.at[next_symbol, "open_interest"]

        missing_oi = bool(
            next_symbol is not None and (pd.isna(held_oi) or pd.isna(next_oi))
        )
        past_deadline = bool(
            position_symbol is not None
            and pd.notna(position_deadline)
            and trade_date >= pd.Timestamp(position_deadline)
        )

        roll_flag = False
        roll_reason: Optional[str] = None
        roll_to_symbol: Optional[str] = None

        if position_symbol is None:
            roll_reason = "no_eligible_contract"
        elif not current_row_available:
            if next_symbol is not None:
                roll_flag = True
                roll_reason = "held_contract_unavailable"
                roll_to_symbol = next_symbol
        elif past_deadline:
            if next_symbol is not None:
                roll_flag = True
                roll_reason = "expiration_deadline"
                roll_to_symbol = next_symbol
            else:
                roll_reason = "past_deadline_no_next_contract"
        elif (
            next_symbol is not None
            and not missing_oi
            and float(next_oi) > float(held_oi)
        ):
            roll_flag = True
            roll_reason = "open_interest_crossover"
            roll_to_symbol = next_symbol

        roll_to_expiration = (
            expirations.get(roll_to_symbol) if roll_to_symbol is not None else pd.NaT
        )
        roll_to_deadline = (
            deadlines.get(roll_to_symbol) if roll_to_symbol is not None else pd.NaT
        )

        # Store the price difference for review, but do not use it in returns.
        roll_price_gap = np.nan
        roll_price_ratio = np.nan
        if roll_flag and pd.notna(held_price) and pd.notna(next_price):
            roll_price_gap = float(next_price) - float(held_price)
            if float(held_price) != 0.0:
                roll_price_ratio = float(next_price) / float(held_price)

        end_of_day_symbol = roll_to_symbol if roll_flag else position_symbol

        records.append(
            {
                "trade_date": trade_date,
                "root": root,
                "category": category,
                "settlement_type": settlement_type,
                "deadline_basis": "expiration_proxy",
                "hard_roll_bdays": hard_roll_bdays,
                # Current contract
                "held_symbol": position_symbol,
                "held_expiration": position_expiration,
                "held_settlement": held_price,
                "prior_held_settlement": prior_held_price,
                "held_open_interest": held_oi,
                "hard_roll_date": position_deadline,
                # Next contract
                "next_symbol": next_symbol,
                "next_expiration": next_expiration,
                "next_settlement": next_price,
                "next_open_interest": next_oi,
                # Roll decision
                "roll_flag": roll_flag,
                "roll_reason": roll_reason,
                "roll_to_symbol": roll_to_symbol,
                "roll_to_expiration": roll_to_expiration,
                "roll_to_hard_roll_date": roll_to_deadline,
                "end_of_day_symbol": end_of_day_symbol,
                # Return series
                "continuous_return": daily_return,
                "continuous_index": continuous_index,
                # Checks
                "roll_price_gap": roll_price_gap,
                "roll_price_ratio": roll_price_ratio,
                "missing_oi": missing_oi,
                "missing_return": bool(missing_return),
                "held_contract_available": bool(current_row_available),
                "past_deadline": past_deadline,
                "past_deadline_no_next": bool(
                    past_deadline and next_symbol is None
                ),
            }
        )

        held_symbol = end_of_day_symbol
        previous_day_prices = day["settlement_price"].to_dict()
        previous_trade_date = trade_date

    return pd.DataFrame.from_records(records)


def build_oi_continuous_series(
    panel: pd.DataFrame,
    *,
    root: Optional[str] = None,
    hard_roll_bdays: HardRollInput = DEFAULT_HARD_ROLL_BDAYS,
    settlement_type_by_root: Optional[Mapping[str, str]] = None,
    initial_index: float = 100.0,
) -> pd.DataFrame:
    """Build open-interest-rolled returns for one or more roots.

    The held contract earns today's return. Any roll takes effect after today's
    settlement and is used for the next return.
    """
    if not np.isfinite(initial_index) or initial_index <= 0:
        raise ValueError("initial_index must be a positive finite number.")

    df = _prepare_panel(panel)
    if root is not None:
        df = df[df["root"] == root].copy()
        if df.empty:
            raise ValueError(f"No rows found for root={root!r}.")

    settlement_type_by_root = settlement_type_by_root or {}
    outputs: list[pd.DataFrame] = []

    for root_name, root_panel in df.groupby("root", sort=True):
        root_name = str(root_name)
        deadline_days = _resolve_root_setting(
            hard_roll_bdays,
            root_name,
            setting_name="hard_roll_bdays",
        )
        settlement_type = str(
            settlement_type_by_root.get(root_name, "unknown")
        )
        outputs.append(
            _build_one_root(
                root_panel,
                hard_roll_bdays=deadline_days,
                initial_index=float(initial_index),
                settlement_type=settlement_type,
            )
        )

    if not outputs:
        return pd.DataFrame()

    result = pd.concat(outputs, ignore_index=True)
    return result.sort_values(["root", "trade_date"]).reset_index(drop=True)


def liquidity_roll_oi(
    panel: pd.DataFrame,
    root: Optional[str] = None,
    oi_col: str = "open_interest",
    *,
    hard_roll_bdays: HardRollInput = DEFAULT_HARD_ROLL_BDAYS,
) -> pd.DataFrame:
    """Run the open-interest rolling method."""
    if oi_col != "open_interest":
        raise ValueError(
            "This data source supports open-interest rolling only; "
            "oi_col must be 'open_interest'."
        )
    return build_oi_continuous_series(
        panel,
        root=root,
        hard_roll_bdays=hard_roll_bdays,
    )



def main() -> None:
    """Load downloaded contracts and save the rolled return series."""
    rolling_dir = Path(__file__).resolve().parent
    src_dir = rolling_dir.parent

    input_path = src_dir / "data_download" / "all_contracts_settlement.parquet"
    output_path = rolling_dir / "continuous_oi_returns.parquet"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Run src/data_download/download_functions.py first."
        )

    source = pd.read_parquet(input_path)

    # bad = source[
    # source["raw_symbol"].isna()
    # | source["expiration"].isna()
    # ]

    # print(len(bad))
    # print(bad.head(25))
    continuous = build_oi_continuous_series(source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    continuous.to_parquet(output_path, index=False)
    print(f"Saved {len(continuous):,} rows to {output_path}")


if __name__ == "__main__":
    main()