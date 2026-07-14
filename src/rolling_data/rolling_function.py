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
    "instrument_id",
    "expiration",
    "settlement_price",
    "open_interest",
}

# Default only; pass per-root values for market-specific deadlines.
DEFAULT_HARD_ROLL_BDAYS = 5


HardRollInput = Union[int, Mapping[str, int]]


def _resolve_root_setting(
    value: HardRollInput,
    root: str,
    *,
    setting_name: str,
) -> int:
    """Return one integer setting for a root."""
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
    """Validate and normalize the input panel."""
    missing = REQUIRED_COLUMNS.difference(panel.columns)
    if missing:
        raise KeyError(f"Panel is missing required columns: {sorted(missing)}")
    df = panel.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.tz_localize(None)
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.tz_localize(None)
    df["settlement_price"] = pd.to_numeric(df["settlement_price"], errors="coerce")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    df["instrument_id"] = pd.to_numeric(df["instrument_id"], errors="coerce")
    invalid_he_price = (
    (df["root"] == "HE")
    & (df["settlement_price"] <= 0)
    )

    if invalid_he_price.any():
        print(
            f"Replacing {invalid_he_price.sum():,} non-positive "
            "HE settlement prices."
        )
        df.loc[invalid_he_price, "settlement_price"] = np.nan

    df = df.sort_values(["instrument_id", "trade_date"])
    he_prices = (
        df.loc[df["root"] == "HE"]
        .groupby("instrument_id")["settlement_price"]
        .ffill(limit=2)
    )

    df.loc[df["root"] == "HE", "settlement_price"] = he_prices
    key_cols = [
        "trade_date",
        "root",
        "raw_symbol",
        "instrument_id",
        "expiration",
    ]
    invalid = df[key_cols].isna().any(axis=1)

    if invalid.any():
        print(f"Dropping {invalid.sum():,} rows with missing contract metadata.")
        df = df.loc[~invalid].copy()

    if df.empty:
        raise ValueError("No valid contract rows remain after cleaning.")

    df["instrument_id"] = df["instrument_id"].astype("int64")

    # Keep one row per instrument and trade date.
    sort_cols = [
        "root",
        "trade_date",
        "expiration",
        "instrument_id",
    ]
    df = df.sort_values(sort_cols)
    df = df.drop_duplicates(
        subset=["root", "trade_date", "instrument_id"],
        keep="last",
    )

    return df.reset_index(drop=True)


def _select_front_contract(
    day: pd.DataFrame,
    trade_date: pd.Timestamp,
    deadlines: Mapping[int, pd.Timestamp],
) -> Optional[int]:
    """Choose the nearest valid contract not past its deadline."""
    candidates = day[
        day["settlement_price"].notna()
        & (day["expiration"] >= trade_date)
    ].sort_values("expiration")

    if candidates.empty:
        return None

    safe = candidates[
        candidates["instrument_id"].map(deadlines).gt(trade_date)
    ]
    selected = safe.iloc[0] if not safe.empty else candidates.iloc[0]
    return int(selected["instrument_id"])


def _select_next_contract(
    day: pd.DataFrame,
    current_expiration: pd.Timestamp,
    trade_date: pd.Timestamp,
    deadlines: Mapping[int, pd.Timestamp],
) -> Optional[int]:
    """Choose the nearest valid later contract."""
    candidates = day[
        day["settlement_price"].notna()
        & (day["expiration"] > current_expiration)
        & (day["expiration"] >= trade_date)
    ].sort_values("expiration")

    if candidates.empty:
        return None

    safe = candidates[
        candidates["instrument_id"].map(deadlines).gt(trade_date)
    ]
    selected = safe.iloc[0] if not safe.empty else candidates.iloc[0]
    return int(selected["instrument_id"])

def _build_one_root(
    root_panel: pd.DataFrame,
    *,
    hard_roll_bdays: int,
    initial_index: float,
    settlement_type: str,
) -> pd.DataFrame:
    """Build the roll path and returns for one root."""
    root = str(root_panel["root"].iloc[0])
    root_panel = root_panel.sort_values(
        ["trade_date", "expiration", "instrument_id"]
    ).reset_index(drop=True)

    # instrument_id is the internal contract key. raw_symbol is display only.
    contract_meta = (
        root_panel[
            ["instrument_id", "raw_symbol", "expiration"]
        ]
        .drop_duplicates("instrument_id")
        .set_index("instrument_id")
    )

    expirations = contract_meta["expiration"].to_dict()
    symbols = contract_meta["raw_symbol"].to_dict()
    deadlines = {
        instrument_id: pd.Timestamp(expiration) - BDay(hard_roll_bdays)
        for instrument_id, expiration in expirations.items()
    }

    category = (
        root_panel["category"].dropna().iloc[0]
        if "category" in root_panel.columns
        and root_panel["category"].notna().any()
        else None
    )

    records: list[dict] = []
    held_contract: Optional[int] = None
    previous_day_prices: dict[int, float] = {}
    previous_trade_date: Optional[pd.Timestamp] = None
    continuous_index = float(initial_index)

    for trade_date, raw_day in root_panel.groupby("trade_date", sort=True):
        trade_date = pd.Timestamp(trade_date)
        day = (
            raw_day.sort_values(["expiration", "instrument_id"])
            .drop_duplicates("instrument_id", keep="last")
            .set_index("instrument_id", drop=False)
        )

        initialized_today = False
        if held_contract is None:
            held_contract = _select_front_contract(
                day,
                trade_date,
                deadlines,
            )
            initialized_today = True

        position_contract = held_contract
        position_symbol = (
            symbols.get(position_contract)
            if position_contract is not None
            else None
        )
        position_expiration = (
            expirations.get(position_contract)
            if position_contract is not None
            else pd.NaT
        )
        position_deadline = (
            deadlines.get(position_contract)
            if position_contract is not None
            else pd.NaT
        )

        current_row_available = (
            position_contract is not None
            and position_contract in day.index
        )

        held_price = (
            day.at[position_contract, "settlement_price"]
            if current_row_available
            else np.nan
        )
        held_oi = (
            day.at[position_contract, "open_interest"]
            if current_row_available
            else np.nan
        )
        prior_held_price = (
            previous_day_prices.get(position_contract, np.nan)
            if position_contract is not None
            else np.nan
        )

        # Use the same contract on both sides of each return.
        if previous_trade_date is None or initialized_today:
            daily_return = np.nan
        elif (
            pd.notna(held_price)
            and pd.notna(prior_held_price)
            and float(prior_held_price) != 0.0
        ):
            daily_return = (
                float(held_price) / float(prior_held_price) - 1.0
            )
        else:
            daily_return = np.nan

        missing_return = (
            previous_trade_date is not None
            and pd.isna(daily_return)
        )

        if pd.notna(daily_return):
            continuous_index *= 1.0 + float(daily_return)

        next_contract: Optional[int] = None
        next_symbol: Optional[str] = None
        next_expiration = pd.NaT
        next_price = np.nan
        next_oi = np.nan

        if (
            position_contract is not None
            and pd.notna(position_expiration)
        ):
            next_contract = _select_next_contract(
                day,
                pd.Timestamp(position_expiration),
                trade_date,
                deadlines,
            )

            if next_contract is not None:
                next_symbol = symbols.get(next_contract)
                next_expiration = expirations[next_contract]
                next_price = day.at[
                    next_contract,
                    "settlement_price",
                ]
                next_oi = day.at[
                    next_contract,
                    "open_interest",
                ]

        missing_oi = bool(
            next_contract is not None
            and (pd.isna(held_oi) or pd.isna(next_oi))
        )

        past_deadline = bool(
            position_contract is not None
            and pd.notna(position_deadline)
            and trade_date >= pd.Timestamp(position_deadline)
        )

        roll_flag = False
        roll_reason: Optional[str] = None
        roll_to_contract: Optional[int] = None

        if position_contract is None:
            roll_reason = "no_eligible_contract"
        elif not current_row_available:
            if next_contract is not None:
                roll_flag = True
                roll_reason = "held_contract_unavailable"
                roll_to_contract = next_contract
        elif past_deadline:
            if next_contract is not None:
                roll_flag = True
                roll_reason = "expiration_deadline"
                roll_to_contract = next_contract
            else:
                roll_reason = "past_deadline_no_next_contract"
        elif (
            next_contract is not None
            and not missing_oi
            and float(next_oi) > float(held_oi)
        ):
            roll_flag = True
            roll_reason = "open_interest_crossover"
            roll_to_contract = next_contract

        roll_to_symbol = (
            symbols.get(roll_to_contract)
            if roll_to_contract is not None
            else None
        )
        roll_to_expiration = (
            expirations.get(roll_to_contract)
            if roll_to_contract is not None
            else pd.NaT
        )
        roll_to_deadline = (
            deadlines.get(roll_to_contract)
            if roll_to_contract is not None
            else pd.NaT
        )

        # Roll-gap fields are diagnostics only.
        roll_price_gap = np.nan
        roll_price_ratio = np.nan

        if (
            roll_flag
            and pd.notna(held_price)
            and pd.notna(next_price)
        ):
            roll_price_gap = float(next_price) - float(held_price)
            if float(held_price) != 0.0:
                roll_price_ratio = (
                    float(next_price) / float(held_price)
                )

        end_of_day_contract = (
            roll_to_contract
            if roll_flag
            else position_contract
        )
        end_of_day_symbol = (
            symbols.get(end_of_day_contract)
            if end_of_day_contract is not None
            else None
        )

        # Keep the existing output schema unchanged.
        records.append(
            {
                "trade_date": trade_date,
                "root": root,
                "category": category,
                "settlement_type": settlement_type,
                "deadline_basis": "expiration_proxy",
                "hard_roll_bdays": hard_roll_bdays,
                "held_symbol": position_symbol,
                "held_expiration": position_expiration,
                "held_settlement": held_price,
                "prior_held_settlement": prior_held_price,
                "held_open_interest": held_oi,
                "hard_roll_date": position_deadline,
                "next_symbol": next_symbol,
                "next_expiration": next_expiration,
                "next_settlement": next_price,
                "next_open_interest": next_oi,
                "roll_flag": roll_flag,
                "roll_reason": roll_reason,
                "roll_to_symbol": roll_to_symbol,
                "roll_to_expiration": roll_to_expiration,
                "roll_to_hard_roll_date": roll_to_deadline,
                "end_of_day_symbol": end_of_day_symbol,
                "continuous_return": daily_return,
                "continuous_index": continuous_index,
                "roll_price_gap": roll_price_gap,
                "roll_price_ratio": roll_price_ratio,
                "missing_oi": missing_oi,
                "missing_return": bool(missing_return),
                "held_contract_available": bool(
                    current_row_available
                ),
                "past_deadline": past_deadline,
                "past_deadline_no_next": bool(
                    past_deadline and next_contract is None
                ),
            }
        )

        held_contract = end_of_day_contract
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
    """Build OI-rolled returns for one or more roots.

    ``held_symbol`` earns today's return. ``end_of_day_symbol`` is used for the
    next return. ``hard_roll_bdays`` may be one integer or a per-root mapping.
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


# Backward-compatible wrapper.
def liquidity_roll_oi(
    panel: pd.DataFrame,
    root: Optional[str] = None,
    oi_col: str = "open_interest",
    *,
    hard_roll_bdays: HardRollInput = DEFAULT_HARD_ROLL_BDAYS,
) -> pd.DataFrame:
    """Run the open-interest-only rolling method."""
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

    continuous = build_oi_continuous_series(source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    continuous.to_parquet(output_path, index=False)
    print(f"Saved {len(continuous):,} rows to {output_path}")


if __name__ == "__main__":
    main()