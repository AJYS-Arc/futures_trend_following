
# download_functions.py
# This code covers issues #1 through #3


# --------------------------------------------------------------------------------------------------------------
# Preliminaries
# --------------------------------------------------------------------------------------------------------------


### Import libraries

import databento as db
import pandas as pd
import os
import time
import set_client
from concurrent.futures import ThreadPoolExecutor, as_completed


### Set variables

# Dataset name
DATASET = "GLBX.MDP3"

# Period for the analysis
START_DATE = "2010-06-06"  
END_DATE = "2026-06-30"    

# Period for data download
DATASET_AVAILABLE_START = pd.Timestamp("2010-06-06", tz="UTC")
DATASET_AVAILABLE_END = pd.Timestamp("2026-07-02", tz="UTC")
SETTLEMENT_CUTOFF_DATE = pd.Timestamp(END_DATE).date() 

# Libaries for saving data
CHECKPOINT_DIR = "checkpoints"                      # one parquet file per root saved here
CONTRACTS_CHECKPOINT_PATH = "contracts_definitions.parquet"
OUTPUT_PATH = "all_contracts_settlement.parquet"    # final merged file

# Relevant data items
OUTRIGHT_CLASS = "F"          # instrument_class for outright futures
SETTLEMENT_STAT_TYPE = 3      # stat_type code for settlement price
OPEN_INTEREST_STAT_TYPE = 9   # stat_type code for open interest


### Universe

UNIVERSE = {
    "fx": [
        "6A",  # Australian Dollar futures (AUD/USD)
        "6C",  # Canadian Dollar futures (CAD/USD)
        "6B",  # British Pound futures (GBP/USD)
        "6E",  # Euro FX futures (EUR/USD)
        "6J",  # Japanese Yen futures (JPY/USD)
        "6S",  # Swiss Franc futures (CHF/USD)
    ],
    "energy": [
        "CL",  # WTI Crude Oil futures
        "HO",  # NY Harbor ULSD (Heating Oil) futures
        "NG",  # Henry Hub Natural Gas futures
        "RB",  # RBOB Gasoline futures
    ],
    "metals": [
        "GC",  # Gold futures
        "SI",  # Silver futures
        "HG",  # Copper futures
        "PL",  # Platinum futures
        "PA",  # Palladium futures
    ],
    "rates": [
        "ZT",  # 2-Year T-Note futures
        "ZF",  # 5-Year T-Note futures
        "ZN",  # 10-Year T-Note futures
        "ZB",  # 30-Year T-Bond futures
    ],
    "equity_index": [
        "ES",   # E-mini S&P 500 futures
        "NQ",   # E-mini Nasdaq-100 futures
        "NIY",  # Nikkei 225 (Yen-denominated) futures
        "RTY",  # E-mini Russell 2000 futures
    ],
    "ags_livestock": [
        "ZC",  # Corn futures
        "ZS",  # Soybean futures
        "ZW",  # Chicago SRW Wheat futures
        "ZL",  # Soybean Oil futures
        "LE",  # Live Cattle futures
        "HE",  # Lean Hog futures
        "GF",  # Feeder Cattle futures
    ],
}



# --------------------------------------------------------------------------------------------------------------
# Retrieve contracts definitions from Definitions schema 
# --------------------------------------------------------------------------------------------------------------

# These functions retrieve the contract data from the Definitions schema. We are interested in symbol, id, expiration.
# We will then use these symbols/expirations to retrieve settlement prices and open interest from Statistics schema.


# Pull definitions for a single monthly snapshot date
def fetch_snapshot(client: db.Historical, sym: str, snap: pd.Timestamp):
    try:
        df = client.timeseries.get_range(
            dataset=DATASET, symbols=[sym], stype_in="parent",
            schema="definition",
            start=snap.strftime("%Y-%m-%d"),
            end=(snap + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        ).to_df().reset_index()
        return df if not df.empty else None
    except Exception as e:
        print(f"    snapshot {snap.date()} failed: {e}")
        return None

# Generate monthly snapshot dates
def generate_snapshots(start_date: str, end_date: str) -> pd.DatetimeIndex:
    snapshots = pd.date_range(start_date, end_date, freq="1MS")
    return pd.DatetimeIndex([
        d if d.weekday() < 5 else d + pd.offsets.BDay(1)
        for d in snapshots
    ])

#   Pull definitions on monthly snapshot dates, fetched in parallel via a thread pool.
def get_outright_contracts(client: db.Historical, root: str, max_workers: int = 15,
                             expiration_cutoff: str = "2026-12-31") -> pd.DataFrame:
    """
    Filters to actual outright contract months only (excludes continuous
    references and calendar spreads), and returns raw_symbol, instrument_id,
    activation, and expiration.
    """
    sym = f"{root}.FUT"
    snapshots = generate_snapshots(START_DATE, END_DATE)
    total = len(snapshots)

    all_frames = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_snapshot, client, sym, snap): snap for snap in snapshots}

        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result is not None:
                all_frames.append(result)

            if completed % 20 == 0 or completed == total:
                print(f"  [{root}] snapshots: {completed}/{total} done")

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    combined = combined[~combined["raw_symbol"].str.contains(r"\.", regex=True)]
    combined = combined[combined["instrument_class"] == OUTRIGHT_CLASS]
    combined = (
        combined.sort_values("ts_event")
        .groupby("instrument_id", as_index=False)
        .tail(1)
    )

     # Filter out contracts expiring after the cutoff
    before_count = len(combined)
    combined = combined[combined["expiration"] <= pd.Timestamp(expiration_cutoff, tz=combined["expiration"].dt.tz)]
    after_count = len(combined)

    print(f"  [{root}] resolved {after_count} outright contracts "
          f"(dropped {before_count - after_count} expiring after {expiration_cutoff}).")

    print(f"  [{root}] resolved {len(combined)} outright contracts.")

    return combined[["raw_symbol", "instrument_id", "activation", "expiration"]].reset_index(drop=True)



# --------------------------------------------------------------------------------------------------------------
# Retrieve contracts settlement prices and open interest from Statistics schema 
# --------------------------------------------------------------------------------------------------------------


# Sort contracts by expiration (earliest first) and split into fixed-size chunks. Each chunk becomes a single batched API call.
def chunk_contracts(contracts: pd.DataFrame, chunk_size: int = 15) -> list[pd.DataFrame]:
    sorted_contracts = contracts.sort_values("expiration").reset_index(drop=True)
    return [
        sorted_contracts.iloc[i:i + chunk_size].reset_index(drop=True)
        for i in range(0, len(sorted_contracts), chunk_size)
    ]


# Pull settlement + OI for a batch of contracts in one call.
def fetch_contract_chunk(client: db.Historical, chunk: pd.DataFrame, root: str, category: str) -> pd.DataFrame:
    chunk_start = max(chunk["activation"].min(), DATASET_AVAILABLE_START)
    chunk_end = min(chunk["expiration"].max() + pd.Timedelta(days=1), DATASET_AVAILABLE_END)

    if chunk_start >= chunk_end:
        return pd.DataFrame()

    symbols = chunk["raw_symbol"].tolist()

    try:
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=symbols,
            schema="statistics",
            stype_in="raw_symbol",
            start=chunk_start,
            end=chunk_end,
        )
    except Exception as e:
        print(f"    FAILED chunk {symbols[0]}..{symbols[-1]}: {e}")
        return pd.DataFrame()

    df = data.to_df()
    if df.empty:
        return pd.DataFrame()

    df = df[df["stat_type"].isin([SETTLEMENT_STAT_TYPE, OPEN_INTEREST_STAT_TYPE])].copy()
    if df.empty:
        return pd.DataFrame()

    df["trade_date"] = pd.to_datetime(df["ts_ref"]).dt.date
    df["value"] = df["price"].where(df["stat_type"] == SETTLEMENT_STAT_TYPE, df["quantity"])
    df["stat_name"] = df["stat_type"].map({
        SETTLEMENT_STAT_TYPE: "settlement_price",
        OPEN_INTEREST_STAT_TYPE: "open_interest",
    })

    df = df.sort_values("ts_event")

    pivoted = df.pivot_table(
        index=["trade_date", "instrument_id"],
        columns="stat_name", values="value", aggfunc="last",
    ).reset_index()

    # Merge against chunk's own contract metadata to attach the right values per row.
    pivoted = pivoted.merge(
        chunk[["instrument_id", "raw_symbol", "expiration"]],
        on="instrument_id", how="left",
    )
    pivoted["root"] = root
    pivoted["category"] = category
    return pivoted


# Pull settlement price and open interest for every outright contract
def pull_settlement_and_oi(client: db.Historical, contracts: pd.DataFrame, root: str, category: str,
                            chunk_size: int = 15, max_workers: int = 8) -> pd.DataFrame:

    chunks = chunk_contracts(contracts, chunk_size)
    frames = []
    total = len(chunks)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_contract_chunk, client, chunk, root, category): chunk["raw_symbol"].tolist()
            for chunk in chunks
        }
        for future in as_completed(futures):
            completed += 1
            df = future.result()
            if not df.empty:
                frames.append(df)

            print(f"  [{root}] chunks: {completed}/{total} done")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    before_count = len(result)
    result = result[result["trade_date"] <= SETTLEMENT_CUTOFF_DATE]
    after_count = len(result)
    if before_count != after_count:
        print(f"  [{root}] dropped {before_count - after_count} rows with trade_date after {SETTLEMENT_CUTOFF_DATE}.")

    return result



# ---------------------------------------------------------------------------
# Main pull function
# -------
# --------------------------------------------------------------------


# Pull settlement price, open interest, and expiration date for every outright contract across the full universe.
def pull_all_settlement_data() -> pd.DataFrame:

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    client = set_client.get_client()
    all_frames = []

    # Load the single contracts checkpoint once, up front, if it exists
    if os.path.exists(CONTRACTS_CHECKPOINT_PATH):
        all_contracts = pd.read_parquet(CONTRACTS_CHECKPOINT_PATH)
    else:
        all_contracts = pd.DataFrame()

    for category, roots in UNIVERSE.items():
        for root in roots:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{root}.parquet")

            if os.path.exists(checkpoint_path):
                print(f"[{category}] {root}: checkpoint found, loading from disk.")
                df = pd.read_parquet(checkpoint_path)
                all_frames.append(df)
                continue

            start_time = time.time()

            if not all_contracts.empty and root in all_contracts["root"].values:
                print(f"[{category}] {root}: contracts checkpoint found, loading from disk.")
                contracts = all_contracts[all_contracts["root"] == root].drop(columns=["root"]).reset_index(drop=True)
            else:
                print(f"[{category}] {root}: resolving outright contracts...")
                contracts = get_outright_contracts(client, root)

                if contracts.empty:
                    print(f"  No outright contracts found for {root}, skipping.")
                    continue

                tagged = contracts.copy()
                tagged["root"] = root
                all_contracts = pd.concat([all_contracts, tagged], ignore_index=True)
                all_contracts.to_parquet(CONTRACTS_CHECKPOINT_PATH, index=False)

            if contracts.empty:
                print(f"  No outright contracts found for {root}, skipping.")
                continue

            print(f"  Found {len(contracts)} outright contract(s): "
                  f"{contracts['raw_symbol'].tolist()[:3]}"
                  f"{'...' if len(contracts) > 3 else ''}")

            df = pull_settlement_and_oi(client, contracts, root, category)
            if df.empty:
                print(f"  No settlement data returned for {root}.")
                continue

            elapsed = time.time() - start_time
            print(f"  Pulled {len(df):,} rows for {root}. Saving checkpoint...")
            df.to_parquet(checkpoint_path, index=False)
            all_frames.append(df)

            print(f"  Done with {root} in {elapsed:.1f} seconds.")

    if not all_frames:
        raise RuntimeError("No data was successfully pulled for any product.")

    result = pd.concat(all_frames, ignore_index=True)

    keep_cols = [
        c for c in [
            "trade_date", "root", "category", "raw_symbol", "instrument_id",
            "expiration", "settlement_price", "open_interest",
        ]
        if c in result.columns
    ]
    result = result[keep_cols]

    # Drop rows fully duplicated on (instrument_id, expiration, trade_date)
    before_count = len(result)
    result = result.drop_duplicates(subset=["instrument_id", "expiration", "trade_date"])
    after_count = len(result)
    if before_count != after_count:
        print(f"Dropped {before_count - after_count} duplicate rows (instrument_id, expiration, trade_date).")

    # Restrict to observations on/after the universe start date
    UNIVERSE_START_DATE = pd.Timestamp("2010-06-04").date()
    before_count = len(result)
    result = result[pd.to_datetime(result["trade_date"]) >= pd.Timestamp(UNIVERSE_START_DATE)]
    after_count = len(result)
    if before_count != after_count:
        print(f"Dropped {before_count - after_count} rows before {UNIVERSE_START_DATE}.")

    result = result.sort_values(["root", "raw_symbol", "trade_date"])

    return result


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pull_all_settlement_data()
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} total rows to {OUTPUT_PATH}")
