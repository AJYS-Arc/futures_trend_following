"""Issue #3 — Set up Databento data access.

Thin wrapper around ``databento.Historical`` that reads the API key from the
environment (loaded from a local ``.env`` via python-dotenv). Keeping this in one
place means the rest of the code never touches the raw key.
"""
from __future__ import annotations

import os
from datetime import date

DATASET = "GLBX.MDP3"  # CME Globex (covers CME/CBOT/NYMEX/COMEX)


def _load_key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # dotenv is optional; env var may be set another way
    key = os.getenv("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError(
            "DATABENTO_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


def get_client():
    """Return an authenticated databento.Historical client."""
    import databento as db  # imported lazily so the package works without it installed

    return db.Historical(_load_key())


def fetch_ohlcv(
    symbols: list[str],
    start: str | date,
    end: str | date,
    schema: str = "ohlcv-1d",
    stype_in: str = "continuous",
):
    """Fetch OHLCV bars for one or more symbols and return a tidy DataFrame.

    ``stype_in="continuous"`` uses Databento's front-month continuous symbology
    (e.g. ``ES.c.0``). Prices are returned already scaled to real units by
    ``.to_df()``.
    """
    client = get_client()
    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=symbols,
        schema=schema,
        stype_in=stype_in,
        start=str(start),
        end=str(end),
    )
    df = data.to_df()
    return df


if __name__ == "__main__":
    # Smoke test: pull one day of ES to confirm auth works.
    df = fetch_ohlcv(["ES.c.0"], "2024-01-02", "2024-01-03")
    print(df.head())
