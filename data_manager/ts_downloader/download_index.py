"""
Index daily data downloader using tushare index_daily API.

Download benchmark indices (上证指数, 深证成指, 沪深300) and store as parquet files.
Supports incremental updates: merges with existing data if parquet already exists.

Usage:
    # As a standalone script (downloads all 3 indices from 2010 to today):
    python data_manager/ts_downloader/download_index.py

    # As a module:
    from data_manager.ts_downloader.download_index import download_all
    download_all(start_date="20100101")
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import tushare as ts

from vnpy.trader.setting import SETTINGS

# Project root relative to this file
PROJECT_ROOT = Path(__file__).parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "core" / "alpha_db" / "index"

# Benchmark indices to download
BENCHMARK_INDICES = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "000300.SH": "沪深300",
}


def _get_pro():
    """Initialize and return tushare pro API."""
    token = SETTINGS["datafeed.password"]
    if not token:
        raise RuntimeError("tushare_token not found in SETTINGS['datafeed.password']")
    ts.set_token(token)
    return ts.pro_api()


def download_index(ts_code: str, start_date: str = "20100101", end_date: str = None) -> pl.DataFrame:
    """
    Download index daily data for a single index.

    Args:
        ts_code: Tushare index code (e.g., "000001.SH")
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format. Defaults to today.

    Returns:
        Polars DataFrame with columns: trade_date, close, open, high, low, pct_chg
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    pro = _get_pro()

    print(f"  Downloading {ts_code} ({BENCHMARK_INDICES.get(ts_code, '')}) from {start_date} to {end_date}...")

    df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

    if df is None or df.empty:
        print(f"  No data returned for {ts_code}")
        return pl.DataFrame()

    # Convert to polars and process
    result = pl.from_pandas(df)

    # Keep only relevant columns and convert trade_date to datetime
    columns_to_keep = ["trade_date", "close", "open", "high", "low", "pct_chg"]
    result = result.select(
        [pl.col(c) for c in columns_to_keep if c in result.columns]
    )

    # Convert trade_date from string (YYYYMMDD) to datetime
    result = result.with_columns(
        pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d")
    )

    # Ensure numeric types
    for col_name in ["close", "open", "high", "low", "pct_chg"]:
        if col_name in result.columns:
            result = result.with_columns(pl.col(col_name).cast(pl.Float64))

    # Sort by date ascending
    result = result.sort("trade_date")

    print(f"  Downloaded {len(result)} rows for {ts_code}")
    return result


def download_all(start_date: str = "20100101", end_date: str = None):
    """
    Download all benchmark indices and store as parquet files.
    Supports incremental updates: merges with existing data.

    Args:
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format. Defaults to today.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Downloading Benchmark Index Data ===")
    print(f"Output directory: {INDEX_DIR}")

    for ts_code, name in BENCHMARK_INDICES.items():
        parquet_path = INDEX_DIR / f"{ts_code}.parquet"

        # Check existing data
        existing_df = None
        existing_end_date = None
        if parquet_path.exists():
            try:
                existing_df = pl.read_parquet(parquet_path)
                if not existing_df.is_empty():
                    existing_end_date = (
                        existing_df["trade_date"].max().strftime("%Y%m%d")
                    )
                    print(f"  Existing data for {ts_code}: {len(existing_df)} rows, latest: {existing_end_date}")
            except Exception as e:
                print(f"  Warning: Could not read existing parquet for {ts_code}: {e}")
                existing_df = None

        # Determine download start date
        if existing_end_date:
            # Incremental: start from the day after last available date
            last_date = datetime.strptime(existing_end_date, "%Y%m%d")
            download_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
            if download_start >= (end_date or datetime.now().strftime("%Y%m%d")):
                print(f"  {ts_code}: Already up to date. Skipping.")
                continue
        else:
            download_start = start_date

        # Download new data
        try:
            new_df = download_index(ts_code, download_start, end_date)
            time.sleep(0.3)  # Rate limiting
        except Exception as e:
            print(f"  Error downloading {ts_code}: {e}")
            continue

        # Merge with existing
        if new_df.is_empty():
            continue

        if existing_df is not None and not existing_df.is_empty():
            merged = pl.concat([existing_df, new_df])
            # Deduplicate by trade_date
            merged = merged.unique(subset=["trade_date"], keep="last")
            merged = merged.sort("trade_date")
        else:
            merged = new_df

        # Save
        merged.write_parquet(parquet_path)
        print(f"  Saved {len(merged)} rows to {parquet_path}")

    print(f"=== Index Download Complete ===")


if __name__ == "__main__":
    import sys

    # Support optional start_date and end_date arguments
    start = "20100101"
    end = None
    if len(sys.argv) > 1:
        start = sys.argv[1]
    if len(sys.argv) > 2:
        end = sys.argv[2]

    download_all(start_date=start, end_date=end)
