
import json
from datetime import datetime, timedelta
from pathlib import Path
import polars as pl
from vnpy.trader.database import get_database
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest, BarData
from vnpy.alpha.lab import AlphaLab

def ingest_all_data(
    config_path: str = "core/data_download/download_config.json",
    lab_path: str = "core/alpha_db"
):
    """
    Load FULL data from vnpy database and save to AlphaLab (parquet).
    """
    # 1. Initialize
    db = get_database()
    lab = AlphaLab(lab_path)
    
    print(f"Initializing Full Data Ingestion...")
    print(f"Source: vnpy database")
    print(f"Destination: {lab_path}")

    # 2. Load Config
    if not Path(config_path).exists():
        print(f"Config file {config_path} not found.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 3. Process each symbol
    for task in config.get("downloads", []):
        symbol_str = task["symbol"]
        exchange_str = task["exchange"]
        start_str = task["start_date"]
        end_str = task["end_date"]
        interval_str = task["interval"]

        # Parse
        try:
            req_symbol = symbol_str.split(".")[0] if "." in symbol_str else symbol_str
            exchange = Exchange(exchange_str)
            interval = Interval(interval_str)
            start = datetime.strptime(start_str, "%Y%m%d")
            if end_str == "latest":
                end = datetime.now()
            else:
                end = datetime.strptime(end_str, "%Y%m%d")
        except Exception as e:
            print(f"Skipping {symbol_str}: Invalid config format ({e})")
            continue

        # Query Database
        req = HistoryRequest(
            symbol=req_symbol,
            exchange=exchange,
            interval=interval,
            start=start,
            end=end
        )
        
        print(f"Querying {req_symbol}.{exchange.value} from DB...")
        bars = db.load_bar_data(req.symbol, req.exchange, req.interval, req.start, req.end) #type: ignore
        
        if not bars:
            print(f"  No data found in DB for {req_symbol}.{exchange.value}")
            continue
            
        print(f"  Loaded {len(bars)} bars. Saving to AlphaLab...")
        
        # Save to AlphaLab
        lab.save_bar_data(bars)
        
        _add_contract_defaults(lab, req_symbol, exchange)

    print("Data ingestion complete.")

def ingest_incremental(
    config_path: str = "core/data_download/download_config.json",
    lab_path: str = "core/alpha_db"
):
    """
    Incrementally load data from vnpy database to AlphaLab.
    Only queries DB for data newer than what is currently in AlphaLab.
    """
    db = get_database()
    lab = AlphaLab(lab_path)
    
    print(f"Initializing Incremental Data Ingestion...")

    if not Path(config_path).exists():
        print(f"Config file {config_path} not found.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    for task in config.get("downloads", []):
        symbol_str = task["symbol"]
        exchange_str = task["exchange"]
        interval_str = task["interval"]
        
        # Parse basic info
        try:
            req_symbol = symbol_str.split(".")[0] if "." in symbol_str else symbol_str
            exchange = Exchange(exchange_str)
            interval = Interval(interval_str)
            
            # Default fallback start date if no data in AlphaLab
            config_start = datetime.strptime(task["start_date"], "%Y%m%d")
        except Exception as e:
            print(f"Skipping {symbol_str}: Invalid config format ({e})")
            continue

        vt_symbol = f"{req_symbol}.{exchange.value}"
        
        # Check AlphaLab for existing data
        latest_dt = None
        
        # We need to construct the path manually to check existence efficiently, 
        # or use a helper if AlphaLab exposed one. 
        # Based on AlphaLab code: daily_path / {vt_symbol}.parquet
        if interval == Interval.DAILY:
            file_path = lab.daily_path.joinpath(f"{vt_symbol}.parquet")
        else:
            print(f"Skipping {vt_symbol}: Only DAILY supported for now.")
            continue
            
        if file_path.exists():
            try:
                # Read just the last date? Polars lazy read might be efficient
                # df = pl.scan_parquet(file_path).select(pl.col("datetime").max()).collect()
                # Actually reading full parquet for daily data is fast enough.
                df = pl.read_parquet(file_path)
                if not df.is_empty():
                    latest_dt = df["datetime"].max()
            except Exception as e:
                print(f"Error reading parquet for {vt_symbol}: {e}")
        
        # Determine Query Start Date
        if latest_dt:
            start = latest_dt + timedelta(days=1)
            # Remove time part for comparison if needed, but datetime is fine
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start = config_start

        end = datetime.now()
        
        if start > end:
            # print(f"  {vt_symbol} is up to date.")
            continue
            
        # Query Database
        req = HistoryRequest(
            symbol=req_symbol,
            exchange=exchange,
            interval=interval,
            start=start,
            end=end
        )
        
        bars = db.load_bar_data(req.symbol, req.exchange, req.interval, req.start, req.end) #type: ignore
        
        if not bars:
            # print(f"  No new data in DB for {vt_symbol} since {start}.")
            continue
            
        print(f"  {vt_symbol}: Found {len(bars)} new bars (from {start}). Saving to AlphaLab...")
        lab.save_bar_data(bars)
        
        # Ensure contract settings
        _add_contract_defaults(lab, req_symbol, exchange)

    print("Incremental ingestion complete.")

def _add_contract_defaults(lab, req_symbol, exchange):
    """Helper to set default contract settings"""
    lab.add_contract_setting(
        vt_symbol=f"{req_symbol}.{exchange.value}",
        long_rate=0.0003,
        short_rate=0.0013,
        size=1,
        pricetick=0.01
    )

if __name__ == "__main__":
    ingest_all_data()
