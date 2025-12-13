
import json
from datetime import datetime
from pathlib import Path
from vnpy.trader.database import get_database
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest, BarData
from vnpy.alpha.lab import AlphaLab

def ingest_data(
    config_path: str = "core/data_download/download_config.json",
    lab_path: str = "core/alpha_db"
):
    """
    Load data from vnpy database and save to AlphaLab (parquet).
    """
    # 1. Initialize
    db = get_database()
    lab = AlphaLab(lab_path)
    
    print(f"Initializing Data Ingestion...")
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
            # vnpy database uses symbol without exchange suffix usually, 
            # but let's check how download_data.py saved it.
            # download_data.py: req_symbol = symbol.split(".")[0]
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
        bars = db.load_bar_data(req.symbol, req.exchange, req.interval, req.start, req.end)
        
        if not bars:
            print(f"  No data found in DB for {req_symbol}.{exchange.value}")
            continue
            
        print(f"  Loaded {len(bars)} bars. Saving to AlphaLab...")
        
        # Save to AlphaLab
        # AlphaLab.save_bar_data takes list[BarData]
        lab.save_bar_data(bars)
        
        # Also set contract setting for backtesting defaults
        # Assuming some defaults for A-shares
        lab.add_contract_setting(
            vt_symbol=f"{req_symbol}.{exchange.value}",
            long_rate=0.0003,   # Est. commission
            short_rate=0.0013,  # Est. commission + tax
            size=1,             # 1 share per unit of volume
            pricetick=0.01
        )

    print("Data ingestion complete.")

if __name__ == "__main__":
    ingest_data()
