import json
from datetime import datetime, time, timedelta
import pytz
from pathlib import Path

from vnpy.trader.datafeed import get_datafeed
from vnpy.trader.database import get_database
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest
from vnpy.alpha.lab import AlphaLab

from ingest_data import ingest_incremental
from data_loader import get_vt_symbols

class DataManager:
    def __init__(self, config_path: str = "core/data_download/download_config.json"):
        self.config_path = config_path
        self.db = get_database()
        self.datafeed = get_datafeed()
        self.lab = AlphaLab("core/alpha_db")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def get_target_date(self) -> datetime:
        """
        Determine the target date for data validity.
        < 17:00 => Yesterday
        >= 17:00 => Today
        
        Adjust for weekends: If target falls on Sat/Sun, revert to Friday.
        """
        now = datetime.now()
        cutoff_time = time(17, 0) # 17:00
        
        if now.time() < cutoff_time:
            target_date = now - timedelta(days=1)
        else:
            target_date = now
            
        # Adjust for weekends (Sat=5, Sun=6)
        while target_date.weekday() >= 5:
            target_date -= timedelta(days=1)
            
        return target_date

    def check_and_update_all(self, force: bool = False):
        """
        Main pipeline to check and update all data layers.
        """
        target_date = self.get_target_date()
        target_date_str = target_date.strftime("%Y-%m-%d")
        print(f"--- Data Manager ---")
        print(f"Target Date: {target_date_str} (Now: {datetime.now().strftime('%Y-%m-%d %H:%M')})")
        
        # 1. Update VNPY DB from Datafeed
        self.update_vnpy_db(target_date)
        
        # 2. Update AlphaLab from VNPY DB
        print("Syncing AlphaLab with VNPY Database...")
        ingest_incremental(self.config_path, "core/alpha_db")
        
        print("--- Data Check & Update Complete ---\n")

    def update_vnpy_db(self, target_date: datetime):
        """
        Ensure VNPY Database has data up to target_date for all symbols.
        """
        print("Checking VNPY Database status...")
        
        # 1. Fetch Overview (Snapshot)
        print("  Fetching DB overview...")
        all_overviews = self.db.get_bar_overview()
        overview_map = {
            (o.symbol, o.exchange, o.interval): o 
            for o in all_overviews
        }
        
        for task in self.config.get("downloads", []):
            symbol_str = task["symbol"]
            exchange_str = task["exchange"]
            interval_str = task["interval"]
            
            try:
                req_symbol = symbol_str.split(".")[0] if "." in symbol_str else symbol_str
                exchange = Exchange(exchange_str)
                interval = Interval(interval_str)
                config_start = datetime.strptime(task["start_date"], "%Y%m%d")
            except Exception:
                continue
                
            # Check latest bar via Overview
            overview = overview_map.get((req_symbol, exchange, interval))
            
            start_download = None
            
            if not overview or not overview.end:
                print(f"  {req_symbol}: No data in DB. initializing download...")
                start_download = config_start
            elif overview.end.date() < target_date.date():
                # Data is stale
                start_download = overview.end + timedelta(days=1)
                # print(f"  {req_symbol}: Stale (Latest: {overview.end.date()}). Downloading from {start_download.date()}...")
            
            if start_download:
                # If start_download is in future (e.g. today is sat, latest is fri, target is sat), 
                # check if start_download <= target_date
                if start_download.date() > target_date.date():
                    continue

                print(f"  Downloading {req_symbol} from {start_download.date()} to {target_date.date()}...")
                req = HistoryRequest(
                    symbol=req_symbol,
                    exchange=exchange,
                    interval=interval,
                    start=start_download,
                    end=target_date
                )
                
                try:
                    bars = self.datafeed.query_bar_history(req)
                    if bars:
                        self.db.save_bar_data(bars)
                        print(f"    Saved {len(bars)} bars.")
                    else:
                        print(f"    No data returned from Datafeed.")
                except Exception as e:
                    print(f"    Download failed: {e}")
            else:
                # Up to date
                pass

if __name__ == "__main__":
    dm = DataManager()
    dm.check_and_update_all()
