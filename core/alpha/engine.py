import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import polars as pl
from vnpy.trader.database import get_database
from vnpy.trader.constant import Interval, Exchange
from vnpy.alpha.lab import AlphaLab
from core.selector import FundamentalSelector

ALPHA_DB_PATH = "core/alpha_db"

class AlphaEngine:
    def __init__(self):
        self.project_root = Path(os.getcwd())
        self.lab_path = self.project_root / ALPHA_DB_PATH
        self.lab = AlphaLab(str(self.lab_path))
        self.selector = FundamentalSelector()
        self.database = get_database()

    def sync_data(self, start_date: Optional[datetime] = None, end_date: datetime = datetime.now()):
        """
        Sync data from vnpy database to AlphaLab parquet files.
        """
        symbols = self.selector.get_candidate_symbols()
        if not symbols:
            print("No symbols found in selector.")
            return

        # Determine overall range if not provided
        if not start_date:
            s, _ = self.selector.get_data_range()
            start_date = s if s else datetime(2020, 12, 24)
        
        print(f"Syncing data for {len(symbols)} symbols from {start_date} to {end_date}...")

        for vt_symbol in symbols:
            symbol, exchange_str = vt_symbol.split(".")
            exchange = Exchange(exchange_str)
            
            bars = self.database.load_bar_data(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.DAILY,
                start=start_date,
                end=end_date
            )
            
            if bars:
                self.lab.save_bar_data(bars)
        
        print("Data sync complete.")

    def calculate_factors(self, factor_names: List[str]):
        """
        Calculate factors and save them as signals/datasets.
        This is a placeholder for the actual research workflow.
        """
        # Example: Load data, compute factors using polars, save result
        pass

    def get_signal_df(self, name: str) -> Optional[pl.DataFrame]:
        return self.lab.load_signal(name)
