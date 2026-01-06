import os
from datetime import datetime
from typing import List, Tuple, Dict
import polars as pl
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database

class FundamentalSelector:
    def __init__(self, vt_symbols: List[str] = None):
        self.database = get_database()
        self.vt_symbols = vt_symbols

    @staticmethod
    def check_stock_filters(factors: Dict[str, float]) -> bool:
        """
        Check if the stock passes fundamental filters (A-share common practices).
        Return True if passed, False if rejected.
        """
        if not factors:
            return True
            
        # 1. Profitability Filter (剔除亏损股)
        # ep_ratio = 1 / PE. If ep_ratio < 0, it means PE is negative (Loss making).
        if "ep_ratio" in factors:
             if factors["ep_ratio"] < 0:
                 return False

        # 2. Liquidity Filter (剔除流动性枯竭)
        # Tushare turnover_rate is usually in %. 
        # A threshold of 1.0 (1%) is a standard minimum for active alpha strategies.
        if "turnover_mean_20d" in factors:
             if factors["turnover_mean_20d"] < 1.0: 
                 return False

        # 4. Market Cap Filter (剔除微盘股/壳股)
        # size_ln_cap = ln(TotalMV). 
        # TotalMV unit is 10,000 RMB (万元).
        # Example: 10亿 RMB = 100,000 万元 -> ln(100,000) ≈ 11.5
        if "size_ln_cap" in factors:
            if factors["size_ln_cap"] < 11.5: 
                return False
                 
        return True

    @staticmethod
    def filter_polars(df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply fundamental filters to a Polars DataFrame.
        """
        # 1. Profitability Filter
        if "ep_ratio" in df.columns:
            df = df.filter(pl.col("ep_ratio") >= 0)
            
        # 2. Liquidity Filter
        if "turnover_mean_20d" in df.columns:
            df = df.filter(pl.col("turnover_mean_20d") >= 1.0)
            
        # 3. Market Cap Filter
        if "size_ln_cap" in df.columns:
            df = df.filter(pl.col("size_ln_cap") >= 11.5)
            
        return df

    def get_candidate_symbols(self) -> List[str]:
        """
        Returns a list of vt_symbols (e.g. '000001.SZSE') available in the database.
        """
        if self.vt_symbols:
            return self.vt_symbols

        symbols = []
        overviews = self.database.get_bar_overview()
        
        for overview in overviews:
            if overview.interval == Interval.DAILY:
                vt_symbol = f"{overview.symbol}.{overview.exchange.value}"
                symbols.append(vt_symbol)
        
        # Filter symbols by market type (only keep '主板')
        if symbols:
            try:
                from data_manager.ts_downloader.stock_info_manager import StockInfoManager
                stock_info_manager = StockInfoManager()
                df = stock_info_manager.load_data(symbols)
                
                if not df.empty:
                    df_filtered = df[df["market"] == "主板"]
                    symbols = df_filtered["vt_symbol"].tolist()
            except Exception as e:
                print(f"Warning: Failed to filter symbols by market type: {e}")
        
        return symbols

    def get_data_range(self) -> Tuple[datetime, datetime]:
        """
        Returns the overall start and end date of the available data in the database.
        """
        overviews = self.database.get_bar_overview()
        if not overviews:
            return None, None
            
        min_start = None
        max_end = None
        
        for overview in overviews:
            if overview.interval == Interval.DAILY:
                if overview.start:
                    if min_start is None or overview.start < min_start:
                        min_start = overview.start
                
                if overview.end:
                    if max_end is None or overview.end > max_end:
                        max_end = overview.end
                
        return min_start, max_end
