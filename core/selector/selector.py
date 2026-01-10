import os
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import polars as pl
import tushare as ts
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.setting import SETTINGS

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

    def get_last_trading_day(self) -> datetime:
        """
        Get the last trading day before current time.
        If today is trading day and time < 17:00, return previous trading day.
        """
        try:
            pro = ts.pro_api(SETTINGS["datafeed.password"])
        except Exception:
            print("Warning: Tushare initialization failed. Check datafeed.password setting.")
            return datetime.now()

        now = datetime.now()
        # Look back 30 days to cover long holidays
        start_date = (now - timedelta(days=30)).strftime("%Y%m%d")
        end_date = now.strftime("%Y%m%d")
        
        try:
            df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date, is_open='1')
        except Exception as e:
            print(f"Warning: Failed to query trade_cal: {e}")
            return now
            
        if df.empty:
            return now
            
        # Ensure sorted
        df = df.sort_values('cal_date')
        trading_days = df['cal_date'].tolist()
        today_str = now.strftime("%Y%m%d")
        
        if not trading_days:
            return now

        last_trading_day = trading_days[-1]
        
        target_date_str = last_trading_day
        
        if last_trading_day == today_str:
            # Today is a trading day
            if now.hour < 17:
                # Return previous one if available
                if len(trading_days) >= 2:
                    target_date_str = trading_days[-2]
                # else: keep today if we can't find previous
        
        return datetime.strptime(target_date_str, "%Y%m%d")
