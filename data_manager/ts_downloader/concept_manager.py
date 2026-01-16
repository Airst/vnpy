import threading
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import pymysql
import tushare as ts
from vnpy.trader.setting import SETTINGS
from core.selector.selector import FundamentalSelector

class RateLimiter:
    """
    Simple thread-safe rate limiter.
    """
    def __init__(self, calls_per_minute):
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()

class ConceptManager:
    """
    Manager for Concept Board Data (Daily & Members).
    """
    def __init__(self):
        self.pro = ts.pro_api(SETTINGS["datafeed.password"])
        self.db_config = {
            "host": SETTINGS["database.host"],
            "port": SETTINGS["database.port"],
            "user": SETTINGS["database.user"],
            "password": SETTINGS["database.password"],
            "database": SETTINGS["database.database"],
            "charset": "utf8mb4",
            "cursorclass": pymysql.cursors.DictCursor
        }
        # Max 200 calls per minute
        self.limiter = RateLimiter(200)
        self.init_db()

    def init_db(self):
        """Initialize database tables for dc_daily and dc_member."""
        conn = pymysql.connect(**self.db_config)
        try:
            with conn.cursor() as cursor:
                # 1. dc_daily: Concept Daily Market Data
                # Updated fields based on dc_daily API
                sql_daily = """
                CREATE TABLE IF NOT EXISTS dc_daily (
                    ts_code VARCHAR(20),
                    trade_date VARCHAR(20),
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    `change` FLOAT,
                    pct_change FLOAT,
                    vol FLOAT,
                    amount FLOAT,
                    swing FLOAT,
                    turnover_rate FLOAT,
                    PRIMARY KEY (ts_code, trade_date)
                )
                """
                cursor.execute(sql_daily)

                # 2. dc_member: Concept Board Members (Historical)
                sql_member = """
                CREATE TABLE IF NOT EXISTS dc_member (
                    ts_code VARCHAR(20),
                    con_code VARCHAR(20),
                    trade_date VARCHAR(20),
                    name VARCHAR(20),
                    PRIMARY KEY (ts_code, con_code, trade_date)
                )
                """
                cursor.execute(sql_member)

            conn.commit()
        finally:
            conn.close()

    def _get_connection(self):
        return pymysql.connect(**self.db_config)

    def _fetch_daily(self, trade_date):
        """Worker to fetch daily data for a specific date."""
        self.limiter.wait()
        try:
            # Using dc_daily API
            df = self.pro.dc_daily(trade_date=trade_date)
            return trade_date, df
        except Exception as e:
            print(f"Error fetching daily data for {trade_date}: {e}")
            return trade_date, None

    def download_daily(self, max_workers=4):
        """
        Incremental download of concept daily data.
        """
        print("Starting Incremental Concept Daily Download...")
        self._incremental_download("dc_daily", self._fetch_daily, self._save_daily, max_workers)

    def _save_daily(self, df):
        """Save daily data to DB."""
        conn = self._get_connection()
        try:
            columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 
                       'change', 'pct_change', 'vol', 'amount', 'swing', 'turnover_rate']
            
            # Note: change is a reserved keyword in MySQL, so we use `change` in SQL
            sql = """
            REPLACE INTO dc_daily 
            (ts_code, trade_date, open, high, low, close, `change`, pct_change, vol, amount, swing, turnover_rate)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = []
            for _, row in df.iterrows():
                val = tuple(row.get(col) for col in columns)
                values.append(val)
                
            with conn.cursor() as cursor:
                cursor.executemany(sql, values)
            conn.commit()
        finally:
            conn.close()

    def _fetch_members_by_date(self, trade_date):
        """Worker to fetch members for all concepts on a specific date."""
        self.limiter.wait()
        try:
            # Using dc_member API
            df = self.pro.dc_member(trade_date=trade_date)
            return trade_date, df
        except Exception as e:
            print(f"Error fetching members for {trade_date}: {e}")
            return trade_date, None

    def download_members(self, max_workers=4):
        """
        Incremental download of concept members data.
        """
        print("Starting Incremental Concept Members Download...")
        self._incremental_download("dc_member", self._fetch_members_by_date, self._save_members_date, max_workers)

    def _save_members_date(self, df):
        """Save members data to DB."""
        if df is None or df.empty:
            return

        conn = self._get_connection()
        try:
            # Columns: trade_date, ts_code, con_code, name
            columns = ['ts_code', 'con_code', 'trade_date', 'name']
            
            sql = """
            REPLACE INTO dc_member (ts_code, con_code, trade_date, name)
            VALUES (%s, %s, %s, %s)
            """
            
            values = []
            for _, row in df.iterrows():
                val = tuple(row.get(col) for col in columns)
                values.append(val)
            
            with conn.cursor() as cursor:
                cursor.executemany(sql, values)
            conn.commit()
        finally:
            conn.close()

    def _incremental_download(self, table_name, fetch_func, save_func, max_workers):
        """
        Generic incremental download helper.
        """
        # 1. Get last trading day from core
        selector = FundamentalSelector(None)
        last_day_dt = selector.get_last_trading_day()
        last_day_str = last_day_dt.strftime("%Y%m%d")
        
        # 2. Get max date in DB
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # Need to handle empty table
                try:
                    cursor.execute(f"SELECT MAX(trade_date) as max_date FROM {table_name}")
                    res = cursor.fetchone()
                    db_max_date = res['max_date']
                except Exception:
                    # Table might not exist or other error, assume empty
                    db_max_date = None
        finally:
            conn.close()

        start_dt = datetime(2010, 1, 1) # Default start
        if db_max_date:
            try:
                db_max_dt = datetime.strptime(db_max_date, "%Y%m%d")
                start_dt = db_max_dt + timedelta(days=1)
            except ValueError:
                pass
        
        if start_dt > last_day_dt:
            print(f"{table_name} is already up to date.")
            return

        # 3. Generate date range
        dates_to_fetch = []
        self.limiter.wait()
        # Fetch calendar
        try:
            cal_df = self.pro.trade_cal(start_date=start_dt.strftime("%Y%m%d"), 
                                        end_date=last_day_str, is_open='1')
            if not cal_df.empty:
                dates_to_fetch = cal_df['cal_date'].tolist()
        except Exception as e:
            print(f"Error fetching calendar: {e}")
            return

        if not dates_to_fetch:
            print("No trading days found in range.")
            return

        print(f"Need to download {len(dates_to_fetch)} days from {dates_to_fetch[0]} to {dates_to_fetch[-1]}.")

        # 4. Multi-threaded Download
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {executor.submit(fetch_func, date): date for date in dates_to_fetch}
            
            completed_count = 0
            for future in as_completed(future_to_date):
                date, df = future.result()
                if df is not None and not df.empty:
                    save_func(df)
                    print(f"[{completed_count+1}/{len(dates_to_fetch)}] Saved {len(df)} records for {date}.")
                else:
                    print(f"[{completed_count+1}/{len(dates_to_fetch)}] No data for {date}.")
                completed_count += 1

if __name__ == "__main__":
    manager = ConceptManager()
    manager.download_daily()
    manager.download_members()