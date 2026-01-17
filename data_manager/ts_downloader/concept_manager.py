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

    def download_members(self, max_workers=4):
        """
        Incremental download of concept members data (Monthly - First Trading Day).
        """
        print("Starting Incremental Concept Members Download (Monthly)...")
        
        # 1. Determine Dates
        # Get max date from DB
        conn = self._get_connection()
        db_max_date = None
        try:
            with conn.cursor() as cursor:
                # Check if table exists and has data
                try:
                    cursor.execute("SELECT MAX(trade_date) as max_date FROM dc_member")
                    res = cursor.fetchone()
                    if res:
                        db_max_date = res['max_date']
                except Exception:
                    pass
        finally:
            conn.close()

        # Get last trading day from core
        selector = FundamentalSelector(None)
        last_day_dt = selector.get_last_trading_day()
        last_day_str = last_day_dt.strftime("%Y%m%d")

        # Start from a safe past date to find monthly firsts
        start_cal_date = "20241220"
        
        # Fetch calendar
        try:
            self.limiter.wait()
            cal_df = self.pro.trade_cal(start_date=start_cal_date, 
                                        end_date=last_day_str, is_open='1')
        except Exception as e:
            print(f"Error fetching calendar: {e}")
            return

        if cal_df.empty:
            print("No trading days found.")
            return

        # Filter for 1st trading day of each month
        cal_df['month'] = cal_df['cal_date'].str.slice(0, 6)
        # Group by month, take first (min)
        all_monthly_dates = cal_df.groupby('month')['cal_date'].min().tolist()
        
        # Filter dates > db_max_date
        if db_max_date:
            target_dates = [d for d in all_monthly_dates if d > db_max_date]
        else:
            target_dates = all_monthly_dates

        if not target_dates:
            print("dc_member is already up to date (Monthly).")
            return

        print(f"Need to download for {len(target_dates)} months: {target_dates[0]} to {target_dates[-1]}")

        # 2. Download
        for trade_date in target_dates:
            # Get concept codes from dc_daily for this date
            conn = self._get_connection()
            ts_codes = []
            try:
                with conn.cursor() as cursor:
                    sql = "SELECT DISTINCT ts_code FROM dc_daily WHERE trade_date = %s"
                    cursor.execute(sql, (trade_date,))
                    results = cursor.fetchall()
                    ts_codes = [row['ts_code'] for row in results]
            except Exception as e:
                print(f"Error fetching concepts from dc_daily for {trade_date}: {e}")
            finally:
                conn.close()

            if not ts_codes:
                print(f"No concept data found in dc_daily for {trade_date}. Skipping.")
                continue

            print(f"Processing {trade_date} ({len(ts_codes)} concepts)...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all concepts for this date
                future_to_code = {
                    executor.submit(self._fetch_concept_member, code, trade_date): code 
                    for code in ts_codes
                }
                
                saved_count = 0
                for future in as_completed(future_to_code):
                    df = future.result()
                    if df is not None and not df.empty:
                        self._save_members_date(df)
                        saved_count += 1
                
                print(f"Finished {trade_date}. Saved data for {saved_count} concepts.")

    def _fetch_concept_member(self, ts_code, trade_date):
        """Worker to fetch members for a specific concept and date."""
        self.limiter.wait()
        try:
            return self.pro.dc_member(ts_code=ts_code, trade_date=trade_date)
        except Exception:
            return None

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

    def load_daily_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load concept daily data from DB.
        """
        conn = self._get_connection()
        try:
            # Use SQL to filter by date
            # format date to match DB (assuming DB uses YYYYMMDD based on download_daily)
            # Input start_date is usually YYYY-MM-DD or YYYYMMDD.
            s_date = start_date.replace("-", "")
            e_date = end_date.replace("-", "")
            
            sql = f"""
            SELECT ts_code, trade_date, close, pct_change, turnover_rate 
            FROM dc_daily 
            WHERE trade_date >= '{s_date}' AND trade_date <= '{e_date}'
            """
            with conn.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()
            return pd.DataFrame(results)
        except Exception as e:
            print(f"Error loading concept daily data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def load_member_data(self) -> pd.DataFrame:
        """
        Load all concept member data.
        Returns: DataFrame with [ts_code, con_code]
        """
        conn = self._get_connection()
        try:
            # We want the latest mapping or historical?
            # dc_member has trade_date. It seems to be a snapshot per date?
            # Or is it "Joined Date"?
            # Tushare dc_member doc: "Concept Board Detail".
            # Usually it returns the *current* members or members at a date.
            # If we downloaded historical, we might have multiple entries.
            # Let's load all and let the consumer handle it.
            sql = "SELECT ts_code, con_code, trade_date FROM dc_member"
            with conn.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()
            return pd.DataFrame(results)
        except Exception as e:
            print(f"Error loading concept member data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

if __name__ == "__main__":
    manager = ConceptManager()
    manager.download_daily()
    manager.download_members()