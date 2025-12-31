import pandas as pd
import pymysql
import tushare as ts
from vnpy.trader.setting import SETTINGS


class StockInfoManager:
    """
    股票基础信息管理类 (stock_basic)
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
        self.init_db()

    def init_db(self):
        """初始化数据库表"""
        conn = pymysql.connect(**self.db_config)
        try:
            with conn.cursor() as cursor:
                # Create table
                # ts_code, symbol, exchange as Primary Key
                sql = """
                CREATE TABLE IF NOT EXISTS stock_basic (
                    ts_code VARCHAR(20),
                    symbol VARCHAR(20),
                    exchange VARCHAR(20),
                    name VARCHAR(50),
                    area VARCHAR(50),
                    industry VARCHAR(50),
                    market VARCHAR(20),
                    list_date VARCHAR(20),
                    list_status VARCHAR(10),
                    delist_date VARCHAR(20),
                    is_hs VARCHAR(10),
                    PRIMARY KEY (ts_code, symbol, exchange)
                )
                """
                cursor.execute(sql)
            conn.commit()
        finally:
            conn.close()

    def save_data(self, df: pd.DataFrame):
        """保存数据到数据库"""
        if df.empty:
            return

        conn = pymysql.connect(**self.db_config)
        try:
            columns = [
                'ts_code', 'symbol', 'exchange', 'name', 'area', 
                'industry', 'market', 'list_date', 'list_status', 
                'delist_date', 'is_hs'
            ]
            
            # Ensure columns exist in df
            for col in columns:
                if col not in df.columns:
                    df[col] = None

            # SQL for replace
            placeholders = ", ".join(["%s"] * len(columns))
            columns_str = ", ".join(columns)
            sql = f"REPLACE INTO stock_basic ({columns_str}) VALUES ({placeholders})"
            
            values = []
            for _, row in df.iterrows():
                row_data = []
                for col in columns:
                    val = row.get(col)
                    if pd.isna(val):
                        val = None
                    
                    # Special handling for PK columns to avoid NULL
                    if col in ['ts_code', 'symbol', 'exchange'] and val is None:
                        val = ""
                        
                    row_data.append(val)
                values.append(tuple(row_data))
                
            with conn.cursor() as cursor:
                cursor.executemany(sql, values)
            conn.commit()
            print(f"成功保存 {len(values)} 条股票基础信息")
        finally:
            conn.close()

    def get_tushare_suffix(self, exchange_str: str) -> str:
        """根据交易所字符串获取Tushare后缀"""
        if exchange_str == "SSE":
            return "SH"
        elif exchange_str == "SZSE":
            return "SZ"
        elif exchange_str == "BSE":
            return "BJ"
        return ""

    def load_data(self, symbols: list[str]) -> pd.DataFrame:
        """
        根据vt_symbol列表查询股票基础信息
        :param symbols: vnpy symbol列表 (e.g. ["000001.SZSE", ...])
        :return: pd.DataFrame
        """
        if not symbols:
            return pd.DataFrame()
            
        ts_codes = []
        ts_to_vt = {}
        
        for vt_symbol in symbols:
            try:
                parts = vt_symbol.split(".")
                if len(parts) != 2:
                    continue
                code, exchange_str = parts
                suffix = self.get_tushare_suffix(exchange_str)
                if suffix:
                    ts_code = f"{code}.{suffix}"
                    ts_codes.append(ts_code)
                    ts_to_vt[ts_code] = vt_symbol
            except Exception:
                continue
                
        if not ts_codes:
            return pd.DataFrame()
            
        conn = pymysql.connect(**self.db_config)
        try:
            with conn.cursor() as cursor:
                # 批量查询
                format_strings = ','.join(['%s'] * len(ts_codes))
                sql = f"SELECT * FROM stock_basic WHERE ts_code IN ({format_strings})"
                cursor.execute(sql, ts_codes)
                result = cursor.fetchall()
                
                if not result:
                    return pd.DataFrame()
                
                df = pd.DataFrame(result)
                # 映射回 vt_symbol
                df['vt_symbol'] = df['ts_code'].map(ts_to_vt)
                return df
        finally:
            conn.close()

    def download_all(self):
        """全量查询并更新"""
        print("正在从 Tushare 下载股票基础信息 (stock_basic)...")
        try:
            # fields defined by requirements
            fields = 'ts_code,symbol,exchange,name,area,industry,market,list_date,list_status,delist_date,is_hs'
            
            # Fetch L (Listed), D (Delisted), P (Paused)
            # Tushare typically accepts comma separated list_status or we loop.
            # We try comma separated first.
            df = self.pro.stock_basic(exchange='', list_status='L,D,P', fields=fields)
            
            if df is not None and not df.empty:
                self.save_data(df)
            else:
                print("未获取到股票基础信息数据")
            
        except Exception as e:
            print(f"下载股票基础信息失败: {e}")
