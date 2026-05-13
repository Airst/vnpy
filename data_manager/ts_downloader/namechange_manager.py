import pandas as pd
import pymysql
import tushare as ts
from datetime import datetime
from vnpy.trader.setting import SETTINGS


class NamechangeManager:
    """
    股票名称变更管理类 (namechange)
    用于获取历史ST状态：name字段包含"ST"则该股票在[start_date, end_date]期间为ST状态
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
                sql = """
                CREATE TABLE IF NOT EXISTS namechange (
                    ts_code VARCHAR(20),
                    name VARCHAR(50),
                    start_date VARCHAR(20),
                    end_date VARCHAR(20),
                    ann_date VARCHAR(20),
                    change_reason VARCHAR(200),
                    PRIMARY KEY (ts_code, start_date)
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
            columns = ['ts_code', 'name', 'start_date', 'end_date', 'ann_date', 'change_reason']
            for col in columns:
                if col not in df.columns:
                    df[col] = None

            placeholders = ", ".join(["%s"] * len(columns))
            columns_str = ", ".join(columns)
            sql = f"REPLACE INTO namechange ({columns_str}) VALUES ({placeholders})"

            values = []
            for _, row in df.iterrows():
                row_data = []
                for col in columns:
                    val = row.get(col)
                    if pd.isna(val):
                        row_data.append(None)
                    else:
                        row_data.append(str(val))
                values.append(tuple(row_data))

            with conn.cursor() as cursor:
                cursor.executemany(sql, values)
            conn.commit()
            print(f"[NamechangeManager] 保存 {len(values)} 条名称变更记录")
        finally:
            conn.close()

    def download_all(self):
        """全量下载名称变更数据（分交易所下载避免单次10000条限制）"""
        print("正在从 Tushare 下载股票名称变更数据 (namechange)...")
        try:
            all_dfs = []
            for exchange in ['SSE', 'SZSE']:
                # tushare namechange 不支持 exchange 参数，需要通过 stock_basic 获取代码列表后逐批下载
                # 但实际测试发现不带参数就有10000限制，改用按起始年份分段
                pass
            
            # 按年份分段下载（每段覆盖一年的公告日期）
            import time
            total = 0
            # namechange从1990年开始有记录
            for year in range(1990, 2027):
                start_date = f"{year}0101"
                end_date = f"{year}1231"
                df = self.pro.namechange(
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,name,start_date,end_date,ann_date,change_reason'
                )
                if df is not None and not df.empty:
                    all_dfs.append(df)
                    total += len(df)
                time.sleep(0.3)  # 避免触发频率限制
            
            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['ts_code', 'start_date'])
                self.save_data(combined_df)
                print(f"[NamechangeManager] 下载完成，共 {len(combined_df)} 条记录")
            else:
                print("[NamechangeManager] 未获取到数据")
        except Exception as e:
            print(f"[NamechangeManager] 下载失败: {e}")

    def load_st_periods(self) -> pd.DataFrame:
        """
        加载所有ST时段记录。
        返回DataFrame: ts_code, start_date, end_date (name包含'ST'的记录)
        """
        conn = pymysql.connect(**self.db_config)
        try:
            with conn.cursor() as cursor:
                sql = """
                SELECT ts_code, name, start_date, end_date 
                FROM namechange 
                WHERE name LIKE '%ST%'
                """
                cursor.execute(sql)
                result = cursor.fetchall()
                if result:
                    return pd.DataFrame(result)
                return pd.DataFrame(columns=['ts_code', 'name', 'start_date', 'end_date'])
        finally:
            conn.close()

    def is_st_on_date(self, ts_code: str, trade_date: str) -> bool:
        """判断某只股票在某个交易日是否为ST状态"""
        conn = pymysql.connect(**self.db_config)
        try:
            with conn.cursor() as cursor:
                sql = """
                SELECT COUNT(*) as cnt FROM namechange 
                WHERE ts_code = %s 
                AND name LIKE '%%ST%%'
                AND start_date <= %s 
                AND (end_date >= %s OR end_date IS NULL OR end_date = '')
                """
                cursor.execute(sql, (ts_code, trade_date, trade_date))
                result = cursor.fetchone()
                return result['cnt'] > 0
        finally:
            conn.close()
