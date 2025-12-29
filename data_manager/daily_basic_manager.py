import pandas as pd
import pymysql
import tushare as ts
from datetime import datetime
from vnpy.trader.setting import SETTINGS
from vnpy.trader.constant import Exchange
from vnpy.trader.database import get_database


class DailyBasicManager:
    """
    每日指标数据管理类
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
                sql = """
                CREATE TABLE IF NOT EXISTS dailybasic (
                    ts_code VARCHAR(20),
                    trade_date VARCHAR(20),
                    close FLOAT,
                    turnover_rate FLOAT,
                    turnover_rate_f FLOAT,
                    volume_ratio FLOAT,
                    pe FLOAT,
                    pe_ttm FLOAT,
                    pb FLOAT,
                    ps FLOAT,
                    ps_ttm FLOAT,
                    dv_ratio FLOAT,
                    dv_ttm FLOAT,
                    total_share FLOAT,
                    float_share FLOAT,
                    free_share FLOAT,
                    total_mv FLOAT,
                    circ_mv FLOAT,
                    PRIMARY KEY (ts_code, trade_date)
                )
                """
                cursor.execute(sql)
            conn.commit()
        finally:
            conn.close()

    def get_vnpy_suffix(self, exchange: Exchange) -> str:
        """获取vnpy交易所对应的Tushare后缀"""
        if exchange == Exchange.SSE:
            return "SH"
        elif exchange == Exchange.SZSE:
            return "SZ"
        elif exchange == Exchange.BSE:
            return "BJ"
        return ""

    def save_data(self, conn, df: pd.DataFrame):
        """保存数据到数据库"""
        columns = [
            'ts_code', 'trade_date', 'close', 'turnover_rate', 'turnover_rate_f', 
            'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 
            'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 
            'free_share', 'total_mv', 'circ_mv'
        ]
        
        # SQL for replace
        placeholders = ", ".join(["%s"] * len(columns))
        columns_str = ", ".join(columns)
        sql = f"REPLACE INTO dailybasic ({columns_str}) VALUES ({placeholders})"
        
        values = []
        for _, row in df.iterrows():
            row_data = []
            for col in columns:
                val = row.get(col)
                if pd.isna(val):
                    val = None
                row_data.append(val)
            values.append(tuple(row_data))
            
        with conn.cursor() as cursor:
            cursor.executemany(sql, values)
        conn.commit()

    def get_latest_date(self, conn, ts_code: str) -> str:
        """获取数据库中最新的日期"""
        with conn.cursor() as cursor:
            sql = "SELECT MAX(trade_date) as max_date FROM dailybasic WHERE ts_code = %s"
            cursor.execute(sql, (ts_code,))
            result = cursor.fetchone()
            if result and result['max_date']:
                return result['max_date']
        return None

    def load_data(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        从数据库加载每日指标数据
        :param symbols: vnpy symbol列表 (e.g. ["000001.SZSE", ...])
        :param start_date: 开始日期 YYYYMMDD
        :param end_date: 结束日期 YYYYMMDD
        :return: pd.DataFrame
        """
        if not symbols:
            return pd.DataFrame()

        # 1. 建立映射关系
        vt_to_ts = {}
        ts_to_vt = {}
        ts_codes = []
        
        for vt_symbol in symbols:
            try:
                code, exchange_str = vt_symbol.split(".")
                exchange = Exchange(exchange_str)
                suffix = self.get_vnpy_suffix(exchange)
                if suffix:
                    ts_code = f"{code}.{suffix}"
                    vt_to_ts[vt_symbol] = ts_code
                    ts_to_vt[ts_code] = vt_symbol
                    ts_codes.append(ts_code)
            except Exception:
                continue
                
        if not ts_codes:
            return pd.DataFrame()

        # 2. 构建查询
        conn = pymysql.connect(**self.db_config)
        try:
            with conn.cursor() as cursor:
                #由于列表可能很长，直接拼SQL可能会超长，但一般几千个也还好。如果实在太多建议分批。
                #这里简单处理，使用IN查询
                format_strings = ','.join(['%s'] * len(ts_codes))
                sql = f"""
                    SELECT * FROM dailybasic 
                    WHERE ts_code IN ({format_strings}) 
                    AND trade_date >= %s 
                    AND trade_date <= %s
                """
                
                params = ts_codes + [start_date, end_date]
                cursor.execute(sql, params)
                data = cursor.fetchall()
                
                if not data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
            
        finally:
            conn.close()

        if df.empty:
            return df

        # 3. 数据处理
        # 映射回vt_symbol
        df['vt_symbol'] = df['ts_code'].map(ts_to_vt)
        
        # Clean trade_date
        df['trade_date'] = df['trade_date'].astype(str).str.strip()

        # 转换日期
        df['datetime'] = pd.to_datetime(df['trade_date'], format='%Y%m%d', errors='coerce')
        
        # Drop invalid dates
        if df['datetime'].isnull().any():
            invalid_rows = df[df['datetime'].isnull()]
            print(f"Warning: Dropping {len(invalid_rows)} rows with invalid dates. Bad values: {invalid_rows['trade_date'].unique()}")
            df.dropna(subset=['datetime'], inplace=True)

        # 移除不需要的列
        # df.drop(columns=['ts_code', 'trade_date'], inplace=True) 
        # 保留原列以便核对也可，这里按需保留
        
        return df

    def download_all(self):
        """下载所有已有数据的每日指标"""
        database = get_database()
        overviews = database.get_bar_overview()
        
        conn = pymysql.connect(**self.db_config)
        
        try:
            print(f"开始更新每日指标数据，共 {len(overviews)} 只标的...")
            for overview in overviews:
                symbol = overview.symbol
                exchange = overview.exchange
                
                # 只处理股票
                if exchange not in [Exchange.SSE, Exchange.SZSE, Exchange.BSE]:
                    continue

                suffix = self.get_vnpy_suffix(exchange)
                if not suffix:
                    continue
                    
                ts_code = f"{symbol}.{suffix}"
                
                if not overview.start or not overview.end:
                    continue
                
                req_start = overview.start
                req_end = overview.end

                # Check incremental
                latest_date_str = self.get_latest_date(conn, ts_code)
                if latest_date_str:
                    try:
                        latest_date = datetime.strptime(latest_date_str, "%Y%m%d")
                        # Start from next day
                        next_start = latest_date + pd.Timedelta(days=1)
                        
                        if next_start > req_end:
                            print(f"  - {ts_code} 数据已是最新 ({latest_date_str})，跳过")
                            continue
                        
                        if next_start > req_start:
                            req_start = next_start
                            
                    except Exception:
                        pass

                start_date = req_start.strftime("%Y%m%d")
                end_date = req_end.strftime("%Y%m%d")
                
                print(f"正在下载 {ts_code}: {start_date} - {end_date}")
                
                try:
                    # 分段下载，避免一次获取过多数据（虽然Tushare单次支持较多，但为了保险可以考虑，这里暂不分段，假设时间跨度不大或API支持）
                    # Tushare daily_basic 限制：单次最大5000行? 通常daily_basic按日期取较多，按票取通常整个历史没问题（几年几千行）
                    # 但是如果按票取，Tushare API通常支持
                    df = self.pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)
                    
                    if df is not None and not df.empty:
                        self.save_data(conn, df)
                        print(f"  - 成功保存 {len(df)} 条数据")
                    else:
                        print(f"  - 无数据")
                        
                except Exception as e:
                    print(f"  - 下载失败: {e}")
                    import time
                    time.sleep(1) # 简单的错误重试等待或限流保护
                    
        finally:
            conn.close()
        
        print("所有数据更新完成")
