import pandas as pd
import pymysql
import tushare as ts
from datetime import datetime
from vnpy.trader.setting import SETTINGS
from vnpy.trader.constant import Exchange
from vnpy.trader.database import get_database


class MoneyFlowManager:
    """
    个股资金流向数据管理类
    数据来源: tushare moneyflow接口
    核心Alpha信号: 大单/超大单净流入反映机构资金动向
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
                CREATE TABLE IF NOT EXISTS moneyflow (
                    ts_code VARCHAR(20),
                    trade_date VARCHAR(20),
                    buy_sm_vol FLOAT,
                    buy_sm_amount FLOAT,
                    sell_sm_vol FLOAT,
                    sell_sm_amount FLOAT,
                    buy_md_vol FLOAT,
                    buy_md_amount FLOAT,
                    sell_md_vol FLOAT,
                    sell_md_amount FLOAT,
                    buy_lg_vol FLOAT,
                    buy_lg_amount FLOAT,
                    sell_lg_vol FLOAT,
                    sell_lg_amount FLOAT,
                    buy_elg_vol FLOAT,
                    buy_elg_amount FLOAT,
                    sell_elg_vol FLOAT,
                    sell_elg_amount FLOAT,
                    net_mf_vol FLOAT,
                    net_mf_amount FLOAT,
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
            'ts_code', 'trade_date', 
            'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount',
            'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount',
            'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount',
            'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
            'net_mf_vol', 'net_mf_amount'
        ]
        
        placeholders = ", ".join(["%s"] * len(columns))
        columns_str = ", ".join(columns)
        sql = f"REPLACE INTO moneyflow ({columns_str}) VALUES ({placeholders})"
        
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

    def load_data(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        从数据库加载资金流向数据
        :param symbols: vnpy symbol列表 (e.g. ["000001.SZSE", ...])
        :param start_date: 开始日期 YYYYMMDD
        :param end_date: 结束日期 YYYYMMDD
        :return: pd.DataFrame
        """
        if not symbols:
            return pd.DataFrame()

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

        conn = pymysql.connect(**self.db_config)
        try:
            with conn.cursor() as cursor:
                format_strings = ','.join(['%s'] * len(ts_codes))
                sql = f"""
                    SELECT * FROM moneyflow 
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

        df['vt_symbol'] = df['ts_code'].map(ts_to_vt)
        df['trade_date'] = df['trade_date'].astype(str).str.strip()
        df['datetime'] = pd.to_datetime(df['trade_date'], format='%Y%m%d', errors='coerce')
        
        if df['datetime'].isnull().any():
            df.dropna(subset=['datetime'], inplace=True)

        return df

    def download_all(self, start_date: str = "20200101"):
        """
        下载资金流向数据（按交易日批量下载）
        每次API调用返回全市场所有股票的资金流向，天然批量
        :param start_date: 历史起始日期 YYYYMMDD，默认2020年
        """
        database = get_database()
        overviews = database.get_bar_overview()
        
        valid_ts_codes = set()
        for overview in overviews:
            if overview.exchange not in [Exchange.SSE, Exchange.SZSE, Exchange.BSE]:
                continue
            suffix = self.get_vnpy_suffix(overview.exchange)
            if suffix:
                valid_ts_codes.add(f"{overview.symbol}.{suffix}")
        
        if not valid_ts_codes:
            print("未找到有效股票配置")
            return

        # 获取全局最新日期（简化：不按个股追踪，按整表最新日期）
        conn = pymysql.connect(**self.db_config)
        try:
            print("正在检查现有资金流向数据进度...")
            with conn.cursor() as cursor:
                sql = "SELECT MAX(trade_date) as last_date FROM moneyflow"
                cursor.execute(sql)
                result = cursor.fetchone()
            
            last_date_str = start_date
            if result and result['last_date']:
                last_date_str = str(result['last_date']).strip()
                # 从下一天开始
                last_dt = datetime.strptime(last_date_str, "%Y%m%d")
                last_date_str = (last_dt + pd.Timedelta(days=1)).strftime("%Y%m%d")
                
            today_str = datetime.now().strftime("%Y%m%d")
            
            if last_date_str > today_str:
                print("资金流向数据已是最新")
                return
            
            # 生成需要下载的日期列表
            all_dates = pd.date_range(start=last_date_str, end=today_str, freq='B')  # B=工作日
            date_list = [d.strftime("%Y%m%d") for d in all_dates]
            
            if not date_list:
                print("资金流向数据已是最新")
                return
                
            print(f"共需下载 {len(date_list)} 个交易日的资金流向数据 ({date_list[0]} ~ {date_list[-1]})")

            import time
            total_days = len(date_list)
            saved_count = 0
            
            for idx, date_str in enumerate(date_list):
                try:
                    # 每次调用返回全市场数据（天然批量）
                    df = self.pro.moneyflow(trade_date=date_str)
                    
                    if df is not None and not df.empty:
                        # 只保留关注的股票
                        df_filtered = df[df['ts_code'].isin(valid_ts_codes)]
                        
                        if not df_filtered.empty:
                            self.save_data(conn, df_filtered)
                            saved_count += len(df_filtered)
                            
                            if (idx + 1) % 20 == 0 or idx == total_days - 1:
                                print(f"  进度: {idx+1}/{total_days}, 已保存 {saved_count} 条记录")
                        
                except Exception as e:
                    # 非交易日会返回空数据，不算错误
                    if "没有数据" not in str(e) and "无数据" not in str(e):
                        print(f"  {date_str} 下载失败: {e}")
                
                time.sleep(0.3)
                    
        finally:
            conn.close()
        
        print(f"资金流向数据更新完成，共保存 {saved_count} 条记录")
