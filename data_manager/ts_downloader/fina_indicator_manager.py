import pandas as pd
import pymysql
import tushare as ts
import time
from datetime import datetime
from vnpy.trader.setting import SETTINGS
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database


class FinaIndicatorManager:
    """
    财务指标数据管理类 (Tushare fina_indicator)
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
        self.fields = [
            "ts_code", "ann_date", "end_date", "eps", "dt_eps", "total_revenue_ps", 
            "revenue_ps", "gross_margin", "netprofit_margin", "roe", "roa", "roic", 
            "netprofit_yoy", "tr_yoy", "current_ratio", "quick_ratio", "assets_turn"
        ]
        self.init_db()

    def init_db(self):
        """初始化数据库表"""
        conn = pymysql.connect(**self.db_config)
        try:
            with conn.cursor() as cursor:
                # Create table
                # Primary key is (ts_code, end_date, ann_date) because sometimes there are corrections
                sql = """
                CREATE TABLE IF NOT EXISTS fina_indicator (
                    ts_code VARCHAR(20),
                    ann_date VARCHAR(20),
                    end_date VARCHAR(20),
                    eps FLOAT,
                    dt_eps FLOAT,
                    total_revenue_ps FLOAT,
                    revenue_ps FLOAT,
                    gross_margin FLOAT,
                    netprofit_margin FLOAT,
                    roe FLOAT,
                    roa FLOAT,
                    roic FLOAT,
                    netprofit_yoy FLOAT,
                    tr_yoy FLOAT,
                    current_ratio FLOAT,
                    quick_ratio FLOAT,
                    assets_turn FLOAT,
                    PRIMARY KEY (ts_code, end_date, ann_date)
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
        if df.empty:
            return
            
        columns = self.fields
        
        # SQL for replace
        placeholders = ", ".join(["%s"] * len(columns))
        columns_str = ", ".join(columns)
        sql = f"REPLACE INTO fina_indicator ({columns_str}) VALUES ({placeholders})"
        
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
        加载财务指标数据
        :param symbols: vt_symbol列表
        :param start_date: 开始日期 YYYYMMDD
        :param end_date: 结束日期 YYYYMMDD
        :return: pd.DataFrame
        """
        if not symbols:
            return pd.DataFrame()

        ts_to_vt = {}
        ts_codes = []
        for vt_symbol in symbols:
            code, exchange_str = vt_symbol.split(".")
            suffix = self.get_vnpy_suffix(Exchange(exchange_str))
            if suffix:
                ts_code = f"{code}.{suffix}"
                ts_to_vt[ts_code] = vt_symbol
                ts_codes.append(ts_code)

        if not ts_codes:
            return pd.DataFrame()

        conn = pymysql.connect(**self.db_config)
        try:
            with conn.cursor() as cursor:
                format_strings = ','.join(['%s'] * len(ts_codes))
                # 关键：我们应该基于 ann_date 来加载数据，因为只有公布了才能用
                sql = f"""
                    SELECT * FROM fina_indicator 
                    WHERE ts_code IN ({format_strings}) 
                    AND ann_date <= %s
                """
                params = ts_codes + [end_date]
                cursor.execute(sql, params)
                data = cursor.fetchall()
                if not data:
                    return pd.DataFrame()
                df = pd.DataFrame(data)
        finally:
            conn.close()

        # 转换日期用于合并
        df['datetime'] = pd.to_datetime(df['ann_date'], format='%Y%m%d')
        df['vt_symbol'] = df['ts_code'].map(ts_to_vt)
        
        return df

    def download_all(self):
        """为数据库中所有股票下载财务指标"""
        database = get_database()
        overviews = database.get_bar_overview()
        
        # 1. 收集需关注的股票
        all_ts_codes = []
        for overview in overviews:
            if overview.interval == Interval.DAILY and overview.exchange in [Exchange.SSE, Exchange.SZSE, Exchange.BSE]:
                suffix = self.get_vnpy_suffix(overview.exchange)
                if suffix:
                    all_ts_codes.append(f"{overview.symbol}.{suffix}")
        
        all_ts_codes = list(set(all_ts_codes))
        if not all_ts_codes:
            print("未找到有效股票配置")
            return

        # 2. 获取每只股票已有的最新进度
        existing_status = {}
        conn = pymysql.connect(**self.db_config)
        try:
            print("正在检查数据库现有财务指标进度...")
            with conn.cursor() as cursor:
                sql = "SELECT ts_code, MAX(end_date) as last_period FROM fina_indicator GROUP BY ts_code"
                cursor.execute(sql)
                rows = cursor.fetchall()
                for row in rows:
                    existing_status[row['ts_code']] = row['last_period']
        finally:
            conn.close()

        print(f"开始增量下载财务指标，全市场共 {len(all_ts_codes)} 只股票")

        # 3. 按下载起点分组，以优化批量请求
        # 相同起点的股票可以合并到同一个 API 请求中（节省点数和次数）
        progress_groups = {} # last_period -> [ts_code]
        for code in all_ts_codes:
            lp = existing_status.get(code, "20190101")
            if lp not in progress_groups:
                progress_groups[lp] = []
            progress_groups[lp].append(code)

        # 4. 分组遍历下载
        conn = pymysql.connect(**self.db_config)
        try:
            total_processed = 0
            for lp, codes in progress_groups.items():
                # 策略：如果起点很老（需要补全历史），每批次股票数小一点以免超过100条限制
                # 如果起点很新（增量更新），可以大批次请求
                if lp < "20240101":
                    batch_size = 2 # 历史数据较多，每批2只股票比较稳妥
                else:
                    batch_size = 40 # 增量数据较少，每批40只股票提高效率

                for i in range(0, len(codes), batch_size):
                    chunk = codes[i : i + batch_size]
                    ts_code_str = ",".join(chunk)
                    
                    total_processed += len(chunk)
                    print(f"正在下载起点为 [{lp}] 的批次 ({total_processed}/{len(all_ts_codes)}): {ts_code_str[:60]}...")
                    
                    try:
                        # 使用 start_date 进行增量过滤
                        df = self.pro.fina_indicator(ts_code=ts_code_str, start_date=lp)
                        
                        if df is not None and not df.empty:
                            # 保存数据
                            self.save_data(conn, df[self.fields])
                            print(f"  - 成功保存 {len(df)} 条记录")
                            
                            if len(df) >= 100:
                                print(f"  - [Warning] 触发100条限制，该批次可能存在数据截断！")
                        else:
                            print("  - 无新公告")
                            
                    except Exception as e:
                        print(f"  - 下载失败: {e}")
                    
                    # 速率限制: Tushare 2000积分用户通常限制 200次/分钟
                    time.sleep(0.35)
                    
        finally:
            conn.close()
        
        print("财务指标更新完成")
