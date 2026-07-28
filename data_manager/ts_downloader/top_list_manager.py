import pandas as pd
import pymysql
import tushare as ts
from datetime import datetime, timedelta
import time
from vnpy.trader.setting import SETTINGS


class TopListManager:
    """
    龙虎榜数据管理类 (Dragon-Tiger Board)
    数据来源: tushare top_list + top_inst 接口

    核心 Alpha 信号:
    - 龙虎榜净买入额/流通市值 → 大资金真实动向（比 moneyflow 更精确，仅涵盖极端异动日）
    - 机构席位买入占比 → 机构 vs 游资博弈信号
    - 上榜频率 → 市场关注度/波动率指标
    - 上榜原因分类 → 涨停/跌停/换手率/振幅区分不同事件类型

    使用:
        mgr = TopListManager()
        mgr.download_history("20200101", "20260714")  # 首次下载
        mgr.download_incremental()  # 增量更新
        df = mgr.load_data(["000001.SZSE", ...], "20220101", "20260714")
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
                # 龙虎榜每日统计
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS top_list (
                    ts_code VARCHAR(20),
                    trade_date VARCHAR(20),
                    name VARCHAR(50),
                    close FLOAT,
                    pct_change FLOAT,
                    turnover_rate FLOAT,
                    amount FLOAT,
                    l_sell FLOAT,
                    l_buy FLOAT,
                    l_amount FLOAT,
                    net_amount FLOAT,
                    net_rate FLOAT,
                    amount_rate FLOAT,
                    float_values FLOAT,
                    reason VARCHAR(200),
                    PRIMARY KEY (ts_code, trade_date, reason)
                )
                """)
                # 龙虎榜机构明细
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS top_inst (
                    ts_code VARCHAR(20),
                    trade_date VARCHAR(20),
                    exalter VARCHAR(200),
                    side VARCHAR(5),
                    buy FLOAT,
                    buy_rate FLOAT,
                    sell FLOAT,
                    sell_rate FLOAT,
                    net_buy FLOAT,
                    reason VARCHAR(200),
                    PRIMARY KEY (ts_code, trade_date, exalter, side, reason)
                )
                """)
            conn.commit()
        finally:
            conn.close()

    def _download_date(self, trade_date: str) -> tuple:
        """下载单日龙虎榜数据 (top_list + top_inst)"""
        top_list_df = None
        top_inst_df = None
        try:
            top_list_df = self.pro.top_list(trade_date=trade_date)
            time.sleep(0.3)  # API rate limit
        except Exception as e:
            print(f"  [TopList] top_list error {trade_date}: {e}")
        try:
            top_inst_df = self.pro.top_inst(trade_date=trade_date)
            time.sleep(0.3)
        except Exception as e:
            print(f"  [TopList] top_inst error {trade_date}: {e}")
        return top_list_df, top_inst_df

    def _save_top_list(self, conn, df: pd.DataFrame):
        """保存 top_list 数据"""
        if df is None or df.empty:
            return
        columns = ['ts_code', 'trade_date', 'name', 'close', 'pct_change',
                   'turnover_rate', 'amount', 'l_sell', 'l_buy', 'l_amount',
                   'net_amount', 'net_rate', 'amount_rate', 'float_values', 'reason']
        placeholders = ", ".join(["%s"] * len(columns))
        columns_str = ", ".join(columns)
        sql = f"REPLACE INTO top_list ({columns_str}) VALUES ({placeholders})"
        values = []
        for _, row in df.iterrows():
            row_data = []
            for col in columns:
                val = row.get(col)
                if pd.isna(val) if not isinstance(val, str) else False:
                    val = None
                row_data.append(val)
            values.append(tuple(row_data))
        with conn.cursor() as cursor:
            cursor.executemany(sql, values)
        conn.commit()

    def _save_top_inst(self, conn, df: pd.DataFrame):
        """保存 top_inst 数据"""
        if df is None or df.empty:
            return
        columns = ['ts_code', 'trade_date', 'exalter', 'side',
                   'buy', 'buy_rate', 'sell', 'sell_rate', 'net_buy', 'reason']
        placeholders = ", ".join(["%s"] * len(columns))
        columns_str = ", ".join(columns)
        sql = f"REPLACE INTO top_inst ({columns_str}) VALUES ({placeholders})"
        values = []
        for _, row in df.iterrows():
            row_data = []
            for col in columns:
                val = row.get(col)
                if pd.isna(val) if not isinstance(val, str) else False:
                    val = None
                row_data.append(val)
            values.append(tuple(row_data))
        with conn.cursor() as cursor:
            cursor.executemany(sql, values)
        conn.commit()

    def download_history(self, start_date: str, end_date: str):
        """按日下载历史数据"""
        print(f"[TopListManager] Downloading history: {start_date} → {end_date}")
        # Get trading calendar
        cal = self.pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
        trading_days = cal[cal['is_open'] == 1]['cal_date'].tolist()
        print(f"  Trading days: {len(trading_days)}")

        conn = pymysql.connect(**self.db_config)
        try:
            for i, td in enumerate(trading_days):
                top_list_df, top_inst_df = self._download_date(td)
                self._save_top_list(conn, top_list_df)
                self._save_top_inst(conn, top_inst_df)
                n_list = len(top_list_df) if top_list_df is not None else 0
                n_inst = len(top_inst_df) if top_inst_df is not None else 0
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(trading_days)}] {td}: {n_list} top_list + {n_inst} top_inst")
        finally:
            conn.close()
        print(f"[TopListManager] Done.")

    def download_incremental(self):
        """增量下载（从数据库最新日期到今天）"""
        conn = pymysql.connect(**self.db_config)
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT MAX(trade_date) as max_date FROM top_list")
                result = cursor.fetchone()
                last_date = result['max_date'] if result and result['max_date'] else '20200101'
        finally:
            conn.close()

        # Next day after last
        start = (datetime.strptime(last_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        end = datetime.now().strftime("%Y%m%d")
        if start > end:
            print(f"[TopListManager] Already up to date ({last_date})")
            return
        self.download_history(start, end)

    def load_data(self, symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载龙虎榜汇总数据（每只股票每日一行，聚合多条 reason）

        返回 DataFrame 列:
        - ts_code, trade_date
        - lhb_net_amount: 龙虎榜净买入额（所有reason合计）
        - lhb_buy: 龙虎榜买入总额
        - lhb_sell: 龙虎榜卖出总额
        - lhb_amount_rate: 龙虎榜成交额占比（max across reasons）
        - lhb_float_values: 当日流通市值
        - lhb_inst_net_buy: 机构席位净买入（from top_inst where exalter contains '机构'）
        - lhb_count: 当日上榜次数（不同reason数）
        """
        if not symbols:
            return pd.DataFrame()

        # Convert vnpy symbols to tushare codes
        ts_codes = []
        ts_to_vt = {}
        for vt_sym in symbols:
            parts = vt_sym.split(".")
            if len(parts) == 2:
                code, exch = parts
                if exch in ("SSE", "SH"):
                    ts_code = f"{code}.SH"
                elif exch in ("SZSE", "SZ"):
                    ts_code = f"{code}.SZ"
                else:
                    ts_code = f"{code}.BJ"
                ts_codes.append(ts_code)
                ts_to_vt[ts_code] = vt_sym

        conn = pymysql.connect(**self.db_config)
        try:
            # Load top_list aggregated per stock per day
            placeholders = ",".join(["%s"] * len(ts_codes))
            sql = f"""
            SELECT ts_code, trade_date,
                   SUM(net_amount) as lhb_net_amount,
                   SUM(l_buy) as lhb_buy,
                   SUM(l_sell) as lhb_sell,
                   MAX(amount_rate) as lhb_amount_rate,
                   MAX(float_values) as lhb_float_values,
                   COUNT(*) as lhb_count
            FROM top_list
            WHERE ts_code IN ({placeholders})
              AND trade_date >= %s AND trade_date <= %s
            GROUP BY ts_code, trade_date
            """
            with conn.cursor() as cursor:
                cursor.execute(sql, ts_codes + [start_date, end_date])
                rows = cursor.fetchall()
            df_list = pd.DataFrame(rows) if rows else pd.DataFrame()

            # Load top_inst for institutional signals
            sql_inst = f"""
            SELECT ts_code, trade_date,
                   SUM(CASE WHEN exalter LIKE '%%机构%%' THEN net_buy ELSE 0 END) as lhb_inst_net_buy
            FROM top_inst
            WHERE ts_code IN ({placeholders})
              AND trade_date >= %s AND trade_date <= %s
            GROUP BY ts_code, trade_date
            """
            with conn.cursor() as cursor:
                cursor.execute(sql_inst, ts_codes + [start_date, end_date])
                rows_inst = cursor.fetchall()
            df_inst = pd.DataFrame(rows_inst) if rows_inst else pd.DataFrame()

        finally:
            conn.close()

        if df_list.empty:
            return pd.DataFrame()

        # Merge inst data
        if not df_inst.empty:
            df_list = df_list.merge(df_inst, on=['ts_code', 'trade_date'], how='left')
        else:
            df_list['lhb_inst_net_buy'] = 0.0

        # Map back to vnpy symbols
        df_list['vt_symbol'] = df_list['ts_code'].map(ts_to_vt)
        df_list = df_list.dropna(subset=['vt_symbol'])

        return df_list


if __name__ == "__main__":
    mgr = TopListManager()
    # Download from 2020 onwards (sufficient for 700-day window starting 2022)
    mgr.download_history("20200101", datetime.now().strftime("%Y%m%d"))
