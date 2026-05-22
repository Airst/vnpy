"""
股东人数数据管理类 (Tushare stk_holdernumber)

数据来源: Tushare pro.stk_holdernumber 接口
数据频率: 季频（随财报披露），按 ann_date 公告日期使用
核心 Alpha 信号:
- holder_num: 股东人数（绝对值）
- 户均持股量 = total_share / holder_num（流动股本/总股本越集中越好）
- 股东人数环比变化率（QoQ）：人数下降通常伴随筹码集中，主力建仓
- 股东人数同比变化率（YoY）：长期筹码结构变化

Tushare 字段:
- ts_code: TS代码
- ann_date: 公告日期 (YYYYMMDD) — 数据可用日期，必须使用此字段对齐 datetime
- end_date: 截止日期（财报区间末尾）
- holder_num: 股东户数

设计决策:
- 主键 (ts_code, end_date) — 同一报告期可能有修正公告，使用 REPLACE 保留最新
- ann_date 为可用日期，load 时按 ann_date <= 当前日期 join_asof（与 fina_indicator 一致）
"""
import pandas as pd
import pymysql
import tushare as ts
import time
from datetime import datetime
from vnpy.trader.setting import SETTINGS
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database


class HolderNumberManager:
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
        self.fields = ["ts_code", "ann_date", "end_date", "holder_num"]
        self.init_db()

    def init_db(self):
        conn = pymysql.connect(**self.db_config)
        try:
            with conn.cursor() as cursor:
                sql = """
                CREATE TABLE IF NOT EXISTS holder_number (
                    ts_code VARCHAR(20),
                    ann_date VARCHAR(20),
                    end_date VARCHAR(20),
                    holder_num BIGINT,
                    PRIMARY KEY (ts_code, end_date)
                )
                """
                cursor.execute(sql)
            conn.commit()
        finally:
            conn.close()

    def get_vnpy_suffix(self, exchange: Exchange) -> str:
        if exchange == Exchange.SSE:
            return "SH"
        elif exchange == Exchange.SZSE:
            return "SZ"
        elif exchange == Exchange.BSE:
            return "BJ"
        return ""

    def save_data(self, conn, df: pd.DataFrame):
        if df.empty:
            return

        columns = self.fields
        placeholders = ", ".join(["%s"] * len(columns))
        columns_str = ", ".join(columns)
        sql = f"REPLACE INTO holder_number ({columns_str}) VALUES ({placeholders})"

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
        加载股东人数数据。
        :param symbols: vt_symbol 列表
        :param start_date: 开始日期 YYYYMMDD（本字段对季频不强约束，主要是 end_date）
        :param end_date: 结束日期 YYYYMMDD（仅取 ann_date <= end_date 的公告，避免未来数据）
        """
        if not symbols:
            return pd.DataFrame()

        ts_to_vt = {}
        ts_codes = []
        for vt_symbol in symbols:
            try:
                code, exchange_str = vt_symbol.split(".")
                suffix = self.get_vnpy_suffix(Exchange(exchange_str))
                if suffix:
                    ts_code = f"{code}.{suffix}"
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
                    SELECT * FROM holder_number
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

        df['datetime'] = pd.to_datetime(df['ann_date'], format='%Y%m%d', errors='coerce')
        df['vt_symbol'] = df['ts_code'].map(ts_to_vt)

        if df['datetime'].isnull().any():
            df.dropna(subset=['datetime'], inplace=True)

        return df

    def download_all(self, start_date: str = "20180101"):
        """
        全市场股东人数数据下载（单股票循环请求）。

        注意：Tushare stk_holdernumber 接口不支持多股票批量（ts_code 传入逗号分隔返回空），
        必须单股票循环请求。
        :param start_date: 历史起始日期 YYYYMMDD（默认2018年开始）
        """
        database = get_database()
        overviews = database.get_bar_overview()

        all_ts_codes = []
        for overview in overviews:
            if overview.interval == Interval.DAILY and overview.exchange in [Exchange.SSE, Exchange.SZSE, Exchange.BSE]:
                suffix = self.get_vnpy_suffix(overview.exchange)
                if suffix:
                    all_ts_codes.append(f"{overview.symbol}.{suffix}")

        all_ts_codes = sorted(set(all_ts_codes))
        if not all_ts_codes:
            print("未找到有效股票配置")
            return

        # 检查每只股票现有进度
        existing_status = {}
        conn = pymysql.connect(**self.db_config)
        try:
            print("正在检查现有股东人数数据进度...")
            with conn.cursor() as cursor:
                sql = "SELECT ts_code, MAX(end_date) as last_period FROM holder_number GROUP BY ts_code"
                cursor.execute(sql)
                rows = cursor.fetchall()
                for row in rows:
                    existing_status[row['ts_code']] = row['last_period']
        finally:
            conn.close()

        print(f"开始下载股东人数数据，全市场共 {len(all_ts_codes)} 只股票")

        conn = pymysql.connect(**self.db_config)
        total_saved = 0
        skip_count = 0
        try:
            for idx, ts_code in enumerate(all_ts_codes):
                last_period = existing_status.get(ts_code)
                # 增量下载: 已有进度的股票，从上次 end_date 之后开始
                if last_period:
                    # last_period 为 YYYYMMDD，加1天作为本次 start_date
                    try:
                        next_dt = datetime.strptime(last_period, "%Y%m%d") + pd.Timedelta(days=1)
                        req_start = next_dt.strftime("%Y%m%d")
                    except Exception:
                        req_start = start_date
                else:
                    req_start = start_date

                try:
                    df = self.pro.stk_holdernumber(
                        ts_code=ts_code,
                        start_date=req_start,
                    )

                    if df is not None and not df.empty:
                        df_save = df[self.fields].copy()
                        self.save_data(conn, df_save)
                        total_saved += len(df_save)
                        if (idx + 1) % 100 == 0 or idx == len(all_ts_codes) - 1:
                            print(f"  进度: {idx+1}/{len(all_ts_codes)}, 已保存 {total_saved} 条, 跳过 {skip_count}")
                    else:
                        skip_count += 1

                except Exception as e:
                    msg = str(e)
                    if "没有数据" not in msg and "无数据" not in msg:
                        print(f"  {ts_code} 下载失败: {e}")

                # Tushare 速率限制：~180次/分钟，留出余量
                time.sleep(0.34)

        finally:
            conn.close()

        print(f"股东人数数据更新完成，共保存 {total_saved} 条记录")


if __name__ == "__main__":
    mgr = HolderNumberManager()
    mgr.download_all(start_date="20180101")
