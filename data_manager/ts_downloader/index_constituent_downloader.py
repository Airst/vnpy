"""
下载指数成分股历史日线数据到数据库。

支持沪深300 (000300.SH)、中证500 (000905.SH)、中证1000 (000852.SH)、中证2000 (932000.CSI)。
获取成分股列表后，检查数据库中缺失的股票，批量下载日线数据。

Usage:
    # 下载四个指数成分股（默认，从20180101开始）
    python -m data_manager.ts_downloader.index_constituent_downloader

    # 指定指数
    python -m data_manager.ts_downloader.index_constituent_downloader --index 000852.SH 932000.CSI

    # 指定起始日期
    python -m data_manager.ts_downloader.index_constituent_downloader --start 20200101
"""

import time
import argparse
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Set, Tuple

import tushare as ts

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData
from vnpy.trader.utility import round_to
from vnpy.trader.setting import SETTINGS

from data_manager.ts_downloader.download_daily import from_tushare_code, to_tushare_code


# 默认指数列表
DEFAULT_INDICES = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "932000.CSI": "中证2000",
}


def get_index_constituents(pro, index_code: str) -> Set[str]:
    """
    获取指数最新成分股列表。

    Returns:
        Set of tushare codes (e.g., {'000001.SZ', '600000.SH'})
    """
    print(f"  获取 {index_code} 成分股...")

    try:
        df = pro.index_weight(index_code=index_code)
        if df is None or df.empty:
            print(f"  警告: {index_code} 无成分股数据")
            return set()

        # 取最新一期的成分股
        latest_date = df["trade_date"].max()
        df_latest = df[df["trade_date"] == latest_date]
        constituents = set(df_latest["con_code"].unique())
        print(f"  {index_code}: {len(constituents)} 只成分股 (日期: {latest_date})")
        return constituents
    except Exception as e:
        print(f"  错误: 获取 {index_code} 成分股失败: {e}")
        return set()


def get_missing_stocks(all_ts_codes: Set[str]) -> List[str]:
    """
    检查数据库，返回尚未下载的股票列表。
    """
    database = get_database()
    overviews = database.get_bar_overview()

    existing = set()
    for o in overviews:
        if o.interval == Interval.DAILY:
            ts_code = to_tushare_code(o.symbol, o.exchange)
            if ts_code:
                existing.add(ts_code)

    missing = all_ts_codes - existing
    return sorted(missing)


def download_stocks(pro, ts_codes: List[str], start_date: str, end_date: str):
    """
    批量下载股票日线数据并存入数据库。
    """
    if not ts_codes:
        print("没有需要下载的股票")
        return

    database = get_database()
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    days = (end_dt - start_dt).days + 1

    # 计算每批次最大股票数（tushare 限制 ~5000 行/次）
    max_per_call = min(max(int(5000 / days), 1), 200)

    print(f"下载 {len(ts_codes)} 只股票, {start_date} - {end_date}, 每批 {max_per_call} 只")

    total_bars = 0
    for i in range(0, len(ts_codes), max_per_call):
        chunk = ts_codes[i:i + max_per_call]
        ts_code_str = ",".join(chunk)

        try:
            df = pro.daily(ts_code=ts_code_str, start_date=start_date, end_date=end_date)

            if df is not None and not df.empty:
                bars_map = defaultdict(list)
                for _, row in df.iterrows():
                    symbol, exchange = from_tushare_code(row["ts_code"])
                    bar = BarData(
                        symbol=symbol,
                        exchange=exchange,
                        datetime=datetime.strptime(row["trade_date"], "%Y%m%d"),
                        interval=Interval.DAILY,
                        volume=float(row["vol"]),
                        turnover=float(row["amount"]),
                        open_interest=0,
                        open_price=round_to(row["open"], 0.000001),
                        high_price=round_to(row["high"], 0.000001),
                        low_price=round_to(row["low"], 0.000001),
                        close_price=round_to(row["close"], 0.000001),
                        gateway_name="TS",
                    )
                    bars_map[row["ts_code"]].append(bar)

                for bars in bars_map.values():
                    bars.sort(key=lambda x: x.datetime)
                    database.save_bar_data(bars)

                total_bars += len(df)
                print(f"  批次 {i // max_per_call + 1}: 保存 {len(df)} 条 ({len(bars_map)} 只股票)")
            else:
                print(f"  批次 {i // max_per_call + 1}: 无数据")
        except Exception as e:
            print(f"  批次 {i // max_per_call + 1} 下载失败: {e}")

        time.sleep(0.3)

    print(f"下载完成, 共保存 {total_bars} 条数据")


def main():
    parser = argparse.ArgumentParser(description="下载指数成分股日线数据")
    parser.add_argument("--index", nargs="+", default=list(DEFAULT_INDICES.keys()),
                        help="指数代码列表, 默认: 000300.SH 000905.SH 000852.SH 932000.CSI")
    parser.add_argument("--start", default="20180101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default=None, help="结束日期 YYYYMMDD, 默认今天")
    parser.add_argument("--all", action="store_true", help="下载所有成分股(包括已有的增量更新)")
    args = parser.parse_args()

    token = SETTINGS["datafeed.password"]
    if not token:
        print("Error: tushare token not found")
        return

    ts.set_token(token)
    pro = ts.pro_api()

    # 1. 收集所有成分股
    all_constituents = set()
    for index_code in args.index:
        name = DEFAULT_INDICES.get(index_code, index_code)
        print(f"=== {name} ({index_code}) ===")
        constituents = get_index_constituents(pro, index_code)
        all_constituents |= constituents
        time.sleep(0.3)

    if not all_constituents:
        print("未获取到任何成分股")
        return

    print(f"\n合计 {len(all_constituents)} 只不重复成分股")

    # 2. 找出数据库中缺失的股票
    if args.all:
        to_download = sorted(all_constituents)
        print(f"全量模式: 下载所有 {len(to_download)} 只股票")
    else:
        to_download = get_missing_stocks(all_constituents)
        print(f"数据库中缺失 {len(to_download)} 只股票")

    if not to_download:
        print("所有成分股数据已存在，无需下载")
        return

    # 3. 下载
    end_date = args.end
    if not end_date:
        now = datetime.now()
        end_date = now.strftime("%Y%m%d") if now.hour >= 16 else (now - timedelta(days=1)).strftime("%Y%m%d")

    download_stocks(pro, to_download, args.start, end_date)
    print("\n=== 全部完成 ===")


if __name__ == "__main__":
    main()
