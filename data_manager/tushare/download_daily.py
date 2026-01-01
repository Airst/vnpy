import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Tuple, Optional

import pandas as pd
import tushare as ts

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData
from vnpy.trader.utility import round_to
from vnpy.trader.setting import SETTINGS


def to_tushare_code(symbol: str, exchange: Exchange) -> Optional[str]:
    """
    Convert vnpy symbol and exchange to tushare code.
    """
    if exchange == Exchange.SZSE:
        return f"{symbol}.SZ"
    elif exchange == Exchange.SSE:
        return f"{symbol}.SH"
    elif exchange == Exchange.BSE:
        return f"{symbol}.BJ"
    return None


def from_tushare_code(ts_code: str) -> Tuple[str, Exchange]:
    """
    Convert tushare code to vnpy symbol and exchange.
    """
    symbol, suffix = ts_code.split(".")
    if suffix == "SZ":
        return symbol, Exchange.SZSE
    elif suffix == "SH":
        return symbol, Exchange.SSE
    elif suffix == "BJ":
        return symbol, Exchange.BSE
    return symbol, Exchange.LOCAL


def download_data(config_path: str = "data_manager/download_daily_config.json", end_date: str = "latest"):
    """
    下载历史数据并存入数据库 (使用 Tushare 批量接口)
    """
    database = get_database()
    print("数据库连接成功")

    # 从json文件加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 初始化 Tushare
    token = SETTINGS["datafeed.password"]
    if not token:
        print("Error: tushare_token not found in config.")
        return
    
    ts.set_token(token)
    pro = ts.pro_api()
    print("Tushare 初始化成功")

    # 1. 整理下载任务
    # Key: (req_symbol, exchange, interval)
    tasks = {}

    # 获取数据库概览，用于增量更新判断
    overviews = database.get_bar_overview()
    overviews_map = {
        (o.symbol, o.exchange, o.interval): o 
        for o in overviews
    }
    
    # 确定 'latest' 对应的截止日期
    if end_date and end_date != "latest":
        latest_date = datetime.strptime(end_date, "%Y%m%d")
    else:
        now = datetime.now()
        if now.hour < 16:
            latest_date = now - timedelta(days=1)
        else:
            latest_date = now
        latest_date = latest_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # 1.1 添加配置文件中的任务
    for task in config.get("downloads", []):
        symbol = task["symbol"]
        exchange = Exchange(task["exchange"])
        interval = Interval(task["interval"])
        req_symbol = symbol.split(".")[0]
        
        # 只处理日线数据
        if interval != Interval.DAILY:
            print(f"跳过非日线任务: {symbol} {interval}")
            continue

        start = datetime.strptime(task["start_date"], "%Y%m%d")
        if task["end_date"] == "latest":
            end = latest_date
        else:
            end = datetime.strptime(task["end_date"], "%Y%m%d")

        # 检查数据库中是否已有数据，若有则更新start时间
        key = (req_symbol, exchange, interval)
        if key in overviews_map:
            overview = overviews_map[key]
            if overview.end:
                db_end = overview.end
                if db_end.tzinfo:
                    db_end = db_end.replace(tzinfo=None)
                
                # 增量更新
                if db_end >= start:
                    start = db_end
        
        if start >= end:
            print(f"数据已是最新，跳过: {symbol} ({start.date()} >= {end.date()})")
            continue

        tasks[key] = {
            "req_symbol": req_symbol,
            "exchange": exchange,
            "interval": interval,
            "start": start,
            "end": end
        }

    # 1.2 添加数据库中已有的任务（增量更新）
    for overview in overviews:
        if overview.interval != Interval.DAILY:
            continue

        key = (overview.symbol, overview.exchange, overview.interval)
        
        if key in tasks:
            continue  # 配置文件优先

        start = overview.end
        if not start:
            start = datetime(2018, 1, 1) # 默认起始时间
            
        if start.tzinfo:
            start = start.replace(tzinfo=None)

        end = latest_date
        
        if start >= end:
            continue
            
        tasks[key] = {
            "req_symbol": overview.symbol,
            "exchange": overview.exchange,
            "interval": overview.interval,
            "start": start,
            "end": end
        }

    print(f"共生成 {len(tasks)} 个下载任务")

    # 2. 任务分组 (按 start_date, end_date 分组)
    # Key: (start_str, end_str) -> List[ts_code]
    batches = defaultdict(list)
    
    for key, task in tasks.items():
        ts_code = to_tushare_code(task["req_symbol"], task["exchange"])
        if not ts_code:
            continue
            
        start_str = task["start"].strftime("%Y%m%d")
        end_str = task["end"].strftime("%Y%m%d")
        batches[(start_str, end_str)].append(ts_code)

    # 3. 批量下载与保存
    total_batches = len(batches)
    processed_count = 0
    
    for (start_str, end_str), ts_codes in batches.items():
        start_date = datetime.strptime(start_str, "%Y%m%d")
        end_date = datetime.strptime(end_str, "%Y%m%d") # type: ignore
        days = (end_date - start_date).days + 1 # type: ignore
        if days < 1: 
            days = 1
            
        # 计算每批次最大股票数量
        # Tushare 限制: 单次请求最多返回 6000-8000 行 (保险起见用 5000)
        # 行数 = 股票数 * 天数
        # 股票数 = 5000 / 天数
        max_symbols_per_call = int(5000 / days)
        if max_symbols_per_call < 1:
            max_symbols_per_call = 1
        if max_symbols_per_call > 100: # 限制URL长度/参数过多
            max_symbols_per_call = 100
            
        print(f"处理时间段 {start_str} - {end_str}, 天数: {days}, 批次大小: {max_symbols_per_call}, 总股票数: {len(ts_codes)}")

        # 分块处理
        for i in range(0, len(ts_codes), max_symbols_per_call):
            chunk = ts_codes[i : i + max_symbols_per_call]
            ts_code_str = ",".join(chunk)
            
            try:
                # 调用 Tushare 接口
                df = pro.daily(ts_code=ts_code_str, start_date=start_str, end_date=end_str)
                
                if df is not None and not df.empty:
                    bars_map = defaultdict(list)
                    for _, row in df.iterrows():
                        symbol, exchange = from_tushare_code(row["ts_code"])
                        
                        bar = BarData(
                            symbol=symbol,
                            exchange=exchange,
                            datetime=datetime.strptime(row["trade_date"], "%Y%m%d"),
                            interval=Interval.DAILY,
                            volume=float(row["vol"]),      # 手 -> 股
                            turnover=float(row["amount"]),# 千元 -> 元
                            open_interest=0,
                            open_price=round_to(row["open"], 0.000001),
                            high_price=round_to(row["high"], 0.000001),
                            low_price=round_to(row["low"], 0.000001),
                            close_price=round_to(row["close"], 0.000001),
                            gateway_name="TS"
                        )
                        bars_map[row["ts_code"]].append(bar)
                    
                    if bars_map:
                        for bars in bars_map.values():
                            # 根据datetime升序排序
                            bars.sort(key=lambda x: x.datetime)
                            database.save_bar_data(bars)
                        print(f"  已保存 {len(df)} 条数据 (涵盖 {len(bars_map)} 只股票)")
                else:
                    print(f"  无数据返回: {ts_code_str} ({start_str}-{end_str})")

            except Exception as e:
                print(f"  下载失败: {e}")
            
            # 速率限制: 每分钟 200 次 -> ~0.3s/次
            time.sleep(0.3)
            
        processed_count += 1
        print(f"进度: {processed_count}/{total_batches} 时间段完成")

    print("所有数据下载任务完成！")


