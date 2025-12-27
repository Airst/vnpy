import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.datafeed import get_datafeed
from vnpy.trader.database import get_database
from vnpy.trader.object import HistoryRequest


def download_data(config_path: str = "data_download/download_config.json"):
    """
    下载历史数据并存入数据库
    """
    # 初始化数据服务和数据库接口
    datafeed = get_datafeed()
    database = get_database()
    print("数据库连接成功")

    # 从json文件加载下载任务配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

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
    # 如果当前时间在16点之前，认为当日数据尚未收盘/就绪，取前一日
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
                
                # 如果数据库已有数据覆盖了当前start，则从db_end开始（增量）
                # 除非配置的start明显晚于db_end（说明中间有断层？通常是希望接着db_end下载）
                # 这里简单处理：取 max(start, db_end)
                if db_end >= start:
                    start = db_end
        
        # 如果调整后的start已经超过end，说明不需要下载
        if start >= end:
            print(f"数据已是最新，跳过: {symbol} ({start} >= {end})")
            continue

        tasks[key] = {
            "symbol": symbol,
            "req_symbol": req_symbol,
            "exchange": exchange,
            "interval": interval,
            "start": start,
            "end": end,
            "source": "config"
        }

    # 1.2 添加数据库中已有的任务（增量更新）
    for overview in overviews:
        key = (overview.symbol, overview.exchange, overview.interval)
        
        if key in tasks:
            continue  # 配置文件优先（已被处理）

        # 增量更新：从最后一条数据的时间开始
        start = overview.end
        if not start:
            start = datetime(2020, 1, 1)
            
        # 如果start带时区，转为naive以便对比（假设系统运行在Local）
        if start.tzinfo:
            start = start.replace(tzinfo=None)

        end = latest_date
        
        # 如果结束时间早于开始时间（或者差距很小），则跳过
        if start >= end:
            continue
            
        tasks[key] = {
            "symbol": f"{overview.symbol}.{overview.exchange.value}",
            "req_symbol": overview.symbol,
            "exchange": overview.exchange,
            "interval": overview.interval,
            "start": start,
            "end": end,
            "source": "database"
        }

    print(f"共生成 {len(tasks)} 个下载任务")

    # 2. 执行下载
    def process_task(task):
        symbol = task["symbol"]
        req_symbol = task["req_symbol"]
        exchange = task["exchange"]
        interval = task["interval"]
        start = task["start"]
        end = task["end"]
        
        print(f"[{task['source']}] 开始下载【{symbol}】... {start} - {end}")
        
        req = HistoryRequest(
            symbol=req_symbol,
            exchange=exchange,
            interval=interval,
            start=start,
            end=end
        )

        try:
            bars = datafeed.query_bar_history(req)
            if bars:
                database.save_bar_data(bars)
                return f"成功保存【{symbol}】: {len(bars)}条"
            else:
                return f"数据为空【{symbol}】"
        except Exception as e:
            return f"下载失败【{symbol}】: {e}"

    # 使用线程池并发下载
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks.values()}
        
        for future in as_completed(futures):
            result = future.result()
            print(result)

    print("所有数据下载完成！")


if __name__ == "__main__":
    download_data()