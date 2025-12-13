import json
from datetime import datetime

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.datafeed import get_datafeed
from vnpy.trader.database import get_database
from vnpy.trader.object import HistoryRequest


def download_data(config_path: str = "core/data_download/download_config.json"):
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

    # 遍历下载配置
    for task in config["downloads"]:
        # 解析任务参数
        symbol: str = task["symbol"]
        exchange: Exchange = Exchange(task["exchange"])
        interval: Interval = Interval(task["interval"])
        start: datetime = datetime.strptime(task["start_date"], "%Y%m%d")
        if task["end_date"] == "latest":
            end = datetime.now()
        else:
            end: datetime = datetime.strptime(task["end_date"], "%Y%m%d")

        # 为了兼容交易所代码后缀（如SH、SZ），统一从配置中读取并拆分
        req_symbol: str = symbol.split(".")[0]

        print(f"开始下载【{symbol}】的【{interval.value}】数据，时间范围：{task['start_date']}-{task['end_date']}")

         # 创建历史数据请求
        req = HistoryRequest(
            symbol=req_symbol,
            exchange=exchange,
            interval=interval,
            start=start,
            end=end
        )

        # 从数据服务获取数据
        bars = datafeed.query_bar_history(req)

        # 存入数据库
        if bars:
            database.save_bar_data(bars)
            print(f"成功保存【{symbol}】分段数据：{len(bars)}条")
        else:
            print(f"分段【{symbol}】数据下载失败或为空")

    print("所有数据下载完成！")


if __name__ == "__main__":
    download_data()