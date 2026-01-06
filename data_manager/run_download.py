
import sys
import os
from pathlib import Path

from ts_downloader.download_daily import download_data
from ts_downloader.daily_basic_manager import DailyBasicManager
from ts_downloader.stock_info_manager import StockInfoManager


if __name__ == "__main__":
    #print("开始更新股票基础信息...")
    #stock_manager = StockInfoManager()
    #stock_manager.download_all()

    print("\n开始下载历史数据...")
    download_data(end_date="latest")

    print("\n开始更新每日指标数据...")
    manager = DailyBasicManager()
    manager.download_all()