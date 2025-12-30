from download_daily import download_data
from daily_basic_manager import DailyBasicManager


if __name__ == "__main__":
    print("开始下载历史数据...")
    download_data(end_date="latest")

    print("\n开始更新每日指标数据...")
    manager = DailyBasicManager()
    manager.download_all()

