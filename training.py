import sys
import os
from datetime import datetime
from pathlib import Path
import re
import argparse

from vnpy.alpha import logger
from core.logger_writer import LoggerWriter
from core.alpha.engine import AlphaEngine
from core.alpha.mlp_signals import MLPSignals
from core.selector.selector import FundamentalSelector

# Import Calculators
from core.alpha.v3_factor_calculator import V3FactorCalculator
from core.alpha.v4_factor_calculator import V4FactorCalculator
from core.alpha.v5_factor_calculator import V5FactorCalculator
from core.alpha.v6_factor_calculator import V6FactorCalculator


from data_manager.ts_downloader.download_daily import download_data
from data_manager.ts_downloader.daily_basic_manager import DailyBasicManager
from data_manager.ts_downloader.stock_info_manager import StockInfoManager


from core.core_service import CoreService

def setup_logger(version: str):
    log_filename = f"log/run_{version}.log"
    
    # Redirect stdout and stderr to log file
    if not hasattr(sys.stdout, 'file') or not isinstance(sys.stdout, LoggerWriter):
        try:
            file = open(log_filename, 'w', encoding='utf-8')

            sys.stdout = LoggerWriter(sys.stdout, file)
            sys.stderr = LoggerWriter(sys.stderr, file)

            # Remove default output
            logger.remove()

            # Add terminal output (which now goes to file via LoggerWriter)
            fmt: str = "{time:YYYY-MM-DD HH:mm:ss} {message}"
            logger.add(sys.stdout, colorize=True, format=fmt)
        except Exception as e:
            print(f"Failed to setup logger redirection: {e}")

# --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Alpha Run Script")
    
    # Version argument: allow -v v3 or -v 3
    parser.add_argument("-v", "--version", required=False, help="Alpha version (e.g., v3, v4, v5)")
    
    # Support direct flags like -v3, -v4, -v5 for convenience/backward compatibility requests
    parser.add_argument("-v3", action="store_true", help="Run V3")
    parser.add_argument("-v4", action="store_true", help="Run V4")
    parser.add_argument("-v5", action="store_true", help="Run V5")
    parser.add_argument("-v6", action="store_true", help="Run V6")
    
    parser.add_argument("-a", "--ans", action="store_true", help="Only calculate factors (no signal model)")
    parser.add_argument("-s", "--skip", action="store_true", help="Skip Sync data before running")

    parser.add_argument("-b", "--basic", action="store_true", help="Sync Stock Basic data before running")
    
    parser.add_argument("-f", "--force", action="store_true", help="Force Sync data to Alpha Lab")

    parser.add_argument("-vt", help="vt_symbol mode")
    
    args = parser.parse_args()
    
    # Determine version
    selected_version = None
    if args.version:
        selected_version = args.version
    elif args.v3:
        selected_version = "v3"
    elif args.v4:
        selected_version = "v4"
    elif args.v5:
        selected_version = "v5"
    elif args.v6:
        selected_version = "v6"
    
    if not selected_version:
        print("Error: Please specify a version using -v [version] or -v3/-v4/-v5 flags.")
        sys.exit(1)
        
    # Normalize version string
    version = selected_version.lower()
    if not version.startswith("v"):
        version = "v" + version
    
    setup_logger(version)
    
    print(f"Initializing Alpha Engine for {version.upper()}...")
    
    if version == "v3":
        calculator = V3FactorCalculator()
        signal_name = "ashare_mlp_signal_v3"
        description = "V3"
    elif version == "v4":
        calculator = V4FactorCalculator()
        signal_name = "ashare_mlp_signal_v4"
        description = "V4 (Alpha101)"
    elif version == "v5":
        calculator = V5FactorCalculator()
        signal_name = "ashare_mlp_signal_v5"
        description = "V5 (Alpha158)"
    elif version == "v6":
        calculator = V6FactorCalculator()
        signal_name = "ashare_mlp_signal_v6"
        description = "V6 (Fusion)"
    else:
        print(f"Error: Unknown version '{version}'. Supported versions: v3, v4, v5, v6")
        sys.exit(1)

    selector=FundamentalSelector([args.vt]) if args.vt else FundamentalSelector()
    earlest_date,latest_date = selector.get_data_range()
    last_trading_date = selector.get_last_trading_day()
    last_trading_date = last_trading_date if last_trading_date else datetime.now()
    engine = AlphaEngine(
        factor_calculator=calculator,
        mlp_signals=MLPSignals(),
        selector=selector,
        signal_name=signal_name if not args.vt else f"{signal_name}_{args.vt}",
        start_date="2019-12-28",
        end_date= last_trading_date.strftime("%Y-%m-%d")
    )

    if args.basic:
        print("开始更新股票基础信息...")
        stock_manager = StockInfoManager()
        stock_manager.download_all()

    manager = DailyBasicManager()
    if latest_date < last_trading_date and not args.skip:
        print("开始下载历史数据...")
        download_data(end_date=last_trading_date.strftime("%Y%m%d"))

        print("开始更新每日指标数据...")
        manager.download_all()
        
        print("数据同步到alpha lab...")
        engine.sync_data()
    else: 
        print("数据已是最新...")
    
    if args.force: 
        print("强制更新数据同步到alpha lab...")
        engine.sync_data()
    
    signal_df = engine.calculate_factors()

    signal_df = engine.analyze_factor_performance(signal_df)

    if not args.ans:
        signal_df = engine.calculate_signals(signal_df)
        engine.save_signals(signal_df)
        
        core_service = CoreService()
        result = core_service.run_backtest(
            strategy_name="MultiFactorStrategy",
            start=datetime.strptime("2022-01-01", "%Y-%m-%d"),
            end=last_trading_date,
            setting={
                "max_holdings": 5,
                "signal_name": signal_name,
            }
        )

