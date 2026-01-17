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
from core.alpha.v7_factor_calculator import V7FactorCalculator


from data_manager.ts_downloader.download_daily import download_data
from data_manager.ts_downloader.daily_basic_manager import DailyBasicManager
from data_manager.ts_downloader.stock_info_manager import StockInfoManager
from data_manager.ts_downloader.concept_manager import ConceptManager


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
    
    # Configuration Map
    VERSION_CONFIG = {
        "v3": (V3FactorCalculator, "V3"),
        "v4": (V4FactorCalculator, "V4 (Alpha101)"),
        "v5": (V5FactorCalculator, "V5 (Alpha158)"),
        "v6": (V6FactorCalculator, "V6 (Fusion)"),
        "v7": (V7FactorCalculator, "V7 (Concept Embedding)"),
    }
    
    parser = argparse.ArgumentParser(description="Unified Alpha Run Script")
    
    # Version argument: allow -v v3 or -v 3
    parser.add_argument("-v", "--version", required=False, help=f"Alpha version (e.g., {', '.join(VERSION_CONFIG.keys())})")
    
    # Dynamic flags for backward compatibility
    for ver in VERSION_CONFIG:
        parser.add_argument(f"-{ver}", action="store_true", help=f"Run {ver.upper()}")
    
    parser.add_argument("-a", "--ans", action="store_true", help="Only calculate factors (no signal model)")
    parser.add_argument("-s", "--skip", action="store_true", help="Skip Sync data before running")

    parser.add_argument("-b", "--basic", action="store_true", help="Sync Stock Basic data before running")
    
    parser.add_argument("-f", "--force", action="store_true", help="Force Sync data to Alpha Lab")

    parser.add_argument("-t", "--total", action="store_true", help="Force retrain all models (Total/Full Rolling)")

    parser.add_argument("-vt", help="vt_symbol mode")
    
    args = parser.parse_args()
    
    # Determine version
    selected_version = args.version
    if not selected_version:
        for ver in VERSION_CONFIG:
            if getattr(args, ver):
                selected_version = ver
                break
    
    if not selected_version:
        print(f"Error: Please specify a version using -v [version] or -{ '/-'.join(VERSION_CONFIG.keys())} flags.")
        sys.exit(1)
        
    # Normalize version string
    version = selected_version.lower()
    if not version.startswith("v"):
        version = "v" + version
        
    if version not in VERSION_CONFIG:
        print(f"Error: Unknown version '{version}'. Supported versions: {', '.join(VERSION_CONFIG.keys())}")
        sys.exit(1)
    
    setup_logger(version)
    
    print(f"Initializing Alpha Engine for {version.upper()}...")
    
    CalcClass, description = VERSION_CONFIG[version]
    calculator = CalcClass()
    signal_name = f"ashare_mlp_signal_{version}"
    
    print(f"Mode: {description}")

    selector=FundamentalSelector([args.vt]) if args.vt else FundamentalSelector()
    earlest_date,latest_date = selector.get_data_range()
    last_trading_date = selector.get_last_trading_day()
    last_trading_date = last_trading_date if last_trading_date else datetime.now()
    
    actual_signal_name = signal_name if not args.vt else f"{signal_name}_{args.vt}"
    
    engine = AlphaEngine(
        factor_calculator=calculator,
        mlp_signals=MLPSignals(signal_name=actual_signal_name, force_retrain=args.total),
        selector=selector,
        signal_name=actual_signal_name,
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
        
        print("开始更新概念板块信息...")
        concept_manager = ConceptManager()
        concept_manager.download_daily()
        concept_manager.download_members()
        
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

