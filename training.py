import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
from core.alpha.v8_factor_calculator import V8FactorCalculator


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

def save_log_dump(version: str):
    log_filename = f"log/run_{version}.log"
    output_filename = f"traning{version.upper()}.txt"
    
    if not os.path.exists(log_filename):
        print(f"Log file {log_filename} not found.")
        return

    try:
        with open(log_filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parts = []
        
        # 1. Extract Factor Performance
        # Find the LAST occurrence of the Factor Analysis Header
        start_marker = "=== 因子绩效分析"
        start_marker_idx = content.rfind(start_marker)
        
        if start_marker_idx != -1:
            # Find the start of the line (to include timestamp)
            section_start = content.rfind('\n', 0, start_marker_idx) + 1
            
            # Find the end: after "Top 5 Factors" block
            top5_marker = "[Top 5 Factors"
            top5_idx = content.find(top5_marker, start_marker_idx)
            
            section_end = -1
            if top5_idx != -1:
                # Top 5 block usually has 6 lines (Header + 5 factors)
                # scan for 6 newlines
                curr = top5_idx
                for _ in range(6):
                    next_nl = content.find('\n', curr + 1)
                    if next_nl == -1:
                        curr = len(content)
                        break
                    curr = next_nl
                section_end = curr + 1 # Include the newline
            else:
                # Fallback if Top 5 not found, take until next double newline or some limit
                section_end = content.find('\n\n', start_marker_idx)
                if section_end == -1: section_end = min(len(content), start_marker_idx + 5000)

            parts.append(content[section_start:section_end])
        
        # 2. Extract Backtest Statistics
        # Look for the start of the stats section
        stats_markers = ["历史数据回放结束", "开始计算逐日盯市盈亏", "开始计算策略统计指标"]
        stats_start_idx = -1
        
        for marker in stats_markers:
            idx = content.rfind(marker)
            if idx != -1:
                stats_start_idx = idx
                break
        
        if stats_start_idx != -1:
            # Find the start of the line
            stats_section_start = content.rfind('\n', 0, stats_start_idx) + 1
            # Add a separator if we have previous parts
            # (Usually logs are contiguous, but we can verify)
            if parts:
                # Check if last part ends with newline, if not add one
                if not parts[-1].endswith('\n'):
                    parts.append("\n")
            parts.append(content[stats_section_start:])
            
        if parts:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write("".join(parts))
            print(f"Log dump saved to {output_filename}")
        else:
            print("No relevant log sections found to dump.")
            
    except Exception as e:
        print(f"Failed to save log dump: {e}")

# --------------------------

if __name__ == "__main__":
    
    # Configuration Map
    VERSION_CONFIG = {
        "v3": (V3FactorCalculator, "V3"),
        "v4": (V4FactorCalculator, "V4 (Alpha101)"),
        "v5": (V5FactorCalculator, "V5 (Alpha158)"),
        "v6": (V6FactorCalculator, "V6 (Fusion)"),
        "v7": (V7FactorCalculator, "V7 (Concept Embedding)"),
        "v8": (V8FactorCalculator, "V8 (Stacking V4)"),
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
    
    save_log_dump(version)

