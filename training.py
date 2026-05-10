import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from datetime import datetime
from pathlib import Path
import re
import argparse
import importlib

from vnpy.alpha import logger
from core.logger_writer import LoggerWriter
from core.alpha.engine import AlphaEngine
from core.alpha.mlp_signals import MLPSignals
from core.selector.selector import FundamentalSelector

# Import Calculators dynamically later


from data_manager.ts_downloader.download_daily import download_data
from data_manager.ts_downloader.daily_basic_manager import DailyBasicManager
from data_manager.ts_downloader.stock_info_manager import StockInfoManager
from data_manager.ts_downloader.concept_manager import ConceptManager
from data_manager.ts_downloader.fina_indicator_manager import FinaIndicatorManager
from data_manager.ts_downloader.moneyflow_manager import MoneyFlowManager
from data_manager.ts_downloader.namechange_manager import NamechangeManager
from data_manager.ts_downloader.download_index import download_all as download_index_all


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

def cleanup_logger():
    """Close log file handles and restore original stdout/stderr"""
    try:
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
        if hasattr(sys.stderr, 'close'):
            sys.stderr.close()
    except Exception:
        pass

def save_log_dump(version: str):
    log_filename = f"log/run_{version}.log"
    output_filename = f"training{version.upper()}.txt"
    
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
    VERSION_CONFIG = {}
    alpha_dir = Path("core/alpha")
    if alpha_dir.exists():
        for file_path in alpha_dir.glob("v*_factor_calculator.py"):
            module_name = file_path.stem
            version_str = module_name.split('_')[0]
            class_name = f"{version_str.capitalize()}FactorCalculator"
            try:
                module = importlib.import_module(f"core.alpha.{module_name}")
                calc_class = getattr(module, class_name)
                VERSION_CONFIG[version_str] = (calc_class, f"{version_str.upper()} Factor Calculator")
            except Exception as e:
                print(f"Warning: Failed to load {class_name} from {module_name}: {e}")
    
    if not VERSION_CONFIG:
        print("Error: No valid factor calculators found in core/alpha/")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Unified Alpha Run Script")
    
    # Version argument: allow -v v3 or -v 3
    parser.add_argument("-v", "--version", required=False, help=f"Alpha version (e.g., {', '.join(VERSION_CONFIG.keys())})")
    
    # Dynamic flags for backward compatibility
    for ver in VERSION_CONFIG:
        parser.add_argument(f"-{ver}", action="store_true", help=f"Run {ver.upper()}")
    
    parser.add_argument("-a", "--ans", action="store_true", help="Only calculate factors (no signal model)")
    parser.add_argument("-s", "--skip", action="store_true", help="Skip Sync data before running")

    parser.add_argument("-b", "--basic", action="store_true", help="Sync Stock Basic data before running")
    parser.add_argument("-d", "--download", action="store_true", help="force download daily data")
    
    parser.add_argument("-f", "--force", action="store_true", help="Force Sync data to Alpha Lab")

    parser.add_argument("-t", "--total", action="store_true", help="Force retrain all models (Total/Full Rolling)")

    parser.add_argument("--gp", action="store_true", help="Run GP factor mining before training")
    parser.add_argument("--gp-only", action="store_true", help="Only run GP factor mining (no training)")

    parser.add_argument("-m", "--max", help="Max Holdings", default=5)

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
    if args.download or (latest_date < last_trading_date and not args.skip):
        print("开始下载历史数据...")
        download_data(end_date=last_trading_date.strftime("%Y%m%d"))

        print("开始更新每日指标数据...")
        manager.download_all()
        
        print("开始更新概念板块信息...")
        concept_manager = ConceptManager()
        concept_manager.download_daily()
        concept_manager.download_members()

        
        print("\n开始更新财务指标数据...")
        fina_manager = FinaIndicatorManager()
        fina_manager.download_all()
        
        print("\n开始更新资金流向数据...")
        mf_manager = MoneyFlowManager()
        mf_manager.download_all()

        print("\n开始更新筹码分布数据...")
        from data_manager.ts_downloader.cyq_manager import CyqPerfManager
        cyq_manager = CyqPerfManager()
        cyq_manager.download_all()

        print("\n开始更新股票名称变更数据(ST过滤用)...")
        nc_manager = NamechangeManager()
        nc_manager.download_all()
        
        print("\n开始下载指数数据...")
        download_index_all()
        
        print("数据同步到alpha lab...")
        engine.sync_data()
    else: 
        print("数据已是最新...")
    
    if args.force: 
        print("强制更新数据同步到alpha lab...")
        engine.sync_data()
    
    data_df = engine.load_data()
    signal_df = engine.calculate_factors(data_df)

    # === GP Factor Mining ===
    if args.gp or args.gp_only:
        from core.alpha.gp_factor_miner import GPFactorMiner
        print("\n[GP Mining] Starting GP factor discovery...")
        
        gp_miner = GPFactorMiner(
            population_size=200,
            n_generations=30,
            max_factors=10,
            min_ic=0.02,
            min_icir=0.3,
        )
        
        # Re-run factor calculation to get GPU tensors for GP
        # We need raw padded tensor and label
        import torch as _torch
        from core.alpha.factor_calculator import device as _device
        
        # Prepare data same way as FactorCalculator.calculate_features
        df_sorted = data_df.sort(["vt_symbol", "datetime"])
        exclude_cols = {"datetime", "vt_symbol", "industry"}
        cols = [c for c in df_sorted.columns if c not in exclude_cols]
        col_map = {name: i for i, name in enumerate(cols)}
        
        import numpy as _np
        raw_data = df_sorted.select(cols).to_numpy().astype(_np.float32)
        symbols = df_sorted["vt_symbol"].to_numpy()
        unique_symbols, inverse_indices, counts = _np.unique(symbols, return_inverse=True, return_counts=True)
        num_stocks = len(unique_symbols)
        max_len = counts.max()
        
        padded_raw = _torch.full((num_stocks, max_len, len(cols)), float('nan'), device=_device, dtype=_torch.float32)
        
        df_idx = df_sorted.select(["vt_symbol"]).with_columns([
            __import__('polars').int_range(0, __import__('polars').len()).over("vt_symbol").alias("t_idx")
        ])
        t_indices = df_idx["t_idx"].to_numpy()
        s_indices = inverse_indices
        
        s_indices_t = _torch.tensor(s_indices, dtype=_torch.long, device=_device)
        t_indices_t = _torch.tensor(t_indices, dtype=_torch.long, device=_device)
        raw_tensor = _torch.tensor(raw_data, device=_device, dtype=_torch.float32)
        padded_raw[s_indices_t, t_indices_t, :] = raw_tensor
        
        # Compute label for fitness evaluation
        from core.alpha.factor_calculator import ts_delay, ts_mean, cs_rank
        C = padded_raw[:, :, col_map['close']]
        raw_ret_5 = ts_delay(C, -5) / C - 1
        label = cs_rank(raw_ret_5)  # Simple forward return rank as GP target
        
        # Collect existing factor tensors for deduplication
        existing_tensors = []
        features_computed = calculator.build_features(padded_raw, col_map)
        for fname, ftensor in features_computed.items():
            if fname != "label":
                existing_tensors.append(ftensor)
        
        # Run GP mining
        results = gp_miner.mine(
            padded_raw=padded_raw,
            col_map=col_map,
            label=label,
            existing_factor_tensors=existing_tensors[:10],  # Use top-10 for speed
        )
        
        # Save discovered factors
        gp_path = "core/alpha_db/gp_factors.json"
        gp_miner.save(gp_path)
        
        del padded_raw, raw_tensor, s_indices_t, t_indices_t, label
        import gc
        gc.collect()
        _torch.cuda.empty_cache() if _torch.cuda.is_available() else None
        
        if args.gp_only:
            print(f"\n[GP Mining] Done. Found {len(results)} factors. Saved to {gp_path}")
            save_log_dump(version)
            cleanup_logger()
            sys.exit(0)
        
        # Reload calculator to pick up new GP factors
        calculator = CalcClass()
        engine.factor_calculator = calculator
        signal_df = engine.calculate_factors(data_df)
        print("[GP Mining] Factors recomputed with GP additions.")

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
                "max_holdings": args.max,
                "signal_name": signal_name,
            }
        )
    
    save_log_dump(version)
    
    # === Memory Cleanup ===
    # Release major objects and force garbage collection
    del engine
    del calculator
    del selector
    del data_df
    del signal_df
    if 'result' in locals():
        del result
    if 'core_service' in locals():
        del core_service
    
    import gc
    gc.collect()
    
    # Close logger file handles
    cleanup_logger()

