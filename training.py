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
from core.alpha.run_manager import RunManager
from core.selector.selector import FundamentalSelector

# Import Calculators dynamically later


from data_manager.ts_downloader.download_daily import download_data
from data_manager.ts_downloader.daily_basic_manager import DailyBasicManager
from data_manager.ts_downloader.stock_info_manager import StockInfoManager
from data_manager.ts_downloader.concept_manager import ConceptManager
from data_manager.ts_downloader.fina_indicator_manager import FinaIndicatorManager
from data_manager.ts_downloader.moneyflow_manager import MoneyFlowManager
from data_manager.ts_downloader.margin_manager import MarginManager
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

def resolve_version_config() -> dict:
    """Scan core/alpha for v*_factor_calculator.py modules → {version: (CalcClass, description)}.

    Reused by the CLI (__main__) AND the auto-research orchestrator (research_runner),
    so both discover factor calculators the same way.
    """
    version_config = {}
    alpha_dir = Path("core/alpha")
    if alpha_dir.exists():
        for file_path in alpha_dir.glob("v*_factor_calculator.py"):
            module_name = file_path.stem
            version_str = module_name.split('_')[0]
            class_name = f"{version_str.capitalize()}FactorCalculator"
            try:
                module = importlib.import_module(f"core.alpha.{module_name}")
                calc_class = getattr(module, class_name)
                version_config[version_str] = (calc_class, f"{version_str.upper()} Factor Calculator")
            except Exception as e:
                print(f"Warning: Failed to load {class_name} from {module_name}: {e}")
    return version_config


def run_training(args, version: str, CalcClass, description: str) -> dict:
    """Execute the full data→factor→signal→backtest pipeline for one CLI run.

    Behavioral twin of the former __main__ body; CLI output is byte-identical
    (same code, same order). Extracted so the pipeline is callable, though the
    auto-research orchestrator builds its OWN engine (for per-seed data reuse)
    and does NOT call this. Returns the backtest result dict ({} if --ans/--skip).
    """
    if args.no_gp:
        gp_filter = []
    elif args.gp_test:
        gp_filter = ["validated", "testing"]
    else:
        gp_filter = None
    calc_kwargs = {}
    if getattr(args, 'label_mode', None):
        calc_kwargs['label_mode'] = args.label_mode
    calculator = CalcClass(gp_status_filter=gp_filter, **calc_kwargs)
    signal_name = f"ashare_mlp_signal_{version}"

    print(f"Mode: {description}")

    selector=FundamentalSelector([args.vt]) if args.vt else FundamentalSelector()
    earlest_date,latest_date = selector.get_data_range()
    last_trading_date = selector.get_last_trading_day()
    last_trading_date = last_trading_date if last_trading_date else datetime.now()

    actual_signal_name = signal_name if not args.vt else f"{signal_name}_{args.vt}"

    # === Run 产物管理 ===
    # -t: 创建新 run (因子/每窗口模型/信号/回测归档到 runs/{run_id}/)
    # 无 -t: 默认补全 active run; --run 指定任意 run。无 active 且无 --run 则回退传统增量模式。
    # -vt 单标调试模式不参与 run 管理。
    run_manager = RunManager()
    run_id = None
    if not args.vt:
        if args.total:
            run_config = {
                "index": args.index,
                "retrain_days": args.retrain_days,
                "backend": args.backend,
                "ensemble": args.ensemble,
                "max_windows": args.max_windows,
                "max_holdings": args.max,
                "start_date": "2019-12-28",
            }
            run_id = run_manager.create_run(version, actual_signal_name, run_config)
        else:
            run_id = args.run if args.run else run_manager.get_active()
            if run_id:
                manifest = run_manager.load_manifest(run_id)
                if manifest is None:
                    print(f"Error: run '{run_id}' 不存在 (runs/ 目录下无 manifest)")
                    cleanup_logger()
                    return {}
                # 补全模式: 从 manifest 恢复训练配置 (窗口网格/股票宇宙必须与原 run 一致)
                cfg = manifest.get("config", {})
                args.index = cfg.get("index", args.index)
                args.retrain_days = cfg.get("retrain_days", args.retrain_days)
                args.backend = cfg.get("backend", args.backend)
                args.ensemble = cfg.get("ensemble", args.ensemble)
                actual_signal_name = manifest.get("signal_name", actual_signal_name)
                print(f"[Run] 增量补全 run: {run_id} (index={args.index}, retrain_days={args.retrain_days}, backend={args.backend}, ensemble={args.ensemble})")
            else:
                print("[Run] 无 active run 且未指定 --run — 使用传统增量模式 (lab 模型路径)")

    engine = AlphaEngine(
        factor_calculator=calculator,
        mlp_signals=MLPSignals(signal_name=actual_signal_name, force_retrain=args.total, max_windows=args.max_windows, model_backend=args.backend, retrain_days=args.retrain_days, ensemble_size=args.ensemble, run_manager=run_manager if run_id else None, run_id=run_id),
        selector=selector,
        signal_name=actual_signal_name,
        start_date="2019-12-28",
        end_date= last_trading_date.strftime("%Y-%m-%d"),
        index_filter=args.index
    )

    if args.basic:
        print("开始更新股票基础信息...")
        stock_manager = StockInfoManager()
        stock_manager.download_all()

    manager = DailyBasicManager()
    if args.download or (latest_date < last_trading_date):
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

        print("\n开始更新融资融券数据...")
        margin_manager = MarginManager()
        margin_manager.download_all()

        print("\n开始更新筹码分布数据...")
        from data_manager.ts_downloader.cyq_manager import CyqPerfManager
        cyq_manager = CyqPerfManager()
        cyq_manager.download_all()

        print("\n开始更新股票名称变更数据(ST过滤用)...")
        nc_manager = NamechangeManager()
        nc_manager.download_all()

        print("\n开始下载指数数据...")
        download_index_all()

        print("\n开始更新分钟数据与SSL embedding...")
        try:
            from scripts.minute_ssl_update import incremental_download, update_embeddings
            incremental_download()
            update_embeddings()
        except Exception as e:
            print(f"SSL embedding 更新失败（不阻断训练，可手动运行 scripts/minute_ssl_update.py）: {e}")

        print("数据同步到alpha lab...")
        engine.sync_data()
    else:
        print("数据已是最新...")

    if args.force:
        print("强制更新数据同步到alpha lab...")
        engine.sync_data()

    if args.skip:
        print("跳过执行")
        return {}

    data_df = engine.load_data()
    signal_df = engine.calculate_factors(data_df)

    # Auto-research: analyze_factor_performance now returns (factors_df, factor_metrics)
    signal_df, _factor_metrics = engine.analyze_factor_performance(signal_df)

    # 因子产物:
    # - 全局因子库 factors/{version}.parquet: 不论全量/增量, 因子都按最新代码重算,
    #   按因子名增量合并仅保留一份 (不再 per-run 快照)
    # - 因子 IC 摘要: 仅全量训练时写入 run manifest
    if not args.vt and not args.no_factor_store:
        try:
            run_manager.save_factors(version, signal_df)
        except Exception as e:
            print(f"[FactorStore] 因子库更新失败 (不阻断训练): {e}")
    if run_id and args.total:
        try:
            run_manager.update_manifest(run_id, {"factors": {
                f: {"ic": round(m.get("ic", 0.0), 4), "icir": round(m.get("icir", 0.0), 4)}
                for f, m in (_factor_metrics or {}).items()
            }})
        except Exception as e:
            print(f"[Run] 因子 IC 摘要写入失败 (不阻断训练): {e}")

    result = {}
    if not args.ans:
        signal_df = engine.calculate_signals(signal_df)

        if run_id:
            # 信号存入 run 目录 (已有信号时只追加新日期)
            run_manager.save_signal(run_id, signal_df)
            if args.total and not args.no_activate:
                # 新全量 run 默认设为生产 run, 信号同步到 signal/{signal_name}.parquet
                run_manager.set_active(run_id)
            elif run_manager.get_active() == run_id:
                run_manager.sync_signal_to_production(run_id)
        else:
            engine.save_signals(signal_df)

        # 非 active run 的补全不回测: 生产信号未变更, 回测读的是生产信号路径
        do_backtest = (run_id is None) or (run_manager.get_active() == run_id)
        if do_backtest:
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
            if run_id and isinstance(result, dict) and result.get("filename"):
                run_manager.add_backtest(run_id, result["filename"])
        else:
            print(f"[Run] run {run_id} 非 active run, 跳过回测 (可先激活该 run 后再回测)")

    # === Memory Cleanup ===
    del engine
    del calculator
    del selector
    del data_df
    del signal_df
    if 'core_service' in locals():
        del core_service

    import gc
    gc.collect()

    # Close logger file handles
    cleanup_logger()
    return result


if __name__ == "__main__":

    # Configuration Map
    VERSION_CONFIG = resolve_version_config()

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
    parser.add_argument("-s", "--skip", action="store_true", help="Skip exec data.")

    parser.add_argument("-b", "--basic", action="store_true", help="Sync Stock Basic data before running")
    parser.add_argument("-d", "--download", action="store_true", help="force download daily data")

    parser.add_argument("-f", "--force", action="store_true", help="Force Sync data to Alpha Lab")

    parser.add_argument("-t", "--total", action="store_true", help="Force retrain all models (Total/Full Rolling)")

    parser.add_argument("--run", help="指定 run_id 进行增量信号补全 (无 -t 时默认补全 active run)")
    parser.add_argument("--no-factor-store", action="store_true", help="不更新全局因子库 factors/{version}.parquet (省写入时间)")
    parser.add_argument("--no-activate", action="store_true", help="全量训练后不将新 run 设为生产 (active) run")

    parser.add_argument("--gp-test", action="store_true", help="Include GP testing factors in training")
    parser.add_argument("--no-gp", action="store_true", help="Disable all GP factors")
    parser.add_argument("--index", help="Filter stocks to index constituents (e.g. 000300.SH)")

    parser.add_argument("-m", "--max", help="Max Holdings", default=5)
    parser.add_argument("--max-windows", type=int, default=0, help="Quick mode: only train last N windows (for fast factor validation)")
    parser.add_argument("--backend", choices=["attention", "lgb", "tabnet"], default="attention", help="ML model backend")
    parser.add_argument("--retrain-days", type=int, default=45, help="Retrain cycle in trading days (default 45)")
    parser.add_argument("--ensemble", type=int, default=1, help="Ensemble size (number of models to average, default 1)")
    parser.add_argument("--label-mode", default=None, help="Override label mode (e.g. 3d_raw, 3d_excess, 5d)")

    parser.add_argument("-vt", help="vt_symbol mode")

    args = parser.parse_args()

    # Determine version
    selected_version = args.version
    if not selected_version:
        for ver in VERSION_CONFIG:
            if getattr(args, ver):
                selected_version = ver
                break

    # --run 补全模式可省略版本参数: 从 run manifest 解析
    if not selected_version and args.run:
        _m = RunManager().load_manifest(args.run)
        if _m and _m.get("version"):
            selected_version = _m["version"]
            print(f"[Run] Version resolved from run manifest: {selected_version}")

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
    run_training(args, version, CalcClass, description)

