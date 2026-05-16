"""
GP Factor Mining - 独立脚本
从 training.py 中拆出的 GP 因子挖掘流程，可独立运行。

== 用法 ==
挖掘:
  python gp_mining.py -v12                          # 默认参数运行GP挖掘
  python gp_mining.py -v12 --pop 500 --gen 60       # 自定义种群和代数
  python gp_mining.py -v12 --max-factors 10         # 最多发现10个因子
  python gp_mining.py -v12 --min-ic 0.03            # IC阈值0.03

注册表管理:
  python gp_mining.py -v12 --status                 # 查看因子注册表状态
  python gp_mining.py -v12 --accept gp_001,gp_002   # 标记为validated
  python gp_mining.py -v12 --reject gp_003 --note "gp_003:redundant with mom_5d"
  python gp_mining.py -v12 --test gp_004            # 标记为testing

== 流程 ==
1. 加载数据 → 构建 GPU 张量 (复用 AlphaEngine + FactorCalculator)
2. 计算 label (未来5日收益率 cs_rank) 用于 fitness 评估
3. 收集当前因子张量用于相关性去重
4. 运行 GP 进化 (种群选择 → 交叉 → 变异 → 适应度评估)
5. 滚动 IC 验证 → 更新注册表状态
"""
import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import gc
import argparse
import importlib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import polars as pl

from core.alpha.gp_factor_miner import GPFactorMiner
from core.alpha.factor_calculator import device, ts_delay, ts_mean, cs_rank
from core.alpha.mlp_signals import MLPSignals
from core.alpha.engine import AlphaEngine
from core.selector.selector import FundamentalSelector


GP_REGISTRY_PATH = "core/alpha/gp_factors.json"


def discover_versions():
    """动态发现所有版本的 FactorCalculator"""
    config = {}
    alpha_dir = Path("core/alpha")
    if alpha_dir.exists():
        for file_path in alpha_dir.glob("v*_factor_calculator.py"):
            module_name = file_path.stem
            version_str = module_name.split('_')[0]
            class_name = f"{version_str.capitalize()}FactorCalculator"
            try:
                module = importlib.import_module(f"core.alpha.{module_name}")
                calc_class = getattr(module, class_name)
                config[version_str] = calc_class
            except Exception as e:
                print(f"Warning: Failed to load {class_name} from {module_name}: {e}")
    return config


def parse_args(version_config):
    parser = argparse.ArgumentParser(description="GP Factor Mining")

    parser.add_argument("-v", "--version", required=False,
                        help=f"Alpha version (e.g., {', '.join(version_config.keys())})")
    for ver in version_config:
        parser.add_argument(f"-{ver}", action="store_true", help=f"Use {ver.upper()} calculator")

    # GP mining parameters
    parser.add_argument("--pop", type=int, default=300, help="Population size (default: 300)")
    parser.add_argument("--gen", type=int, default=40, help="Number of generations (default: 40)")
    parser.add_argument("--max-factors", type=int, default=5, help="Max factors to discover (default: 5)")
    parser.add_argument("--min-ic", type=float, default=0.02, help="Min IC threshold (default: 0.02)")
    parser.add_argument("--min-icir", type=float, default=0.3, help="Min ICIR threshold (default: 0.3)")

    # Registry management
    parser.add_argument("--status", action="store_true", help="Show GP factor registry status and exit")
    parser.add_argument("--accept", type=str, metavar="IDS", help="Mark factor IDs as validated (comma-separated)")
    parser.add_argument("--reject", type=str, metavar="IDS", help="Mark factor IDs as rejected (comma-separated)")
    parser.add_argument("--test", type=str, metavar="IDS", help="Mark factor IDs as testing (comma-separated)")
    parser.add_argument("--note", type=str, metavar="ID:NOTE", help="Add note to a factor (format: 'gp_001:note text')")

    parser.add_argument("-s", "--skip", action="store_true", help="Skip data sync")

    return parser.parse_args()


def resolve_version(args, version_config):
    version = args.version
    if not version:
        for ver in version_config:
            if getattr(args, ver):
                version = ver
                break
    if not version:
        print(f"Error: Please specify a version using -v [version] or -{'/-'.join(version_config.keys())} flags.")
        sys.exit(1)
    version = version.lower()
    if not version.startswith("v"):
        version = "v" + version
    if version not in version_config:
        print(f"Error: Unknown version '{version}'. Supported: {', '.join(version_config.keys())}")
        sys.exit(1)
    return version


def handle_registry_commands(args):
    """处理注册表管理命令（不需要加载数据）"""
    if args.status:
        GPFactorMiner.print_status(GP_REGISTRY_PATH)
        return True

    if args.accept:
        ids = [x.strip() for x in args.accept.split(",")]
        GPFactorMiner.set_status(GP_REGISTRY_PATH, ids, "validated")
        return True

    if args.reject:
        ids = [x.strip() for x in args.reject.split(",")]
        note = ""
        if args.note:
            parts = args.note.split(":", 1)
            note = parts[1] if len(parts) > 1 else parts[0]
        GPFactorMiner.set_status(GP_REGISTRY_PATH, ids, "rejected", note=note)
        return True

    if args.test:
        ids = [x.strip() for x in args.test.split(",")]
        GPFactorMiner.set_status(GP_REGISTRY_PATH, ids, "testing")
        return True

    if args.note:
        parts = args.note.split(":", 1)
        if len(parts) == 2:
            factor_id, note = parts[0].strip(), parts[1].strip()
            registry = GPFactorMiner.load_registry(GP_REGISTRY_PATH)
            for f in registry["factors"]:
                if f["id"] == factor_id:
                    f["note"] = note
                    GPFactorMiner.save_registry(GP_REGISTRY_PATH, registry)
                    print(f"[GP] Note added to {factor_id}: {note}")
                    break
            else:
                print(f"[GP] Factor {factor_id} not found")
        return True

    return False


def prepare_data(version, CalcClass):
    """加载数据并准备 GPU 张量"""
    selector = FundamentalSelector()
    _, _ = selector.get_data_range()
    last_trading_date = selector.get_last_trading_day()
    last_trading_date = last_trading_date if last_trading_date else datetime.now()

    calculator = CalcClass()
    signal_name = f"ashare_mlp_signal_{version}"

    engine = AlphaEngine(
        factor_calculator=calculator,
        mlp_signals=MLPSignals(signal_name=signal_name, force_retrain=False),
        selector=selector,
        signal_name=signal_name,
        start_date="2019-12-28",
        end_date=last_trading_date.strftime("%Y-%m-%d"),
    )

    data_df = engine.load_data()
    _ = engine.calculate_factors(data_df)

    # Build padded tensor (same as training.py)
    df_sorted = data_df.sort(["vt_symbol", "datetime"])
    exclude_cols = {"datetime", "vt_symbol", "industry"}
    cols = [c for c in df_sorted.columns if c not in exclude_cols]
    col_map = {name: i for i, name in enumerate(cols)}

    raw_data = df_sorted.select(cols).to_numpy().astype(np.float32)
    symbols = df_sorted["vt_symbol"].to_numpy()
    unique_symbols, inverse_indices, counts = np.unique(symbols, return_inverse=True, return_counts=True)
    num_stocks = len(unique_symbols)
    max_len = counts.max()

    padded_raw = torch.full((num_stocks, max_len, len(cols)), float('nan'), device=device, dtype=torch.float32)

    df_idx = df_sorted.select(["vt_symbol"]).with_columns([
        pl.int_range(0, pl.len()).over("vt_symbol").alias("t_idx")
    ])
    t_indices = df_idx["t_idx"].to_numpy()
    s_indices = inverse_indices

    s_indices_t = torch.tensor(s_indices, dtype=torch.long, device=device)
    t_indices_t = torch.tensor(t_indices, dtype=torch.long, device=device)
    raw_tensor = torch.tensor(raw_data, device=device, dtype=torch.float32)
    padded_raw[s_indices_t, t_indices_t, :] = raw_tensor

    del raw_data, raw_tensor, s_indices_t, t_indices_t, df_sorted, df_idx

    # Label: simple forward return rank
    C = padded_raw[:, :, col_map['close']]
    raw_ret_5 = ts_delay(C, -5) / C - 1
    label = cs_rank(raw_ret_5)
    del C, raw_ret_5

    # Existing factor tensors for dedup
    existing_tensors = []
    features_computed = calculator.build_features(padded_raw, col_map)
    for fname, ftensor in features_computed.items():
        if fname != "label":
            existing_tensors.append(ftensor)
    del features_computed

    gc.collect()
    torch.cuda.empty_cache()

    return padded_raw, col_map, label, existing_tensors


def run_mining(args, padded_raw, col_map, label, existing_tensors):
    """执行 GP 因子挖掘"""
    gp_miner = GPFactorMiner(
        population_size=args.pop,
        n_generations=args.gen,
        max_factors=args.max_factors,
        min_ic=args.min_ic,
        min_icir=args.min_icir,
    )

    results = gp_miner.mine(
        padded_raw=padded_raw,
        col_map=col_map,
        label=label,
        existing_factor_tensors=existing_tensors[:10],
        registry_path=GP_REGISTRY_PATH,
    )

    # Append to registry
    gp_miner.append_discovered(GP_REGISTRY_PATH, gp_miner.discovered_factors)

    # Auto-validation gate: rolling IC filter
    if gp_miner.discovered_factors:
        print("\n[GP Validate] Running rolling IC validation on discovered factors...")
        accepted, rejected = GPFactorMiner.validate_discovered(
            path=GP_REGISTRY_PATH,
            padded_raw=padded_raw,
            col_map=col_map,
            label=label,
            min_rolling_ic=0.03,
            n_windows=5,
            window_size=200,
        )
        print(f"[GP Validate] Result: {len(accepted)} accepted, {len(rejected)} rejected")

    print(f"\n[GP Mining] Done. Found {len(results)} new factors.")
    GPFactorMiner.print_status(GP_REGISTRY_PATH)


if __name__ == "__main__":
    version_config = discover_versions()
    if not version_config:
        print("Error: No valid factor calculators found in core/alpha/")
        sys.exit(1)

    args = parse_args(version_config)
    version = resolve_version(args, version_config)
    CalcClass = version_config[version]

    # Handle registry-only commands (no data loading needed)
    if handle_registry_commands(args):
        sys.exit(0)

    print(f"[GP Mining] Version: {version.upper()}, Pop: {args.pop}, Gen: {args.gen}")

    # Load data and prepare tensors
    padded_raw, col_map, label, existing_tensors = prepare_data(version, CalcClass)

    # Run mining
    run_mining(args, padded_raw, col_map, label, existing_tensors)

    # Cleanup
    del padded_raw, label, existing_tensors
    gc.collect()
    torch.cuda.empty_cache()
