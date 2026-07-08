"""
LLM-driven Factor Mining - 独立脚本

使用 LLM（GLM 模型）驱动因子挖掘，替代传统 GP 随机搜索。

== 用法 ==
挖掘（迭代式，默认 5 轮，目标 10 个因子）:
  python gp_mining_llm.py -v15 --n-candidates 20
  python gp_mining_llm.py -v15 --api-key your_zhipuai_key
  python gp_mining_llm.py -v15 --model glm-5.2 --max-rounds 5 --target-count 10

验证（对 discovered 因子执行滚动 IC 门禁）:
  python gp_mining_llm.py -v15 --validate

== 流程 ==
1. 构建知识库（GP 注册表 + 迭代文档 + 准则 + 模型 docstring）
2. 加载数据 → 构建 GPU 张量
3. LLM 分析知识库 → 生成因子假设（每轮 n_candidates 条）
4. Co-STEER 转译 → Node 表达式树
5. GPU 评估 → Rank IC + ICIR + 换手率 + 去重 → 注册表更新
6. 将本轮结果反馈给 LLM，进入下一轮，直到达到 target_count 或 max_rounds
7. 可选 --validate：对 discovered 因子执行滚动 IC 门禁，通过者升级为 testing
"""
import sys
import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import argparse
import importlib
import gc
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import polars as pl

from vnpy.alpha import logger
from core.logger_writer import LoggerWriter
from core.alpha.llm_factor_miner import LLMFactorMiner
from core.alpha.factor_calculator import device, ts_delay, cs_rank
from core.alpha.mlp_signals import MLPSignals
from core.alpha.engine import AlphaEngine
from core.selector.selector import FundamentalSelector


def setup_logger(version: str):
    log_filename = f"log/llm_mining_{version}.log"
    if not hasattr(sys.stdout, 'file') or not isinstance(sys.stdout, LoggerWriter):
        try:
            file = open(log_filename, 'w', encoding='utf-8')
            sys.stdout = LoggerWriter(sys.stdout, file)
            sys.stderr = LoggerWriter(sys.stderr, file)
            logger.remove()
            fmt: str = "{time:YYYY-MM-DD HH:mm:ss} {message}"
            logger.add(sys.stdout, colorize=True, format=fmt)
        except Exception as e:
            print(f"Failed to setup logger redirection: {e}")


def cleanup_logger():
    try:
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
        if hasattr(sys.stderr, 'close'):
            sys.stderr.close()
    except Exception:
        pass


GP_REGISTRY_PATH = "core/alpha/gp_factors.json"


def discover_versions():
    config = {}
    alpha_dir = Path("core/alpha")
    if alpha_dir.exists():
        for file_path in alpha_dir.glob("v*_factor_calculator.py"):
            module_name = file_path.stem
            version_str = module_name.split("_")[0]
            class_name = f"{version_str.capitalize()}FactorCalculator"
            try:
                module = importlib.import_module(f"core.alpha.{module_name}")
                calc_class = getattr(module, class_name)
                config[version_str] = calc_class
            except Exception as e:
                print(f"Warning: Failed to load {class_name} from {module_name}: {e}")
    return config


def parse_args(version_config):
    parser = argparse.ArgumentParser(description="LLM-driven Factor Mining")

    parser.add_argument("-v", "--version", required=False,
                        help=f"Alpha version (e.g., {', '.join(version_config.keys())})")
    for ver in version_config:
        parser.add_argument(f"-{ver}", action="store_true", help=f"Use {ver.upper()} calculator")

    parser.add_argument("--n-candidates", type=int, default=20,
                        help="Number of factor hypotheses to request from LLM (default: 20)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="LLM API key (or set DASHSCOPE_API_KEY / OPENAI_API_KEY env var)")
    parser.add_argument("--base-url", type=str,
                        default="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                        help="LLM API base URL (default: 百炼 Token Plan OpenAI-compatible)")
    parser.add_argument("--model", type=str, default="glm-5.2",
                        help="LLM model name (e.g., glm-5.2, qwen-plus, qwen-max)")

    parser.add_argument("--min-ic", type=float, default=0.02, help="Min IC threshold")
    parser.add_argument("--min-icir", type=float, default=0.3, help="Min ICIR threshold")
    parser.add_argument("--max-corr", type=float, default=0.5, help="Max correlation with existing factors")
    parser.add_argument("--max-rounds", type=int, default=5, help="Max iteration rounds (default: 5)")
    parser.add_argument("--target-count", type=int, default=10, help="Target number of new factors (default: 10)")

    parser.add_argument("-s", "--skip", action="store_true", help="Skip data sync")
    parser.add_argument("--validate", action="store_true",
                        help="Validate discovered factors with rolling IC gate (promote to testing)")

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


def prepare_data(version, CalcClass):
    """加载数据并准备 GPU 张量（与 gp_mining.py 相同）"""
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

    del raw_data, raw_tensor, s_indices_t, t_indices_t, df_idx

    all_dates = sorted(data_df["datetime"].unique().to_list())
    num_days_pad = padded_raw.shape[1]
    aligned_dates = all_dates[-num_days_pad:]
    recent_start_idx = 0
    for i, d in enumerate(aligned_dates):
        if hasattr(d, 'year') and d.year >= 2026:
            recent_start_idx = i
            break
    print(f"[LLM Mining] Recent start idx: {recent_start_idx} "
          f"(date={aligned_dates[recent_start_idx] if recent_start_idx > 0 else 'N/A'})")

    del df_sorted

    C = padded_raw[:, :, col_map['close']]
    raw_ret_5 = ts_delay(C, -5) / C - 1
    label = cs_rank(raw_ret_5)
    del C, raw_ret_5

    existing_tensors = []
    features_computed = calculator.build_features(padded_raw, col_map)
    for fname, ftensor in features_computed.items():
        if fname != "label":
            existing_tensors.append(ftensor)
    del features_computed

    gc.collect()
    torch.cuda.empty_cache()

    return padded_raw, col_map, label, existing_tensors, recent_start_idx


def run_llm_mining(args, padded_raw, col_map, label, existing_tensors, recent_start_idx):
    """执行 LLM 驱动的因子挖掘"""
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: API key not set.")
        print("For 百炼 DashScope: export DASHSCOPE_API_KEY=your_key")
        print("Or use: --api-key your_key")
        sys.exit(1)

    miner = LLMFactorMiner(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        min_ic=args.min_ic,
        min_icir=args.min_icir,
        max_corr=args.max_corr,
        n_candidates=args.n_candidates,
    )

    results = miner.mine(
        padded_raw=padded_raw,
        col_map=col_map,
        label=label,
        existing_tensors=existing_tensors[:10],
        recent_start_idx=recent_start_idx,
        max_rounds=args.max_rounds,
        target_count=args.target_count,
    )

    print(f"\n[LLM Mining] Complete. {len(results)} factors discovered.")


def run_validation(padded_raw, col_map, label, recent_start_idx):
    """对 discovered 因子执行滚动 IC 门禁，通过者升级为 testing，失败者标记 rejected"""
    from core.alpha.gp_factor_miner import GPFactorMiner

    print("\n[LLM Validate] Running rolling IC validation on discovered factors...")
    accepted, rejected = GPFactorMiner.validate_discovered(
        path=GP_REGISTRY_PATH,
        padded_raw=padded_raw,
        col_map=col_map,
        label=label,
        min_rolling_ic=0.03,
        n_windows=5,
        window_size=200,
        recent_start_idx=recent_start_idx,
    )
    print(f"[LLM Validate] Result: {len(accepted)} accepted, {len(rejected)} rejected")
    if accepted:
        print(f"  Promoted to testing: {', '.join(accepted)}")
    if rejected:
        print(f"  Rejected: {', '.join(rejected)}")
    GPFactorMiner.print_status(GP_REGISTRY_PATH)


if __name__ == "__main__":
    version_config = discover_versions()
    if not version_config:
        print("Error: No valid factor calculators found in core/alpha/")
        sys.exit(1)

    args = parse_args(version_config)
    version = resolve_version(args, version_config)
    CalcClass = version_config[version]

    setup_logger(version)

    print(f"[LLM Mining] Version: {version.upper()}, Model: {args.model}, Candidates: {args.n_candidates}")

    padded_raw, col_map, label, existing_tensors, recent_start_idx = prepare_data(version, CalcClass)

    if args.validate:
        run_validation(padded_raw, col_map, label, recent_start_idx)
    else:
        run_llm_mining(args, padded_raw, col_map, label, existing_tensors, recent_start_idx)

    del padded_raw, label, existing_tensors
    gc.collect()
    torch.cuda.empty_cache()
    cleanup_logger()
