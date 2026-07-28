"""
模型稳定性测试 — 多 Seed 全量训练对比

用不同随机种子跑多次训练，对比回测指标的方差，评估模型是否稳定。

使用方法:
  # 快速模式 (最近5窗口, ~15分钟/seed)
  python scripts/stability_test.py --max-windows 5 --index 000852.SH,399303.SZ

  # 完整模式 (~1小时/seed)
  python scripts/stability_test.py --index 000852.SH,399303.SZ
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import json
import argparse
import gc
import torch
from datetime import datetime
from pathlib import Path

from vnpy.alpha import logger
from core.alpha.engine import AlphaEngine
from core.alpha.mlp_signals import MLPSignals
from core.selector.selector import FundamentalSelector
from core.core_service import CoreService


def run_single_seed(seed: int, args, CalcClass, selector, last_trading_date) -> dict:
    """执行单个 seed 的完整训练+回测 pipeline"""
    print(f"\n{'='*60}")
    print(f"  SEED = {seed}")
    print(f"{'='*60}")

    calculator = CalcClass(gp_status_filter=None)
    signal_name = f"stability_test_s{seed}"

    engine = AlphaEngine(
        factor_calculator=calculator,
        mlp_signals=MLPSignals(
            signal_name=signal_name,
            force_retrain=True,
            max_windows=args.max_windows,
            model_backend="attention",
            retrain_days=45,
            ensemble_size=1,
            seed=seed,
        ),
        selector=selector,
        signal_name=signal_name,
        start_date="2019-12-28",
        end_date=last_trading_date.strftime("%Y-%m-%d"),
        index_filter=args.index,
    )

    # Load data
    data_df = engine.load_data()
    signal_df = engine.calculate_factors(data_df)
    signal_df, _ = engine.analyze_factor_performance(signal_df)

    # Train & generate signals
    signal_df = engine.calculate_signals(signal_df)
    engine.save_signals(signal_df)

    # Backtest
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

    # Cleanup
    del engine, calculator, data_df, signal_df, core_service
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Model Stability Test (Multi-Seed)")
    parser.add_argument("--seeds", type=str, default="42,123,2024",
                        help="Comma-separated seeds to test (default: 42,123,2024)")
    parser.add_argument("--max-windows", type=int, default=5,
                        help="Quick mode: last N windows only (0=full, default 5)")
    parser.add_argument("--index", type=str, default="000852.SH,399303.SZ",
                        help="Index filter")
    parser.add_argument("--version", type=str, default="v15",
                        help="Factor calculator version (default: v15)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print(f"=== Model Stability Test ===")
    print(f"Seeds: {seeds}")
    print(f"Max windows: {args.max_windows} ({'FULL' if args.max_windows == 0 else f'last {args.max_windows}'})")
    print(f"Index: {args.index}")
    print(f"Version: {args.version}")
    print()

    # Load factor calculator class
    from training import resolve_version_config
    version_config = resolve_version_config()
    version = args.version.lower()
    if not version.startswith("v"):
        version = "v" + version
    if version not in version_config:
        print(f"ERROR: Version '{version}' not found. Available: {list(version_config.keys())}")
        return
    CalcClass, _ = version_config[version]

    # Setup selector once (shared across seeds)
    selector = FundamentalSelector()
    last_trading_date = selector.get_last_trading_day() or datetime.now()

    # Run all seeds
    results = {}
    for seed in seeds:
        try:
            result = run_single_seed(seed, args, CalcClass, selector, last_trading_date)
            if result and "statistics" in result:
                results[seed] = result["statistics"]
            elif result:
                results[seed] = result
            else:
                print(f"  WARNING: Seed {seed} returned no results")
        except Exception as e:
            print(f"  ERROR: Seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()

    # ---- Report ----
    if not results:
        print("\nERROR: No successful runs!")
        return

    print(f"\n\n{'='*70}")
    print(f"  STABILITY REPORT ({len(results)} seeds)")
    print(f"{'='*70}")

    # Key metrics to compare
    metrics = ["sharpe_ratio", "annual_return", "total_return", "max_ddpercent",
               "max_drawdown_duration", "return_drawdown_ratio"]
    labels = ["Sharpe Ratio", "Annual Return (%)", "Total Return (%)", "Max DD (%)",
              "Max DD Duration (days)", "Return/DD Ratio"]

    # Print table header
    seed_headers = [f"Seed {s}" for s in results.keys()]
    print(f"\n{'Metric':<25} {'|'.join(f'{h:>14}' for h in seed_headers)}  |{'Mean':>10} {'Std':>10}")
    print("-" * (25 + 15 * len(results) + 25))

    import numpy as np

    for metric, label in zip(metrics, labels):
        values = []
        for seed, stats in results.items():
            val = stats.get(metric, 0)
            values.append(val)

        arr = np.array(values)
        mean = arr.mean()
        std = arr.std()

        val_strs = [f"{v:>14.2f}" for v in values]
        print(f"{label:<25} {'|'.join(val_strs)}  |{mean:>10.2f} {std:>10.2f}")

    # Stability verdict
    print(f"\n{'='*70}")
    sharpe_values = [results[s].get("sharpe_ratio", 0) for s in results]
    sharpe_arr = np.array(sharpe_values)
    sharpe_mean = sharpe_arr.mean()
    sharpe_std = sharpe_arr.std()
    sharpe_cv = sharpe_std / (abs(sharpe_mean) + 1e-8)

    print(f"  Sharpe: mean={sharpe_mean:.3f}, std={sharpe_std:.3f}, CV={sharpe_cv:.2%}")

    if sharpe_cv < 0.05:
        print(f"  ★ HIGHLY STABLE — Sharpe CV < 5%")
    elif sharpe_cv < 0.10:
        print(f"  ● STABLE — Sharpe CV < 10%")
    elif sharpe_cv < 0.20:
        print(f"  ○ MODERATE variance — Sharpe CV < 20%")
    else:
        print(f"  ✗ UNSTABLE — Sharpe CV >= 20%, model is sensitive to randomness")
    print(f"{'='*70}")

    # Save results
    output_path = f"log/stability_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": {"seeds": seeds, "max_windows": args.max_windows, "index": args.index, "version": version},
            "results": {str(k): v for k, v in results.items()},
            "summary": {"sharpe_mean": sharpe_mean, "sharpe_std": sharpe_std, "sharpe_cv": sharpe_cv},
        }, f, indent=2, default=str)
    print(f"\n  Full results saved to {output_path}")


if __name__ == "__main__":
    main()
