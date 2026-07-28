"""
Strategy parameter sweep on existing signals (no retraining needed).
Tests previously-kept Tier-1 strategy params at full 35-window scale.
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

from core.core_service import CoreService
from datetime import datetime
import json

core = CoreService()
start = datetime(2022, 1, 1)
end = datetime(2026, 7, 14)
signal = "ashare_mlp_signal_v15"

configs = [
    ("baseline (mh=5, sl=5%, st=1.54)", {"signal_name": signal, "max_holdings": "5"}),
    ("max_holdings=3", {"signal_name": signal, "max_holdings": "3"}),
    ("max_holdings=4", {"signal_name": signal, "max_holdings": "4"}),
    ("mh=3 + sl=3%", {"signal_name": signal, "max_holdings": "3", "stop_loss_pct": "0.03"}),
    ("mh=3 + st=2.0", {"signal_name": signal, "max_holdings": "3", "sell_threshold": "2.0"}),
    ("mh=3 + sl=3% + st=2.0", {"signal_name": signal, "max_holdings": "3", "stop_loss_pct": "0.03", "sell_threshold": "2.0"}),
    ("mh=4 + sl=3% + st=2.0", {"signal_name": signal, "max_holdings": "4", "stop_loss_pct": "0.03", "sell_threshold": "2.0"}),
    ("mh=3 + sl=4%", {"signal_name": signal, "max_holdings": "3", "stop_loss_pct": "0.04"}),
]

print(f"Strategy Parameter Sweep ({len(configs)} configs)")
print(f"Signal: {signal}, Period: {start.date()} to {end.date()}")
print(f"{'='*80}")
print(f"{'Config':<35} {'RDD':>8} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8}")
print(f"{'-'*80}")

results = []
for name, setting in configs:
    try:
        r = core.run_backtest("MultiFactorStrategy", start, end, setting=setting)
        s = r["statistics"]
        rdd = s.get("return_drawdown_ratio", 0)
        sharpe = s.get("sharpe_ratio", 0)
        ret = s.get("total_return", 0)
        mdd = s.get("max_ddpercent", 0)
        print(f"{name:<35} {rdd:>8.3f} {sharpe:>8.3f} {ret:>9.1f}% {mdd:>7.1f}%")
        results.append({"name": name, "rdd": rdd, "sharpe": sharpe, "return": ret, "mdd": mdd})
    except Exception as e:
        print(f"{name:<35} ERROR: {e}")

print(f"{'='*80}")
if results:
    best = max(results, key=lambda x: x["rdd"])
    print(f"\nBest: {best['name']} (RDD={best['rdd']:.3f}, Sharpe={best['sharpe']:.3f})")
    baseline_rdd = results[0]["rdd"] if results else 0
    print(f"Improvement over baseline: {best['rdd'] - baseline_rdd:+.3f}")
