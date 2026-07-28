"""
持有期策略配对回测 — 验证信号评测矩阵结论 (2026-07-28)

矩阵结论: alpha 在持有第 1~7 天持续兑现 (前 5 天 0.107%/天, 第 6~7 天
0.098%/天), 第 8~10 天钝化到 0.042%/天; Top-1~10 准确率持平, Top-3 超额略优。

对应策略改造 (multifactor_strategy.py 新参数, 默认关闭):
- min_hold_days: 最小持有期内屏蔽信号卖出 (止损/风控不受限)
- max_hold_days: 持满后不在当日 Top-K 强制换仓, 仍在 Top-K 重置计时

所有配置同信号同区间, 只动持有期参数, 与基线可直接配对比较。
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
    ("baseline (mh=5)",          {}),
    ("min_hold=3",               {"min_hold_days": "3"}),
    ("min_hold=5",               {"min_hold_days": "5"}),
    ("max_hold=7",               {"max_hold_days": "7"}),
    ("min=3 max=7",              {"min_hold_days": "3", "max_hold_days": "7"}),
    ("min=5 max=7",              {"min_hold_days": "5", "max_hold_days": "7"}),
    ("min=5 max=10",             {"min_hold_days": "5", "max_hold_days": "10"}),
    ("mh=3 min=5 max=7",         {"max_holdings": "3", "min_hold_days": "5", "max_hold_days": "7"}),
]

print(f"Holding Period Sweep ({len(configs)} configs)")
print(f"Signal: {signal}, Period: {start.date()} to {end.date()}")
print(f"{'='*88}")
print(f"{'Config':<24} {'RDD':>8} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'Trades':>8}")
print(f"{'-'*88}")

results = []
for name, extra in configs:
    setting = {"signal_name": signal, **extra}
    try:
        r = core.run_backtest("MultiFactorStrategy", start, end, setting=setting)
        s = r["statistics"]
        rdd = s.get("return_drawdown_ratio", 0)
        sharpe = s.get("sharpe_ratio", 0)
        ret = s.get("total_return", 0)
        mdd = s.get("max_ddpercent", 0)
        trades = s.get("total_trade_count", 0)
        print(f"{name:<24} {rdd:>8.3f} {sharpe:>8.3f} {ret:>9.1f}% {mdd:>7.1f}% {trades:>8}",
              flush=True)
        results.append({"name": name, "setting": setting, "rdd": rdd, "sharpe": sharpe,
                        "return": ret, "mdd": mdd, "trades": trades})
    except Exception as e:
        print(f"{name:<24} ERROR: {e}", flush=True)

print(f"{'='*88}")
if results:
    base = results[0]
    print(f"\n对比基线 (Sharpe {base['sharpe']:.3f}, RDD {base['rdd']:.3f}):")
    for r in results[1:]:
        print(f"  {r['name']:<24} dSharpe {r['sharpe']-base['sharpe']:+.3f}  "
              f"dRDD {r['rdd']-base['rdd']:+.3f}  dRet {r['return']-base['return']:+.1f}pp  "
              f"dTrades {r['trades']-base['trades']:+d}")
    with open("log/holding_period_sweep.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n结果已保存: log/holding_period_sweep.json")
