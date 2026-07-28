"""
持仓广度扫描 — 同一信号、不同 max_holdings 的纯净 A/B 测试

方法学说明:
- 信号固定（seed=42 生产信号 ashare_mlp_signal_v15，全时段 2022-2026）
  → N 之间的差异完全来自组合构建层，无信号噪声混入
- 另对 3 个 stability seed 信号（仅覆盖最近 ~5 窗口）做同样扫描
  → 检验广度结论在近期弱 regime 下是否跨 seed 稳健
- 每个配置保存完整回测 JSON + 年度分解，供事后分析

用法:
  /home/airst/Workspace/.venv/bin/python scripts/breadth_sweep.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
import numpy as np
from datetime import datetime
from core.core_service import CoreService


def yearly_breakdown(daily_data):
    """从 daily_data 计算逐年 return / sharpe"""
    dates = [x["date"] for x in daily_data]
    bal = np.array([x["balance"] for x in daily_data])
    if len(bal) < 2:
        return {}
    ret = np.diff(bal) / bal[:-1]
    years = {}
    for i, dt in enumerate(dates[1:]):
        years.setdefault(dt[:4], []).append(ret[i])
    out = {}
    for y, r in sorted(years.items()):
        r = np.array(r)
        cum = float(np.prod(1 + r) - 1)
        sharpe = float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else 0.0
        out[y] = {"return_pct": round(cum * 100, 2), "sharpe": round(sharpe, 2), "days": len(r)}
    return out


def run_one(core, signal, n, start, end):
    setting = {"signal_name": signal, "max_holdings": str(n)}
    r = core.run_backtest("MultiFactorStrategy", start, end, setting=setting)
    s = r["statistics"]
    return {
        "signal": signal,
        "max_holdings": n,
        "sharpe": s.get("sharpe_ratio", 0),
        "annual_return": s.get("annual_return", 0),
        "total_return": s.get("total_return", 0),
        "max_ddpercent": s.get("max_ddpercent", 0),
        "max_dd_duration": s.get("max_drawdown_duration", 0),
        "rdd": s.get("return_drawdown_ratio", 0),
        "trade_count": s.get("total_trade_count", 0),
        "commission": s.get("total_commission", 0),
        "slippage": s.get("total_slippage", 0),
        "yearly": yearly_breakdown(r.get("daily_data", [])),
    }


def main():
    core = CoreService()
    start = datetime(2022, 1, 1)
    end = datetime(2026, 7, 16)

    ns = [5, 10, 15, 20]
    # 全时段生产信号 + 近期 regime 的 3 个 seed 信号
    signals = [
        "ashare_mlp_signal_v15",
        "stability_test_s42",
        "stability_test_s123",
        "stability_test_s2024",
    ]

    all_results = []
    for signal in signals:
        print(f"\n{'='*90}")
        print(f"Signal: {signal}")
        print(f"{'N':>4} {'RDD':>8} {'Sharpe':>8} {'Annual':>9} {'Total':>9} {'MaxDD':>8} {'DDdur':>6} {'Trades':>7} {'Cost':>9}")
        print("-" * 90)
        for n in ns:
            try:
                res = run_one(core, signal, n, start, end)
                all_results.append(res)
                cost = res["commission"] + res["slippage"]
                print(f"{n:>4} {res['rdd']:>8.3f} {res['sharpe']:>8.3f} {res['annual_return']:>8.1f}% "
                      f"{res['total_return']:>8.1f}% {res['max_ddpercent']:>7.1f}% {res['max_dd_duration']:>6} "
                      f"{res['trade_count']:>7} {cost:>9.0f}")
            except Exception as e:
                print(f"{n:>4} ERROR: {e}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"log/breadth_sweep_{ts}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")

    # 汇总：全时段信号上 N 的单调性
    full = [r for r in all_results if r["signal"] == "ashare_mlp_signal_v15"]
    if len(full) == len(ns):
        print("\n=== 全时段广度单调性（生产信号）===")
        for r in full:
            y26 = r["yearly"].get("2026", {})
            print(f"N={r['max_holdings']:>2}: RDD={r['rdd']:.2f} Sharpe={r['sharpe']:.2f} "
                  f"MaxDD={r['max_ddpercent']:.1f}% | 2026: {y26.get('return_pct', '?')}% (sharpe {y26.get('sharpe', '?')})")


if __name__ == "__main__":
    main()
