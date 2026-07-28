"""
计算 alpha 效能门控序列 — 信号滚动 IC → 每日允许持仓数

机制:
- 每日全截面 5 日 forward IC（信号排名质量）
- 滚动 10 日均值，滞后 5 日（只用已实现的 IC，防未来函数）
- 映射到持仓数: ic_roll >= 0.01 → 5; -0.03 <= ic_roll < 0.01 → 3; < -0.03 → 1
- 策略经 ic_gate_enabled=True 读取，在风控 dynamic_max 上再钳制

用法:
  /home/airst/Workspace/.venv/bin/python scripts/compute_ic_gate.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import polars as pl
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr
from vnpy.trader.database import get_database
from vnpy.trader.constant import Exchange, Interval

SIGNAL = "ashare_mlp_signal_v15"
OUT = "core/alpha_db/signal/ic_gate.parquet"


def main():
    sig = pl.read_parquet(f"core/alpha_db/signal/{SIGNAL}.parquet")
    dates = sorted(sig["datetime"].unique().to_list())

    db = get_database()
    price = {}
    for s in sig["vt_symbol"].unique().to_list():
        code, ex = s.split(".")
        bars = db.load_bar_data(code, Exchange(ex), Interval.DAILY, dates[0], dates[-1])
        if bars:
            price[s] = {b.datetime.strftime("%Y-%m-%d"): b.close_price for b in bars}

    # 每日 5 日 IC
    daily_ic = {}
    for idx, d in enumerate(dates):
        if idx + 5 >= len(dates):
            break
        dstr = d.strftime("%Y-%m-%d")
        d5 = dates[idx + 5].strftime("%Y-%m-%d")
        day = sig.filter(pl.col("datetime") == d)
        xs, ys = [], []
        for row in day.iter_rows(named=True):
            s = row["vt_symbol"]
            if s in price and dstr in price[s] and d5 in price[s]:
                xs.append(row["total_score"])
                ys.append(price[s][d5] / price[s][dstr] - 1)
        if len(xs) > 50:
            ic, _ = spearmanr(xs, ys)
            if not np.isnan(ic):
                daily_ic[d] = ic

    ic_dates = sorted(daily_ic.keys())
    ic_vals = np.array([daily_ic[d] for d in ic_dates])

    # 滚动 10 日均值 + 滞后 5 日（gate 在第 i 日用第 i-5 日及之前已实现的 IC）
    roll = {}
    for i in range(len(ic_dates)):
        # 可用的最新 IC 是 ic_dates[i-5]（其 forward 收益在第 i 日已实现）
        j = i - 5
        if j < 9:
            roll[ic_dates[i]] = 5  # 数据不足时不限（满仓）
            continue
        window = ic_vals[j - 9: j + 1]
        m = float(np.mean(window))
        if m >= 0.01:
            roll[ic_dates[i]] = 5
        elif m >= -0.03:
            roll[ic_dates[i]] = 3
        else:
            roll[ic_dates[i]] = 1

    out = pl.DataFrame({
        "datetime": list(roll.keys()),
        "ic_gate_holdings": list(roll.values()),
    })
    out.write_parquet(OUT)
    n = out["ic_gate_holdings"].to_numpy()
    print(f"gate saved: {OUT}, {len(out)} days")
    print(f"持仓分布: 5仓 {(n == 5).mean():.0%}, 3仓 {(n == 3).mean():.0%}, 1仓 {(n == 1).mean():.0%}")
    # 2026 年分布
    recent = out.filter(pl.col("datetime") >= datetime(2026, 1, 1))["ic_gate_holdings"].to_numpy()
    print(f"2026 分布: 5仓 {(recent == 5).mean():.0%}, 3仓 {(recent == 3).mean():.0%}, 1仓 {(recent == 1).mean():.0%}")


if __name__ == "__main__":
    main()
