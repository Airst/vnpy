"""模型信号 2026 逐月 IC 分析 — 验证 6 月 alpha 失手的 regime 机制"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import polars as pl
import numpy as np
import collections
from datetime import datetime
from scipy.stats import spearmanr
from vnpy.trader.database import get_database
from vnpy.trader.constant import Exchange, Interval

sig = pl.read_parquet("core/alpha_db/signal/ashare_mlp_signal_v15.parquet")
sig = sig.filter(pl.col("datetime") >= datetime(2026, 1, 1))
dates = sorted(sig["datetime"].unique().to_list())

db = get_database()
syms = sig["vt_symbol"].unique().to_list()
price = {}
for s in syms:
    code, ex = s.split(".")
    bars = db.load_bar_data(code, Exchange(ex), Interval.DAILY, datetime(2025, 12, 20), datetime(2026, 7, 31))
    if bars:
        price[s] = {b.datetime.strftime("%Y-%m-%d"): b.close_price for b in bars}

ics = []
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
            ics.append((dstr, ic))

monthly = collections.defaultdict(list)
for dstr, ic in ics:
    monthly[dstr[:7]].append(ic)

print("=== 模型信号 5 日 IC（2026 月度）===")
for m in sorted(monthly):
    v = np.array(monthly[m])
    print(f"{m}: IC={v.mean():+.3f}  (正IC占比 {(v > 0).mean():.0%}, {len(v)}天)")

print()
print("=== 全时段参考（2022-2025 信号 IC 均值 ~+0.03~0.05 为健康）===")
