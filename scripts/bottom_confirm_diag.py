"""
探底企稳状态 × 买入胜率 条件诊断（只读分析，不训练）

问题: V15 的抄底由超卖深度因子驱动，无企稳确认特征。诊断买入时点的企稳状态
是否区分后续收益——若区分，企稳过滤作为策略执行层规则有价值；若不区分，
说明抄底时机非问题所在，方向关闭。

方法:
- 从生产回测 JSON 取 1222 笔成交，按 symbol 时序配对 买(多)→卖(空) 得 round-trip 收益
- 企稳状态用买入日 T 的前一日 T-1 及更早的 bar 判定（决策时点可见信息，无前视）:
  ① vol_dry:    V[T-1] < mean(V[T-6..T-2])          缩量
  ② higher_low: L[T-1] > min(L[T-6..T-2])           低点抬高
  ③ clv_strong: mean(CLV[T-3..T-1]) > 0.5, CLV=(C-L)/(H-L)  收盘位置偏强（针的连续化）
- 按单条件/组合评分(0-3)分组: 笔数、胜率、均值、中位数；再按 2026 前后分段

用法: /home/airst/Workspace/.venv/bin/python scripts/bottom_confirm_diag.py
"""
import sys, os, json, glob
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")
import numpy as np
import polars as pl

BT = sorted(glob.glob("core/alpha_db/backtest/ashare_mlp_signal_v15_2022*.json"))[-1]
print(f"backtest: {BT}")
d = json.load(open(BT))
trades = d["trades"]

# --- 配对 round-trip ---
from collections import defaultdict, deque
opens = defaultdict(deque)
rts = []  # (symbol, buy_date, buy_price, sell_date, sell_price)
for t in trades:
    sym, dt, px = t["symbol"], t["date"][:10], t["price"]
    if t["direction"] == "多":
        opens[sym].append((dt, px))
    else:
        if opens[sym]:
            bd, bp = opens[sym].popleft()
            rts.append((sym, bd, bp, dt, px))
print(f"round-trips: {len(rts)}, 未平仓: {sum(len(v) for v in opens.values())}")

# --- 逐笔计算企稳状态 ---
bars_cache = {}
def get_bars(sym):
    if sym not in bars_cache:
        p = f"core/alpha_db/daily/{sym}.parquet"
        bars_cache[sym] = pl.read_parquet(p) if os.path.exists(p) else None
    return bars_cache[sym]

rows = []
skipped = 0
for sym, bd, bp, sd, sp in rts:
    bars = get_bars(sym)
    if bars is None:
        skipped += 1
        continue
    dts = bars["datetime"].dt.strftime("%Y-%m-%d").to_list()
    try:
        i = dts.index(bd)  # 买入日 T
    except ValueError:
        skipped += 1
        continue
    if i < 7:
        skipped += 1
        continue
    V = bars["volume"].to_numpy(); L = bars["low"].to_numpy()
    H = bars["high"].to_numpy(); C = bars["close"].to_numpy()
    # T-1 = i-1, 参照窗 T-6..T-2 = i-6..i-2
    vol_dry = V[i-1] < np.mean(V[i-6:i-1])
    higher_low = L[i-1] > np.min(L[i-6:i-1])
    rng = H[i-3:i] - L[i-3:i]
    clv = np.where(rng > 0, (C[i-3:i] - L[i-3:i]) / np.where(rng > 0, rng, 1.0), 0.5)
    clv_strong = float(np.mean(clv)) > 0.5
    ret = sp / bp - 1
    rows.append({"symbol": sym, "buy_date": bd, "ret": ret,
                 "vol_dry": bool(vol_dry), "higher_low": bool(higher_low),
                 "clv_strong": bool(clv_strong),
                 "score": int(vol_dry) + int(higher_low) + int(clv_strong)})
print(f"有效样本: {len(rows)}, 跳过: {skipped}\n")
df = pl.DataFrame(rows)

def stat(sub, name):
    if len(sub) == 0:
        print(f"  {name:<28s} n=0")
        return
    r = sub["ret"]
    print(f"  {name:<28s} n={len(sub):4d}  胜率={100*(r>0).mean():5.1f}%  "
          f"均值={100*r.mean():+6.2f}%  中位={100*r.median():+6.2f}%")

def report(df, title):
    print(f"=== {title} (n={len(df)}) ===")
    stat(df, "全部买入")
    for c in ["vol_dry", "higher_low", "clv_strong"]:
        stat(df.filter(pl.col(c)), f"{c}=True")
        stat(df.filter(~pl.col(c)), f"{c}=False")
    for s in range(4):
        stat(df.filter(pl.col("score") == s), f"企稳评分={s}")
    stat(df.filter(pl.col("score") >= 2), "评分>=2 (已企稳)")
    stat(df.filter(pl.col("score") <= 1), "评分<=1 (未企稳)")
    print()

report(df, "全时段 2022-2026")
report(df.filter(pl.col("buy_date") < "2026-01-01"), "2026 之前")
report(df.filter(pl.col("buy_date") >= "2026-01-01"), "2026 年（逆风期）")
report(df.filter(pl.col("buy_date") >= "2026-04-01"), "2026-04 以来（alpha 反转期）")
