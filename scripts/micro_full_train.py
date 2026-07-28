"""
含微观因子的全量生产训练（35 窗 × attention × swa × vl=100 × seed=42，双池）

- 基线对照: 当前生产 SWA 信号（双池 seed=42, N=5: Sharpe 1.16, RDD 3.27, total 228.7%）
- 候选: 143 因子 + 6 个 Tier-0 过门微观因子（tail30_ret, first60_vol_r, realized_vol_5m,
  vol_of_vol, kyle_lambda, ushape_vol），join 后 label 重排回最后一列（防泄漏）
- 微观因子覆盖 2019-07 起（回补后），早期缺口 null 填 0
- 输出: 候选全时段回测 + 逐年分解 vs 生产基线

用法:
  /home/airst/Workspace/.venv/bin/python scripts/micro_full_train.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
import numpy as np
import polars as pl
import collections
from datetime import datetime
from core.alpha.research_runner import load_session, _run_single_seed

KEEP = ["tail30_ret", "first60_vol_r", "realized_vol_5m", "vol_of_vol", "kyle_lambda", "ushape_vol"]


def main():
    print("load_session (双池) ...")
    session = load_session(index="000852.SH,399303.SZ", version="v15")
    base = session.factor_df
    print(f"基线 {len(base.columns)} 列")

    micro = pl.read_parquet("core/alpha_db/micro_factors.parquet")
    micro = micro.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
    micro = micro.select(["datetime", "vt_symbol"] + KEEP)

    cand = base.join(micro, on=["datetime", "vt_symbol"], how="left")
    cand = cand.with_columns([pl.col(k).fill_null(0.0) for k in KEEP])
    fac_cols = [c for c in cand.columns if c not in ("datetime", "vt_symbol", "label", "industry")]
    cand = cand.select(["datetime", "vt_symbol"] + fac_cols + ["label"])
    assert cand.columns[-1] == "label", "label 必须在最后一列"
    print(f"候选 {len(cand.columns)} 列 (143 + {len(KEEP)} micro)")

    print("=== 候选全量训练 (35 窗 × attention × swa × vl=100 × seed=42) ===")
    r = _run_single_seed(
        session, cand, 42, "micro_full",
        backend="attention", max_windows=0,
        end_date_filter=None,
    )
    s = r["stats"] if "stats" in r else r.get("statistics", {})
    print(f"\n=== 候选(143+6micro) 全时段 ===")
    print(f"Sharpe={r['sharpe']:.3f} RDD={r['score']:.3f}")
    print("对照 生产基线(SWA 无micro): Sharpe=1.16 RDD=3.27 total=228.7%")

    # 逐年分解
    daily = r.get("daily_data", [])
    if daily:
        dates = [x["date"] for x in daily]
        bal = np.array([x["balance"] for x in daily])
        ret = np.diff(bal) / bal[:-1]
        years = collections.defaultdict(list)
        for i, dt in enumerate(dates[1:]):
            years[dt[:4]].append(ret[i])
        print("\n逐年:")
        for y in sorted(years):
            rr = np.array(years[y])
            cum = np.prod(1 + rr) - 1
            sharpe = rr.mean() / rr.std() * np.sqrt(244) if rr.std() > 0 else 0
            print(f"  {y}: {cum*100:+7.1f}%  sharpe={sharpe:5.2f}")
    with open("log/micro_full_train.json", "w") as f:
        json.dump({"sharpe": r["sharpe"], "rdd": r["score"], "stats": s,
                   "daily_data": daily}, f, default=str)


if __name__ == "__main__":
    main()
