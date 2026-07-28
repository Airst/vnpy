"""
三方向回测对比 — 生产(SWA反转) / TimesFM(动量) / 组合(rank平均) @ 2026 YTD

验证假设: TimesFM 动量信号与生产反转信号镜像互补，rank 平均组合能抹平 regime。
回测窗口 2026 YTD（含 4 月反转 rally、6 月动量 rally、7 月崩盘 三种 regime），N=5。

用法:
  /home/airst/Workspace/.venv/bin/python scripts/timesfm_combo_backtest.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import numpy as np
import polars as pl
import collections
from datetime import datetime
from core.core_service import CoreService

PROD = "ashare_mlp_signal_v15"
FM = "timesfm_zeroshot"
COMBO = "timesfm_combo"
START = datetime(2026, 1, 1)
END = datetime(2026, 7, 17)


def build_combo():
    prod = pl.read_parquet(f"core/alpha_db/signal/{PROD}.parquet")
    fm = pl.read_parquet(f"core/alpha_db/signal/{FM}.parquet")
    # fm 的 datetime 是字符串，统一转为 Datetime 再 join
    if fm.schema["datetime"] == pl.String:
        fm = fm.with_columns(pl.col("datetime").str.to_datetime())
    if prod.schema["datetime"] == pl.String:
        prod = prod.with_columns(pl.col("datetime").str.to_datetime())
    j = prod.join(fm.select(["datetime", "vt_symbol", pl.col("total_score").alias("fm_score")]),
                  on=["datetime", "vt_symbol"], how="inner")
    # 逐日截面 rank 平均
    j = j.with_columns([
        pl.col("total_score").rank(method="average").over("datetime").alias("r1"),
        pl.col("fm_score").rank(method="average").over("datetime").alias("r2"),
    ])
    j = j.with_columns(((pl.col("r1") + pl.col("r2")) / 2).alias("total_score"))
    j = j.with_columns([
        pl.col("total_score").rank(method="average").over("datetime").alias("rank"),
        pl.col("total_score").count().over("datetime").alias("count"),
    ])
    j = j.with_columns([(((pl.col("rank") / pl.col("count")) - 0.5) * 3.46).clip(-3, 3).alias("final_signal")])
    out = j.select(["datetime", "vt_symbol", "total_score", "final_signal"])
    out.write_parquet(f"core/alpha_db/signal/{COMBO}.parquet")
    print(f"combo signal: {len(out)} rows, {out['datetime'].n_unique()} days")


def run(core, signal, label):
    r = core.run_backtest("MultiFactorStrategy", START, END,
                          setting={"signal_name": signal, "max_holdings": "5"})
    s = r["statistics"]
    daily = r["daily_data"]
    dates = [x["date"] for x in daily]
    bal = np.array([x["balance"] for x in daily])
    ret = np.diff(bal) / bal[:-1]
    months = collections.defaultdict(list)
    for i, dt in enumerate(dates[1:]):
        months[dt[:7]].append(ret[i])
    mline = " | ".join(f"{m}:{np.prod(1+np.array(v))-1:+.1%}" for m, v in sorted(months.items()))
    print(f"{label:<12} Sharpe={s['sharpe_ratio']:>5.2f} RDD={s['return_drawdown_ratio']:>5.2f} "
          f"total={s['total_return']:>6.1f}% MaxDD={s['max_ddpercent']:>6.1f}%  {mline}")
    return s


def main():
    build_combo()
    core = CoreService()
    print(f"\n=== 三方向回测 @ 2026 YTD, N=5 ===")
    run(core, PROD, "生产(SWA反转)")
    run(core, FM, "TimesFM(动量)")
    run(core, COMBO, "组合(rank平均)")


if __name__ == "__main__":
    main()
