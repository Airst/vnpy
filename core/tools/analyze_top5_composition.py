"""
模型 Top-5 选股结构分析

诊断近期因子失效的问题：观察模型选股在 size / industry / amihud 维度上的演化，
对比近期表现差的时段（2026-04~ 至今）与历史类似亏损时段（2024 Q2 -16%）。

用法:
    python core/tools/analyze_top5_composition.py
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from collections import Counter

from data_manager.ts_downloader.daily_basic_manager import DailyBasicManager
from data_manager.ts_downloader.stock_info_manager import StockInfoManager

SIGNAL_PATH = "/home/airst/Workspace/vnpy/core/alpha_db/signal/ashare_mlp_signal_v15.parquet"
DAILY_DIR = Path("/home/airst/Workspace/vnpy/core/alpha_db/daily")
TOP_N = 5


def load_signals_and_meta():
    print("[1/4] Loading signals...")
    sig = pl.read_parquet(SIGNAL_PATH)
    print(f"      shape={sig.shape}, date range={sig['datetime'].min()} → {sig['datetime'].max()}")

    print("[2/4] Loading daily basic (total_mv, circ_mv, turnover_rate)...")
    all_symbols = [f.stem for f in DAILY_DIR.glob("*.parquet")]
    dmin = sig["datetime"].min()
    dmax = sig["datetime"].max()
    start = dmin.strftime("%Y%m%d")
    end = dmax.strftime("%Y%m%d")
    db_manager = DailyBasicManager()
    basic_pd = db_manager.load_data(all_symbols, start, end)
    basic = pl.from_pandas(basic_pd).with_columns(pl.col("datetime").cast(pl.Datetime("us")))
    basic = basic.select(["vt_symbol", "datetime", "total_mv", "circ_mv", "turnover_rate"])
    print(f"      shape={basic.shape}")

    print("[3/4] Loading industry info...")
    info_mgr = StockInfoManager()
    info_pd = info_mgr.load_data(all_symbols)
    info = pl.from_pandas(info_pd).select(["vt_symbol", "industry", "name"])
    print(f"      shape={info.shape}")

    return sig, basic, info


def top_n_per_day(sig: pl.DataFrame, n: int = TOP_N) -> pl.DataFrame:
    return (
        sig.sort(["datetime", "total_score"], descending=[False, True])
        .group_by("datetime", maintain_order=True)
        .head(n)
    )


def analyze_period(df: pl.DataFrame, label: str):
    print(f"\n{'=' * 78}")
    print(f"[{label}] N={len(df)} rows, {df['datetime'].n_unique()} days")
    print("=" * 78)

    # Size 分布
    total_mv_ok = df.filter(pl.col("total_mv").is_not_null())
    if len(total_mv_ok) > 0:
        mv_stats = total_mv_ok["total_mv"].describe()
        print(f"\n[Total MV 万元] "
              f"mean={total_mv_ok['total_mv'].mean():.0f}, "
              f"median={total_mv_ok['total_mv'].median():.0f}, "
              f"p10={total_mv_ok['total_mv'].quantile(0.1):.0f}, "
              f"p90={total_mv_ok['total_mv'].quantile(0.9):.0f}")
        ln_mv = np.log(total_mv_ok["total_mv"].to_numpy())
        print(f"[ln(MV)]        mean={ln_mv.mean():.3f}, std={ln_mv.std():.3f}")

    # 换手率
    tr_ok = df.filter(pl.col("turnover_rate").is_not_null())
    if len(tr_ok) > 0:
        print(f"[Turnover %]    mean={tr_ok['turnover_rate'].mean():.2f}, "
              f"median={tr_ok['turnover_rate'].median():.2f}, "
              f"p90={tr_ok['turnover_rate'].quantile(0.9):.2f}")

    # 行业分布 top 10
    if "industry" in df.columns:
        ind_ok = df.filter(pl.col("industry").is_not_null())
        if len(ind_ok) > 0:
            counter = Counter(ind_ok["industry"].to_list())
            total = sum(counter.values())
            print(f"\n[Top 10 行业] (占比%)")
            for ind, cnt in counter.most_common(10):
                print(f"    {ind}: {cnt} ({100 * cnt / total:.1f}%)")


def main():
    sig, basic, info = load_signals_and_meta()

    print("\n[4/4] Computing daily top-5 selection...")
    top5 = top_n_per_day(sig, TOP_N)
    top5 = top5.join(basic, on=["vt_symbol", "datetime"], how="left")
    top5 = top5.join(info, on="vt_symbol", how="left")
    print(f"      top5 rows={len(top5)}")

    # 时段划分
    periods = {
        "2022 全年":       ("2022-01-01", "2022-12-31"),
        "2023 全年":       ("2023-01-01", "2023-12-31"),
        "2024 Q2 (亏损期-16%)": ("2024-04-08", "2024-07-08"),
        "2024 Q4":          ("2024-10-01", "2024-12-31"),
        "2025 Q1-Q2 (爆发期)": ("2025-01-01", "2025-06-30"),
        "2025 Q3":          ("2025-07-01", "2025-09-30"),
        "2025 Q4":          ("2025-10-01", "2025-12-31"),
        "2026 Q1":          ("2026-01-01", "2026-03-31"),
        "2026 Q2 (近期)":   ("2026-04-01", "2026-07-01"),
    }

    for label, (s, e) in periods.items():
        sub = top5.filter((pl.col("datetime") >= datetime.fromisoformat(s))
                          & (pl.col("datetime") <= datetime.fromisoformat(e)))
        if len(sub) == 0:
            print(f"\n[{label}] (no data)")
            continue
        analyze_period(sub, label)

    print(f"\n{'=' * 78}")
    print("[Global] Size 分布随时间演化 (月度均值 ln(MV))")
    print("=" * 78)
    monthly = (
        top5.filter(pl.col("total_mv").is_not_null())
        .with_columns([
            pl.col("datetime").dt.strftime("%Y-%m").alias("month"),
            pl.col("total_mv").log().alias("ln_mv"),
        ])
        .group_by("month")
        .agg([
            pl.col("ln_mv").mean().alias("ln_mv_mean"),
            pl.col("ln_mv").std().alias("ln_mv_std"),
            pl.col("turnover_rate").mean().alias("turnover_mean"),
            pl.col("vt_symbol").n_unique().alias("unique_stocks"),
        ])
        .sort("month")
    )
    for row in monthly.iter_rows(named=True):
        print(f"  {row['month']}: ln_mv={row['ln_mv_mean']:.2f}±{row['ln_mv_std']:.2f}, "
              f"turnover={row['turnover_mean']:.2f}%, "
              f"unique={row['unique_stocks']}")


if __name__ == "__main__":
    main()
