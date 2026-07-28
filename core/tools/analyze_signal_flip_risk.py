"""
信号决策边界翻转概率分析

诊断训练方差导致回测大幅波动的原因：量化两个决策边界的翻转风险
1. 入场边界: rank 5 vs rank 6/7/8 的 final_signal 间隔（top-5 选股竞争）
2. 退出边界: 持仓股（前一日 top-5）final_signal 距 sell_threshold=1.54 的距离

用法:
    python core/tools/analyze_signal_flip_risk.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl
import numpy as np
from datetime import datetime

SIGNAL_PATH = "core/alpha_db/signal/ashare_mlp_signal_v15.parquet"
BUY_THRESHOLD = 1.0
SELL_THRESHOLD = 1.54
RECOVERY_THRESHOLD = 1.5  # buy_threshold + 0.5
TOP_N = 5
NOISE_LEVELS = [0.01, 0.02, 0.05]  # 模拟训练噪声幅度


def compute_entry_flip_risk(sig: pl.DataFrame) -> pl.DataFrame:
    """每天 rank 5 vs rank 6/7/8 的分数间隔"""
    top8 = (
        sig.sort(["datetime", "final_signal"], descending=[False, True])
        .group_by("datetime", maintain_order=True)
        .head(8)
        .with_columns(
            pl.col("final_signal").rank("ordinal", descending=True)
            .over("datetime").alias("rank")
        )
    )
    pivot = top8.pivot(
        values="final_signal",
        index="datetime",
        on="rank",
        aggregate_function="first",
    ).sort("datetime")
    # pivot columns are integers 1..8, rename to s1..s8
    rename_map = {c: f"s{c}" for c in pivot.columns if isinstance(c, int) or (isinstance(c, str) and c.isdigit())}
    pivot = pivot.rename(rename_map)
    pivot = pivot.with_columns([
        (pl.col("s5") - pl.col("s6")).alias("gap_5_6"),
        (pl.col("s5") - pl.col("s7")).alias("gap_5_7"),
        (pl.col("s5") - pl.col("s8")).alias("gap_5_8"),
    ])
    return pivot


def compute_exit_flip_risk(sig: pl.DataFrame) -> pl.DataFrame:
    """前一日 top-5 持仓在当日的 final_signal 距 sell_threshold 的距离"""
    top5 = (
        sig.sort(["datetime", "final_signal"], descending=[False, True])
        .group_by("datetime", maintain_order=True)
        .head(TOP_N)
        .select(["datetime", "vt_symbol"])
        .with_columns(pl.col("datetime").alias("hold_date"))
    )
    # 持仓日 = hold_date, 评估日 = hold_date + 1 交易日
    top5 = top5.with_columns(
        pl.col("datetime").shift(-1).over("vt_symbol").alias("next_date")
    ).filter(pl.col("next_date").is_not_null())

    # join next day's signal
    next_sig = sig.select(["datetime", "vt_symbol", "final_signal"]).rename({
        "datetime": "next_date",
        "final_signal": "next_signal",
    })
    held = top5.join(next_sig, on=["vt_symbol", "next_date"], how="left")
    held = held.with_columns(
        (pl.col("next_signal") - SELL_THRESHOLD).alias("dist_to_sell"),
        (pl.col("next_signal") - RECOVERY_THRESHOLD).alias("dist_to_recovery"),
    )
    return held


def summarize_entry_risk(pivot: pl.DataFrame):
    print("\n" + "=" * 78)
    print("[1] 入场边界翻转风险: rank 5 vs rank 6/7/8 间隔")
    print(f"    (策略选 top-5, rank 6+ 落选. 间隔越小 → 训练噪声越易翻转选股)")
    print("=" * 78)

    for gap_col, label in [("gap_5_6", "rank5→6"), ("gap_5_7", "rank5→7"), ("gap_5_8", "rank5→8")]:
        vals = pivot[gap_col].drop_nulls()
        print(f"\n  {label} 间隔 (final_signal units):")
        print(f"    mean={vals.mean():.4f}, median={vals.median():.4f}, "
              f"p10={vals.quantile(0.1):.4f}, p90={vals.quantile(0.9):.4f}")
        for noise in NOISE_LEVELS:
            n_flip = (vals < noise).sum()
            pct = 100.0 * n_flip / len(vals)
            print(f"    噪声 ±{noise:.2f}: {n_flip:>4}/{len(vals)} 天 ({pct:.1f}%) 会翻转选股")

    # 时间序列: 每月高风险天比例
    print("\n  月度高风险天比例 (gap_5_6 < 0.02):")
    monthly = pivot.with_columns(
        pl.col("datetime").dt.strftime("%Y-%m").alias("month"),
        (pl.col("gap_5_6") < 0.02).alias("flip_risk"),
    ).group_by("month").agg([
        pl.col("flip_risk").mean().alias("flip_rate"),
        pl.col("gap_5_6").mean().alias("avg_gap"),
        pl.len().alias("days"),
    ]).sort("month")
    for row in monthly.iter_rows(named=True):
        flag = " !!" if row["flip_rate"] > 0.3 else ""
        print(f"    {row['month']}: flip_rate={100*row['flip_rate']:.1f}%, "
              f"avg_gap={row['avg_gap']:.4f}, days={row['days']}{flag}")


def summarize_exit_risk(held: pl.DataFrame):
    print("\n" + "=" * 78)
    print("[2] 退出边界翻转风险: 持仓股 final_signal 距 sell_threshold=1.54")
    print(f"    (距阈值越近 → 训练噪声越易翻转 hold/sell 决策)")
    print("=" * 78)

    dist = held["dist_to_sell"].drop_nulls()
    print(f"\n  持仓股距 sell 阈值距离 (final_signal - 1.54):")
    print(f"    mean={dist.mean():.4f}, median={dist.median():.4f}, "
          f"p10={dist.quantile(0.1):.4f}, p90={dist.quantile(0.9):.4f}")

    for noise in NOISE_LEVELS:
        # 持仓股 current signal 在 [1.54-noise, 1.54+noise] 区间 → 噪声会翻转
        n_flip = ((dist > -noise) & (dist < noise)).sum()
        pct = 100.0 * n_flip / len(dist)
        print(f"    噪声 ±{noise:.2f}: {n_flip:>5}/{len(dist)} 持仓天 ({pct:.1f}%) 会翻转 hold/sell")

    # 距 recovery 阈值 1.5 的距离 (cancel pending sell)
    dist_rec = held["dist_to_recovery"].drop_nulls()
    print(f"\n  持仓股距 recovery 阈值距离 (final_signal - 1.50):")
    print(f"    mean={dist_rec.mean():.4f}, median={dist_rec.median():.4f}")
    for noise in NOISE_LEVELS:
        n_flip = ((dist_rec > -noise) & (dist_rec < noise)).sum()
        pct = 100.0 * n_flip / len(dist_rec)
        print(f"    噪声 ±{noise:.2f}: {n_flip:>5}/{len(dist_rec)} 持仓天 ({pct:.1f}%) 会翻转 cancel/pending")


def compare_backtest_overlap():
    """对比 4 次训练的回测结果, 看同一天的持仓重合度"""
    import json
    import glob
    import os
    files = sorted(glob.glob("core/alpha_db/backtest/*v15*.json"),
                   key=lambda f: os.path.getmtime(f), reverse=True)[:4]
    if len(files) < 2:
        print("\n[3] 回测持仓对比: 不足 2 个回测文件, 跳过")
        return

    print("\n" + "=" * 78)
    print("[3] 多次训练回测的持仓重合度 (同一天不同训练选了几只相同股票)")
    print("=" * 78)

    runs = []
    for f in files:
        d = json.load(open(f))
        trades = d.get("trades", d.get("records", []))
        # 提取每日持仓
        daily_holds = {}
        for t in trades:
            dt = t.get("datetime", t.get("date", ""))
            sym = t.get("vt_symbol", "")
            if dt and sym:
                daily_holds.setdefault(dt, set()).add(sym)
        ts = os.path.basename(f).split("_")[-1].replace(".json", "")
        runs.append((ts, daily_holds))
        print(f"  {ts}: {len(daily_holds)} 天有交易记录")

    # 找共同日期
    common = None
    for _, dh in runs:
        if common is None:
            common = set(dh.keys())
        else:
            common &= set(dh.keys())
    print(f"  共同日期: {len(common)} 天")
    if len(common) < 5:
        print("  共同日期不足, 跳过重合度分析")
        return

    overlaps = []
    for dt in sorted(common)[:50]:
        sets = [dh[dt] for _, dh in runs]
        if all(len(s) > 0 for s in sets):
            inter = set.intersection(*sets)
            union = set.union(*sets)
            overlaps.append(len(inter) / len(union) if len(union) > 0 else 1.0)
    if overlaps:
        print(f"\n  持仓 Jaccard 重合度 (前50天均值): {np.mean(overlaps):.3f}")
        print(f"  (0=完全不同, 1=完全相同)")


def main():
    print("Loading signals...")
    sig = pl.read_parquet(SIGNAL_PATH)
    print(f"  shape={sig.shape}, range={sig['datetime'].min()} → {sig['datetime'].max()}")

    pivot = compute_entry_flip_risk(sig)
    held = compute_exit_flip_risk(sig)

    summarize_entry_risk(pivot)
    summarize_exit_risk(held)
    compare_backtest_overlap()

    print("\n" + "=" * 78)
    print("[结论]")
    print("=" * 78)
    gap_median = pivot["gap_5_6"].median()
    exit_close = ((held["dist_to_sell"].abs() < 0.05) & held["dist_to_sell"].is_not_null()).sum()
    print(f"  入场: rank5→6 中位间隔 {gap_median:.4f}")
    print(f"  退出: {exit_close} 持仓天距 sell 阈值 < 0.05")
    print(f"  → 训练噪声 > {gap_median:.3f} 即可翻转 top-5 选股")


if __name__ == "__main__":
    main()
