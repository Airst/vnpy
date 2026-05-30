"""
信号分布诊断脚本 — 量化"横线问题"严重程度

分析维度：
1. 每只股票信号时序 std（低 std = 横线股）
2. 每日截面 std（低 dispersion = 全体扁平）
3. Top-K 稳定度（连续两日 Top5 重叠率）
4. total_score 原始分数分布 vs rank-norm 后的分布
"""
import polars as pl
import numpy as np
from pathlib import Path

SIGNAL_PATH = Path(__file__).parent.parent / "alpha_db" / "signal" / "ashare_mlp_signal_v15.parquet"


def main():
    df = pl.read_parquet(SIGNAL_PATH)
    print(f"Signal file: {SIGNAL_PATH.name}")
    print(f"Records: {len(df):,}, Stocks: {df['vt_symbol'].n_unique()}, Days: {df['datetime'].n_unique()}")
    print()

    # === 1. Per-stock signal std (time-series) ===
    stock_stats = df.group_by("vt_symbol").agg([
        pl.col("final_signal").std().alias("signal_std"),
        pl.col("final_signal").mean().alias("signal_mean"),
        pl.col("total_score").std().alias("raw_std"),
        pl.col("final_signal").count().alias("n_days"),
    ]).filter(pl.col("n_days") >= 20)  # 至少20天数据

    stds = stock_stats["signal_std"].to_numpy()
    raw_stds = stock_stats["raw_std"].to_numpy()
    
    print("=" * 60)
    print("1. Per-Stock Signal Std (时序维度 — 横线检测)")
    print("=" * 60)
    print(f"  Stocks analyzed: {len(stds)}")
    print(f"  final_signal std: mean={np.nanmean(stds):.4f}, median={np.nanmedian(stds):.4f}")
    print(f"  total_score std:  mean={np.nanmean(raw_stds):.6f}, median={np.nanmedian(raw_stds):.6f}")
    
    # 横线定义：std < 0.3 (在 [-3,3] 范围内，std < 0.3 意味着信号基本不动)
    thresholds = [0.2, 0.3, 0.5, 0.8]
    print(f"\n  横线股票占比 (final_signal std < threshold):")
    for t in thresholds:
        pct = np.mean(stds < t) * 100
        print(f"    std < {t}: {pct:.1f}%")
    
    print(f"\n  Percentiles: P10={np.nanpercentile(stds, 10):.3f}, P25={np.nanpercentile(stds, 25):.3f}, "
          f"P50={np.nanpercentile(stds, 50):.3f}, P75={np.nanpercentile(stds, 75):.3f}, P90={np.nanpercentile(stds, 90):.3f}")

    # === 2. Daily cross-sectional std ===
    daily_stats = df.group_by("datetime").agg([
        pl.col("final_signal").std().alias("cs_std"),
        pl.col("total_score").std().alias("cs_raw_std"),
        pl.col("final_signal").max().alias("cs_max"),
        pl.col("final_signal").min().alias("cs_min"),
        pl.col("total_score").max().alias("raw_max"),
        pl.col("total_score").min().alias("raw_min"),
        pl.col("vt_symbol").count().alias("n_stocks"),
    ]).sort("datetime")

    cs_stds = daily_stats["cs_std"].to_numpy()
    cs_raw_stds = daily_stats["cs_raw_std"].to_numpy()

    print()
    print("=" * 60)
    print("2. Daily Cross-Section Std (截面维度 — dispersion)")
    print("=" * 60)
    print(f"  Days analyzed: {len(cs_stds)}")
    print(f"  final_signal CS-std: mean={np.nanmean(cs_stds):.4f}, min={np.nanmin(cs_stds):.4f}, max={np.nanmax(cs_stds):.4f}")
    print(f"  total_score CS-std:  mean={np.nanmean(cs_raw_stds):.6f}, min={np.nanmin(cs_raw_stds):.6f}, max={np.nanmax(cs_raw_stds):.6f}")
    print(f"  final_signal range:  [{daily_stats['cs_min'].mean():.2f}, {daily_stats['cs_max'].mean():.2f}]")
    print(f"  total_score range:   [{daily_stats['raw_min'].mean():.6f}, {daily_stats['raw_max'].mean():.6f}]")
    
    # === 3. Top-K stability ===
    print()
    print("=" * 60)
    print("3. Top-K Stability (连续两日选股重叠率)")
    print("=" * 60)
    
    dates = df["datetime"].unique().sort().to_list()
    
    for k in [5, 10, 20]:
        overlaps = []
        for i in range(1, len(dates)):
            prev_day = df.filter(pl.col("datetime") == dates[i-1]).sort("final_signal", descending=True).head(k)["vt_symbol"].to_list()
            curr_day = df.filter(pl.col("datetime") == dates[i]).sort("final_signal", descending=True).head(k)["vt_symbol"].to_list()
            overlap = len(set(prev_day) & set(curr_day))
            overlaps.append(overlap / k)
        
        overlaps = np.array(overlaps)
        print(f"  Top-{k} overlap: mean={overlaps.mean():.1%}, median={np.median(overlaps):.1%}, "
              f"P10={np.percentile(overlaps, 10):.1%}, P90={np.percentile(overlaps, 90):.1%}")

    # === 4. Raw score concentration ===
    print()
    print("=" * 60)
    print("4. Total Score Distribution (原始模型输出)")
    print("=" * 60)
    
    raw_scores = df["total_score"].to_numpy()
    print(f"  mean={np.nanmean(raw_scores):.6f}, std={np.nanstd(raw_scores):.6f}")
    print(f"  min={np.nanmin(raw_scores):.6f}, max={np.nanmax(raw_scores):.6f}")
    print(f"  Percentiles: P1={np.nanpercentile(raw_scores, 1):.6f}, P5={np.nanpercentile(raw_scores, 5):.6f}, "
          f"P50={np.nanpercentile(raw_scores, 50):.6f}, P95={np.nanpercentile(raw_scores, 95):.6f}, P99={np.nanpercentile(raw_scores, 99):.6f}")
    
    # Top/Bottom 分数间距
    print(f"\n  Top-Bottom score gap per day:")
    gap_stats = daily_stats.with_columns([
        (pl.col("raw_max") - pl.col("raw_min")).alias("raw_gap")
    ])
    gaps = gap_stats["raw_gap"].to_numpy()
    print(f"    mean={np.nanmean(gaps):.6f}, min={np.nanmin(gaps):.6f}, max={np.nanmax(gaps):.6f}")

    # === 5. Signal flip analysis (rank instability) ===
    print()
    print("=" * 60)
    print("5. Signal Flip (排名剧变检测)")
    print("=" * 60)
    
    # 计算每只股票相邻两天 rank 变化
    rank_df = df.with_columns([
        pl.col("final_signal").rank(method="average", descending=True).over("datetime").alias("daily_rank")
    ]).sort(["vt_symbol", "datetime"])
    
    rank_df = rank_df.with_columns([
        pl.col("daily_rank").diff().over("vt_symbol").alias("rank_change")
    ])
    
    rank_changes = rank_df["rank_change"].drop_nulls().to_numpy()
    abs_changes = np.abs(rank_changes)
    
    n_stocks_per_day = daily_stats["n_stocks"].mean()
    print(f"  Average stocks/day: {n_stocks_per_day:.0f}")
    print(f"  Daily rank change: mean={np.mean(abs_changes):.1f}, median={np.median(abs_changes):.1f}")
    print(f"  P90={np.percentile(abs_changes, 90):.0f}, P95={np.percentile(abs_changes, 95):.0f}, P99={np.percentile(abs_changes, 99):.0f}")
    
    # 从 Top-20 掉出 Top-50 的频率
    big_drop = np.mean(abs_changes > n_stocks_per_day * 0.1) * 100  # rank jump > 10% of universe
    print(f"  Rank jump > 10% of universe: {big_drop:.1f}% of all observations")


if __name__ == "__main__":
    main()
