"""
股票池 Regime 分析

对比 CSI 300 / 500 / 1000 / 2000 四个池近半年表现，验证：
1. 是否小盘池被抛弃（收益远弱于大盘池）
2. 波动率、成交额、拥挤度是否显著变化
3. 是否存在明确的 regime 切换点

用法:
    python core/tools/analyze_pool_regime.py
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from core.tools.n_limit_up_analysis import get_index_constituents

DAILY_DIR = Path("/home/airst/Workspace/vnpy/core/alpha_db/daily")

POOLS = {
    "沪深300":    "000300.SH",
    "中证500":    "000905.SH",
    "中证1000":   "000852.SH",
    "中证2000":   "932000.CSI",
}


def load_pool_daily(index_code: str, start: datetime, end: datetime) -> pl.DataFrame:
    """加载股票池每日数据（等权聚合）"""
    print(f"  Loading constituents for {index_code}...")
    constituents = get_index_constituents(index_code)
    files = [DAILY_DIR / f"{s}.parquet" for s in constituents
             if (DAILY_DIR / f"{s}.parquet").exists()]
    print(f"    got {len(constituents)} constituents, {len(files)} on disk")

    dfs = []
    for f in files:
        try:
            d = pl.read_parquet(f, columns=["datetime", "close", "volume", "turnover"])
            d = d.with_columns(pl.lit(f.stem).alias("vt_symbol"))
            d = d.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))
            if len(d) > 0:
                dfs.append(d)
        except Exception as e:
            continue

    if not dfs:
        return pl.DataFrame()

    price = pl.concat(dfs)
    # per-stock daily return
    price = price.sort(["vt_symbol", "datetime"]).with_columns([
        (pl.col("close") / pl.col("close").shift(1).over("vt_symbol") - 1).alias("ret"),
    ])
    return price


def compute_pool_metrics(price: pl.DataFrame) -> pl.DataFrame:
    """按日聚合成池指标（等权）"""
    daily = (
        price.group_by("datetime")
        .agg([
            pl.col("ret").mean().alias("pool_ret"),
            pl.col("ret").std().alias("cross_std"),   # 池内截面波动
            pl.col("turnover").sum().alias("pool_turnover"),  # 总成交额
            pl.col("vt_symbol").n_unique().alias("n_stocks"),
        ])
        .sort("datetime")
    )
    return daily


def summarize_period(daily: pl.DataFrame, s: str, e: str, label: str):
    sub = daily.filter(
        (pl.col("datetime") >= datetime.fromisoformat(s))
        & (pl.col("datetime") <= datetime.fromisoformat(e))
    )
    if len(sub) == 0:
        return None

    rets = sub["pool_ret"].drop_nulls().to_numpy()
    if len(rets) == 0:
        return None

    total_ret = float(np.prod(1 + rets) - 1)
    ann_ret = float((1 + total_ret) ** (252 / len(rets)) - 1)
    daily_std = float(np.std(rets))
    ann_vol = daily_std * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = compute_max_dd(rets)

    turnover_avg = float(sub["pool_turnover"].drop_nulls().mean() or 0)  # 元
    cross_std = float(sub["cross_std"].drop_nulls().mean() or 0)

    return {
        "label": label,
        "days": len(rets),
        "total_ret": total_ret,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "turnover_avg_yi": turnover_avg / 1e8,  # 亿元
        "cross_std": cross_std,
    }


def compute_max_dd(rets: np.ndarray) -> float:
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min())


def main():
    end_dt = datetime(2026, 7, 1)
    start_dt = datetime(2022, 1, 1)

    pools_daily = {}
    for name, code in POOLS.items():
        print(f"\n[{name}] {code}")
        price = load_pool_daily(code, start_dt, end_dt)
        if len(price) == 0:
            print(f"    (no data)")
            continue
        daily = compute_pool_metrics(price)
        pools_daily[name] = daily
        print(f"    daily rows={len(daily)}, dates={daily['datetime'].min()} → {daily['datetime'].max()}")

    periods = [
        ("2022 全年",           "2022-01-01", "2022-12-31"),
        ("2023 全年",           "2023-01-01", "2023-12-31"),
        ("2024 Q1Q2",           "2024-01-01", "2024-06-30"),
        ("2024 Q3Q4 (924 转折)", "2024-07-01", "2024-12-31"),
        ("2025 Q1Q2 (爆发期)",   "2025-01-01", "2025-06-30"),
        ("2025 Q3",             "2025-07-01", "2025-09-30"),
        ("2025 Q4",             "2025-10-01", "2025-12-31"),
        ("2026 Q1",             "2026-01-01", "2026-03-31"),
        ("2026 Q2 (近期弱势)",   "2026-04-01", "2026-07-01"),
    ]

    for period_label, s, e in periods:
        print(f"\n{'=' * 92}")
        print(f"[{period_label}]")
        print(f"{'=' * 92}")
        header = f"{'Pool':<10}{'Days':>6}{'TotRet':>10}{'AnnRet':>10}{'AnnVol':>10}{'Sharpe':>8}{'MaxDD':>10}{'成交额亿':>12}{'截面σ':>10}"
        print(header)
        print("-" * 92)
        for name, daily in pools_daily.items():
            m = summarize_period(daily, s, e, name)
            if m is None:
                continue
            line = f"{name:<10}{m['days']:>6}"
            line += f"{m['total_ret']:>+10.2%}"
            line += f"{m['ann_ret']:>+10.2%}"
            line += f"{m['ann_vol']:>10.2%}"
            line += f"{m['sharpe']:>+8.2f}"
            line += f"{m['max_dd']:>+10.2%}"
            line += f"{m['turnover_avg_yi']:>12.1f}"
            line += f"{m['cross_std']:>10.4f}"
            print(line)

    print(f"\n{'=' * 92}")
    print("[Relative Strength] 小盘 vs 大盘：中证2000 - 沪深300 累计收益")
    print("=" * 92)
    if "沪深300" in pools_daily and "中证2000" in pools_daily:
        d300 = pools_daily["沪深300"].select(["datetime", "pool_ret"]).rename({"pool_ret": "ret_300"})
        d2000 = pools_daily["中证2000"].select(["datetime", "pool_ret"]).rename({"pool_ret": "ret_2000"})
        merged = d300.join(d2000, on="datetime", how="inner").sort("datetime")
        merged = merged.with_columns([
            (pl.col("ret_2000") - pl.col("ret_300")).alias("rs_2000_vs_300")
        ])
        # 月度累积 relative strength
        merged = merged.with_columns([
            pl.col("datetime").dt.strftime("%Y-%m").alias("month")
        ])
        monthly_rs = (
            merged.group_by("month")
            .agg([
                pl.col("rs_2000_vs_300").sum().alias("rs_month_sum"),
                pl.col("ret_300").sum().alias("ret_300_month"),
                pl.col("ret_2000").sum().alias("ret_2000_month"),
            ])
            .sort("month")
        )
        cum_rs = 0
        print(f"{'Month':<10}{'CSI300':>10}{'CSI2000':>10}{'RS(小-大)':>12}{'CumRS':>10}")
        print("-" * 52)
        for row in monthly_rs.iter_rows(named=True):
            cum_rs += row["rs_month_sum"]
            print(f"{row['month']:<10}{row['ret_300_month']:>+10.2%}{row['ret_2000_month']:>+10.2%}{row['rs_month_sum']:>+12.2%}{cum_rs:>+10.2%}")


if __name__ == "__main__":
    main()
