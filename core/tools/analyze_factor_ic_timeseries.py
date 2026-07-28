"""
V15 单因子 IC 时间序列分析

针对近期因子失效问题：
1. 加载 V15 全部因子（含新增 GP 因子）
2. 用面积标签（10日）作为 target
3. 计算每个因子的日 IC，做 30 日滚动均值
4. 输出各时段（2024/2025-Q1Q2/2025-Q3Q4/2026）的因子 IC 变化对比

用法:
    python core/tools/analyze_factor_ic_timeseries.py
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl
import numpy as np
import torch
from datetime import datetime

from core.alpha.factor_calculator import device, cs_rank, ts_delay
from core.alpha.engine import AlphaEngine
from core.alpha.mlp_signals import MLPSignals
from core.alpha.v15_factor_calculator import V15FactorCalculator
from core.selector.selector import FundamentalSelector


def daily_rank_ic(factor: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Compute daily Spearman IC.
    factor / label: (num_stocks, num_days) tensors.
    returns: (num_days,) IC series.
    """
    fr = cs_rank(factor)
    lr = cs_rank(label)
    mask = ~(torch.isnan(fr) | torch.isnan(lr))
    ics = torch.full((factor.shape[1],), float('nan'), device=factor.device)
    for d in range(factor.shape[1]):
        m = mask[:, d]
        if m.sum() < 30:
            continue
        f_v = fr[m, d]
        l_v = lr[m, d]
        f_v = f_v - f_v.mean()
        l_v = l_v - l_v.mean()
        denom = f_v.std() * l_v.std()
        if denom < 1e-12:
            continue
        ics[d] = (f_v * l_v).mean() / denom
    return ics


def main():
    print("[Setup] Loading data...")
    selector = FundamentalSelector()
    last_trading_date = selector.get_last_trading_day() or datetime.now()

    calculator = V15FactorCalculator()
    engine = AlphaEngine(
        factor_calculator=calculator,
        mlp_signals=MLPSignals(signal_name="ashare_mlp_signal_v15", force_retrain=False),
        selector=selector,
        signal_name="ashare_mlp_signal_v15",
        start_date="2019-12-28",
        end_date=last_trading_date.strftime("%Y-%m-%d"),
    )
    data_df = engine.load_data()
    _ = engine.calculate_factors(data_df)

    df_sorted = data_df.sort(["vt_symbol", "datetime"])
    exclude_cols = {"datetime", "vt_symbol", "industry"}
    cols = [c for c in df_sorted.columns if c not in exclude_cols]
    col_map = {name: i for i, name in enumerate(cols)}

    raw_data = df_sorted.select(cols).to_numpy().astype(np.float32)
    symbols = df_sorted["vt_symbol"].to_numpy()
    unique_symbols, inverse_indices, counts = np.unique(symbols, return_inverse=True, return_counts=True)
    num_stocks = len(unique_symbols)
    max_len = counts.max()

    padded_raw = torch.full((num_stocks, max_len, len(cols)), float('nan'),
                            device=device, dtype=torch.float32)
    df_idx = df_sorted.select(["vt_symbol"]).with_columns([
        pl.int_range(0, pl.len()).over("vt_symbol").alias("t_idx")
    ])
    t_indices = df_idx["t_idx"].to_numpy()
    s_indices = inverse_indices

    s_indices_t = torch.tensor(s_indices, dtype=torch.long, device=device)
    t_indices_t = torch.tensor(t_indices, dtype=torch.long, device=device)
    raw_tensor = torch.tensor(raw_data, device=device, dtype=torch.float32)
    padded_raw[s_indices_t, t_indices_t, :] = raw_tensor
    del raw_data, raw_tensor, s_indices_t, t_indices_t

    # Compute date-per-column mapping (使用第一支股票的 datetime 序列)
    print("[Setup] Building date index...")
    first_sym = unique_symbols[0]
    first_dates = (df_sorted.filter(pl.col("vt_symbol") == first_sym)
                   .select("datetime").to_series().to_list())
    # Pad to max_len (some stocks have longer histories; take union)
    all_dates = sorted(df_sorted["datetime"].unique().to_list())
    date_array = np.array(all_dates)
    # Assume padded_raw uses per-stock local time index — we need to align on global dates.
    # However, all stocks in this system share the same trading calendar boundary (holidays are shared);
    # the padding uses t_idx per symbol. Different listing dates → different t_idx=0.
    # Simplification: use the LATEST num_days from padded_raw and align to end date.
    num_days_pad = padded_raw.shape[1]
    # Take last N global trading dates matching num_days_pad
    if len(date_array) < num_days_pad:
        # Should not happen but guard
        aligned_dates = list(date_array) + [None] * (num_days_pad - len(date_array))
    else:
        aligned_dates = list(date_array[-num_days_pad:])
    print(f"      num_stocks={num_stocks}, num_days={num_days_pad}, "
          f"date range={aligned_dates[0]} → {aligned_dates[-1]}")

    print("[Compute] Building V15 features...")
    features = calculator.build_features(padded_raw, col_map)
    label = features["label"]  # 面积标签 (cs_rank of excess area 10d)

    factor_names = [n for n in features.keys() if n != "label"]
    print(f"      factor count = {len(factor_names)}")

    # Time period boundaries
    periods = {
        "2022":            ("2022-01-01", "2022-12-31"),
        "2023":            ("2023-01-01", "2023-12-31"),
        "2024-Q1Q2":       ("2024-01-01", "2024-06-30"),
        "2024-Q3Q4":       ("2024-07-01", "2024-12-31"),
        "2025-Q1Q2":       ("2025-01-01", "2025-06-30"),
        "2025-Q3":         ("2025-07-01", "2025-09-30"),
        "2025-Q4":         ("2025-10-01", "2025-12-31"),
        "2026-Q1":         ("2026-01-01", "2026-03-31"),
        "2026-Q2 (近期)":  ("2026-04-01", "2026-07-01"),
    }
    dates_np = np.array([pd.timestamp() if hasattr(pd, "timestamp") else pd for pd in aligned_dates])
    aligned_dates_ts = np.array([d.timestamp() if d is not None else 0 for d in aligned_dates])

    period_masks = {}
    for label_str, (s, e) in periods.items():
        s_ts = datetime.fromisoformat(s).timestamp()
        e_ts = datetime.fromisoformat(e).timestamp() + 86400
        mask = (aligned_dates_ts >= s_ts) & (aligned_dates_ts < e_ts)
        period_masks[label_str] = mask
        # print(f"  {label_str}: {mask.sum()} days")

    print("\n[Compute] Computing per-factor IC series...")
    results = {}  # factor_name -> {period_label: mean_ic}
    for fname in factor_names:
        ftensor = features[fname]
        if ftensor is None or ftensor.shape != label.shape:
            continue
        ics = daily_rank_ic(ftensor, label).cpu().numpy()  # (num_days,)
        row = {}
        for plabel, pmask in period_masks.items():
            valid = pmask & ~np.isnan(ics)
            if valid.sum() < 5:
                row[plabel] = float('nan')
            else:
                row[plabel] = float(np.nanmean(ics[valid]))
        row["FULL"] = float(np.nanmean(ics[~np.isnan(ics)]))
        results[fname] = row

    # Print result table
    period_cols = list(periods.keys()) + ["FULL"]
    print("\n" + "=" * 140)
    print(f"{'Factor':<40}" + "".join(f"{p:>13}" for p in period_cols))
    print("=" * 140)

    # Sort by FULL IC abs value
    sorted_factors = sorted(results.items(),
                            key=lambda kv: abs(kv[1].get("FULL", 0) or 0),
                            reverse=True)
    for fname, row in sorted_factors:
        line = f"{fname:<40}"
        for p in period_cols:
            v = row.get(p)
            if v is None or np.isnan(v):
                line += f"{'-':>13}"
            else:
                line += f"{v:>+13.4f}"
        print(line)

    print("\n" + "=" * 140)
    print("[Analysis] 因子 IC 衰减对比：近期 vs 历史")
    print("=" * 140)
    print(f"{'Factor':<40}{'历史(22-24)':>14}{'2025-Q1Q2':>14}{'2025-Q3':>14}{'2025-Q4':>14}{'2026-Q1':>14}{'2026-Q2':>14}{'衰减比':>14}")
    print("-" * 140)

    for fname, row in sorted_factors:
        hist_periods = ["2022", "2023", "2024-Q1Q2", "2024-Q3Q4"]
        hist_vals = [row.get(p) for p in hist_periods if row.get(p) is not None and not np.isnan(row.get(p) or float('nan'))]
        if not hist_vals:
            continue
        hist_mean = np.mean(hist_vals)
        recent = row.get("2026-Q2 (近期)")
        if recent is None or np.isnan(recent):
            continue
        decay = (abs(recent) / abs(hist_mean)) if abs(hist_mean) > 0.001 else float('nan')
        line = f"{fname:<40}"
        line += f"{hist_mean:>+14.4f}"
        for p in ["2025-Q1Q2", "2025-Q3", "2025-Q4", "2026-Q1", "2026-Q2 (近期)"]:
            v = row.get(p)
            if v is None or np.isnan(v):
                line += f"{'-':>14}"
            else:
                line += f"{v:>+14.4f}"
        if not np.isnan(decay):
            line += f"{decay:>14.2f}"
        else:
            line += f"{'-':>14}"
        print(line)


if __name__ == "__main__":
    main()
