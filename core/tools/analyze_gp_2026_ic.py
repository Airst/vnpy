"""
GP 因子 2026 IC 诊断

分析 16 个 validated GP 因子在 2026 Q1/Q2 的 IC 表现，
识别 IC 反转的因子（历史正 IC → 2026 负 IC）。

用法:
    python core/tools/analyze_gp_2026_ic.py
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
from core.alpha.gp_factor_miner import GPFactorMiner
from core.selector.selector import FundamentalSelector


def daily_rank_ic(factor: torch.Tensor, label: torch.Tensor) -> np.ndarray:
    fr = cs_rank(factor)
    lr = cs_rank(label)
    mask = ~(torch.isnan(fr) | torch.isnan(lr))
    ics = np.full(factor.shape[1], np.nan)
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
        ics[d] = (f_v * l_v).mean().item() / denom.item()
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

    # Date alignment
    all_dates = sorted(df_sorted["datetime"].unique().to_list())
    num_days_pad = padded_raw.shape[1]
    aligned_dates = list(all_dates[-num_days_pad:])
    print(f"  num_stocks={num_stocks}, num_days={num_days_pad}, "
          f"dates={aligned_dates[0]} → {aligned_dates[-1]}")

    # Build features
    print("[Compute] Building V15 features (including GP factors)...")
    features = calculator.build_features(padded_raw, col_map)
    label = features["label"]

    # Identify GP factors
    gp_factor_names = sorted([n for n in features.keys() if n.startswith("gp_")])
    non_gp_names = sorted([n for n in features.keys() if n != "label" and not n.startswith("gp_")])
    print(f"  GP factors: {len(gp_factor_names)}, Non-GP factors: {len(non_gp_names)}")

    # Period definitions
    periods = {
        "2022-2024":      ("2022-01-01", "2024-12-31"),
        "2025-Q1Q2":      ("2025-01-01", "2025-06-30"),
        "2025-Q3Q4":      ("2025-07-01", "2025-12-31"),
        "2026-Q1":        ("2026-01-01", "2026-03-31"),
        "2026-Q2":        ("2026-04-01", "2026-07-01"),
    }
    aligned_dates_ts = np.array([d.timestamp() if d is not None else 0 for d in aligned_dates])
    period_masks = {}
    for plabel, (s, e) in periods.items():
        s_ts = datetime.fromisoformat(s).timestamp()
        e_ts = datetime.fromisoformat(e).timestamp() + 86400
        mask = (aligned_dates_ts >= s_ts) & (aligned_dates_ts < e_ts)
        period_masks[plabel] = mask

    # Analyze GP factors
    print(f"\n[Compute] Computing IC for {len(gp_factor_names)} GP factors...")
    gp_results = {}
    for fname in gp_factor_names:
        ftensor = features[fname]
        if ftensor is None or ftensor.shape != label.shape:
            continue
        ics = daily_rank_ic(ftensor, label)
        row = {}
        for plabel, pmask in period_masks.items():
            valid = pmask & ~np.isnan(ics)
            row[plabel] = float(np.nanmean(ics[valid])) if valid.sum() >= 5 else float('nan')
        row["FULL"] = float(np.nanmean(ics[~np.isnan(ics)]))
        gp_results[fname] = row

    # Analyze top non-GP factors for comparison
    print(f"[Compute] Computing IC for {len(non_gp_names)} non-GP factors...")
    non_gp_results = {}
    for fname in non_gp_names:
        ftensor = features[fname]
        if ftensor is None or ftensor.shape != label.shape:
            continue
        ics = daily_rank_ic(ftensor, label)
        row = {}
        for plabel, pmask in period_masks.items():
            valid = pmask & ~np.isnan(ics)
            row[plabel] = float(np.nanmean(ics[valid])) if valid.sum() >= 5 else float('nan')
        row["FULL"] = float(np.nanmean(ics[~np.isnan(ics)]))
        non_gp_results[fname] = row

    # Print GP factor table
    period_cols = list(periods.keys()) + ["FULL"]
    print("\n" + "=" * 120)
    print("=== GP 因子 IC 分析 ===")
    print("=" * 120)
    print(f"{'Factor':<20}" + "".join(f"{p:>14}" for p in period_cols))
    print("-" * 120)
    for fname in sorted(gp_results.keys(), key=lambda k: abs(gp_results[k].get("FULL", 0) or 0), reverse=True):
        row = gp_results[fname]
        line = f"{fname:<20}"
        for p in period_cols:
            v = row.get(p)
            if v is None or np.isnan(v):
                line += f"{'-':>14}"
            else:
                line += f"{v:>+14.4f}"
        print(line)

    # IC reversal analysis
    print("\n" + "=" * 120)
    print("=== IC 反转分析（历史 vs 2026-Q2）===")
    print("=" * 120)
    print(f"{'Factor':<20}{'历史IC':>14}{'2026-Q1':>14}{'2026-Q2':>14}{'反转?':>10}{'状态':>10}")
    print("-" * 120)

    all_results = {**gp_results, **non_gp_results}
    reversal_factors = []
    for fname in sorted(all_results.keys(), key=lambda k: abs(all_results[k].get("FULL", 0) or 0), reverse=True):
        row = all_results[fname]
        hist = row.get("2022-2024")
        q1 = row.get("2026-Q1")
        q2 = row.get("2026-Q2")
        if hist is None or np.isnan(hist) or q2 is None or np.isnan(q2):
            continue
        reversed_flag = "YES" if (abs(hist) > 0.01 and np.sign(hist) != np.sign(q2) and abs(q2) > 0.01) else ""
        is_gp = "GP" if fname.startswith("gp_") else ""
        if reversed_flag:
            reversal_factors.append(fname)
        line = f"{fname:<20}{hist:>+14.4f}{q1:>+14.4f}{q2:>+14.4f}{reversed_flag:>10}{is_gp:>10}"
        print(line)

    # Summary
    print(f"\n=== 汇总 ===")
    print(f"IC 反转因子数: {len(reversal_factors)}")
    if reversal_factors:
        print(f"反转因子: {', '.join(reversal_factors)}")

    # GP factor 2026 effectiveness
    gp_effective = []
    gp_reversed = []
    for fname, row in gp_results.items():
        q2 = row.get("2026-Q2")
        hist = row.get("2022-2024")
        if q2 is None or np.isnan(q2):
            continue
        if hist is not None and not np.isnan(hist) and abs(hist) > 0.01:
            if np.sign(hist) != np.sign(q2) and abs(q2) > 0.01:
                gp_reversed.append(fname)
            elif abs(q2) > 0.01:
                gp_effective.append(fname)
    print(f"\nGP 因子在 2026-Q2 仍有效: {len(gp_effective)} ({', '.join(gp_effective) if gp_effective else '无'})")
    print(f"GP 因子在 2026-Q2 IC反转: {len(gp_reversed)} ({', '.join(gp_reversed) if gp_reversed else '无'})")


if __name__ == "__main__":
    main()
