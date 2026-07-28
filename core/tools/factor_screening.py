"""
因子快速筛选工具 — 全因子相关性矩阵 + IC + 共线聚类

无需训练，快速识别冗余因子。复用 check_gp_collinearity.py 的数据加载模式，
但覆盖全部 V15 因子（不限 GP）。

用法: python core/tools/factor_screening.py --index 399303.SZ
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

import argparse
import torch
import numpy as np
import polars as pl
from datetime import datetime
from collections import defaultdict

from core.alpha.factor_calculator import device, cs_rank
from core.alpha.engine import AlphaEngine
from core.alpha.mlp_signals import MLPSignals
from core.alpha.v15_factor_calculator import V15FactorCalculator
from core.selector.selector import FundamentalSelector


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    xr = cs_rank(x)
    yr = cs_rank(y)
    mask = ~(torch.isnan(xr) | torch.isnan(yr))
    if mask.sum() < 100:
        return float('nan')
    xr_v = xr[mask]
    yr_v = yr[mask]
    xr_v = xr_v - xr_v.mean()
    yr_v = yr_v - yr_v.mean()
    denom = (xr_v.std() * yr_v.std())
    if denom < 1e-12:
        return float('nan')
    return float((xr_v * yr_v).mean() / denom)


def daily_rank_ic(factor: torch.Tensor, label: torch.Tensor) -> float:
    """Mean daily Spearman IC: average of per-day cross-sectional rank correlations."""
    num_stocks, num_days = factor.shape
    ics = []
    for t in range(num_days):
        f_t = factor[:, t]
        l_t = label[:, t]
        mask = ~(torch.isnan(f_t) | torch.isnan(l_t))
        if mask.sum() < 30:
            continue
        f_r = cs_rank(f_t.unsqueeze(1)).squeeze(1)
        l_r = cs_rank(l_t.unsqueeze(1)).squeeze(1)
        f_v = f_r[mask] - f_r[mask].mean()
        l_v = l_r[mask] - l_r[mask].mean()
        denom = f_v.std() * l_v.std()
        if denom < 1e-12:
            continue
        ics.append(float((f_v * l_v).mean() / denom))
    return np.mean(ics) if ics else float('nan')


def greedy_pairwise_removal(names, corr_matrix, ic_dict, threshold=0.5):
    """For each highly correlated pair (|corr| > threshold), remove the one with lower |IC|.

    Unlike union-find clustering, this does NOT use transitivity.
    Each removal decision is based on actual pairwise correlation.
    """
    n = len(names)
    keep = set(range(n))
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            c = corr_matrix[i][j]
            if abs(c) >= threshold:
                pairs.append((abs(c), i, j))
    pairs.sort(reverse=True)

    removals = []
    for _, i, j in pairs:
        if i in keep and j in keep:
            ic_i = abs(ic_dict[names[i]]) if not np.isnan(ic_dict[names[i]]) else 0
            ic_j = abs(ic_dict[names[j]]) if not np.isnan(ic_dict[names[j]]) else 0
            if ic_i >= ic_j:
                remove_idx, keep_idx = j, i
            else:
                remove_idx, keep_idx = i, j
            keep.remove(remove_idx)
            removals.append((
                names[remove_idx],
                names[keep_idx],
                corr_matrix[i][j],
                ic_dict[names[remove_idx]],
                ic_dict[names[keep_idx]],
            ))

    return keep, removals


def load_factor_tensors(index_filter=None):
    """Load V15 factors as {name: tensor(num_stocks, num_days)}."""
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
        index_filter=index_filter,
    )
    data_df = engine.load_data()

    df_sorted = data_df.sort(["vt_symbol", "datetime"])
    exclude_cols = {"datetime", "vt_symbol", "industry"}
    cols = [c for c in df_sorted.columns if c not in exclude_cols]
    col_map = {name: i for i, name in enumerate(cols)}

    raw_data = df_sorted.select(cols).to_numpy().astype(np.float32)
    symbols = df_sorted["vt_symbol"].to_numpy()
    unique_symbols, inverse_indices, counts = np.unique(symbols, return_inverse=True, return_counts=True)
    num_stocks = len(unique_symbols)
    max_len = counts.max()

    padded_raw = torch.full((num_stocks, max_len, len(cols)), float('nan'), device=device, dtype=torch.float32)
    df_idx = df_sorted.select(["vt_symbol"]).with_columns([
        pl.int_range(0, pl.len()).over("vt_symbol").alias("t_idx")
    ])
    t_indices = df_idx["t_idx"].to_numpy()
    s_indices = inverse_indices

    s_indices_t = torch.tensor(s_indices, dtype=torch.long, device=device)
    t_indices_t = torch.tensor(t_indices, dtype=torch.long, device=device)
    raw_tensor = torch.tensor(raw_data, device=device, dtype=torch.float32)
    padded_raw[s_indices_t, t_indices_t, :] = raw_tensor

    print("[FactorScreening] Computing features...")
    features = calculator.build_features(padded_raw, col_map)

    label = features.pop("label", None)
    if label is None:
        raise ValueError("Label not found in features")

    return features, label


def main():
    parser = argparse.ArgumentParser(description="Factor Screening Tool")
    parser.add_argument("--index", default="399303.SZ", help="Index filter")
    parser.add_argument("--corr-threshold", type=float, default=0.5, help="Correlation threshold for clustering")
    parser.add_argument("--ic-threshold", type=float, default=0.01, help="IC threshold for noise flagging")
    args = parser.parse_args()

    print(f"[FactorScreening] Loading data (index={args.index})...", flush=True)
    features, label = load_factor_tensors(index_filter=args.index)

    factor_names = sorted(features.keys())
    n = len(factor_names)
    print(f"[FactorScreening] {n} factors loaded. Computing {n*(n-1)//2} correlations...", flush=True)

    # Compute IC for each factor
    print("[FactorScreening] Computing IC...", flush=True)
    ic_dict = {}
    for name in factor_names:
        ic = daily_rank_ic(features[name], label)
        ic_dict[name] = ic

    # Compute correlation matrix
    print("[FactorScreening] Computing correlation matrix...", flush=True)
    corr_matrix = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = spearman_corr(features[factor_names[i]], features[factor_names[j]])
            corr_matrix[i][j] = c
            corr_matrix[j][i] = c
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n}...", flush=True)

    # Greedy pairwise removal
    keep_indices, removals = greedy_pairwise_removal(factor_names, corr_matrix, ic_dict, args.corr_threshold)

    removal_set = set()
    print("\n" + "=" * 80)
    print(f"因子筛选报告 (corr_threshold={args.corr_threshold}, ic_threshold={args.ic_threshold})")
    print("=" * 80)
    print(f"总因子数: {n}")

    if removals:
        print(f"\n共线移除 (|corr| > {args.corr_threshold}, 保留 |IC| 更高者):")
        for remove_name, keep_name, corr, remove_ic, keep_ic in removals:
            print(f"  移除: {remove_name:30s} (|IC|={abs(remove_ic):.4f}) "
                  f"← 保留: {keep_name:30s} (|IC|={abs(keep_ic):.4f}, corr={corr:+.3f})")
            removal_set.add(remove_name)

    # Low IC factors (not already in removal set)
    low_ic = [(name, ic) for name, ic in ic_dict.items()
              if abs(ic) < args.ic_threshold and name not in removal_set and not np.isnan(ic)]
    if low_ic:
        print(f"\n低 IC 因子 (|IC| < {args.ic_threshold}):")
        for name, ic in sorted(low_ic, key=lambda x: abs(x[1])):
            print(f"  {name}: |IC|={abs(ic):.4f}")
            removal_set.add(name)

    # NaN IC factors
    nan_ic = [name for name, ic in ic_dict.items() if np.isnan(ic)]
    if nan_ic:
        print(f"\nNaN IC 因子 (可能全 NaN 或常量):")
        for name in nan_ic:
            print(f"  {name}")
            removal_set.add(name)

    # Summary
    keep_count = n - len(removal_set)
    print(f"\n{'=' * 80}")
    print(f"汇总: 建议移除 {len(removal_set)} 个因子, 保留 {keep_count} 个")

    if removal_set:
        removal_list = sorted(removal_set)
        print(f"\n=== 复制以下代码到 v15_factor_calculator.py 末尾(return features之前) ===")
        print(f'for _f in {removal_list!r}:')
        print(f'    features.pop(_f, None)')

    return removal_set


if __name__ == "__main__":
    main()
