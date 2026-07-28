"""
GP 因子共线性检查

计算新候选因子（testing 状态）之间、以及与已有 validated 因子之间的
Spearman 相关性，识别高度共线的因子对。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

import json
import torch
import numpy as np
import polars as pl
from datetime import datetime

from core.alpha.gp_factor_miner import GPFactorMiner, evaluate_tree, Node
from core.alpha.factor_calculator import device, cs_rank
from core.alpha.engine import AlphaEngine
from core.alpha.mlp_signals import MLPSignals
from core.alpha.v15_factor_calculator import V15FactorCalculator
from core.selector.selector import FundamentalSelector

GP_REGISTRY_PATH = "/home/airst/Workspace/vnpy/core/alpha/gp_factors.json"


def deserialize_tree(d: dict) -> Node:
    children = [deserialize_tree(c) for c in d.get('children', [])]
    return Node(op=d['op'], children=children, value=d.get('value'), window=d.get('window'))


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """全样本 Spearman 相关（先做截面 rank）"""
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

    miner = GPFactorMiner()
    terminals = miner._prepare_terminals(padded_raw, col_map)

    registry = GPFactorMiner.load_registry(GP_REGISTRY_PATH)
    validated = [f for f in registry["factors"] if f["status"] == "validated"]
    testing = [f for f in registry["factors"] if f["status"] == "testing"]

    print(f"[Setup] validated={len(validated)}, testing={len(testing)}")

    def eval_factor(f):
        tree = deserialize_tree(f["tree"])
        t = evaluate_tree(tree, terminals)
        if t is None:
            return None
        return t

    print("\n[Eval] Computing testing factors...")
    testing_tensors = {}
    for f in testing:
        t = eval_factor(f)
        if t is not None:
            testing_tensors[f["id"]] = (t, f["expr"])
            print(f"  {f['id']}: OK")
        else:
            print(f"  {f['id']}: FAILED")

    print("\n[Eval] Computing validated factors...")
    validated_tensors = {}
    for f in validated:
        t = eval_factor(f)
        if t is not None:
            validated_tensors[f["id"]] = (t, f["expr"])
            print(f"  {f['id']}: OK")
        else:
            print(f"  {f['id']}: FAILED")

    print("\n" + "=" * 78)
    print("[Result] Testing x Testing Spearman correlation (|corr| >= 0.5 flagged)")
    print("=" * 78)
    testing_ids = list(testing_tensors.keys())
    for i, id1 in enumerate(testing_ids):
        for j, id2 in enumerate(testing_ids):
            if j <= i:
                continue
            t1, e1 = testing_tensors[id1]
            t2, e2 = testing_tensors[id2]
            corr = spearman_corr(t1, t2)
            flag = " !!" if abs(corr) >= 0.5 else ""
            if abs(corr) >= 0.3:
                print(f"  {id1} vs {id2}: {corr:+.3f}{flag}")

    print("\n" + "=" * 78)
    print("[Result] Testing x Validated Spearman correlation (top-3 highest |corr| per testing)")
    print("=" * 78)
    for tid, (tt, texpr) in testing_tensors.items():
        rows = []
        for vid, (vt, vexpr) in validated_tensors.items():
            corr = spearman_corr(tt, vt)
            rows.append((vid, corr, vexpr))
        rows.sort(key=lambda r: abs(r[1]), reverse=True)
        print(f"\n  {tid}: {texpr}")
        for vid, corr, vexpr in rows[:3]:
            flag = " !!" if abs(corr) >= 0.5 else ""
            print(f"    vs {vid} ({corr:+.3f}){flag}: {vexpr}")


if __name__ == "__main__":
    main()
