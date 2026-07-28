"""
SWA + 微观因子 seed=42 全量结果可复现性测试

设计（同会话配对，消除会话间非确定性）:
- 每轮: load_session 一次（基线因子计算共享）→ 基线(143) 与 候选(143+13micro)
  同 seed=42 全量 35 窗 × attention × swa，各回测 RDD
- 跑 3 轮，看 delta = cand - base 是否一致为正
- 判定: median(delta) > 0 且 ≥2/3 轮为正 → 改善可复现（真）；否则 seed 运气

用法:
  /home/airst/Workspace/.venv/bin/python scripts/micro_repro_test.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
import gc
import numpy as np
import polars as pl
from datetime import datetime
from core.alpha.research_runner import load_session, _run_single_seed

KEEP = ["kyle_lambda", "vol_weighted_vol", "vol_autocorr", "mean_bar_range", "tail30_ret",
        "intraday_mdd", "ushape_vol", "big_bar_vol_r", "obv_slope", "first60_vol_r",
        "close_auct_vol_r", "trend_persist", "vol_price_corr"]
OUT = "log/micro_repro.jsonl"
N_ROUNDS = 3


def build_candidate(base):
    micro = pl.read_parquet("core/alpha_db/micro_factors_v2.parquet")
    micro = micro.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
    micro = micro.select(["datetime", "vt_symbol"] + KEEP)
    for k in KEEP:
        lo = micro[k].quantile(0.001)
        hi = micro[k].quantile(0.999)
        micro = micro.with_columns(pl.col(k).clip(lo, hi))
    cand = base.join(micro, on=["datetime", "vt_symbol"], how="left")
    cand = cand.with_columns([
        ((pl.when(pl.col(k).is_infinite()).then(0.0).otherwise(pl.col(k)).fill_nan(0.0).fill_null(0.0)
          - pl.col(k).fill_nan(0.0).fill_null(0.0).mean().over("datetime"))
         / (pl.col(k).fill_nan(0.0).fill_null(0.0).std().over("datetime") + 1e-8))
        .clip(-5.0, 5.0).alias(k)
        for k in KEEP
    ])
    fac_cols = [c for c in cand.columns if c not in ("datetime", "vt_symbol", "label", "industry")]
    cand = cand.select(["datetime", "vt_symbol"] + fac_cols + ["label"])
    assert cand.columns[-1] == "label"
    return cand


def main():
    results = []
    for r in range(1, N_ROUNDS + 1):
        print(f"\n{'='*55}\n=== 第 {r}/{N_ROUNDS} 轮（同会话配对）===\n{'='*55}")
        session = load_session(index="000852.SH,399303.SZ", version="v15")
        cand = build_candidate(session.factor_df)

        print(f"[轮{r}] 基线(143) seed=42 全量...")
        rb = _run_single_seed(session, session.factor_df, 42, f"repro_b_r{r}",
                              backend="attention", max_windows=0, end_date_filter=None)
        gc.collect()
        print(f"[轮{r}] 基线: RDD={rb['score']:.3f} Sharpe={rb['sharpe']:.3f}")

        print(f"[轮{r}] 候选(143+{len(KEEP)}micro) seed=42 全量...")
        rc = _run_single_seed(session, cand, 42, f"repro_c_r{r}",
                              backend="attention", max_windows=0, end_date_filter=None)
        gc.collect()
        print(f"[轮{r}] 候选: RDD={rc['score']:.3f} Sharpe={rc['sharpe']:.3f}")

        delta = rc["score"] - rb["score"]
        results.append({"round": r, "base": rb["score"], "cand": rc["score"],
                        "base_sharpe": rb["sharpe"], "cand_sharpe": rc["sharpe"], "delta": delta})
        with open(OUT, "a") as f:
            f.write(json.dumps(results[-1]) + "\n")
        print(f"[轮{r}] delta = {delta:+.3f}")
        del session, cand
        gc.collect()

    deltas = [x["delta"] for x in results]
    med = float(np.median(deltas))
    n_pos = sum(1 for d in deltas if d > 0)
    print(f"\n{'='*55}\n=== 可复现性判定 ===\n{'='*55}")
    for x in results:
        print(f"  轮{x['round']}: base={x['base']:.3f} cand={x['cand']:.3f} delta={x['delta']:+.3f} "
              f"(Sharpe {x['base_sharpe']:.2f}→{x['cand_sharpe']:.2f})")
    verdict = "改善可复现（真）" if (med > 0 and n_pos >= 2) else "seed 运气（不可复现）"
    print(f"median delta={med:+.3f}, {n_pos}/{N_ROUNDS} 轮为正 → {verdict}")
    with open(OUT, "a") as f:
        f.write(json.dumps({"verdict": verdict, "median_delta": med, "n_positive": n_pos, "results": results}) + "\n")


if __name__ == "__main__":
    main()
