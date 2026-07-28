"""
微观因子 v2 候选-only 验证（省内存版）

- 复用已有的 13 个去重后因子（不再跑耗内存的共线分析）+ 已有基线 [3.19, 7.30, 5.99]
- 候选 = 基线 + 13 清洗后微观因子（winsorize + fill_nan + 截面 z-score + clip ±5）
- 3 seeds × 8 窗 × attention 配对
- 注意: 与之前 raw 列 bug 无关——micro_clean 只含 keep 因子

用法:
  /home/airst/Workspace/.venv/bin/python scripts/micro_v2_cand_only.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
import gc
import numpy as np
import polars as pl
from datetime import datetime
from core.alpha.research_runner import load_session, _run_single_seed, DEFAULT_SEEDS

KEEP = ["kyle_lambda", "vol_weighted_vol", "vol_autocorr", "mean_bar_range", "tail30_ret",
        "intraday_mdd", "ushape_vol", "big_bar_vol_r", "obv_slope", "first60_vol_r",
        "close_auct_vol_r", "trend_persist", "vol_price_corr"]
BASE = {42: 3.194540031978386, 123: 7.297423276058438, 2024: 5.985355919024772}
OUT = "log/micro_v2_cand.jsonl"


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
    print("load_session ...")
    session = load_session(index="399303.SZ", version="v15")
    cand = build_candidate(session.factor_df)
    print(f"候选 {len(cand.columns)} 列 (基线+{len(KEEP)}micro)")
    gc.collect()

    print(f"=== 候选(143+{len(KEEP)}micro) 3 seeds × 8 窗 × attention ===")
    s_cand = {}
    for seed in DEFAULT_SEEDS:
        r = _run_single_seed(session, cand, seed, "v2cand", backend="attention",
                             max_windows=8, end_date_filter=session.oos_start)
        s_cand[seed] = r["score"]
        with open(OUT, "a") as f:
            f.write(json.dumps({"seed": seed, "score": r["score"], "sharpe": r["sharpe"],
                                "ts": datetime.now().isoformat(timespec="seconds")}) + "\n")
        print(f"  seed={seed}: RDD={r['score']:.3f} Sharpe={r['sharpe']:.3f}")
        gc.collect()

    deltas = {s: s_cand[s] - BASE[s] for s in DEFAULT_SEEDS}
    vals = sorted(deltas.values())
    med, n_pos = vals[len(vals) // 2], sum(1 for d in deltas.values() if d > 0)
    verdict = "KEEP" if (med > 0.05 and n_pos >= 2) else "REVERT"
    print(f"\n=== PAIRED: deltas={ {s: round(d, 3) for s, d in deltas.items()} } median={med:+.3f} {n_pos}/3 → {verdict} ===")
    with open(OUT, "a") as f:
        f.write(json.dumps({"verdict": verdict, "median_delta": med, "n_positive": n_pos,
                            "deltas": deltas, "baseline": BASE, "cand": s_cand}) + "\n")


if __name__ == "__main__":
    main()
