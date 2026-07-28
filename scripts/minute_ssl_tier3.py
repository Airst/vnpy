"""
SSL embedding Tier-3 终验: 同会话配对全量复现测试（3 轮）

与 scripts/micro_repro_test.py 完全同尺（判微观因子死刑的那把）:
- 每轮 load_session 一次（共享因子计算），基线 143 因子 vs 候选 143+10 SSL 维度
- 同 seed=42, 全量 35 窗（max_windows=0）, backend=attention
- 判定: median delta > 0 且 >=2/3 轮为正 → 可集成; 否则 Tier-1 KEEP 视为噪声

前置: Tier-1 KEEP（median +0.075, deltas {42:+0.776, 123:+0.075, 2024:-0.036}）
用法: /home/airst/Workspace/.venv/bin/python scripts/minute_ssl_tier3.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")
import gc
import json
import numpy as np
import polars as pl

EMB = "core/alpha_db/minute_ssl_emb.parquet"
PASSED = ["ssl_emb_1", "ssl_emb_2", "ssl_emb_6", "ssl_emb_7", "ssl_emb_8",
          "ssl_emb_9", "ssl_emb_10", "ssl_emb_12", "ssl_emb_14", "ssl_emb_15"]
OUT = "log/minute_ssl_tier3.jsonl"
N_ROUNDS = 3


def build_candidate(base: pl.DataFrame) -> pl.DataFrame:
    emb = pl.read_parquet(EMB, columns=["datetime", "vt_symbol"] + PASSED)
    emb = emb.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
    cand = base.join(emb, on=["datetime", "vt_symbol"], how="left")
    cand = cand.with_columns([
        ((pl.when(pl.col(k).is_infinite()).then(0.0).otherwise(pl.col(k)).fill_nan(0.0).fill_null(0.0)
          - pl.col(k).fill_nan(0.0).fill_null(0.0).mean().over("datetime"))
         / (pl.col(k).fill_nan(0.0).fill_null(0.0).std().over("datetime") + 1e-8))
        .clip(-5.0, 5.0).alias(k)
        for k in PASSED
    ])
    fac_cols = [c for c in cand.columns if c not in ("datetime", "vt_symbol", "label", "industry")]
    cand = cand.select(["datetime", "vt_symbol"] + fac_cols + ["label"])
    assert cand.columns[-1] == "label", "label 必须最后一列"
    return cand


def main():
    from core.alpha.research_runner import load_session, _run_single_seed

    deltas = []
    for r in range(1, N_ROUNDS + 1):
        print(f"\n===== Round {r}/{N_ROUNDS} =====")
        session = load_session(index="000852.SH,399303.SZ", version="v15")
        base = session.factor_df
        cand = build_candidate(base)

        rb = _run_single_seed(session, base, 42, f"ssl3_b_r{r}", backend="attention",
                              max_windows=0, end_date_filter=None)
        gc.collect()
        rc = _run_single_seed(session, cand, 42, f"ssl3_c_r{r}", backend="attention",
                              max_windows=0, end_date_filter=None)
        gc.collect()
        d = rc["score"] - rb["score"]
        deltas.append(d)
        print(f"Round {r}: base={rb['score']:.3f} cand={rc['score']:.3f} delta={d:+.3f}")
        with open(OUT, "a") as f:
            f.write(json.dumps({"round": r, "base": rb["score"], "cand": rc["score"],
                                "delta": d, "base_detail": rb.get("detail"),
                                "cand_detail": rc.get("detail")}, default=str) + "\n")
        del session, base, cand
        gc.collect()

    med = float(np.median(deltas))
    n_pos = sum(1 for d in deltas if d > 0)
    verdict = "REPRODUCIBLE" if (med > 0 and n_pos >= 2) else "NOT_REPRODUCIBLE"
    print(f"\n===== Tier-3 终判: deltas={[f'{d:+.3f}' for d in deltas]}, median={med:+.3f}, {n_pos}/{N_ROUNDS} 为正 → {verdict}")
    with open(OUT, "a") as f:
        f.write(json.dumps({"stage": "verdict", "deltas": deltas, "median": med,
                            "n_pos": n_pos, "verdict": verdict}) + "\n")


if __name__ == "__main__":
    main()
