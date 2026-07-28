"""
SSL embedding Tier-0 IC 扫描 + Tier-1 配对验证

流程:
1. Tier-0: 16 维 ssl_emb 逐维 vs 5日 label 的日度 RankIC（|IC|>=0.02, |ICIR|>=0.3, dir>=0.6），
   并检查与 143 生产因子的最大截面相关（>0.9 视为冗余）
2. 过门维度 → cs_zscore+clip±5 → 追加到基线因子（label 保持最后一列！）
3. Tier-1: 3 seeds × 8 窗配对（复用 research_runner），门禁 median delta>0.05 且 >=2/3 为正

用法: /home/airst/Workspace/.venv/bin/python scripts/minute_ssl_tier1.py [--tier0-only]
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")
import gc
import json
import numpy as np
import polars as pl

EMB = "core/alpha_db/minute_ssl_emb.parquet"
Z = 16
OUT = "log/minute_ssl_tier1.jsonl"


def cs_rank_ic(df: pl.DataFrame, col: str) -> tuple[float, float, float]:
    """日度 RankIC 均值 / ICIR / 方向一致率"""
    d = df.select(["datetime", col, "label"]).drop_nulls()
    d = d.with_columns([
        pl.col(col).rank().over("datetime").alias("_rx"),
        pl.col("label").rank().over("datetime").alias("_ry"),
    ])
    ic = d.group_by("datetime").agg(pl.corr("_rx", "_ry").alias("ic"))["ic"].drop_nulls().to_numpy()
    ic = ic[np.isfinite(ic)]  # polars corr 可能产 NaN(非 null), drop_nulls 拦不住
    if len(ic) < 50:
        return 0.0, 0.0, 0.5
    m, s = float(np.mean(ic)), float(np.std(ic))
    icir = m / (s + 1e-12)
    direction = float(np.mean(np.sign(ic) == np.sign(m))) if m != 0 else 0.5
    return m, icir, direction


def max_corr_with_base(df: pl.DataFrame, emb_col: str, base_cols: list[str], sample_dates: list) -> float:
    """embedding 维与生产因子的最大 |截面秩相关|（抽样日期上算，逐列 finite 掩码防 NaN 污染）"""
    sub = df.filter(pl.col("datetime").is_in(sample_dates)).drop_nulls(subset=[emb_col])
    ranked = sub.with_columns([pl.col(c).rank().over("datetime") for c in [emb_col] + base_cols])
    arr = ranked.select([emb_col] + base_cols).to_numpy().astype(np.float64)
    y, X = arr[:, 0], arr[:, 1:]
    ym = np.isfinite(y)
    best = 0.0
    for j in range(X.shape[1]):
        m = ym & np.isfinite(X[:, j])
        if m.sum() < 100:
            continue
        c = np.corrcoef(y[m], X[m, j])[0, 1]
        if np.isfinite(c):
            best = max(best, abs(float(c)))
    return best


def main():
    tier0_only = "--tier0-only" in sys.argv
    from core.alpha.research_runner import load_session, _run_single_seed

    print("[1] load_session ...")
    session = load_session(index="000852.SH,399303.SZ", version="v15")
    base = session.factor_df
    emb = pl.read_parquet(EMB).with_columns(pl.col("datetime").cast(pl.Datetime("us")))
    emb_cols = [f"ssl_emb_{i}" for i in range(Z)]

    joined = base.join(emb, on=["datetime", "vt_symbol"], how="left")
    cover = 1 - joined[emb_cols[0]].null_count() / len(joined)
    print(f"[1] 覆盖率: {cover:.1%}")

    # ---- Tier-0 ----
    print("[2] Tier-0 IC 扫描（评估期 2022+，与回测期一致）...")
    eval_df = joined.filter(pl.col("datetime") >= pl.datetime(2022, 1, 1))
    base_cols = [c for c in base.columns if c not in ("datetime", "vt_symbol", "label", "industry")]
    dates = eval_df["datetime"].unique().sort().to_list()
    sample_dates = dates[:: max(1, len(dates) // 60)]

    passed = []
    for ec in emb_cols:
        m, icir, dirr = cs_rank_ic(eval_df, ec)
        ok_ic = abs(m) >= 0.02 and abs(icir) >= 0.3 and dirr >= 0.6
        mc = max_corr_with_base(eval_df, ec, base_cols, sample_dates) if ok_ic else float("nan")
        ok = ok_ic and mc < 0.9
        print(f"  {ec:<12s} IC={m:+.4f} ICIR={icir:+.3f} dir={dirr:.2f} max_corr={mc if mc==mc else float('nan'):.3f} {'PASS' if ok else 'fail'}")
        if ok:
            passed.append(ec)
        with open(OUT, "a") as f:
            f.write(json.dumps({"stage": "tier0", "col": ec, "ic": m, "icir": icir,
                                "dir": dirr, "max_corr": mc if mc == mc else None, "pass": ok}) + "\n")
    print(f"[2] Tier-0: {len(passed)}/{Z} 过门: {passed}")
    if not passed or tier0_only:
        print("结束（无过门维度或 --tier0-only）")
        return

    # ---- 构建候选（label 必须最后一列）----
    print("[3] 构建候选 ...")
    emb_keep = emb.select(["datetime", "vt_symbol"] + passed)
    cand = base.join(emb_keep, on=["datetime", "vt_symbol"], how="left")
    cand = cand.with_columns([
        ((pl.when(pl.col(k).is_infinite()).then(0.0).otherwise(pl.col(k)).fill_nan(0.0).fill_null(0.0)
          - pl.col(k).fill_nan(0.0).fill_null(0.0).mean().over("datetime"))
         / (pl.col(k).fill_nan(0.0).fill_null(0.0).std().over("datetime") + 1e-8))
        .clip(-5.0, 5.0).alias(k)
        for k in passed
    ])
    fac_cols = [c for c in cand.columns if c not in ("datetime", "vt_symbol", "label", "industry")]
    cand = cand.select(["datetime", "vt_symbol"] + fac_cols + ["label"])
    assert cand.columns[-1] == "label", "label 必须最后一列"
    del joined, eval_df, emb
    gc.collect()

    # ---- Tier-1 配对 ----
    print("[4] Tier-1 配对（3 seeds × 8 窗 × attention）...")
    deltas = {}
    for seed in (42, 123, 2024):
        rb = _run_single_seed(session, base, seed, f"sslq_b_s{seed}", backend="attention",
                              max_windows=8, end_date_filter=None)
        gc.collect()
        rc = _run_single_seed(session, cand, seed, f"sslq_c_s{seed}", backend="attention",
                              max_windows=8, end_date_filter=None)
        gc.collect()
        deltas[seed] = rc["score"] - rb["score"]
        print(f"  seed {seed}: base={rb['score']:.3f} cand={rc['score']:.3f} delta={deltas[seed]:+.3f}")
        with open(OUT, "a") as f:
            f.write(json.dumps({"stage": "tier1", "seed": seed, "base": rb["score"],
                                "cand": rc["score"], "delta": deltas[seed]}) + "\n")

    ds = list(deltas.values())
    med = float(np.median(ds))
    n_pos = sum(1 for d in ds if d > 0)
    verdict = "KEEP" if (med > 0.05 and n_pos >= 2) else "REVERT"
    print(f"\n[4] Tier-1 判定: median={med:+.3f}, {n_pos}/3 为正 → {verdict}")
    with open(OUT, "a") as f:
        f.write(json.dumps({"stage": "verdict", "median": med, "n_pos": n_pos,
                            "verdict": verdict, "passed_dims": passed}) + "\n")


if __name__ == "__main__":
    main()
