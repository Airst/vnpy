"""
微观因子 v2 去重 + Tier-1 配对验证

- 21 个 Tier-0 过门因子按 |ICIR| 降序 greedy 去重:
  与现有 143 因子 |corr|>=0.90 判冗余; 与已选 micro |corr|>=0.80 判冗余(低波动簇)
- 选出代表性因子组后, 基线(143) vs 候选(143+代表micro) 3 seeds × 8 窗 × attention 配对

用法:
  /home/airst/Workspace/.venv/bin/python scripts/micro_v2_dedup_tier1.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
import numpy as np
import polars as pl
from datetime import datetime
from scipy.stats import spearmanr
from core.alpha.research_runner import load_session, _run_single_seed, DEFAULT_SEEDS

OUT = "log/micro_v2_tier1.jsonl"


def cs_corr(df, a, b):
    """两因子逐日截面 spearman 相关均值（polars 向量化，比 Python 循环快数百倍）
    rank 按日截面，Pearson(rankA,rankB)=spearman；rank 的均值/方差每日固定
    """
    sub = df.select(["datetime", a, b]).drop_nulls()
    if len(sub) == 0:
        return 0.0
    sub = sub.with_columns([
        pl.col(a).rank(method="average").over("datetime").alias("_ra"),
        pl.col(b).rank(method="average").over("datetime").alias("_rb"),
    ])
    agg = sub.group_by("datetime").agg([
        (pl.col("_ra") * pl.col("_rb")).mean().alias("mab"),
        pl.col("_ra").mean().alias("ma"),
        pl.col("_rb").mean().alias("mb"),
        pl.col("_ra").std().alias("sa"),
        pl.col("_rb").std().alias("sb"),
        pl.len().alias("n"),
    ])
    corr = ((agg["mab"] - agg["ma"] * agg["mb"]) / (agg["sa"] * agg["sb"] + 1e-12)).to_numpy()
    n = agg["n"].to_numpy()
    valid = (~np.isnan(corr)) & (n > 100)
    return float(np.mean(corr[valid])) if valid.sum() else 0.0


def main():
    t0 = json.load(open("core/alpha_db/micro_factors_tier0_v2.json"))["results"]
    passed = json.load(open("core/alpha_db/micro_factors_tier0_v2.json"))["passed"]
    # 按 |ICIR| 降序
    passed = sorted(passed, key=lambda f: -abs(t0[f]["icir"]))
    print(f"过门 {len(passed)} 因子, 按 ICIR 降序去重")

    micro = pl.read_parquet("core/alpha_db/micro_factors_v2.parquet")
    micro = micro.with_columns(pl.col("datetime").cast(pl.Datetime("us")))

    print("load_session ...")
    session = load_session(index="399303.SZ", version="v15")
    base = session.factor_df
    base_cols = [c for c in base.columns if c not in ("datetime", "vt_symbol", "label", "industry")]

    joined = base.join(micro, on=["datetime", "vt_symbol"], how="left")
    covered = joined.filter(pl.col(passed[0]).is_not_null())

    # 向量化共线: 判定时直接计算与已选 micro 的相关（micro 对少，重算也便宜）
    keep = []
    for mf in passed:
        max_e, with_e = 0.0, ""
        for bc in base_cols:
            c = abs(cs_corr(covered, mf, bc))
            if c > max_e:
                max_e, with_e = c, bc
        max_m = max([abs(cs_corr(covered, mf, k)) for k in keep], default=0.0)
        redundant = max_e >= 0.90 or max_m >= 0.80
        if not redundant:
            keep.append(mf)
        print(f"  {mf:<18} ICIR={abs(t0[mf]['icir']):.2f} max|corr|现有={max_e:.2f}({with_e}) micro={max_m:.2f} → {'冗余' if redundant else '保留'}")
    print(f"去重后保留 {len(keep)}: {keep}")

    # 微观因子必须与生产一致做截面 z-score 归一化（原始值直接 join 会致量纲不一致/梯度爆炸 nan）
    # 候选 = 基线 + 清洗后的 keep 因子（micro_clean 只含 keep，避免引入未清洗的原始微观列）
    micro_clean = micro.select(["datetime", "vt_symbol"] + keep)
    for k in keep:
        lo = micro_clean[k].quantile(0.001)
        hi = micro_clean[k].quantile(0.999)
        micro_clean = micro_clean.with_columns(pl.col(k).clip(lo, hi))
    cand = base.join(micro_clean, on=["datetime", "vt_symbol"], how="left")
    # 截面 z-score 归一化（与 factor_calculator._normalize_data 一致：inf→0, fill nan/null 0）
    cand = cand.with_columns([
        ((pl.when(pl.col(k).is_infinite()).then(0.0).otherwise(pl.col(k)).fill_nan(0.0).fill_null(0.0)
          - pl.col(k).fill_nan(0.0).fill_null(0.0).mean().over("datetime"))
         / (pl.col(k).fill_nan(0.0).fill_null(0.0).std().over("datetime") + 1e-8))
        .clip(-5.0, 5.0).alias(k)
        for k in keep
    ])
    fac_cols = [c for c in cand.columns if c not in ("datetime", "vt_symbol", "label", "industry")]
    cand = cand.select(["datetime", "vt_symbol"] + fac_cols + ["label"])
    assert cand.columns[-1] == "label"

    def run(df, suffix):
        scores = {}
        for seed in DEFAULT_SEEDS:
            r = _run_single_seed(session, df, seed, suffix, backend="attention",
                                 max_windows=8, end_date_filter=session.oos_start)
            scores[seed] = r["score"]
            with open(OUT, "a") as f:
                f.write(json.dumps({"suffix": suffix, "seed": seed, "score": r["score"],
                                    "sharpe": r["sharpe"], "ts": datetime.now().isoformat(timespec="seconds")}) + "\n")
            print(f"  {suffix} seed={seed}: RDD={r['score']:.3f} Sharpe={r['sharpe']:.3f}")
        return scores

    print("\n=== 基线(143) ===")
    s_base = run(base, "v2_base")
    print(f"\n=== 候选(143+{len(keep)}micro) ===")
    s_cand = run(cand, "v2_cand")
    deltas = {s: s_cand[s] - s_base[s] for s in DEFAULT_SEEDS}
    vals = sorted(deltas.values())
    med, n_pos = vals[len(vals) // 2], sum(1 for d in deltas.values() if d > 0)
    verdict = "KEEP" if (med > 0.05 and n_pos >= 2) else "REVERT"
    print(f"\n=== PAIRED: deltas={ {s: round(d, 3) for s, d in deltas.items()} } median={med:+.3f} {n_pos}/3 → {verdict} ===")
    with open(OUT, "a") as f:
        f.write(json.dumps({"verdict": verdict, "median_delta": med, "n_positive": n_pos,
                            "deltas": deltas, "keep_factors": keep}) + "\n")


if __name__ == "__main__":
    main()
