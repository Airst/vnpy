"""
微观结构因子 Tier-1 集成验证（共线去重 + 配对验证）

流程:
1. load_session 取基线 factor_df（143 因子）
2. 重算微观因子（扩展数据 2022-07 起），join 到基线
3. 共线去重: 与 143 因子 |corr|>=0.9 的判冗余; micro 间 greedy 去重
4. Tier-1: 基线(143) vs 候选(143+选中micro)，3 seeds × 8 窗 × attention 配对门禁

用法:
  /home/airst/Workspace/.venv/bin/python scripts/micro_factors_tier1.py
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

OUT = "log/micro_tier1.jsonl"


def cs_corr(df, a, b):
    """两因子列的逐日截面 spearman 相关均值"""
    vals = []
    for _, g in df.group_by("datetime", maintain_order=True):
        x, y = g[a].to_numpy(), g[b].to_numpy()
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() > 100 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
            c, _ = spearmanr(x[m], y[m])
            if not np.isnan(c):
                vals.append(c)
    return float(np.mean(vals)) if vals else 0.0


def main():
    micro = pl.read_parquet("core/alpha_db/micro_factors.parquet")
    micro = micro.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
    passed = json.load(open("core/alpha_db/micro_factors_tier0.json"))["passed"]
    print(f"tier0 通过因子: {passed}")

    print("load_session (基线 143 因子) ...")
    session = load_session(index="399303.SZ", version="v15")
    base = session.factor_df
    base_cols = [c for c in base.columns if c not in ("datetime", "vt_symbol", "label", "industry")]
    print(f"基线 {len(base_cols)} 因子")

    # join micro（左连，未覆盖日期为 null）
    joined = base.join(micro, on=["datetime", "vt_symbol"], how="left")

    # 共线分析（在覆盖日期上）
    covered = joined.filter(pl.col(passed[0]).is_not_null())
    print(f"覆盖日期 {covered['datetime'].n_unique()} 天，用于共线分析")
    keep, dropped = [], {}
    for mf in passed:
        # 与现有因子的最大相关
        max_c, max_with = 0.0, ""
        for bc in base_cols:
            c = abs(cs_corr(covered, mf, bc))
            if c > max_c:
                max_c, max_with = c, bc
        # 与已选 micro 的相关
        mm = max([abs(cs_corr(covered, mf, k)) for k in keep], default=0.0)
        redundant = max_c >= 0.90 or mm >= 0.85
        dropped[mf] = {"max_corr_with_existing": round(max_c, 3), "with": max_with,
                       "max_corr_with_selected_micro": round(mm, 3), "redundant": redundant}
        if not redundant:
            keep.append(mf)
        print(f"  {mf:<18} max|corr|现有={max_c:.2f}({max_with}) micro={mm:.2f} → {'冗余' if redundant else '保留'}")
    print(f"去重后保留 {len(keep)}: {keep}")

    # 候选 factor_df = 基线 + 保留 micro（null 填 0）
    cand = joined.with_columns([pl.col(k).fill_null(0.0) for k in keep]) if keep else base
    # 关键: join 把 micro 列追加到 label 之后，模型 df.columns[2:-1] 会把 label 当特征（泄漏）
    # 必须重排让 label 回到最后一列
    if keep:
        all_fac = [c for c in cand.columns if c not in ("datetime", "vt_symbol", "label", "industry")]
        cand = cand.select(["datetime", "vt_symbol"] + all_fac + ["label"])
        assert cand.columns[-1] == "label", "label 必须在最后一列（防泄漏）"

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
    s_base = run(base, "micro_base")
    if keep:
        print(f"\n=== 候选(143+{len(keep)}micro) ===")
        s_cand = run(cand, "micro_cand")
        deltas = {s: s_cand[s] - s_base[s] for s in DEFAULT_SEEDS}
        vals = sorted(deltas.values())
        med, n_pos = vals[len(vals) // 2], sum(1 for d in deltas.values() if d > 0)
        verdict = "KEEP" if (med > 0.05 and n_pos >= 2) else "REVERT"
        print(f"\n=== PAIRED: deltas={ {s: round(d, 3) for s, d in deltas.items()} } median={med:+.3f} {n_pos}/3 → {verdict} ===")
        with open(OUT, "a") as f:
            f.write(json.dumps({"verdict": verdict, "median_delta": med, "n_positive": n_pos,
                                "deltas": deltas, "keep_factors": keep, "dropped": dropped}) + "\n")
    else:
        print("无保留因子，跳过 Tier-1")


if __name__ == "__main__":
    main()
