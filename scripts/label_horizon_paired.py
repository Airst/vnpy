"""
标签地平线配对验证 — 10 日 vs 5 日 beta-neutral 标签

背景（用户 2026-07-18 诊断）: 5 日反转标签把模型铸成"超跌反弹猎手"，在 2026 动量
regime 结构性押不中反弹（信号 IC 连续 3 月为负，7 月 -0.239）。V14 的 10 日标签曾是
前代最优（Sharpe 1.42），更慢更不接飞刀。准则 #3：标签设计优先于因子工程。

设计:
- 两 session 分别算因子（label_horizon=5 / 10），因子完全相同、仅标签不同
- 各 3 seeds × 8 窗 × attention，回测 N=5，in-sample（OOS 留出）
- 判定: 配对 delta = score(10d) - score(5d)，median > 0.05 且 ≥2/3 为正 → 10 日胜
- 结果增量写 log/label_horizon_paired.jsonl

预计: 2 session 数据加载(~5min) + 2 配置 × 3 seeds × 8 窗 × attention(~1.6h)

用法:
  /home/airst/Workspace/.venv/bin/python scripts/label_horizon_paired.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
from datetime import datetime
from core.alpha.research_runner import load_session, _run_single_seed, DEFAULT_SEEDS

OUT = "log/label_horizon_paired.jsonl"


def run_label(label_horizon: int) -> dict:
    print(f"\n{'='*60}\n=== label_horizon={label_horizon} ===\n{'='*60}")
    session = load_session(index="399303.SZ", version="v15",
                           calc_kwargs={"label_horizon": label_horizon})
    scores = {}
    for seed in DEFAULT_SEEDS:
        print(f"--- label={label_horizon} seed={seed} ({datetime.now().strftime('%H:%M:%S')}) ---")
        r = _run_single_seed(
            session, session.factor_df, seed, f"lh{label_horizon}",
            backend="attention", max_windows=8,
            end_date_filter=session.oos_start,
        )
        scores[seed] = r["score"]
        rec = {"label_horizon": label_horizon, "seed": seed, "score": r["score"],
               "sharpe": r["sharpe"], "max_drawdown": r["max_drawdown"],
               "ts": datetime.now().isoformat(timespec="seconds")}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"label={label_horizon} seed={seed}: RDD={r['score']:.3f} Sharpe={r['sharpe']:.3f}")
    return scores


def main():
    s5 = run_label(5)
    s10 = run_label(10)

    print("\n=== PAIRED VERDICT (10d vs 5d) ===")
    deltas = {s: s10[s] - s5[s] for s in DEFAULT_SEEDS}
    for s in DEFAULT_SEEDS:
        print(f"seed {s}: 5d={s5[s]:.3f}  10d={s10[s]:.3f}  delta={deltas[s]:+.3f}")
    vals = sorted(deltas.values())
    med = vals[len(vals) // 2]
    n_pos = sum(1 for d in deltas.values() if d > 0)
    verdict = "10d WIN" if (med > 0.05 and n_pos >= 2) else "5d HOLD"
    print(f"median delta={med:+.3f}, {n_pos}/3 positive → {verdict}")
    with open(OUT, "a") as f:
        f.write(json.dumps({"verdict": verdict, "median_delta": med, "n_positive": n_pos,
                            "deltas": {str(k): v for k, v in deltas.items()},
                            "scores_5d": s5, "scores_10d": s10}) + "\n")


if __name__ == "__main__":
    main()
