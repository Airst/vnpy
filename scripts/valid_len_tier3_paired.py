"""
exp_050 (valid_len=50) 配对 Tier-3 终验 — 全量 35 窗 × 3 seeds × attention

设计:
- 同一 session（数据+因子只算一次），两组配置配对：
  A) valid_len=100（旧基线，准则 #17）
  B) valid_len=50（exp_050 keep，当前生产）
- 每组 3 seeds (42/123/2024) × 35 窗 × attention，全时段（含 OOS）
- 判定: 配对 delta = score(50) - score(100)，median > 0 且 ≥2/3 seeds 为正 → 确认 keep
- 同时输出 OOS 切片（最近 6 月）RDD 供人工判读
- 结果增量写入 log/valid_len_tier3.jsonl（每 seed 一行，中断不丢）

预计耗时: ~6.5h（2 配置 × 3 seeds × ~1h + 回测）

用法:
  /home/airst/Workspace/.venv/bin/python scripts/valid_len_tier3_paired.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
from datetime import datetime
from core.alpha.research_runner import (
    load_session, _run_single_seed, _rdd_from_daily, DEFAULT_SEEDS,
)


def main():
    out_path = "log/valid_len_tier3.jsonl"
    print("=== valid_len paired Tier-3: 100 vs 50 ===")
    session = load_session(index="399303.SZ", version="v15")
    end_dt = datetime.strptime(session.engine.end_date, "%Y-%m-%d")
    print(f"session loaded. OOS start: {session.oos_start}, end: {end_dt.date()}")

    results = {}  # {valid_len: {seed: score}}
    for vl in [100, 50]:
        results[vl] = {}
        for seed in DEFAULT_SEEDS:
            print(f"\n--- valid_len={vl} seed={seed} ({datetime.now().strftime('%H:%M:%S')}) ---")
            r = _run_single_seed(
                session, session.factor_df, seed, f"t3v{vl}",
                backend="attention", max_windows=0,
                end_date_filter=None,
                hparam_overrides={"valid_len": vl},
            )
            oos_rdd = _rdd_from_daily(r["daily_data"], session.oos_start, end_dt)
            rec = {
                "valid_len": vl, "seed": seed,
                "score": r["score"], "sharpe": r["sharpe"],
                "max_drawdown": r["max_drawdown"], "oos_rdd": oos_rdd,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            results[vl][seed] = r["score"]
            print(f"valid_len={vl} seed={seed}: RDD={r['score']:.3f} Sharpe={r['sharpe']:.3f} OOS_RDD={oos_rdd}")

    # 配对判定
    print("\n=== PAIRED VERDICT (50 vs 100) ===")
    deltas = {}
    for seed in DEFAULT_SEEDS:
        d = results[50].get(seed, 0) - results[100].get(seed, 0)
        deltas[seed] = d
        print(f"seed {seed}: 100→{results[100].get(seed, 0):.3f}  50→{results[50].get(seed, 0):.3f}  delta={d:+.3f}")
    vals = sorted(deltas.values())
    med = vals[len(vals) // 2]
    n_pos = sum(1 for d in deltas.values() if d > 0)
    verdict = "CONFIRM keep valid_len=50" if (med > 0 and n_pos >= 2) else "REVERT to valid_len=100"
    print(f"median delta={med:+.3f}, {n_pos}/3 positive → {verdict}")

    with open(out_path, "a") as f:
        f.write(json.dumps({"verdict": verdict, "median_delta": med,
                            "n_positive": n_pos, "deltas": {str(k): v for k, v in deltas.items()}}) + "\n")


if __name__ == "__main__":
    main()
