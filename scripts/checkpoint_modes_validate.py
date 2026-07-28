"""
检查点平均化三候选配对验证（attention 后端 × 8 窗 × 3 seeds）

候选（model_settings.checkpoint_mode）:
  ① topk_pred — top-3 valid 检查点分别预测 + 逐日 rank 平均
  ② swa       — top-3 检查点权重平均（greedy model soup）
  ③ ema       — 权重指数滑动平均（decay=0.999），best EMA 检查点
基线: checkpoint_mode="best"（当前生产配置，valid_len=100）

判定: 配对 delta = score(candidate) - score(baseline)，median > 0.05 且 ≥2/3 为正 → keep
结果增量写 log/checkpoint_modes_validate.jsonl

预计: 4 配置 × 3 seeds × 8 窗 × attention ≈ 3.2h

用法:
  /home/airst/Workspace/.venv/bin/python scripts/checkpoint_modes_validate.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
from datetime import datetime
from core.alpha.research_runner import load_session, _run_single_seed, DEFAULT_SEEDS

OUT = "log/checkpoint_modes_validate.jsonl"
MODES = ["best", "topk_pred", "swa", "ema"]


def main():
    print("=== Checkpoint Modes Paired Validation ===")
    session = load_session(index="399303.SZ", version="v15")
    print(f"session loaded ({datetime.now().strftime('%H:%M:%S')})")

    results = {}  # {mode: {seed: score}}
    for mode in MODES:
        results[mode] = {}
        for seed in DEFAULT_SEEDS:
            print(f"\n--- mode={mode} seed={seed} ({datetime.now().strftime('%H:%M:%S')}) ---")
            r = _run_single_seed(
                session, session.factor_df, seed, f"ckpt_{mode}",
                backend="attention", max_windows=8,
                end_date_filter=session.oos_start,
                hparam_overrides={"model_settings": {"checkpoint_mode": mode}},
            )
            rec = {"mode": mode, "seed": seed, "score": r["score"],
                   "sharpe": r["sharpe"], "max_drawdown": r["max_drawdown"],
                   "ts": datetime.now().isoformat(timespec="seconds")}
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            results[mode][seed] = r["score"]
            print(f"mode={mode} seed={seed}: RDD={r['score']:.3f} Sharpe={r['sharpe']:.3f}")

    print("\n=== PAIRED VERDICTS (vs best) ===")
    base = results["best"]
    for mode in ["topk_pred", "swa", "ema"]:
        deltas = {s: results[mode][s] - base[s] for s in DEFAULT_SEEDS}
        vals = sorted(deltas.values())
        med = vals[len(vals) // 2]
        n_pos = sum(1 for d in deltas.values() if d > 0)
        keep = med > 0.05 and n_pos >= 2
        print(f"{mode}: deltas={ {s: round(d, 3) for s, d in deltas.items()} } "
              f"median={med:+.3f} {n_pos}/3 → {'KEEP' if keep else 'REVERT'}")
        with open(OUT, "a") as f:
            f.write(json.dumps({"verdict_for": mode, "deltas": {str(s): d for s, d in deltas.items()},
                                "median_delta": med, "n_positive": n_pos,
                                "verdict": "keep" if keep else "revert"}) + "\n")


if __name__ == "__main__":
    main()
