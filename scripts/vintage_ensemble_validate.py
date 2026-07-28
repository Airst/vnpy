"""
Vintage Ensemble Tier-1 配对验证（auto-research 协议）

实验设计:
- Fresh baseline: 3 seeds × 8 窗 × lgb, vintage_ensemble=0（当前因子集 + valid_len=50）
- Candidate: 同 seeds/窗口, vintage_ensemble=2（当前窗口模型 + 过去 2 个窗口模型 rank 平均）
- 配对门槛: median(delta) > 0.05 且 ≥2/3 seeds delta > 0 → keep
- 注意: 8 窗中前 2 窗为 vintage warmup（无/部分历史模型），对候选略不利（保守方向）

用法:
  /home/airst/Workspace/.venv/bin/python scripts/vintage_ensemble_validate.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

from core.alpha.research_loop import load_and_baseline, run_one_iteration
from core.alpha.research_runner import FactorChange


def main():
    print("=== Vintage Ensemble Tier-1 Validation ===")
    s = load_and_baseline(index="399303.SZ", version="v15", max_windows=8,
                          backend="lgb", margin=0.05)
    print(f"Baseline: median={s.baseline_scores['median_score']:.3f} "
          f"spread={s.baseline_scores['spread']:.3f} "
          f"seeds={s.baseline_scores['seed_scores']}")

    change = FactorChange(
        change_type="hparam",
        factors=["vintage_ensemble=2"],
        desc="vintage ensemble K=2: rank-average current + 2 past window models",
        hparam_overrides={"vintage_ensemble": 2},
    )
    result = run_one_iteration(s, change, commit_on_keep=False)

    print("\n=== RESULT ===")
    v = result.get("verdict", {})
    print(f"verdict: {v.get('verdict')} — {v.get('detail')}")
    t1 = result.get("tier1", {})
    print(f"tier1 seed_scores:    {t1.get('seed_scores')}")
    print(f"baseline seed_scores: {s.baseline_scores['seed_scores']}")
    print(f"paired_deltas: {v.get('paired_deltas')}")
    print(f"exp_id: {result.get('exp_id')}")


if __name__ == "__main__":
    main()
