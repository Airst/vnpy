"""Auto-research: systematic experiments to improve 26 Q2 performance.

Runs multiple hparam/strategy candidates sequentially against the 26 Q2 attention baseline.
Each experiment: 3 seeds × 2 windows (attention) → paired-seed verdict.
"""
import sys
from datetime import datetime
sys.path.insert(0, "/home/airst/Workspace/vnpy")

from core.alpha.research_runner import (
    FactorChange, load_session, compute_baseline,
    tier0_factor_gate, tier1_quick_validate,
    variance_keep_or_revert, record_experiment,
)


EXPERIMENTS = [
    {
        "desc": "26Q2: max_windows=4 (more recent training)",
        "hparam_overrides": {},
        "max_windows": 4,
    },
    {
        "desc": "26Q2: retrain_days=30 (faster adaptation)",
        "hparam_overrides": {"retrain_days": 30},
        "max_windows": 2,
    },
    {
        "desc": "26Q2: stop_loss=3% (tight risk control)",
        "hparam_overrides": {"strategy_settings": {"stop_loss_pct": 0.03}},
        "max_windows": 2,
    },
    {
        "desc": "26Q2: stop_loss=5% + trailing=8%",
        "hparam_overrides": {"strategy_settings": {"stop_loss_pct": 0.05, "trailing_stop_pct": 0.08}},
        "max_windows": 2,
    },
    {
        "desc": "26Q2: sell_threshold=2.0 (reduce turnover)",
        "hparam_overrides": {"strategy_settings": {"sell_threshold": 2.0}},
        "max_windows": 2,
    },
    {
        "desc": "26Q2: max_holdings=3 (concentrate on strongest)",
        "hparam_overrides": {"max_holdings": 3},
        "max_windows": 2,
    },
    {
        "desc": "26Q2: max_windows=6 + retrain_days=30",
        "hparam_overrides": {"retrain_days": 30},
        "max_windows": 6,
    },
    {
        "desc": "26Q2: buy_threshold=1.5 (stricter entry)",
        "hparam_overrides": {"strategy_settings": {"buy_threshold": 1.5}},
        "max_windows": 2,
    },
]


def main():
    session = load_session(
        index="399303.SZ", version="v15", oos_months=6,
        eval_start=datetime(2026, 4, 1),
        eval_end=datetime(2026, 6, 30),
    )

    print("=== ATTN BASELINE (26 Q2) ===")
    compute_baseline(session, seeds=[42, 123, 2024],
                     max_windows=2, backend="attention", margin=0.05)
    print(f"Baseline: {session.baseline_scores}")
    print()

    results = []
    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i+1}/{len(EXPERIMENTS)}: {exp['desc']}")
        print(f"{'='*60}")

        change = FactorChange(
            change_type="hparam",
            factors=[],
            desc=exp["desc"],
            hparam_overrides=exp["hparam_overrides"],
        )

        t0 = tier0_factor_gate(session, change)
        if not t0["pass"]:
            print(f"  TIER-0 REJECT: {t0['reason']}")
            results.append({"desc": exp["desc"], "verdict": "tier0_reject"})
            continue

        t1 = tier1_quick_validate(session, change,
                                  seeds=[42, 123, 2024],
                                  max_windows=exp.get("max_windows", 2),
                                  backend="attention")
        print(f"  Tier-1 scores: {t1['seed_scores']}, median={t1['median_score']:.4f}")

        verdict = variance_keep_or_revert(t1, session.baseline_scores, margin=0.05)
        print(f"  Verdict: {verdict['verdict']} — {verdict['detail']}")

        exp_id = record_experiment(session, change, t1, verdict,
                                   tier3_result=None, commit_hash=None,
                                   baseline=session.baseline_scores)
        print(f"  Recorded: {exp_id}")

        results.append({
            "desc": exp["desc"],
            "exp_id": exp_id,
            "median_score": t1["median_score"],
            "seed_scores": t1["seed_scores"],
            "verdict": verdict["verdict"],
            "delta": verdict.get("delta", 0),
        })

    print(f"\n\n{'='*60}")
    print("SUMMARY — 26 Q2 Experiments")
    print(f"{'='*60}")
    print(f"Baseline median RDD: {session.baseline_scores['median_score']:.4f}")
    print()
    for r in results:
        v = r.get("verdict", "?")
        d = r.get("delta", 0)
        m = r.get("median_score", 0)
        symbol = "✓" if v == "keep" else "✗"
        print(f"  {symbol} {r['desc']}")
        print(f"    median={m:.4f}, delta={d:+.4f}, verdict={v}")
    print()


if __name__ == "__main__":
    main()
