"""Combined experiment: 10 GP testing factors + best strategy params from Round 1.

Baseline (validated only, default strategy): median RDD -0.7377
Best single-effect:
  - All 10 GP factors:       delta +0.308 (median -0.633)
  - max_holdings=3:          delta +0.255 (Round 1)
  - stop_loss=3%:            delta +0.110 (Round 1)
  - sell_threshold=2.0:      delta +0.166 (Round 1)

If effects stack, combined could push median RDD close to zero.
"""
import sys
from datetime import datetime
sys.path.insert(0, "/home/airst/Workspace/vnpy")

import json
from core.alpha.research_runner import (
    FactorChange, load_session, compute_baseline,
    tier1_quick_validate, variance_keep_or_revert, record_experiment,
)

ALL_TESTING = ["gp_077", "gp_078", "gp_081", "gp_082", "gp_083",
               "gp_084", "gp_086", "gp_087", "gp_088", "gp_092"]

EXPERIMENTS = [
    {
        "desc": "26Q2-C1: 10 GP + max_holdings=3",
        "hparam_overrides": {"max_holdings": 3},
        "max_windows": 2,
    },
    {
        "desc": "26Q2-C2: 10 GP + max_holdings=3 + stop_loss=3%",
        "hparam_overrides": {"max_holdings": 3, "strategy_settings": {"stop_loss_pct": 0.03}},
        "max_windows": 2,
    },
    {
        "desc": "26Q2-C3: 10 GP + max_holdings=3 + sell_threshold=2.0",
        "hparam_overrides": {"max_holdings": 3, "strategy_settings": {"sell_threshold": 2.0}},
        "max_windows": 2,
    },
    {
        "desc": "26Q2-C4: 10 GP + full stack (mh=3, sl=3%, st=2.0)",
        "hparam_overrides": {"max_holdings": 3, "strategy_settings": {"stop_loss_pct": 0.03, "sell_threshold": 2.0}},
        "max_windows": 2,
    },
    {
        "desc": "26Q2-C5: 10 GP + max_windows=6",
        "hparam_overrides": {},
        "max_windows": 6,
    },
]


def main():
    # Baseline: validated only (no testing factors)
    session_base = load_session(
        index="399303.SZ", version="v15", oos_months=6,
        eval_start=datetime(2026, 4, 1),
        eval_end=datetime(2026, 6, 30),
        gp_status_filter=["validated"],
    )
    print("=== BASELINE (validated only, 26 Q2) ===")
    compute_baseline(session_base, seeds=[42, 123, 2024],
                     max_windows=2, backend="attention", margin=0.05)
    print(f"Baseline: {session_base.baseline_scores}")

    # Experiment session: include testing factors
    session_exp = load_session(
        index="399303.SZ", version="v15", oos_months=6,
        eval_start=datetime(2026, 4, 1),
        eval_end=datetime(2026, 6, 30),
        gp_status_filter=["validated", "testing"],
    )
    session_exp.baseline_scores = session_base.baseline_scores

    results = []
    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i+1}/{len(EXPERIMENTS)}: {exp['desc']}")
        print(f"{'='*60}")

        change = FactorChange(
            change_type="hparam",  # skip Tier-0 factor gate (add uses candidate_factor_df)
            factors=ALL_TESTING,
            desc=exp["desc"],
            candidate_factor_df=session_exp.factor_df,
            hparam_overrides=exp["hparam_overrides"],
        )

        t1 = tier1_quick_validate(session_exp, change,
                                  seeds=[42, 123, 2024],
                                  max_windows=exp.get("max_windows", 2),
                                  backend="attention")
        print(f"  Tier-1 scores: {t1['seed_scores']}, median={t1['median_score']:.4f}")

        verdict = variance_keep_or_revert(t1, session_base.baseline_scores, margin=0.05)
        print(f"  Verdict: {verdict['verdict']} — {verdict['detail']}")

        exp_id = record_experiment(session_exp, change, t1, verdict,
                                   tier3_result=None, commit_hash=None,
                                   baseline=session_base.baseline_scores)
        print(f"  Recorded: {exp_id}")

        results.append({
            "desc": exp["desc"], "exp_id": exp_id,
            "median": t1["median_score"], "verdict": verdict["verdict"],
            "delta": verdict.get("delta", 0),
            "seed_scores": t1["seed_scores"],
        })

    print(f"\n\n{'='*60}")
    print("SUMMARY — 10 GP + Strategy Combined (26 Q2)")
    print(f"{'='*60}")
    print(f"Baseline median RDD: {session_base.baseline_scores['median_score']:.4f}")
    for r in sorted(results, key=lambda x: -x.get("delta", 0)):
        v, d, m = r.get("verdict", "?"), r.get("delta", 0), r.get("median", 0)
        symbol = "✓" if v == "keep" else "✗"
        print(f"  {symbol} delta={d:+.4f} median={m:.4f}  {r['desc']}")
        print(f"    seeds: {r['seed_scores']}")


if __name__ == "__main__":
    main()
