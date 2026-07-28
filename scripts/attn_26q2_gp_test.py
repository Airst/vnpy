"""26 Q2 validation: add testing GP factors to v15, check attention delta.

Batch A: top 3 (gp_086, gp_081, gp_092) - highest |IC|, different info dimensions
Batch B: top 5 (+ gp_087, gp_083)
Batch C: all 10 testing factors

Uses gp_status_filter to include testing factors automatically.
Per principle #16/#27: incremental batches, max 3 first.
"""
import sys
from datetime import datetime
sys.path.insert(0, "/home/airst/Workspace/vnpy")

import json
from core.alpha.research_runner import (
    FactorChange, load_session, compute_baseline,
    tier0_factor_gate, tier1_quick_validate,
    variance_keep_or_revert, record_experiment,
)


GP_REGISTRY = "/home/airst/Workspace/vnpy/core/alpha/gp_factors.json"


def set_testing_subset(factor_ids: list):
    """Temporarily set only specific testing factors to 'testing', rest to 'discovered'."""
    with open(GP_REGISTRY, "r") as f:
        reg = json.load(f)
    
    for factor in reg["factors"]:
        if factor["status"] == "testing":
            if factor["id"] in factor_ids:
                factor["status"] = "testing"
            else:
                factor["status"] = "discovered"  # temporarily hide
    
    with open(GP_REGISTRY, "w") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


def restore_all_testing(all_testing_ids: list):
    """Restore all originally testing factors back to testing."""
    with open(GP_REGISTRY, "r") as f:
        reg = json.load(f)
    
    for factor in reg["factors"]:
        if factor["id"] in all_testing_ids:
            factor["status"] = "testing"
    
    with open(GP_REGISTRY, "w") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


ALL_TESTING = ["gp_077", "gp_078", "gp_081", "gp_082", "gp_083",
               "gp_084", "gp_086", "gp_087", "gp_088", "gp_092"]

BATCHES = [
    ("Batch A: top 3 (gp_086, gp_081, gp_092)", ["gp_086", "gp_081", "gp_092"]),
    ("Batch B: top 5 (+ gp_087, gp_083)", ["gp_086", "gp_081", "gp_092", "gp_087", "gp_083"]),
    ("Batch C: all 10 testing", ALL_TESTING),
]


def main():
    # First: baseline with ONLY validated factors (no testing)
    set_testing_subset([])  # hide all testing
    
    session = load_session(
        index="399303.SZ", version="v15", oos_months=6,
        eval_start=datetime(2026, 4, 1),
        eval_end=datetime(2026, 6, 30),
        gp_status_filter=["validated"],  # baseline = validated only
    )
    
    print("=== BASELINE (validated only, 26 Q2) ===")
    compute_baseline(session, seeds=[42, 123, 2024],
                     max_windows=2, backend="attention", margin=0.05)
    print(f"Baseline: {session.baseline_scores}")
    
    results = []
    
    for batch_name, batch_ids in BATCHES:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {batch_name}")
        print(f"{'='*60}")
        
        # Set only this batch's factors to testing
        set_testing_subset(batch_ids)
        
        # Reload session with testing factors included
        batch_session = load_session(
            index="399303.SZ", version="v15", oos_months=6,
            eval_start=datetime(2026, 4, 1),
            eval_end=datetime(2026, 6, 30),
            gp_status_filter=["validated", "testing"],
        )
        # Reuse the baseline from the no-testing session
        batch_session.baseline_scores = session.baseline_scores
        
        change = FactorChange(
            change_type="add",
            factors=batch_ids,
            desc=f"add GP testing: {batch_name}",
            candidate_factor_df=batch_session.factor_df,
        )

        t1 = tier1_quick_validate(batch_session, change,
                                  seeds=[42, 123, 2024],
                                  max_windows=2,
                                  backend="attention")
        print(f"  Tier-1 scores: {t1['seed_scores']}, median={t1['median_score']:.4f}")

        verdict = variance_keep_or_revert(t1, session.baseline_scores, margin=0.05)
        print(f"  Verdict: {verdict['verdict']} — {verdict['detail']}")

        exp_id = record_experiment(session, change, t1, verdict,
                                   tier3_result=None, commit_hash=None,
                                   baseline=session.baseline_scores)
        print(f"  Recorded: {exp_id}")

        results.append({
            "desc": batch_name, "exp_id": exp_id,
            "median_score": t1["median_score"],
            "verdict": verdict["verdict"],
            "delta": verdict.get("delta", 0),
            "seed_scores": t1["seed_scores"],
        })
    
    # Restore all testing
    restore_all_testing(ALL_TESTING)
    
    print(f"\n\n{'='*60}")
    print("SUMMARY — GP Factor Addition (26 Q2)")
    print(f"{'='*60}")
    print(f"Baseline median RDD: {session.baseline_scores['median_score']:.4f}")
    for r in results:
        v = r.get("verdict", "?")
        d = r.get("delta", 0)
        m = r.get("median_score", 0)
        symbol = "✓" if v == "keep" else "✗"
        print(f"  {symbol} delta={d:+.4f} median={m:.4f}  {r['desc']}")
        print(f"    seeds: {r['seed_scores']}")


if __name__ == "__main__":
    main()
