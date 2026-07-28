"""Attention-backend Tier-1 hparam research (single iteration).

Runs attention baseline then one hparam candidate on the 2026 Q2 window.
"""
import sys
from datetime import datetime
sys.path.insert(0, "/home/airst/Workspace/vnpy")

from core.alpha.research_runner import (
    FactorChange, load_session, compute_baseline,
    tier0_factor_gate, tier1_quick_validate,
    variance_keep_or_revert, record_experiment,
)


def main():
    session = load_session(
        index="399303.SZ", version="v15", oos_months=6,
        eval_start=datetime(2026, 4, 1),
        eval_end=datetime(2026, 6, 30),
    )

    # Attention baseline (2 windows to keep cost bounded)
    print("=== ATTN BASELINE (26 Q2 window) ===")
    compute_baseline(session, seeds=[42, 123, 2024],
                     max_windows=2, backend="attention", margin=0.05)
    print(f"Baseline: {session.baseline_scores}")

    # Candidate: weight_decay=0.001 (attention-only param, lgb-exp_009 was invalid)
    change = FactorChange(
        change_type="hparam",
        factors=[],
        desc="attn: weight_decay 0.002 -> 0.001 (26Q2)",
        hparam_overrides={"model_settings": {"weight_decay": 0.001}},
    )

    t0 = tier0_factor_gate(session, change)
    print(f"Tier-0: {t0}")

    t1 = tier1_quick_validate(session, change,
                              seeds=[42, 123, 2024], max_windows=2,
                              backend="attention")
    print(f"Tier-1: {t1}")

    verdict = variance_keep_or_revert(t1, session.baseline_scores, margin=0.05)
    print(f"Verdict: {verdict}")

    exp_id = record_experiment(session, change, t1, verdict,
                               tier3_result=None, commit_hash=None,
                               baseline=session.baseline_scores)
    print(f"Recorded: {exp_id}")


if __name__ == "__main__":
    main()
