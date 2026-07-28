"""Round 2: combine top Q2 winners.

Best single-change deltas (Round 1):
  #1 max_holdings=3            → delta=+0.255
  #2 max_windows=6 + rd=30     → delta=+0.188
  #3 sell_threshold=2.0        → delta=+0.166
  #4 stop_loss=3%              → delta=+0.110
  #5 max_windows=4             → delta=+0.103

Round 2 combines these to check if effects stack.
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
        "desc": "26Q2-R2: max_holdings=3 + stop_loss=3%",
        "hparam_overrides": {"max_holdings": 3, "strategy_settings": {"stop_loss_pct": 0.03}},
        "max_windows": 2,
    },
    {
        "desc": "26Q2-R2: max_holdings=3 + max_windows=6",
        "hparam_overrides": {"max_holdings": 3},
        "max_windows": 6,
    },
    {
        "desc": "26Q2-R2: max_holdings=3 + stop_loss=3% + max_windows=6",
        "hparam_overrides": {"max_holdings": 3, "strategy_settings": {"stop_loss_pct": 0.03}},
        "max_windows": 6,
    },
    {
        "desc": "26Q2-R2: max_holdings=3 + sell_threshold=2.0",
        "hparam_overrides": {"max_holdings": 3, "strategy_settings": {"sell_threshold": 2.0}},
        "max_windows": 2,
    },
    {
        "desc": "26Q2-R2: max_holdings=2 (extreme concentration)",
        "hparam_overrides": {"max_holdings": 2},
        "max_windows": 2,
    },
    {
        "desc": "26Q2-R2: stop_loss=2% (extreme tight)",
        "hparam_overrides": {"strategy_settings": {"stop_loss_pct": 0.02}},
        "max_windows": 2,
    },
    {
        "desc": "26Q2-R2: full stack (mh=3, sl=3%, mw=6, st=2.0)",
        "hparam_overrides": {
            "max_holdings": 3,
            "strategy_settings": {"stop_loss_pct": 0.03, "sell_threshold": 2.0},
        },
        "max_windows": 6,
    },
]


def main():
    session = load_session(
        index="399303.SZ", version="v15", oos_months=6,
        eval_start=datetime(2026, 4, 1),
        eval_end=datetime(2026, 6, 30),
    )

    print("=== ATTN BASELINE (26 Q2, R2) ===")
    compute_baseline(session, seeds=[42, 123, 2024],
                     max_windows=2, backend="attention", margin=0.05)
    print(f"Baseline: {session.baseline_scores}")

    results = []
    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n{'='*60}")
        print(f"R2 EXPERIMENT {i+1}/{len(EXPERIMENTS)}: {exp['desc']}")
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
            "desc": exp["desc"], "exp_id": exp_id,
            "median_score": t1["median_score"],
            "verdict": verdict["verdict"],
            "delta": verdict.get("delta", 0),
        })

    print(f"\n\n{'='*60}")
    print("R2 SUMMARY — 26 Q2 Combined Experiments")
    print(f"{'='*60}")
    print(f"Baseline median RDD: {session.baseline_scores['median_score']:.4f}")
    for r in sorted(results, key=lambda x: -x.get("delta", 0)):
        v = r.get("verdict", "?")
        d = r.get("delta", 0)
        m = r.get("median_score", 0)
        symbol = "✓" if v == "keep" else "✗"
        print(f"  {symbol} delta={d:+.4f} median={m:.4f}  {r['desc']}")


if __name__ == "__main__":
    main()
