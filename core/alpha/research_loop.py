"""
research_loop — Auto-Research agent entrypoint.

The programmatic loop the Claude agent (or a human) drives. It reads the steering
contract (docs/loop/auto_research.md) and the 27 research principles
(docs/knowledge/research_principles.md, mirrored into the LLM prompt via
knowledge_base.build_criteria_list), then for each proposed FactorChange runs:

  tier0 (cheap factor gate) → tier1 (multi-seed quick validate, lgb, in-sample)
  → variance_keep_or_revert (delta > noise floor?) → record ledger →
  [if keep] git commit. Tier-3 (full attention + OOS) is human-gated.

The agent proposes changes; this module VALIDATES and decides keep/revert.
Multi-seed + variance threshold + OOS holdout are the guardrails that make
autonomy safe despite "high cost of being wrong" (overfitting + single-seed noise).

USAGE (CLI, single experiment):
  .venv/bin/python -m core.alpha.research_loop --index 399303.SZ --version v15 \
      --remove pool_size_x_regime
  .venv/bin/python -m core.alpha.research_loop --index 399303.SZ --baseline-only   # just measure baseline

The full autonomous loop is driven in-process by the agent importing:
  from core.alpha.research_loop import load_and_baseline, run_one_iteration, FactorChange
"""
import argparse
import subprocess
from typing import Optional, List

from core.alpha.research_runner import (
    FactorChange, ResearchSession,
    load_session, compute_baseline, tier0_factor_gate,
    tier1_quick_validate, variance_keep_or_revert,
    tier3_full_validate, record_experiment,
)


def load_and_baseline(index: str, version: str = "v15", oos_months: int = 6,
                      max_windows: int = 2, backend: str = "lgb",
                      margin: float = 0.15, seeds: Optional[List[int]] = None
                      ) -> ResearchSession:
    """Convenience: load session once + measure the baseline anchor."""
    session = load_session(index=index, version=version, oos_months=oos_months)
    compute_baseline(session, seeds=seeds, max_windows=max_windows, backend=backend, margin=margin)
    return session


def run_one_iteration(session: ResearchSession, change: FactorChange,
                      commit_on_keep: bool = False) -> dict:
    """One full Tier-0 → Tier-1 → keep/revert → record cycle. Returns the experiment summary.

    Tier-3 (full attention + OOS) is NOT run here — it's human-gated and called
    separately via run_tier3() only after the user signs off.
    """
    # 1. Tier-0: cheap factor-side gate (#27 guardrail + add-factor IC/ICIR/direction).
    t0 = tier0_factor_gate(session, change)
    if not t0["pass"]:
        print(f"[research_loop] TIER-0 REJECT: {t0['reason']}")
        return {"exp_id": None, "tier0": t0, "verdict": "rejected_at_tier0"}
    if t0.get("flags"):
        print(f"[research_loop] TIER-0 warnings: {t0['flags']}")

    # 2. Tier-1: multi-seed quick validate (lgb, in-sample, OOS held out).
    t1 = tier1_quick_validate(session, change,
                               seeds=session.baseline_scores["seeds"],
                               max_windows=session.baseline_scores["max_windows"],
                               backend=session.baseline_scores["backend"])

    # 3. Variance-threshold keep/discard (signal > noise, principle #18).
    verdict = variance_keep_or_revert(t1, session.baseline_scores,
                                      margin=session.baseline_scores["margin"])

    # 4. Record to ledger (and git-commit on keep if requested).
    commit_hash = None
    if verdict["verdict"] == "keep" and commit_on_keep:
        commit_hash = _git_commit(change)
    exp_id = record_experiment(session, change, t1, verdict,
                               tier3_result=None, commit_hash=commit_hash,
                               baseline=session.baseline_scores)

    print(f"[research_loop] {exp_id}: {verdict['verdict'].upper()} — {verdict['detail']}")
    return {"exp_id": exp_id, "tier0": t0, "tier1": t1, "verdict": verdict, "commit_hash": commit_hash}


def run_tier3(session: ResearchSession, change: FactorChange, exp_id: str,
              human_approved: bool = False) -> dict:
    """Human-gated full attention retrain + OOS measurement for a kept experiment."""
    t3 = tier3_full_validate(session, change, human_approved=human_approved)
    oos_pass = t3.get("oos_score") is not None and t3["oos_score"] > (session.baseline_scores["median_score"] or 0)
    print(f"[research_loop] TIER-3 OOS score={t3.get('oos_score')} → {'PASS' if oos_pass else 'FAIL'}")
    # Update ledger entry with tier3 result; commit/keep or note failure.
    commit_hash = _git_commit(change) if oos_pass else None
    record_experiment(session, change,
                      {"median_score": t3["median_score"], "spread": t3["spread"],
                       "seed_scores": t3["seed_scores"], "backend": "attention", "max_windows": 0,
                       "seeds": t3.get("seeds", [])},
                      {"verdict": "oos_passed" if oos_pass else "oos_failed",
                       "noise_floor": t3["spread"], "margin": 0.0,
                       "delta": (t3.get("oos_score") or 0) - (session.baseline_scores["median_score"] or 0),
                       "detail": f"OOS {t3.get('oos_score')} vs baseline {session.baseline_scores['median_score']}"},
                      tier3_result=t3, commit_hash=commit_hash, baseline=session.baseline_scores)
    return {"exp_id": exp_id, "tier3": t3, "oos_pass": oos_pass, "commit_hash": commit_hash}


def _git_commit(change: FactorChange) -> Optional[str]:
    """Commit the working tree on keep (zero-cost revert via git revert). Best-effort."""
    try:
        subprocess.run(["git", "add", "-A"], check=False)
        msg = f"auto-research keep: {change.change_type} {','.join(change.factors) or change.desc}"
        subprocess.run(["git", "commit", "-m", msg], check=False, capture_output=True)
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return out.stdout.strip() or None
    except Exception as e:
        print(f"[research_loop] git commit skipped (non-fatal): {e}")
        return None


# ============================================================================
# CLI (single-experiment driver, for testing/manual runs)
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Auto-Research loop driver")
    p.add_argument("--index", required=True, help="Index filter (MANDATORY, GPU-OOM guardrail). e.g. 399303.SZ")
    p.add_argument("--version", default="v15")
    p.add_argument("--oos-months", type=int, default=6)
    p.add_argument("--max-windows", type=int, default=2)
    p.add_argument("--backend", default="lgb", choices=["lgb", "attention", "tabnet"])
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--seeds", default="42,123,2024", help="comma-separated seeds")
    p.add_argument("--baseline-only", action="store_true", help="just measure baseline, no change")
    p.add_argument("--remove", default="", help="comma-separated factors to remove (≤3, principle #27)")
    p.add_argument("--commit-on-keep", action="store_true")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    session = load_and_baseline(
        index=args.index, version=args.version, oos_months=args.oos_months,
        max_windows=args.max_windows, backend=args.backend, margin=args.margin, seeds=seeds,
    )

    if args.baseline_only:
        print("[research_loop] baseline-only done.")
        return

    remove_factors = [f.strip() for f in args.remove.split(",") if f.strip()]
    if not remove_factors:
        print("[research_loop] no --remove specified; pass --baseline-only or --remove f1,f2")
        return

    change = FactorChange(change_type="remove", factors=remove_factors,
                          desc=f"remove {len(remove_factors)} factor(s): {','.join(remove_factors)}")
    run_one_iteration(session, change, commit_on_keep=args.commit_on_keep)


if __name__ == "__main__":
    main()
