"""
Auto-research: Model Layer Exploration (post-factor-saturation)

Per principle #20/#26: factor space is saturated (25 validated, 100 rejected).
Shift to model-layer improvements.

Hypotheses to test (attention backend, 2 windows × 3 seeds):
1. d_token=96 (more representation capacity per factor token)
2. d_ffn=256 (wider FFN in attention block)  
3. lr=0.0005 (slower learning, more careful optimization)
4. batch_size=1024 (smaller batches → more gradient steps → better generalization)
5. early_stop_rounds=60 (more patience before stopping)
6. d_token=128 + d_ffn=256 (wider model overall)
7. attn_dropout=0.20 (more regularization)
"""
import sys
import os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

from core.alpha.research_runner import (
    load_session, compute_baseline, tier1_quick_validate,
    variance_keep_or_revert, load_ledger, save_ledger, _next_exp_id,
    DEFAULT_SEEDS, DEFAULT_MARGIN
)
from core.alpha.research_loop import FactorChange
from datetime import datetime
import json

INDEX = "000852.SH,399303.SZ"
VERSION = "v15"
BACKEND = "attention"
MAX_WINDOWS = 2
MARGIN = 0.05

# Model-layer hypotheses
EXPERIMENTS = [
    {
        "desc": "model: d_token=96 (more capacity per factor)",
        "overrides": {"model_settings": {"d_token": 96}}
    },
    {
        "desc": "model: d_ffn=256 (wider FFN)",
        "overrides": {"model_settings": {"d_ffn": 256}}
    },
    {
        "desc": "model: lr=0.0005 (slower learning)",
        "overrides": {"model_settings": {"lr": 0.0005}}
    },
    {
        "desc": "model: batch_size=1024 (smaller batches)",
        "overrides": {"model_settings": {"batch_size": 1024}}
    },
    {
        "desc": "model: early_stop=60 (more patience)",
        "overrides": {"model_settings": {"early_stop_rounds": 60}}
    },
    {
        "desc": "model: d_token=128 + d_ffn=256 (wider model)",
        "overrides": {"model_settings": {"d_token": 128, "d_ffn": 256}}
    },
    {
        "desc": "model: attn_dropout=0.25 + ffn_dropout=0.25 (stronger reg)",
        "overrides": {"model_settings": {"attn_dropout": 0.25, "ffn_dropout": 0.25}}
    },
]

def run_experiment(session, exp_config, baseline_scores):
    """Run a single hparam experiment through Tier-1 + variance gate."""
    desc = exp_config["desc"]
    overrides = exp_config["overrides"]
    
    change = FactorChange(
        change_type="hparam",
        factors=[],
        desc=desc,
        hparam_overrides=overrides,
    )
    
    # Tier-1: 3 seeds × 2 windows × attention
    tier1 = tier1_quick_validate(
        session, change,
        seeds=DEFAULT_SEEDS,
        max_windows=MAX_WINDOWS,
        backend=BACKEND,
    )
    
    # Variance gate
    gate = variance_keep_or_revert(tier1, baseline_scores, margin=MARGIN)
    
    # Record
    ledger = load_ledger()
    exp_id = _next_exp_id(ledger)
    record = {
        "exp_id": exp_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "change_desc": desc,
        "change_type": "hparam",
        "factor_delta": list(overrides.get("model_settings", {}).keys()),
        "backend": BACKEND,
        "max_windows": MAX_WINDOWS,
        "seeds": DEFAULT_SEEDS,
        "seed_scores": tier1["seed_scores"],
        "median_score": tier1["median_score"],
        "spread": tier1["spread"],
        "baseline_score": baseline_scores["median_score"],
        "baseline_spread": baseline_scores["spread"],
        "baseline_seed_scores": baseline_scores["seed_scores"],
        "paired_deltas": gate.get("paired_deltas", {}),
        "n_positive": gate.get("n_positive", 0),
        "n_seeds": gate.get("n_seeds", 3),
        "sign_pass": gate.get("sign_pass", False),
        "paired": gate.get("paired", True),
        "margin": MARGIN,
        "delta_vs_baseline": gate.get("delta", 0.0),
        "verdict": gate["verdict"],
        "tier1_pass": gate["verdict"] == "keep",
        "tier3_result": None,
        "commit_hash": None,
        "note": gate.get("detail", ""),
    }
    ledger["experiments"].append(record)
    save_ledger(ledger)
    
    print(f"\n{'='*60}")
    print(f"[{exp_id}] {desc}")
    print(f"  Verdict: {gate['verdict'].upper()}")
    print(f"  Median delta: {gate.get('delta', 0):.4f} (margin={MARGIN})")
    print(f"  Seeds positive: {gate.get('n_positive',0)}/{gate.get('n_seeds',3)}")
    print(f"  Detail: {gate.get('detail','')}")
    print(f"{'='*60}\n")
    
    return gate["verdict"] == "keep", record


def main():
    print(f"[auto_research_model_layer] Starting model-layer exploration")
    print(f"  Backend: {BACKEND}, Windows: {MAX_WINDOWS}, Seeds: {DEFAULT_SEEDS}")
    print(f"  Index: {INDEX}, Version: {VERSION}")
    print(f"  Hypotheses: {len(EXPERIMENTS)}")
    print()
    
    # Load session (data + factors computed once, ~3 min)
    session = load_session(index=INDEX, version=VERSION)
    
    # Compute baseline (3 seeds × 2 windows × attention, ~30 min)
    print("\n[Phase 1] Computing baseline...")
    baseline = compute_baseline(session, seeds=DEFAULT_SEEDS, max_windows=MAX_WINDOWS, backend=BACKEND, margin=MARGIN)
    print(f"\n  Baseline: median={baseline['median_score']:.4f}, spread={baseline['spread']:.4f}")
    print(f"  Seed scores: {baseline['seed_scores']}")
    
    # Run experiments sequentially (GPU constraint: one at a time)
    print(f"\n[Phase 2] Running {len(EXPERIMENTS)} experiments...")
    results = []
    keeps = []
    
    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n{'#'*60}")
        print(f"# Experiment {i+1}/{len(EXPERIMENTS)}: {exp['desc']}")
        print(f"{'#'*60}")
        
        try:
            is_keep, record = run_experiment(session, exp, baseline)
            results.append(record)
            if is_keep:
                keeps.append(record)
                print(f"  >>> KEEP! Continuing with more experiments...")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"desc": exp["desc"], "error": str(e)})
    
    # Summary
    print(f"\n{'='*60}")
    print(f"[SUMMARY] {len(results)} experiments completed")
    print(f"  Keeps: {len(keeps)}")
    for k in keeps:
        print(f"    - {k['exp_id']}: {k['change_desc']} (delta={k['delta_vs_baseline']:.4f})")
    if not keeps:
        print("  No improvements found in this batch. Consider:")
        print("    - Different model architecture (e.g., 2-layer with residual)")
        print("    - Label engineering (different prediction horizon)")
        print("    - Data augmentation / sample weighting")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
