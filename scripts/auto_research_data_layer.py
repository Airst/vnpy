"""
Auto-research: Data Layer Exploration

Hypotheses:
1. Training window 500 days (vs 700): faster adaptation, enough regime diversity
2. 3-day beta-neutral label (vs 5-day): more responsive to short-term signals
3. Combined: 500-day window + 3-day label

Method: Full 35-window attention training per hypothesis (no Tier-1 shortcut —
learned from d_ffn=256 that 2-window quick validates are unreliable for 
fundamental data/architecture changes). Each run ~75 min.
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
from datetime import datetime
from core.alpha.research_runner import load_ledger, save_ledger, _next_exp_id

# We'll modify mlp_signals.py and v15_factor_calculator.py, run training.py,
# then revert. This script orchestrates the sequence.

def record_result(desc, rdd, sharpe, total_return, baseline_rdd=3.02, notes=""):
    ledger = load_ledger()
    exp_id = _next_exp_id(ledger)
    delta = rdd - baseline_rdd
    record = {
        "exp_id": exp_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "change_desc": desc,
        "change_type": "data_layer",
        "factor_delta": [],
        "backend": "attention",
        "max_windows": 35,
        "seeds": [42],
        "seed_scores": [rdd],
        "median_score": rdd,
        "spread": 0.0,
        "baseline_score": baseline_rdd,
        "baseline_spread": 0.0,
        "baseline_seed_scores": [baseline_rdd],
        "paired_deltas": {"42": delta},
        "n_positive": 1 if delta > 0 else 0,
        "n_seeds": 1,
        "sign_pass": delta > 0.05,
        "paired": True,
        "margin": 0.05,
        "delta_vs_baseline": delta,
        "verdict": "keep" if delta > 0.05 else "revert",
        "tier1_pass": None,
        "tier3_result": None,
        "commit_hash": None,
        "note": notes,
    }
    ledger["experiments"].append(record)
    save_ledger(ledger)
    print(f"  Recorded {exp_id}: verdict={'KEEP' if delta > 0.05 else 'REVERT'} (delta={delta:+.3f})")
    return exp_id, delta > 0.05

# Hypothesis 1: Training window 500 days
# Modify: mlp_signals.py lines with "700" → "500"
# This changes: train_start_idx calculation, minimum history requirement

print("="*60)
print("Data Layer Experiment 1: Training Window 500 days (vs 700)")
print("="*60)
print()
print("Modifying mlp_signals.py: 700 → 500 day window")
print("Then running full 35-window training...")
print()

# Apply modification
mlp_path = "core/alpha/mlp_signals.py"
with open(mlp_path, "r") as f:
    original_mlp = f.read()

modified_mlp = original_mlp
# Key changes: training window from 700 to 500
# Line ~155: if len(dates) < 750:  → 550
modified_mlp = modified_mlp.replace("if len(dates) < 750:", "if len(dates) < 550:")
# Line ~171-174: start_idx < 700 → 500, dates[700] → dates[500]
modified_mlp = modified_mlp.replace("if start_idx < 700:", "if start_idx < 500:")
modified_mlp = modified_mlp.replace('for 700-day training.")', 'for 500-day training.")')
modified_mlp = modified_mlp.replace("start_idx = 700", "start_idx = 500")
modified_mlp = modified_mlp.replace("print(f\"[MLPSignals] Adjusting start index to 700 (Date: {dates[700]})\")",
                                     "print(f\"[MLPSignals] Adjusting start index to 500 (Date: {dates[500]})\")")
# Line ~209-210: Previous 700 indices → 500, train_end_idx - 699 → -499
modified_mlp = modified_mlp.replace("# Define Training Window (Previous 700 indices)",
                                     "# Define Training Window (Previous 500 indices)")
modified_mlp = modified_mlp.replace("# 700 days total (0 to 699)",
                                     "# 500 days total (0 to 499)")
modified_mlp = modified_mlp.replace("train_end_idx - 699", "train_end_idx - 499")
# Window print line
modified_mlp = modified_mlp.replace("Window: Train [700 days pre", "Window: Train [500 days pre")

with open(mlp_path, "w") as f:
    f.write(modified_mlp)

print("Modification applied. Starting training...")

import subprocess
result = subprocess.run(
    ["/home/airst/Workspace/.venv/bin/python", "training.py", "-v15", "-t", "--index", "000852.SH,399303.SZ"],
    capture_output=False, cwd="/home/airst/Workspace/vnpy"
)

# Revert
with open(mlp_path, "w") as f:
    f.write(original_mlp)
print("Reverted mlp_signals.py to original (700-day window)")

# Check backtest result
import glob
bt_files = sorted(glob.glob("core/alpha_db/backtest/ashare_mlp_signal_v15_*.json"), key=os.path.getmtime)
if bt_files:
    with open(bt_files[-1]) as f:
        bt = json.load(f)
    stats = bt["statistics"]
    rdd = stats["return_drawdown_ratio"]
    sharpe = stats["sharpe_ratio"]
    total_ret = stats["total_return"]
    ann_ret = stats["annual_return"]
    max_dd = stats["max_ddpercent"]
    print(f"\n  Results (500-day window):")
    print(f"    RDD: {rdd:.3f}  Sharpe: {sharpe:.3f}  Return: {total_ret:.1f}%  Annual: {ann_ret:.1f}%  MaxDD: {max_dd:.1f}%")
    exp_id, is_keep = record_result(
        "data_layer: train_window=500d (vs 700d)", rdd, sharpe, total_ret,
        notes=f"500-day window for faster adaptation. Sharpe={sharpe:.3f}, Return={total_ret:.1f}%, MaxDD={max_dd:.1f}%"
    )
    if is_keep:
        print(f"\n  >>> KEEP! Training window 500 improves over 700!")
    else:
        print(f"\n  REVERT. 500-day window doesn't beat 700.")
else:
    print("  ERROR: No backtest file found")
