"""Recalibrate baseline for V15.11 (25 validated factors).

Uses lgb backend, w=8 windows, seeds [42, 123, 2024] to match previous calibration.
Writes result to core/alpha_db/baseline_calibration.json.
"""
import sys
import json
from datetime import datetime as _dt
sys.path.insert(0, "/home/airst/Workspace/vnpy")

from core.alpha.research_runner import load_session, compute_baseline


def main():
    session = load_session(index="399303.SZ", version="v15", oos_months=6)
    print(f"=== BASELINE RECAL (V15.11, 25 factors, lgb, w=8) ===")
    compute_baseline(session, seeds=[42, 123, 2024],
                     max_windows=8, backend="lgb", margin=0.05)
    s = session.baseline_scores
    print(f"\nResult: {s}")

    cal = {
        "version": 3,
        "seeds": s["seeds"],
        "seed_scores": {str(sd): sc for sd, sc in zip(s["seeds"], s["seed_scores"])},
        "baseline_median": s["median_score"],
        "baseline_spread": s["spread"],
        "max_windows": s["max_windows"],
        "backend": s["backend"],
        "metric": "return_drawdown_ratio",
        "gate": "paired_seed_sign_test",
        "paired_margin": 0.05,
        "note": "V15.11: 25 validated factors (15 original + 10 new LLM GP factors from 2026-07-13). Uses lgb backend to match previous calibration methodology.",
        "calibrated_at": "2026-07-14T" + _dt.now().strftime("%H:%M:%S"),
    }
    out = "/home/airst/Workspace/vnpy/core/alpha_db/baseline_calibration.json"
    with open(out, "w") as f:
        json.dump(cal, f, ensure_ascii=False, indent=2)
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
