"""
research_runner — Auto-Research in-process orchestrator.

WHY THIS EXISTS
---------------
One full training+backtest validation cycle ≈ 1 hour (the MLP rolling-window
loop, 35 windows). Factor calc itself is ~10 sec — NOT the bottleneck. The user
falls back to the 1hr full retrain because the cheap `--max-windows 2` result is
untrustworthy (single seed, in-sample). This module makes the cheap path
TRUSTWORTHY so 10-min validation replaces 1hr:

  • data + factors load ONCE per session, reused across all 3 seeds (in-process)
  • multi-seed = ensemble_size=1 × 3 distinct seeds → 3 independent scores + spread
  • variance-threshold keep/discard: keep iff improvement > noise floor (principle #18)
  • OOS evaluation-holdout: Tier-1 scores in-sample only; Tier-3 measures OOS
  • ledger (experiments.json) + git-commit-on-keep → zero-cost revert
  • hard guardrails: #27 (≤3 removals, never bulk), --index mandatory, ≥3 seeds

Loop (driven by research_loop.py / the agent):
  load_session → compute_baseline → [propose change → tier0 → tier1 →
  variance_keep_or_revert → record → (if keep) commit] → ... → tier3 (human-gated)

Design modeled on gp_factor_miner (propose→evaluate→keep/reject→record with hard
IC/ICIR/direction gates + gp_factors.json atomic-save registry) and karpathy/
autoresearch (cheap-budget + scalar-metric + keep/discard + git-revert).

See docs/loop/auto_research.md for the steering contract.
"""
import os
import json
import gc
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median
from typing import Optional, List, Dict, Any

import polars as pl

from core.alpha.engine import AlphaEngine
from core.alpha.mlp_signals import MLPSignals
from core.selector import FundamentalSelector
from training import resolve_version_config

# --- constants mirroring gp_factor_miner._try_add_factor thresholds ---
GATE_MIN_IC = 0.02          # |mean IC| floor for a factor to be worth a train run
GATE_MIN_ICIR = 0.3         # |ICIR| floor
GATE_MIN_DIRECTION = 0.6   # fraction of windows whose IC sign matches overall
GATE_MAX_BULK_REMOVAL = 3   # principle #27: never remove >3 factors per experiment

DEFAULT_SEEDS = [42, 123, 2024]
DEFAULT_MARGIN = 0.05       # paired-seed delta scale (shared seed noise cancels; was 0.15 for the
                             # old independent-median gate, which calibration showed unusable).

EXPERIMENTS_PATH = "core/alpha/experiments.json"


# ============================================================================
# FactorChange + ResearchSession
# ============================================================================

@dataclass
class FactorChange:
    """A single proposed change to the factor set or model hyperparams."""
    change_type: str            # "remove" | "add" | "prune" | "hparam"
    factors: List[str] = field(default_factory=list)   # factor names involved
    desc: str = ""              # human-readable one-liner
    # For "add": pre-built factor_df that already includes the new factor(s).
    # For "remove"/"prune": the orchestrator drops `factors` columns from session.factor_df.
    # For "hparam": no factor change; `factors` carries e.g. ["backend=lgb"] style notes.
    candidate_factor_df: Optional[pl.DataFrame] = None
    # For "add": structured metrics of the NEW factors (computed by a quick re-calc),
    # used by tier0 to gate IC/ICIR/direction before spending a train run.
    candidate_metrics: Optional[Dict[str, Dict[str, float]]] = None
    # For "hparam": overrides passed to _run_single_seed (model_settings, retrain_days, max_holdings).
    hparam_overrides: Optional[Dict[str, Any]] = None


@dataclass
class ResearchSession:
    """Holds the expensive state (data + factors) loaded once and reused across seeds."""
    engine: AlphaEngine
    data_df: pl.DataFrame
    factor_df: pl.DataFrame                 # baseline factor set (current production)
    version: str
    index: str
    oos_start: datetime                     # OOS evaluation-holdout boundary
    eval_start: Optional[datetime] = None   # backtest window start for tier1/baseline
    eval_end: Optional[datetime] = None     # backtest window end for tier1/baseline
    baseline_factor_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    baseline_scores: Optional[Dict[str, Any]] = None   # filled by compute_baseline()


# ============================================================================
# Ledger I/O (mirrors gp_factor_miner registry: atomic os.replace, append-only)
# ============================================================================

def load_ledger() -> Dict:
    if not os.path.exists(EXPERIMENTS_PATH):
        return {"version": 1, "next_id": 1, "experiments": []}
    try:
        with open(EXPERIMENTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"version": 1, "next_id": 1, "experiments": []}


def save_ledger(ledger: Dict) -> None:
    """Atomic write (write .tmp then os.replace), mirroring gp_factor_miner.save_registry."""
    os.makedirs(os.path.dirname(EXPERIMENTS_PATH), exist_ok=True)
    tmp = EXPERIMENTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)
    os.replace(tmp, EXPERIMENTS_PATH)


def _next_exp_id(ledger: Dict) -> str:
    n = ledger.get("next_id", 1)
    ledger["next_id"] = n + 1
    return f"exp_{n:03d}"


# ============================================================================
# Session loading (data + factors computed ONCE)
# ============================================================================

def load_session(index: str, version: str = "v15", oos_months: int = 6,
                 gp_status_filter: Optional[List[str]] = None,
                 eval_start: Optional[datetime] = None,
                 eval_end: Optional[datetime] = None,
                 calc_kwargs: Optional[Dict[str, Any]] = None) -> ResearchSession:
    """Load data + compute baseline factors ONCE. Holds them on the session for reuse.

    `index` is MANDATORY (AGENTS.md GPU-OOM guardrail) — raises if None.
    `gp_status_filter`: None→["validated"] (production default); pass ["validated","testing"]
    to include GP testing factors (e.g. when validating a freshly-mined factor).
    `eval_start`/`eval_end`: restrict Tier-1/baseline backtest to this window (default: 2022-01-01 → oos_start).
    """
    if not index:
        raise ValueError("load_session: `index` is MANDATORY — full-market training GPU-OOMs (AGENTS.md).")

    version_config = resolve_version_config()
    if version not in version_config:
        raise ValueError(f"Unknown version '{version}'. Available: {list(version_config.keys())}")
    CalcClass, _description = version_config[version]

    if gp_status_filter is None:
        gp_status_filter = ["validated"]
    calculator = CalcClass(gp_status_filter=gp_status_filter, **(calc_kwargs or {}))

    selector = FundamentalSelector()
    _, latest_date = selector.get_data_range()
    last_trading_date = selector.get_last_trading_day() or datetime.now()
    end_date = last_trading_date.strftime("%Y-%m-%d")

    # NOTE: research_runner constructs its OWN engine (not run_training) so it controls
    # per-seed MLPSignals and skips CLI/download plumbing. mlp_signals here is a placeholder;
    # _run_single_seed builds fresh per-seed MLPSignals and calls generate_signals directly,
    # bypassing engine.mlp_signals (which holds a single instance).
    engine = AlphaEngine(
        factor_calculator=calculator,
        mlp_signals=MLPSignals(signal_name=f"scratch_{version}", force_retrain=True, max_windows=2),
        selector=selector,
        signal_name=f"scratch_{version}",
        start_date="2019-12-28",
        end_date=end_date,
        index_filter=index,
    )

    print(f"[research_runner] Loading data (once) for {version} index={index} ...")
    data_df = engine.load_data()
    print(f"[research_runner] Computing baseline factors (once) ...")
    factor_df = engine.calculate_factors(data_df)
    print(f"[research_runner] Baseline factor analysis ...")
    _, baseline_factor_metrics = engine.analyze_factor_performance(factor_df)

    oos_start = last_trading_date - timedelta(days=oos_months * 30)

    return ResearchSession(
        engine=engine,
        data_df=data_df,
        factor_df=factor_df,
        version=version,
        index=index,
        oos_start=oos_start,
        eval_start=eval_start,
        eval_end=eval_end,
        baseline_factor_metrics=baseline_factor_metrics,
    )


# ============================================================================
# Atomic unit: one (train → save signal → backtest) on a given factor_df + seed
# ============================================================================

def _run_single_seed(session: ResearchSession, factor_df: pl.DataFrame, seed: int,
                     name_suffix: str, backend: str = "lgb", max_windows: int = 2,
                     end_date_filter: Optional[datetime] = None,
                     max_holdings: int = 5,
                     hparam_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Train one MLPSignals(ensemble_size=1, seed) on `factor_df`, save a scratch signal, backtest it.

    - seed-suffixed signal_name isolates artifacts (lab.save_signal overwrites by name).
    - end_date_filter: if set, truncate factor_df to < this date (OOS evaluation-holdout for Tier-1).
    - bypasses engine.mlp_signals (builds its own instance) to avoid shared-state across seeds.
    - hparam_overrides: optional dict with keys like "model_settings" (dict merged into MLPSignals),
      "retrain_days" (int), "max_holdings" (int, overrides the parameter above).
    """
    from core.core_service import CoreService

    overrides = hparam_overrides or {}
    retrain_days = overrides.get("retrain_days", 45)
    bt_max_holdings = overrides.get("max_holdings", max_holdings)

    signal_name = f"ar_{session.version}_{name_suffix}_s{seed}"
    df = factor_df
    # When a custom eval window is set, use eval_end (not oos_start) as the training-data
    # cutoff, so predictions cover the eval window. Rolling loop ensures no per-window leakage.
    effective_end_filter = session.eval_end if session.eval_end is not None else end_date_filter
    if effective_end_filter is not None:
        df = factor_df.filter(pl.col("datetime") < effective_end_filter)

    mlp = MLPSignals(
        signal_name=signal_name,
        force_retrain=True,        # remove any prior scratch signal of this name
        ensemble_size=1,           # single-member → one independent signal for this seed
        max_windows=max_windows,
        model_backend=backend,
        seed=seed,
        retrain_days=retrain_days,
        vintage_ensemble=overrides.get("vintage_ensemble", 0),
        valid_len=overrides.get("valid_len", 100),
    )
    if "model_settings" in overrides:
        mlp.model_settings.update(overrides["model_settings"])

    signal_df = mlp.generate_signals(df, session.engine.start_date, session.engine.lab)
    # Direct parquet write (no date-merge) — scratch signals are per-experiment, isolated by name.
    session.engine.lab.save_signal(signal_name, signal_df)

    core = CoreService()
    bt_start = session.eval_start if session.eval_start is not None else datetime.strptime("2022-01-01", "%Y-%m-%d")
    if session.eval_end is not None:
        bt_end = session.eval_end
    elif end_date_filter is not None:
        bt_end = end_date_filter
    else:
        bt_end = datetime.strptime(session.engine.end_date, "%Y-%m-%d")
    bt_setting = {"max_holdings": str(bt_max_holdings), "signal_name": signal_name}
    if "strategy_settings" in overrides:
        bt_setting.update({k: str(v) for k, v in overrides["strategy_settings"].items()})
    result = core.run_backtest(
        strategy_name="MultiFactorStrategy",
        start=bt_start,
        end=bt_end,
        setting=bt_setting,
    )
    stats = result.get("statistics", {}) if isinstance(result, dict) else {}
    score = stats.get("return_drawdown_ratio", 0.0) or 0.0

    out = {
        "seed": seed,
        "signal_name": signal_name,
        "score": float(score),
        "sharpe": float(stats.get("sharpe_ratio", 0.0) or 0.0),
        "max_drawdown": float(stats.get("max_ddpercent", 0.0) or 0.0),
        "stats": stats,
        "daily_data": result.get("daily_data", []) if isinstance(result, dict) else [],
    }

    # Free GPU/CPU between seeds (mirrors mlp_signals memory cleanup).
    del mlp, signal_df, result, core
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    return out


def _summarize_seeds(seed_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [r["score"] for r in seed_runs]
    return {
        "seed_scores": scores,
        "median_score": float(median(scores)) if scores else 0.0,
        "spread": float(max(scores) - min(scores)) if scores else 0.0,
        "sharpe_median": float(median([r["sharpe"] for r in seed_runs])) if seed_runs else 0.0,
    }


# ============================================================================
# Baseline (the de-risk anchor)
# ============================================================================

def compute_baseline(session: ResearchSession, seeds: List[int] = None,
                     max_windows: int = 2, backend: str = "lgb",
                     margin: float = DEFAULT_MARGIN) -> Dict[str, Any]:
    """Run 3 seeds on the CURRENT (unmodified) factor set, in-sample (OOS held out).
    Cached on session.baseline_scores. The candidate must beat this by more than the noise."""
    seeds = seeds or DEFAULT_SEEDS
    print(f"[research_runner] === BASELINE: {len(seeds)} seeds × {max_windows} windows × {backend} (in-sample) ===")
    runs = []
    for s in seeds:
        print(f"[research_runner] baseline seed={s} ...")
        runs.append(_run_single_seed(session, session.factor_df, s, "baseline",
                                     backend=backend, max_windows=max_windows,
                                     end_date_filter=session.oos_start))
    summary = _summarize_seeds(runs)
    summary.update({"seeds": seeds, "backend": backend, "max_windows": max_windows, "margin": margin})
    session.baseline_scores = summary
    print(f"[research_runner] baseline: median={summary['median_score']:.3f} spread={summary['spread']:.3f}")
    return summary


# ============================================================================
# Tier-0: cheap factor-side gate (seconds, no training)
# ============================================================================

def tier0_factor_gate(session: ResearchSession, change: FactorChange) -> Dict[str, Any]:
    """Cheap pre-filter using baseline factor metrics (no training). Advisory + hard #27 guardrail.

    - remove/prune: enforce ≤ GATE_MAX_BULK_REMOVAL (#27); flag if removing a high-IC factor.
    - add: gate candidate_metrics (IC/ICIR/direction) — mirror gp_factor_miner._try_add_factor.
    """
    if change.change_type in ("remove", "prune"):
        if len(change.factors) > GATE_MAX_BULK_REMOVAL:
            return {"pass": False, "reason": f"bulk removal of {len(change.factors)} factors violates principle #27 (max {GATE_MAX_BULK_REMOVAL})"}
        flags = []
        for f in change.factors:
            m = session.baseline_factor_metrics.get(f, {})
            ic = abs(m.get("ic", 0.0))
            if ic > 0.15:
                flags.append(f"{f} has |IC|={ic:.3f} (high-value — removal risky)")
        return {"pass": True, "reason": "ok", "flags": flags}

    if change.change_type == "add":
        cm = change.candidate_metrics or {}
        if not cm:
            return {"pass": False, "reason": "add requires candidate_metrics (compute via quick factor re-calc)"}
        failures = []
        for fname, m in cm.items():
            ic = abs(m.get("ic", 0.0))
            icir = abs(m.get("icir", 0.0))
            dr = m.get("direction_ratio", 0.0)
            if ic < GATE_MIN_IC:
                failures.append(f"{fname}: |IC|={ic:.4f} < {GATE_MIN_IC}")
            if icir < GATE_MIN_ICIR:
                failures.append(f"{fname}: |ICIR|={icir:.4f} < {GATE_MIN_ICIR}")
            if dr < GATE_MIN_DIRECTION:
                failures.append(f"{fname}: direction_ratio={dr:.2f} < {GATE_MIN_DIRECTION}")
        if failures:
            return {"pass": False, "reason": "; ".join(failures)}
        return {"pass": True, "reason": "ok"}

    # hparam: no factor gate (let Tier-1 decide)
    return {"pass": True, "reason": "hparam change — no factor-side gate"}


# ============================================================================
# Tier-1: quick multi-seed validation on lgb (minutes, in-sample, OOS held out)
# ============================================================================

def _build_experiment_factor_df(session: ResearchSession, change: FactorChange) -> pl.DataFrame:
    """Apply the change to a COPY of session.factor_df."""
    if change.change_type in ("remove", "prune"):
        present = [f for f in change.factors if f in session.factor_df.columns]
        return session.factor_df.drop(present) if present else session.factor_df
    if change.change_type == "add" and change.candidate_factor_df is not None:
        return change.candidate_factor_df
    return session.factor_df  # hparam or no-op


def tier1_quick_validate(session: ResearchSession, change: FactorChange,
                         seeds: List[int] = None, max_windows: int = 2,
                         backend: str = "lgb") -> Dict[str, Any]:
    """3 seeds × N windows × lgb on the MODIFIED factor set, in-sample (OOS held out)."""
    seeds = seeds or DEFAULT_SEEDS
    if session.baseline_scores is None:
        raise RuntimeError("call compute_baseline() before tier1_quick_validate()")
    exp_df = _build_experiment_factor_df(session, change)
    print(f"[research_runner] === TIER-1 '{change.desc}': {len(seeds)} seeds × {max_windows} windows × {backend} (in-sample) ===")
    runs = []
    overrides = change.hparam_overrides if change.change_type == "hparam" else None
    for s in seeds:
        print(f"[research_runner] tier1 seed={s} ...")
        runs.append(_run_single_seed(session, exp_df, s, "tier1",
                                     backend=backend, max_windows=max_windows,
                                     end_date_filter=session.oos_start,
                                     hparam_overrides=overrides))
    summary = _summarize_seeds(runs)
    summary.update({"change_type": change.change_type, "factors": change.factors,
                    "desc": change.desc, "backend": backend, "max_windows": max_windows,
                    "seeds": seeds})
    return summary


# ============================================================================
# Variance-threshold keep/discard (the core de-risking, principle #18)
# ============================================================================

def variance_keep_or_revert(candidate: Dict[str, Any], baseline: Dict[str, Any],
                            margin: float = 0.05) -> Dict[str, Any]:
    """Paired-seed keep/discard (principle #18 done right).

    Calibration showed return_drawdown_ratio is a ratio of two noisy quantities —
    its between-seed spread (~0.7-1.2) ≈ its median, so an independent median-vs-
    median comparison has no signal. BUT baseline and candidate run the SAME seeds
    ([42,123,2024]) and _run_single_seed is deterministic per seed, so the "seed 123
    is unlucky" variance is SHARED — it cancels in the per-seed delta. The relevant
    noise is the variance of the delta across seeds for an equivalent change, which
    is far smaller.

    Gate (paired + sign test):
        delta_s = score_cand(seed_s) - score_base(seed_s)
        keep iff  median(delta_s) > margin  AND  ≥2/3 of delta_s > 0

    The sign test (≥2/3 consistent direction) is the robust core; `margin` guards
    against a tiny-but-consistent delta being noise. Default margin 0.05 on the
    paired-delta scale (much smaller than the 0.15 independent-median margin, since
    shared seed noise cancels). For a truly-null change, same seed → same score →
    delta=0 exactly; residual delta spread comes from the change perturbing training
    in seed-correlated ways. Calibrate margin via a null-change run if needed.
    """
    cand_seeds = candidate.get("seeds", [])
    base_seeds = baseline.get("seeds", [])
    cand_map = dict(zip(cand_seeds, candidate.get("seed_scores", [])))
    base_map = dict(zip(base_seeds, baseline.get("seed_scores", [])))
    common = [s for s in cand_seeds if s in base_map] if cand_seeds else list(base_map.keys())

    if not common or len(candidate.get("seed_scores", [])) != len(base_seeds):
        # Fallback: independent median comparison (low power — flag it).
        med_c, med_b = candidate["median_score"], baseline["median_score"]
        delta = med_c - med_b
        keep = delta > margin
        return {"verdict": "keep" if keep else "revert", "delta": float(delta),
                "margin": float(margin), "paired": False,
                "detail": f"FALLBACK independent median: delta {delta:.4f} > margin {margin:.4f}? {keep}"}

    deltas = {s: cand_map[s] - base_map[s] for s in common}
    delta_vals = list(deltas.values())
    med_delta = median(delta_vals)
    n_pos = sum(1 for d in delta_vals if d > 0)
    sign_pass = n_pos >= (2 * len(delta_vals) + 2) // 3  # ≥2 of 3, ≥3 of 4-5, etc.
    keep = (med_delta > margin) and sign_pass
    return {
        "verdict": "keep" if keep else "revert",
        "delta": float(med_delta),
        "paired_deltas": {str(s): float(d) for s, d in deltas.items()},
        "n_positive": int(n_pos),
        "n_seeds": len(common),
        "sign_pass": bool(sign_pass),
        "margin": float(margin),
        "paired": True,
        "detail": (f"paired median_delta={med_delta:.4f} {'>' if med_delta>margin else '<='} margin {margin}; "
                   f"{n_pos}/{len(common)} seeds improved (need ≥{(2*len(common)+2)//3}) → {'KEEP' if keep else 'REVERT'}"),
    }


# ============================================================================
# Tier-3: full attention retrain + OOS measurement (human-gated)
# ============================================================================

def _rdd_from_daily(daily_data: List[Dict], start_dt: datetime, end_dt: datetime) -> Optional[float]:
    """Recompute return_drawdown_ratio for the [start_dt, end_dt] slice from backtest daily_data."""
    rows = []
    for d in daily_data:
        dt = d.get("date") or d.get("datetime")
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt.replace("Z", ""))
            except Exception:
                continue
        if dt is None:
            continue
        if start_dt <= dt <= end_dt:
            rows.append((dt, float(d.get("balance", 0.0)), float(d.get("drawdown", 0.0))))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r[0])
    balances = [r[1] for r in rows]
    peak = balances[0]
    max_dd = 0.0
    for b in balances:
        peak = max(peak, b)
        dd = (peak - b) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    total_return = (balances[-1] - balances[0]) / balances[0] if balances[0] > 0 else 0.0
    if max_dd <= 0:
        return None
    return total_return / max_dd


def tier3_full_validate(session: ResearchSession, change: FactorChange,
                        human_approved: bool = False, seeds: List[int] = None
                        ) -> Dict[str, Any]:
    """Full attention retrain (35 windows) + OOS measurement. REQUIRES human sign-off."""
    if not human_approved:
        raise PermissionError("Tier-3 requires explicit human sign-off (human_approved=True).")
    seeds = seeds or DEFAULT_SEEDS
    exp_df = _build_experiment_factor_df(session, change)
    print(f"[research_runner] === TIER-3 '{change.desc}': {len(seeds)} seeds × FULL windows × attention (incl. OOS) ===")
    runs = []
    for s in seeds:
        print(f"[research_runner] tier3 seed={s} ... (long)")
        runs.append(_run_single_seed(session, exp_df, s, "tier3",
                                     backend="attention", max_windows=0,
                                     end_date_filter=None))  # full range incl. OOS
    summary = _summarize_seeds(runs)
    # OOS slice from the first seed's daily_data (representative; full period covers OOS).
    oos = _rdd_from_daily(runs[0]["daily_data"], session.oos_start,
                          datetime.strptime(session.engine.end_date, "%Y-%m-%d"))
    summary["oos_score"] = float(oos) if oos is not None else None
    summary["change_type"] = change.change_type
    summary["factors"] = change.factors
    summary["desc"] = change.desc
    return summary


# ============================================================================
# Ledger recording + verification_log distillation
# ============================================================================

def record_experiment(session: ResearchSession, change: FactorChange,
                      tier1_result: Dict[str, Any], verdict: Dict[str, Any],
                      tier3_result: Optional[Dict[str, Any]] = None,
                      commit_hash: Optional[str] = None,
                      baseline: Optional[Dict[str, Any]] = None) -> str:
    """Append one structured experiment to experiments.json (atomic). Returns exp_id.

    On a Tier-3 oos_passed, distills a row into docs/loop/verification_log.md.
    """
    ledger = load_ledger()
    exp_id = _next_exp_id(ledger)
    b = baseline or session.baseline_scores or {}
    entry = {
        "exp_id": exp_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "change_desc": change.desc,
        "change_type": change.change_type,
        "factor_delta": change.factors,
        "backend": tier1_result.get("backend"),
        "max_windows": tier1_result.get("max_windows"),
        "seeds": tier1_result.get("seeds"),
        "seed_scores": tier1_result.get("seed_scores"),
        "median_score": tier1_result.get("median_score"),
        "spread": tier1_result.get("spread"),
        "baseline_score": b.get("median_score"),
        "baseline_spread": b.get("spread"),
        "baseline_seed_scores": b.get("seed_scores"),
        # Paired-seed gate (principle #18, calibrated): per-seed deltas cancel shared seed noise.
        "paired_deltas": verdict.get("paired_deltas"),
        "n_positive": verdict.get("n_positive"),
        "n_seeds": verdict.get("n_seeds"),
        "sign_pass": verdict.get("sign_pass"),
        "paired": verdict.get("paired"),
        "margin": verdict.get("margin"),
        "delta_vs_baseline": verdict.get("delta"),
        "verdict": verdict.get("verdict"),
        "tier1_pass": verdict.get("verdict") == "keep",
        "tier3_result": {
            "median_score": tier3_result.get("median_score") if tier3_result else None,
            "oos_score": tier3_result.get("oos_score") if tier3_result else None,
            "spread": tier3_result.get("spread") if tier3_result else None,
        } if tier3_result else None,
        "commit_hash": commit_hash,
        "note": verdict.get("detail"),
    }
    ledger["experiments"].append(entry)
    save_ledger(ledger)
    print(f"[research_runner] recorded {exp_id}: {verdict.get('verdict')} ({verdict.get('detail')})")

    # Distill to human-readable verification_log.md only on a final OOS pass.
    if tier3_result and tier3_result.get("oos_score") is not None and commit_hash:
        _distill_to_verification_log(entry, tier3_result, commit_hash)
    return exp_id


def _distill_to_verification_log(entry: Dict, tier3_result: Dict, commit_hash: str) -> None:
    """Append a free-form row to docs/loop/verification_log.md (time-reverse, top)."""
    log_path = "docs/loop/verification_log.md"
    ts = entry["timestamp"][:10]
    header = f"## {ts} {entry['exp_id']} — {entry['change_desc']}"
    block = (
        f"\n{header}\n"
        f"- **基线**：收益回撤比 {entry.get('baseline_score')}\n"
        f"- **本次(Tier-1, lgb, in-sample)**：收益回撤比 {entry.get('median_score')} (spread {entry.get('spread')})\n"
        f"- **Tier-3(attention, OOS)**：收益回撤比 {tier3_result.get('oos_score')}\n"
        f"- **判定**：{'通过' if tier3_result.get('oos_score', 0) > (entry.get('baseline_score') or 0) else 'OOS未达标'}\n"
        f"- **结论**：{'保留' if tier3_result.get('oos_score', 0) > (entry.get('baseline_score') or 0) else '回退'} (auto-research)\n"
        f"- **去向**：experiments.json {entry['exp_id']}\n"
        f"- **关联**：{entry['exp_id']}, commit {commit_hash}\n"
    )
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        existing = ""
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                existing = f.read()
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(block + existing)
    except Exception as e:
        print(f"[research_runner] verification_log distillation failed (non-fatal): {e}")
