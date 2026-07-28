"""
波动率缩放半剂量 R4 配对验证 — 生产双池, 26-04~07 定向

背景: R3 vol_scaled (满缩放 p=1) Tier-1 WIN → Tier-3 子区间首次转正 (+21.9pp)
但 Sharpe 1.36→0.71 腰斩, 不采纳。信号真实、剂量过猛。R4 寻找中间点。

R4 设计 (3 配置, 因子完全相同、仅标签不同):
- 5d:        生产基线; 重跑保证严格配对
- vol_sqrt:  5日超额收益 / √(20日波动率) — p=0.5 幂插值
- vol_blend: 0.5*rank(原始超额) + 0.5*rank(满缩放) — rank 空间线性插值
- 各 3 seeds × 3 窗 × attention, 池=000852.SH,399303.SZ
- 判定: median>0 且 ≥2/3 为正且 seed-42 delta>0 → 升级 Tier-3
  (Tier-3 判定已修订: RDD 不受损 + 子区间改善 + Sharpe 不显著受损)
- 结果增量写 log/label_rebound_paired_r4.jsonl

用法:
  /home/airst/Workspace/.venv/bin/python scripts/label_rebound_paired_r4.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
from datetime import datetime
from core.alpha.research_runner import load_session, _run_single_seed, DEFAULT_SEEDS

OUT = "log/label_rebound_paired_r4.jsonl"
EVAL_START = datetime(2026, 4, 1)
INDEX = "000852.SH,399303.SZ"
MODES = ["5d", "vol_sqrt", "vol_blend"]


def run_mode(label_mode: str) -> dict:
    print(f"\n{'='*60}\n=== label_mode={label_mode} (pool={INDEX}) ===\n{'='*60}")
    session = load_session(index=INDEX, version="v15",
                           eval_start=EVAL_START,
                           calc_kwargs={"label_mode": label_mode})
    lbl = session.factor_df["label"]
    print(f"[label check] rows={len(lbl)}, null_ratio={lbl.null_count()/len(lbl):.4f}, "
          f"min={lbl.min():.3f}, max={lbl.max():.3f}, mean={lbl.mean():.3f}")

    rets = {}
    for seed in DEFAULT_SEEDS:
        print(f"--- mode={label_mode} seed={seed} ({datetime.now().strftime('%H:%M:%S')}) ---")
        r = _run_single_seed(
            session, session.factor_df, seed, f"lr4_{label_mode}",
            backend="attention", max_windows=3,
        )
        total_ret = float(r["stats"].get("total_return", 0.0) or 0.0)
        rets[seed] = total_ret
        rec = {"round": "r4", "pool": INDEX, "label_mode": label_mode, "seed": seed,
               "total_return": total_ret, "score": r["score"],
               "sharpe": r["sharpe"], "max_drawdown": r["max_drawdown"],
               "ts": datetime.now().isoformat(timespec="seconds")}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"mode={label_mode} seed={seed}: TotalRet={total_ret:.2f}% "
              f"RDD={r['score']:.3f} Sharpe={r['sharpe']:.3f} MaxDD={r['max_drawdown']:.2f}%")
    del session
    return rets


def main():
    results = {m: run_mode(m) for m in MODES}

    print("\n=== R4 PAIRED VERDICT (dual pool, 26-04~07 total_return, vs 5d) ===")
    base = results["5d"]
    summary = {"round": "r4", "pool": INDEX,
               "eval_window": "2026-04-01 ~ latest", "results": results}
    for m in MODES[1:]:
        deltas = {s: results[m][s] - base[s] for s in DEFAULT_SEEDS}
        for s in DEFAULT_SEEDS:
            print(f"[{m}] seed {s}: 5d={base[s]:.2f}%  {m}={results[m][s]:.2f}%  delta={deltas[s]:+.2f}pp")
        vals = sorted(deltas.values())
        med = vals[len(vals) // 2]
        n_pos = sum(1 for d in deltas.values() if d > 0)
        seed42_pos = deltas[42] > 0
        verdict = (f"{m} WIN" if (med > 0 and n_pos >= 2 and seed42_pos)
                   else f"{m} WEAK-WIN (seed42<0, 不升级)" if (med > 0 and n_pos >= 2)
                   else "5d HOLD")
        print(f"[{m}] median delta={med:+.2f}pp, {n_pos}/3 positive, "
              f"seed42 {'+' if seed42_pos else '-'} → {verdict}")
        summary[m] = {"median_delta": med, "n_positive": n_pos,
                      "seed42_positive": seed42_pos, "verdict": verdict,
                      "deltas": {str(k): v for k, v in deltas.items()}}
    with open(OUT, "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
