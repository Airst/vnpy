"""
反弹趋势确认标签 R2 配对验证 — 生产双池 + 改良变体, 26-04~07 定向

背景: R1 (label_rebound_paired.py, 单池 399303.SZ) 中 rebound_avg/confirm 均
3/3 WIN, 但 rebound_avg 在 Tier-3 全量双池下 REJECT (目标区间 -11.5pp 反转,
RDD 3.22→2.01)。教训: Tier-1 必须直接在生产双池上跑, 砍掉单池→双池迁移假设。

R2 设计 (4 配置, 因子完全相同、仅标签不同):
- 5d:              生产基线 (5日 beta-neutral 终点收益 rank)
- rebound_confirm: R1 双 WIN 但未进 Tier-3 的 min 链变体 (双池重测)
- rebound_2h:      只融合 5/10 日 (0.60/0.40), 砍掉噪音/前视重叠最大的 20d
- rebound_path:    未来 10 日每日累计收益均值 (路径标签, 奖励持续走强)
- 各 3 seeds × 3 窗 × attention, 池=000852.SH,399303.SZ (与生产/Tier-3 一致)
- 回测窗口: 2026-04-01 → 最新, 主指标 total_return
- 判定: 配对 delta median > 0 且 ≥2/3 seeds 为正 → WIN (再升级 Tier-3)
- 结果增量写 log/label_rebound_paired_r2.jsonl

用法:
  /home/airst/Workspace/.venv/bin/python scripts/label_rebound_paired_r2.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
from datetime import datetime
from core.alpha.research_runner import load_session, _run_single_seed, DEFAULT_SEEDS

OUT = "log/label_rebound_paired_r2.jsonl"
EVAL_START = datetime(2026, 4, 1)
INDEX = "000852.SH,399303.SZ"  # 生产双池 (R1 教训: 不再用单池代理)
MODES = ["5d", "rebound_confirm", "rebound_2h", "rebound_path"]


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
            session, session.factor_df, seed, f"lr2_{label_mode}",
            backend="attention", max_windows=3,
        )
        total_ret = float(r["stats"].get("total_return", 0.0) or 0.0)
        rets[seed] = total_ret
        rec = {"round": "r2", "pool": INDEX, "label_mode": label_mode, "seed": seed,
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

    print("\n=== R2 PAIRED VERDICT (dual pool, 26-04~07 total_return, vs 5d) ===")
    base = results["5d"]
    summary = {"round": "r2", "pool": INDEX,
               "eval_window": "2026-04-01 ~ latest", "results": results}
    for m in MODES[1:]:
        deltas = {s: results[m][s] - base[s] for s in DEFAULT_SEEDS}
        for s in DEFAULT_SEEDS:
            print(f"[{m}] seed {s}: 5d={base[s]:.2f}%  {m}={results[m][s]:.2f}%  delta={deltas[s]:+.2f}pp")
        vals = sorted(deltas.values())
        med = vals[len(vals) // 2]
        n_pos = sum(1 for d in deltas.values() if d > 0)
        verdict = f"{m} WIN" if (med > 0 and n_pos >= 2) else "5d HOLD"
        print(f"[{m}] median delta={med:+.2f}pp, {n_pos}/3 positive → {verdict}")
        summary[m] = {"median_delta": med, "n_positive": n_pos, "verdict": verdict,
                      "deltas": {str(k): v for k, v in deltas.items()}}
    with open(OUT, "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
