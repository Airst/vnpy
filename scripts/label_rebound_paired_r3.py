"""
波动率缩放标签族 R3 配对验证 — 生产双池, 26-04~07 定向

背景: R1/R2 共 4 个地平线/路径变体在生产宇宙上无一稳健优于 5d 基线
(见 docs/loop/verification_log.md 07-26/07-27 条目), 地平线方向关闭。
R3 换标签族: 不改"看多远"(仍 5 日, 无额外 NaN 损失), 改"奖励什么风险形态"。

R3 设计 (3 配置, 因子完全相同、仅标签不同):
- 5d:         生产基线 (5日 beta-neutral 终点收益 rank); 重跑保证严格配对
- vol_scaled: 5日超额收益 / 过去20日波动率 — 偏好单位风险收益高的股票,
              直接针对 26Q2 高波动杀跌环境
- fwd_sharpe: 5日超额收益 / 未来5日实现波动率 — 奖励平稳上涨、惩罚锯齿
- 各 3 seeds × 3 窗 × attention, 池=000852.SH,399303.SZ
- 回测窗口: 2026-04-01 → 最新, 主指标 total_return
- 判定: 配对 delta median > 0 且 ≥2/3 seeds 为正 → WIN; 且 seed-42 delta
  须为正才考虑升级 Tier-3 (R2 教训: 双池 seed-42 ≈ Tier-3 子区间代理)
- 结果增量写 log/label_rebound_paired_r3.jsonl

用法:
  /home/airst/Workspace/.venv/bin/python scripts/label_rebound_paired_r3.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
from datetime import datetime
from core.alpha.research_runner import load_session, _run_single_seed, DEFAULT_SEEDS

OUT = "log/label_rebound_paired_r3.jsonl"
EVAL_START = datetime(2026, 4, 1)
INDEX = "000852.SH,399303.SZ"
MODES = ["5d", "vol_scaled", "fwd_sharpe"]


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
            session, session.factor_df, seed, f"lr3_{label_mode}",
            backend="attention", max_windows=3,
        )
        total_ret = float(r["stats"].get("total_return", 0.0) or 0.0)
        rets[seed] = total_ret
        rec = {"round": "r3", "pool": INDEX, "label_mode": label_mode, "seed": seed,
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

    print("\n=== R3 PAIRED VERDICT (dual pool, 26-04~07 total_return, vs 5d) ===")
    base = results["5d"]
    summary = {"round": "r3", "pool": INDEX,
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
