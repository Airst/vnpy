"""
风格中性标签 R5 配对验证 — 生产双池, 26-04~07 定向

背景 (四轮证据链):
- R1 多地平线融合: Tier-3 REJECT (目标区间反转 -11.5pp)
- R2 confirm/2h/path: 双池 Tier-1 噪声级, 关闭
- R3 vol_scaled: 唯一让 26-04~07 转正的变体 (+21.9pp), 但除法钝器
  摧毁反转 alpha, Sharpe 1.36→0.71 腰斩
- R4 半剂量插值 (vol_sqrt/vol_blend): Tier-1 弱 WIN → Tier-3 REJECT,
  证明剂量插值救不回来

R5 假设: vol_scaled 转正的机理是削掉了标签里的风格暴露 (26q2 高波
小票崩、低波大票稳), 病因是风格轮动而非收益形态。外科手术做法 =
组内排名中性化: 不奖励"押对风格", 只保留风格内相对强弱, alpha 完整。

R5 设计 (4 配置, 因子完全相同、仅标签不同):
- 5d:            生产基线; 重跑保证严格配对
- size_neutral:  大/小市值组内各自 cs_rank
- vol_neutral:   高/低波动 (20d 中位数分组) 组内各自 cs_rank
- style_neutral: size × vol 2×2 四组内各自 cs_rank
- 各 3 seeds × 3 窗 × attention, 池=000852.SH,399303.SZ
- 判定: median>0 且 ≥2/3 为正且 seed-42 delta>0 → 升级 Tier-3
- 结果增量写 log/label_rebound_paired_r5.jsonl

用法:
  /home/airst/Workspace/.venv/bin/python scripts/label_rebound_paired_r5.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
from datetime import datetime
from core.alpha.research_runner import load_session, _run_single_seed, DEFAULT_SEEDS

OUT = "log/label_rebound_paired_r5.jsonl"
EVAL_START = datetime(2026, 4, 1)
INDEX = "000852.SH,399303.SZ"
MODES = ["5d", "size_neutral", "vol_neutral", "style_neutral"]


def load_done() -> dict:
    """从 jsonl 恢复已完成的 (mode, seed) — 支持崩溃后断点续跑。"""
    done = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("round") == "r5" and "seed" in rec:
                    done.setdefault(rec["label_mode"], {})[rec["seed"]] = rec["total_return"]
    return done


def run_mode(label_mode: str, done: dict) -> dict:
    rets = dict(done.get(label_mode, {}))
    todo = [s for s in DEFAULT_SEEDS if s not in rets]
    if not todo:
        print(f"[resume] {label_mode} 全部完成, 跳过: {rets}")
        return rets
    print(f"\n{'='*60}\n=== label_mode={label_mode} (pool={INDEX}) ===\n{'='*60}")
    session = load_session(index=INDEX, version="v15",
                           eval_start=EVAL_START,
                           calc_kwargs={"label_mode": label_mode})
    lbl = session.factor_df["label"]
    print(f"[label check] rows={len(lbl)}, null_ratio={lbl.null_count()/len(lbl):.4f}, "
          f"min={lbl.min():.3f}, max={lbl.max():.3f}, mean={lbl.mean():.3f}")

    for seed in todo:
        print(f"--- mode={label_mode} seed={seed} ({datetime.now().strftime('%H:%M:%S')}) ---")
        r = _run_single_seed(
            session, session.factor_df, seed, f"lr5_{label_mode}",
            backend="attention", max_windows=3,
        )
        total_ret = float(r["stats"].get("total_return", 0.0) or 0.0)
        rets[seed] = total_ret
        rec = {"round": "r5", "pool": INDEX, "label_mode": label_mode, "seed": seed,
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
    done = load_done()
    results = {m: run_mode(m, done) for m in MODES}

    print("\n=== R5 PAIRED VERDICT (dual pool, 26-04~07 total_return, vs 5d) ===")
    base = results["5d"]
    summary = {"round": "r5", "pool": INDEX,
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
