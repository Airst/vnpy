"""
反弹趋势确认标签配对验证 — 2026-04 ~ 2026-07 定向选股收益

背景（用户 2026-07-24 指令）: 实验修改标签，用未来 5/10/20 日走势构造"反弹趋势
确认"标签，快速验证 26 年 4~7 月的选股收益。前次单地平线实验（10d vs 5d 全
in-sample, 2026-07-19）结论 5d HOLD；本次改为多地平线融合 + 聚焦近期区间。

设计:
- 3 配置各自算因子（因子完全相同、仅标签不同）:
  - 5d:              生产基线 (5日 beta-neutral 终点收益 rank)
  - rebound_avg:     5/10/20 日超额收益 rank 加权融合 (0.40/0.35/0.25)
  - rebound_confirm: 三地平线 rank 取 min (最差地平线决定标签)
- 各 3 seeds × 3 窗 (retrain_days=45, 覆盖 ~2026-03 起) × attention, 回测 N=5
- 回测窗口: 2026-04-01 → 数据最新日, 主指标 total_return (区间选股收益)
- 判定: 配对 delta = total_return(变体) - total_return(5d)，
  median > 0 且 ≥2/3 seeds 为正 → 变体胜出（再考虑全量验证）
- 结果增量写 log/label_rebound_paired.jsonl

注意: 20 日标签在训练/测试窗口边界处的前视重叠比 5 日标签多 15 天，配对双方
框架处理一致，但长地平线一侧结果略偏乐观，结论需留安全边际。

用法:
  /home/airst/Workspace/.venv/bin/python scripts/label_rebound_paired.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
from datetime import datetime
from core.alpha.research_runner import load_session, _run_single_seed, DEFAULT_SEEDS

OUT = "log/label_rebound_paired.jsonl"
EVAL_START = datetime(2026, 4, 1)
MODES = ["5d", "rebound_avg", "rebound_confirm"]


def run_mode(label_mode: str) -> dict:
    print(f"\n{'='*60}\n=== label_mode={label_mode} ===\n{'='*60}")
    session = load_session(index="399303.SZ", version="v15",
                           eval_start=EVAL_START,
                           calc_kwargs={"label_mode": label_mode})
    # 标签健康检查: 非空比例（rebound 变体末尾 20 日应为 NaN）
    lbl = session.factor_df["label"]
    print(f"[label check] rows={len(lbl)}, null_ratio={lbl.null_count()/len(lbl):.4f}, "
          f"min={lbl.min():.3f}, max={lbl.max():.3f}, mean={lbl.mean():.3f}")

    rets = {}
    for seed in DEFAULT_SEEDS:
        print(f"--- mode={label_mode} seed={seed} ({datetime.now().strftime('%H:%M:%S')}) ---")
        r = _run_single_seed(
            session, session.factor_df, seed, f"lr_{label_mode}",
            backend="attention", max_windows=3,
        )
        total_ret = float(r["stats"].get("total_return", 0.0) or 0.0)
        rets[seed] = total_ret
        rec = {"label_mode": label_mode, "seed": seed,
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

    print("\n=== PAIRED VERDICT (26-04~07 total_return, vs 5d baseline) ===")
    base = results["5d"]
    summary = {"eval_window": "2026-04-01 ~ latest", "results": results}
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
