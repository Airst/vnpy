"""
R5x 扩种子消歧 — 5d vs vol_neutral, 补 seeds 7/777 至 5 seeds

背景: R5 三变体全为 WEAK-WIN (seed42<0 不升级), 但基线 seed42=-6.86%
是明显离群 (另两 seed -27.9/-19.1%), vol_neutral 两个 WIN 幅度 (+22.5/
+21.6pp) 为整个战役最强。扩到 5 seeds 判断 seed42 基线是否离群抽样。

判定 (5 seeds): median>0 且 ≥4/5 为正 → 升级 Tier-3 (替代 seed-42 单点门禁)
结果追加写 log/label_rebound_paired_r5.jsonl (round=r5x)

用法:
  /home/airst/Workspace/.venv/bin/python scripts/label_rebound_paired_r5x.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
from datetime import datetime
from core.alpha.research_runner import load_session, _run_single_seed

OUT = "log/label_rebound_paired_r5.jsonl"
EVAL_START = datetime(2026, 4, 1)
INDEX = "000852.SH,399303.SZ"
MODES = ["5d", "vol_neutral"]
EXTRA_SEEDS = [7, 777]


def main():
    # 恢复 R5 已有结果 (3 seeds) + R5x 已跑部分
    rets = {m: {} for m in MODES}
    with open(OUT) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("round") in ("r5", "r5x") and "seed" in rec \
                    and rec["label_mode"] in rets:
                rets[rec["label_mode"]][rec["seed"]] = rec["total_return"]

    for m in MODES:
        todo = [s for s in EXTRA_SEEDS if s not in rets[m]]
        if not todo:
            print(f"[resume] {m} 扩种子已完成")
            continue
        print(f"\n=== label_mode={m} extra seeds {todo} ===")
        session = load_session(index=INDEX, version="v15",
                               eval_start=EVAL_START,
                               calc_kwargs={"label_mode": m})
        for seed in todo:
            print(f"--- mode={m} seed={seed} ({datetime.now().strftime('%H:%M:%S')}) ---")
            r = _run_single_seed(session, session.factor_df, seed,
                                 f"lr5x_{m}", backend="attention", max_windows=3)
            total_ret = float(r["stats"].get("total_return", 0.0) or 0.0)
            rets[m][seed] = total_ret
            rec = {"round": "r5x", "pool": INDEX, "label_mode": m, "seed": seed,
                   "total_return": total_ret, "score": r["score"],
                   "sharpe": r["sharpe"], "max_drawdown": r["max_drawdown"],
                   "ts": datetime.now().isoformat(timespec="seconds")}
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"mode={m} seed={seed}: TotalRet={total_ret:.2f}%")
        del session

    print("\n=== R5x VERDICT (5 seeds, vol_neutral vs 5d) ===")
    seeds = sorted(rets["5d"].keys() & rets["vol_neutral"].keys())
    deltas = {s: rets["vol_neutral"][s] - rets["5d"][s] for s in seeds}
    for s in seeds:
        print(f"seed {s}: 5d={rets['5d'][s]:.2f}%  vol_neutral={rets['vol_neutral'][s]:.2f}%  "
              f"delta={deltas[s]:+.2f}pp")
    vals = sorted(deltas.values())
    med = vals[len(vals) // 2]
    n_pos = sum(1 for d in deltas.values() if d > 0)
    verdict = ("UPGRADE Tier-3" if (med > 0 and n_pos >= 4)
               else "5d HOLD (信号不稳健)")
    print(f"median delta={med:+.2f}pp, {n_pos}/{len(seeds)} positive → {verdict}")
    summary = {"round": "r5x", "pool": INDEX, "n_seeds": len(seeds),
               "median_delta": med, "n_positive": n_pos, "verdict": verdict,
               "deltas": {str(k): v for k, v in deltas.items()}}
    with open(OUT, "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
