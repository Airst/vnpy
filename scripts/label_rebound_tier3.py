"""
标签变体 Tier-3 终验: 5d 基线 vs 候选 label_mode 同 seed 配对全量训练

历史: R1 rebound_avg (单池 Tier-1 3/3 WIN) 在此 REJECT (07-26, 目标区间 -11.5pp
反转)。R3 vol_scaled (双池 Tier-1 WIN +10.91/+3.92/-0.83pp, seed-42 门槛通过)
升级至此。每次两侧都重跑 (数据随交易日更新, 保证严格配对)。

与 scripts/minute_ssl_tier3.py 同尺:
- 每配置 load_session 一次 (因子相同仅标签不同), 生产双池 000852.SH,399303.SZ
- 同 seed=42, 全量 35 窗 (max_windows=0), backend=attention, 回测 2022-01-01→最新
- scratch 信号隔离 (ar_v15_lt3_*), 不动生产 run / 全局因子库
- 判定 (07-28 修订, 堵 vol_scaled 暴露的 RDD 口径漏洞): 全时段 RDD 不受损
  且 26-04~07 子区间改善 且 Sharpe 不显著受损 (delta > -0.10) → CANDIDATE
- 结果增量写 log/label_rebound_tier3.jsonl

用法: /home/airst/Workspace/.venv/bin/python scripts/label_rebound_tier3.py [candidate_mode]
      (默认候选 vol_scaled)
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")
import gc
import json
from datetime import datetime

OUT = "log/label_rebound_tier3.jsonl"
SUB_START = "2026-04-01"


def sub_period_return(daily_data: list) -> float:
    """26-04~07 子区间收益: 以 2026-04-01 前最后一日 balance 为基准."""
    if not daily_data:
        return 0.0
    base = None
    for d in daily_data:
        if d["date"] < SUB_START:
            base = d["balance"]
    if base is None or base <= 0:
        return 0.0
    return (daily_data[-1]["balance"] / base - 1) * 100


def run_config(label_mode: str) -> dict:
    from core.alpha.research_runner import load_session, _run_single_seed
    print(f"\n{'='*60}\n=== Tier-3 label_mode={label_mode} ({datetime.now().strftime('%H:%M:%S')}) ===\n{'='*60}")
    session = load_session(index="000852.SH,399303.SZ", version="v15",
                           calc_kwargs={"label_mode": label_mode})
    r = _run_single_seed(session, session.factor_df, 42, f"lt3_{label_mode}",
                         backend="attention", max_windows=0, end_date_filter=None)
    stats = r["stats"]
    rec = {
        "label_mode": label_mode, "seed": 42,
        "score_rdd": r["score"], "sharpe": r["sharpe"],
        "max_drawdown": r["max_drawdown"],
        "total_return": float(stats.get("total_return", 0.0) or 0.0),
        "annual_return": float(stats.get("annual_return", 0.0) or 0.0),
        "sub_return_26q2q3": sub_period_return(r["daily_data"]),
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[Tier-3] {label_mode}: Total={rec['total_return']:.1f}% Sharpe={rec['sharpe']:.3f} "
          f"RDD={rec['score_rdd']:.2f} MaxDD={rec['max_drawdown']:.1f}% "
          f"26-04~07子区间={rec['sub_return_26q2q3']:+.2f}%")
    del session
    gc.collect()
    return rec


def main():
    candidate_mode = sys.argv[1] if len(sys.argv) > 1 else "vol_scaled"
    base = run_config("5d")
    cand = run_config(candidate_mode)

    print(f"\n=== TIER-3 PAIRED VERDICT ({candidate_mode} vs 5d, seed 42, full windows, dual pool) ===")
    d_rdd = cand["score_rdd"] - base["score_rdd"]
    d_sharpe = cand["sharpe"] - base["sharpe"]
    d_sub = cand["sub_return_26q2q3"] - base["sub_return_26q2q3"]
    print(f"RDD:    {base['score_rdd']:.2f} → {cand['score_rdd']:.2f} (delta {d_rdd:+.2f})")
    print(f"Sharpe: {base['sharpe']:.3f} → {cand['sharpe']:.3f} (delta {d_sharpe:+.3f})")
    print(f"26-04~07: {base['sub_return_26q2q3']:+.2f}% → {cand['sub_return_26q2q3']:+.2f}% (delta {d_sub:+.2f}pp)")
    verdict = ("CANDIDATE" if (d_rdd > 0 and d_sub > 0 and d_sharpe > -0.10)
               else "MIXED" if (d_sub > 0) else "REJECT")
    print(f"verdict: {verdict} (RDD不受损且近期改善且Sharpe delta>-0.10→CANDIDATE; 仅近期改善→MIXED; 近期未复现→REJECT)")
    with open(OUT, "a") as f:
        f.write(json.dumps({"verdict": verdict, "delta_rdd": d_rdd,
                            "delta_sharpe": d_sharpe, "delta_sub": d_sub,
                            "baseline": base, "candidate": cand}) + "\n")


if __name__ == "__main__":
    main()
