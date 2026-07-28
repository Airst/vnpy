"""
微观结构因子 Tier-0 IC 门（auto-research 标准）

门槛（research_runner）: |IC|>=0.02, |ICIR|>=0.3, direction_ratio>=0.6
- 每个因子的日频截面 rank IC（vs 5 日前向收益），滚动 60 日窗口算 ICIR 与方向比
- 输出: core/alpha_db/micro_factors_tier0.json + 控制台报告

用法:
  /home/airst/Workspace/.venv/bin/python scripts/micro_factors_tier0.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
import numpy as np
import polars as pl
from datetime import datetime
from scipy.stats import spearmanr
from vnpy.trader.database import get_database
from vnpy.trader.constant import Exchange, Interval

MIN_IC, MIN_ICIR, MIN_DIR = 0.02, 0.3, 0.6
HORIZON = 5


def main():
    fac = pl.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "core/alpha_db/micro_factors.parquet")
    factor_cols = [c for c in fac.columns if c not in ("datetime", "vt_symbol")]
    dates = sorted(fac["datetime"].unique().to_list())
    print(f"factors {len(factor_cols)}, dates {dates[0]}~{dates[-1]}")

    # 前向收益（用日线 close）
    db = get_database()
    price = {}
    for s in fac["vt_symbol"].unique().to_list():
        code, ex = s.split(".")
        bars = db.load_bar_data(code, Exchange(ex), Interval.DAILY, dates[0], dates[-1])
        if bars:
            price[s] = {b.datetime.strftime("%Y-%m-%d"): b.close_price for b in bars}

    # 逐日 fwd return map
    fwd = {}
    dstrs = [d.strftime("%Y-%m-%d") for d in dates]
    for i in range(len(dates) - HORIZON):
        d, d5 = dstrs[i], dstrs[i + HORIZON]
        for s, pc in price.items():
            if d in pc and d5 in pc:
                fwd[(d, s)] = pc[d5] / pc[d] - 1

    results = {}
    for fc in factor_cols:
        ics = []
        for d in dates[: -HORIZON]:
            dstr = d.strftime("%Y-%m-%d")
            day = fac.filter(pl.col("datetime") == d)
            xs, ys = [], []
            for row in day.iter_rows(named=True):
                v = row[fc]
                key = (dstr, row["vt_symbol"])
                if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) and key in fwd:
                    xs.append(v)
                    ys.append(fwd[key])
            if len(xs) > 100 and np.std(xs) > 0:
                ic, _ = spearmanr(xs, ys)
                if not np.isnan(ic):
                    ics.append(ic)
        if not ics:
            results[fc] = {"ic": 0, "icir": 0, "dir": 0, "pass": False}
            continue
        v = np.array(ics)
        # 滚动 60 日 ICIR 与方向比
        roll = [v[i: i + 60].mean() for i in range(0, max(1, len(v) - 60), 20)]
        ic_mean = v.mean()
        icir = ic_mean / (v.std() + 1e-9)
        dir_ratio = float(np.mean([1 if (r > 0) == (ic_mean > 0) else 0 for r in roll])) if roll else 0
        results[fc] = {
            "ic": round(float(ic_mean), 4),
            "icir": round(float(icir), 3),
            "dir": round(dir_ratio, 3),
            "pass": bool(abs(ic_mean) >= MIN_IC and abs(icir) >= MIN_ICIR and dir_ratio >= MIN_DIR),
        }

    passed = [f for f, r in results.items() if r["pass"]]
    print(f"\n{'factor':<18} {'IC':>8} {'ICIR':>7} {'方向比':>6}  判定")
    for fc, r in sorted(results.items(), key=lambda x: -abs(x[1]["ic"])):
        mark = "PASS" if r["pass"] else "  -"
        print(f"{fc:<18} {r['ic']:>+8.4f} {r['icir']:>7.2f} {r['dir']:>6.2f}  {mark}")
    print(f"\n通过 {len(passed)}/{len(factor_cols)}: {passed}")

    out_json = sys.argv[2] if len(sys.argv) > 2 else "core/alpha_db/micro_factors_tier0.json"
    with open(out_json, "w") as f:
        json.dump({"results": results, "passed": passed}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
