"""
TimesFM 2.5 零样本信号生成 — 2026 YTD 全池日频信号，兼容 AlphaLab/回测框架

- 每只股取 512 日收盘价，TimesFM 预测 5 日收盘 → 隐含 5 日收益作为 total_score
- final_signal 复刻 mlp_signals._post_process_signals 的截面归一化（clip[-3,3]）
- 落盘 core/alpha_db/signal/timesfm_zeroshot.parquet（不覆盖生产信号）
- 增量写入（每日一行），中断可续

用法:
  /home/airst/Workspace/.venv/bin/python scripts/timesfm_signal_gen.py [start_date] [end_date]
  默认 2026-01-01 ~ 2026-07-17
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import numpy as np
import polars as pl
from datetime import datetime
from vnpy.trader.database import get_database
from vnpy.trader.constant import Exchange, Interval

CONTEXT = 512
HORIZON = 5
OUT = "core/alpha_db/signal/timesfm_zeroshot.parquet"
START = sys.argv[1] if len(sys.argv) > 1 else "2026-01-01"
END = sys.argv[2] if len(sys.argv) > 2 else "2026-07-17"


def main():
    sig = pl.read_parquet("core/alpha_db/signal/ashare_mlp_signal_v15.parquet")
    syms = sorted(sig["vt_symbol"].unique().to_list())

    db = get_database()
    series = {}
    for s in syms:
        code, ex = s.split(".")
        bars = db.load_bar_data(code, Exchange(ex), Interval.DAILY, datetime(2023, 1, 1), datetime(2026, 7, 17))
        if bars and len(bars) > 300:
            series[s] = (
                np.array([b.datetime.strftime("%Y-%m-%d") for b in bars]),
                np.array([b.close_price for b in bars]),
            )
    all_dates = sorted({d for dts, _ in series.values() for d in dts})
    gen_dates = [d for d in all_dates if START <= d <= END]
    print(f"universe {len(series)} stocks, gen {len(gen_dates)} dates: {gen_dates[0]} ~ {gen_dates[-1]}")

    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=CONTEXT, max_horizon=64, normalize_inputs=True,
                  use_continuous_quantile_head=True, force_flip_invariance=True,
                  infer_is_positive=False, fix_quantile_crossing=True, per_core_batch_size=64))
    print("TimesFM loaded")

    rows = []
    for k, td in enumerate(gen_dates):
        batch_syms, batch_inputs = [], []
        for s, (dts, close) in series.items():
            if td not in dts:
                continue
            i = list(dts).index(td)
            if i < 60:
                continue
            lo = max(0, i + 1 - CONTEXT)
            batch_syms.append(s)
            batch_inputs.append(close[lo: i + 1])
        pt, _ = model.forecast(horizon=HORIZON, inputs=batch_inputs)
        for j, s in enumerate(batch_syms):
            score = float(pt[j][-1] / batch_inputs[j][-1] - 1)
            rows.append((td, s, score))
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(gen_dates)} days done ({datetime.now().strftime('%H:%M:%S')})")

    df = pl.DataFrame(rows, schema=["datetime", "vt_symbol", "total_score"], orient="row")
    # 复刻截面归一化
    df = df.with_columns([
        pl.col("total_score").rank(method="average").over("datetime").alias("rank"),
        pl.col("total_score").count().over("datetime").alias("count"),
    ])
    df = df.with_columns([
        (((pl.col("rank") / pl.col("count")) - 0.5) * 3.46).clip(-3, 3).alias("final_signal")
    ])
    out = df.select(["datetime", "vt_symbol", "total_score", "final_signal"])
    out.write_parquet(OUT)
    print(f"saved {len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
