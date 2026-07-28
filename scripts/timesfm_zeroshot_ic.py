"""
TimesFM 2.5 零样本截面 IC 快速验证 — 判断该范式在股票上是否有任何信号

方法:
- 双池宇宙（与生产一致），每只股取截至测试日的 512 日收盘价
- TimesFM 零样本预测未来 5 日收盘 → 隐含 5 日收益
- 全池按隐含收益排序，与真实 5 日前向收益算 Spearman IC
- 最近 ~20 个可算前向收益的交易日，报均值/正占比
- 对照: 同期生产模型（SWA）的 IC（7 月曾 -0.239）

用法:
  /home/airst/Workspace/.venv/bin/python scripts/timesfm_zeroshot_ic.py [n_dates]
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import numpy as np
from datetime import datetime
from scipy.stats import spearmanr
from vnpy.trader.database import get_database
from vnpy.trader.constant import Exchange, Interval
import polars as pl

CONTEXT = 512
HORIZON = 5
N_DATES = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def main():
    # 双池宇宙 = 生产信号覆盖的股票
    sig = pl.read_parquet("core/alpha_db/signal/ashare_mlp_signal_v15.parquet")
    syms = sorted(sig["vt_symbol"].unique().to_list())
    print(f"universe: {len(syms)} stocks")

    db = get_database()
    series = {}
    for s in syms:
        code, ex = s.split(".")
        bars = db.load_bar_data(code, Exchange(ex), Interval.DAILY, datetime(2023, 1, 1), datetime(2026, 7, 17))
        if bars and len(bars) > 300 + HORIZON:
            series[s] = (
                np.array([b.datetime.strftime("%Y-%m-%d") for b in bars]),
                np.array([b.close_price for b in bars]),
            )
    all_dates = sorted({d for dts, _ in series.values() for d in dts})
    print(f"loaded {len(series)} series, {len(all_dates)} dates")

    # 测试日: 最近 N_DATES 个有 5 日前向收益的交易日
    test_dates = all_dates[-(N_DATES + HORIZON): -HORIZON]
    print(f"test dates: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)} days)")

    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=CONTEXT, max_horizon=64, normalize_inputs=True,
                  use_continuous_quantile_head=True, force_flip_invariance=True,
                  infer_is_positive=False, fix_quantile_crossing=True, per_core_batch_size=64))
    print("TimesFM loaded")

    ics = []
    for td in test_dates:
        batch_syms, batch_inputs, fwd = [], [], {}
        for s, (dts, close) in series.items():
            if td not in dts:
                continue
            i = list(dts).index(td)
            if i + HORIZON >= len(close) or i < 60:
                continue
            lo = max(0, i + 1 - CONTEXT)
            batch_syms.append(s)
            batch_inputs.append(close[lo: i + 1])
            fwd[s] = close[i + HORIZON] / close[i] - 1
        if len(batch_syms) < 100:
            continue
        pt, _ = model.forecast(horizon=HORIZON, inputs=batch_inputs)
        pred_ret = {s: pt[k][-1] / batch_inputs[k][-1] - 1 for k, s in enumerate(batch_syms)}
        common = [s for s in batch_syms if s in fwd]
        ic, _ = spearmanr([pred_ret[s] for s in common], [fwd[s] for s in common])
        if not np.isnan(ic):
            ics.append((td, ic))
        print(f"{td}: IC={ic:+.3f} ({len(common)} stocks)")

    v = np.array([ic for _, ic in ics])
    print(f"\n=== TimesFM 零样本 {len(ics)} 日: mean IC={v.mean():+.4f}, 正占比 {(v > 0).mean():.0%}, ICIR={v.mean() / (v.std() + 1e-9):+.2f} ===")
    print("对照: 生产模型(SWA) 2026-07 IC=-0.239, 健康值 +0.03~0.05")


if __name__ == "__main__":
    main()
