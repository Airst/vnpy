"""
三池并集实验（沪深300+中证1000+中证2000） — 2026 H1 风格切换大盘化验证

背景:
- 当前宇宙 CSI1000+2000（小盘）与沪深300（大盘）完全不交，2026-06 沪深300 +1.8%
  而策略 -8.3%——小盘 alpha 在风格切换期失效，大盘段完全无敞口
- 本实验: 同因子集（V15, 25 validated GP）+ 同配置（attention, swa, vl=100, seed=42）
  仅换股票池为沪深300，全量 35 窗训练 + 回测
- 信号名 ashare_mlp_signal_v15_tripool（不覆盖生产信号）

用法:
  /home/airst/Workspace/.venv/bin/python scripts/tripool_experiment.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import json
import numpy as np
import collections
from datetime import datetime

from core.alpha.engine import AlphaEngine
from core.alpha.mlp_signals import MLPSignals
from core.selector.selector import FundamentalSelector
from core.core_service import CoreService

SIGNAL = "ashare_mlp_signal_v15_tripool"


def main():
    from training import resolve_version_config
    CalcClass, _ = resolve_version_config()["v15"]
    calculator = CalcClass(gp_status_filter=None)  # → ["validated"]

    selector = FundamentalSelector()
    last_trading_date = selector.get_last_trading_day() or datetime.now()

    engine = AlphaEngine(
        factor_calculator=calculator,
        mlp_signals=MLPSignals(
            signal_name=SIGNAL,
            force_retrain=True,
            model_backend="attention",
            retrain_days=45,
            ensemble_size=1,
            seed=42,
        ),
        selector=selector,
        signal_name=SIGNAL,
        start_date="2019-12-28",
        end_date=last_trading_date.strftime("%Y-%m-%d"),
        index_filter="000300.SH,000852.SH,399303.SZ",
    )

    print("=== HS300: load data ===")
    data_df = engine.load_data()
    print("=== HS300: calculate factors ===")
    signal_df = engine.calculate_factors(data_df)
    signal_df, _ = engine.analyze_factor_performance(signal_df)
    print("=== HS300: train (35 windows, attention, swa, vl=100, seed=42) ===")
    signal_df = engine.calculate_signals(signal_df)
    engine.save_signals(signal_df)

    print("=== HS300: backtest N=5 / N=10 ===")
    core = CoreService()
    start = datetime(2022, 1, 1)
    for n in [5, 10]:
        r = core.run_backtest("MultiFactorStrategy", start, last_trading_date,
                              setting={"signal_name": SIGNAL, "max_holdings": str(n)})
        s = r["statistics"]
        print(f"HS300 N={n}: RDD={s['return_drawdown_ratio']:.2f} Sharpe={s['sharpe_ratio']:.2f} "
              f"annual={s['annual_return']:.1f}% total={s['total_return']:.1f}% "
              f"MaxDD={s['max_ddpercent']:.1f}% DD持续={s['max_drawdown_duration']}天")
        # 2026 月度
        daily = r["daily_data"]
        dates = [x["date"] for x in daily]
        bal = np.array([x["balance"] for x in daily])
        ret = np.diff(bal) / bal[:-1]
        months = collections.defaultdict(list)
        for i, dt in enumerate(dates[1:]):
            if dt >= "2026-01-01":
                months[dt[:7]].append(ret[i])
        line = " | ".join(f"{m}:{np.prod(1+np.array(v))-1:+.1%}" for m, v in sorted(months.items()))
        print(f"  2026月度: {line}")
        with open(f"log/tripool_experiment_n{n}.json", "w") as f:
            json.dump({"statistics": s, "daily_data": daily}, f, default=str)


if __name__ == "__main__":
    main()
