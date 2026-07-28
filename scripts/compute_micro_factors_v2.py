"""
微观结构因子扩展挖掘（v2）— 在 v1 的 14 个基础上新增 ~15 个机制不同的因子

新机制（正交于 v1）:
- 订单流/大单: 大单量能占比、量大棒计数、量自相关、收益自相关
- 日内趋势结构: 趋势持续度、反转次数、日内最大回撤、路径效率
- 价差/流动性: 平均bar振幅、Corwin-Schultz价差、量加权波动
- 时段效应: 隔夜跳空、开盘动量、午间跳空、尾盘vs早盘
- 量价: 量价相关、下跌量能占比、OBV 斜率
- 输出: core/alpha_db/micro_factors_v2.parquet (datetime, vt_symbol, ~29 因子)

用法:
  /home/airst/Workspace/.venv/bin/python scripts/compute_micro_factors_v2.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import numpy as np
import polars as pl
from pathlib import Path

MIN_DIR = Path("core/alpha_db/minute")
OUT = "core/alpha_db/micro_factors_v2.parquet"
EPS = 1e-9


def safe_std(x):
    return float(np.std(x)) if len(x) > 1 else 0.0


def autocorr(x):
    if len(x) < 3 or np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def per_stock(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort("trade_time")

    def agg(g: pl.DataFrame) -> dict:
        n = len(g)
        close = g["close"].to_numpy()
        open_ = g["open"].to_numpy()
        high = g["high"].to_numpy()
        low = g["low"].to_numpy()
        vol = g["vol"].to_numpy().astype(float)
        amt = g["amount"].to_numpy().astype(float)
        ret = np.diff(close) / close[:-1] if n > 1 else np.array([0.0])
        day_vol = vol.sum() + EPS
        day_open = open_[0]
        day_close = close[-1]

        # 订单流/大单
        vol_med = np.median(vol) + EPS
        big_mask = vol > 3 * vol_med
        up_mask = ret > 0
        obv = np.cumsum(np.where(ret > 0, vol[1:], -vol[1:])) if n > 1 else np.array([0.0])

        # 日内趋势结构
        signs = np.sign(ret)
        persist = float(np.mean(signs[1:] == signs[:-1])) if len(signs) > 1 else 0.0
        n_rev = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        # 日内最大回撤
        peak = np.maximum.accumulate(close)
        mdd = float(np.max((peak - close) / (peak + EPS))) if n > 1 else 0.0
        path_eff = abs(day_close - day_open) / (np.sum(np.abs(ret)) + EPS)

        # 价差/流动性
        bar_range = (high - low) / (close + EPS)
        # Corwin-Schultz spread (简化高频版)
        hl = np.log((high + EPS) / (low + EPS)) ** 2
        cs_spread = float(np.sqrt(np.mean(hl)) ) if n > 1 else 0.0
        # 量加权波动（vw[1:] 与 ret 同长 n-1）
        vw = vol / day_vol
        vol_weighted_vol = float(np.sqrt(np.sum(vw[1:] * (ret ** 2)))) if n > 1 else 0.0

        # 时段
        n1h = min(12, n)  # 首小时
        ntail = min(12, n)  # 尾小时
        morning_mom = close[n1h - 1] / day_open - 1 if n1h > 0 else 0.0
        tail_ret = day_close / close[-ntail - 1] - 1 if n > ntail else 0.0
        tail_vs_body = tail_ret - morning_mom

        return {
            # v1 的 14 个（保持一致）
            "close_auct_vol_r": vol[-1] / day_vol,
            "tail30_vol_r": vol[-6:].sum() / day_vol,
            "tail30_ret": day_close / (close[-7] if n > 6 else day_open) - 1,
            "first30_ret": close[min(6, n) - 1] / day_open - 1,
            "first60_vol_r": vol[:min(12, n)].sum() / day_vol,
            "vwap_dev": day_close / (amt.sum() / day_vol + EPS) - 1,
            "close_pos_range": (day_close - low.min()) / (high.max() - low.min() + EPS),
            "intraday_range": (high.max() - low.min()) / (day_close + EPS),
            "realized_vol_5m": safe_std(ret),
            "vol_of_vol": safe_std(vol) / (np.mean(vol) + EPS),
            "up_bar_vol_r": vol[1:][up_mask].sum() / day_vol if n > 1 else 0.0,
            "kyle_lambda": float(np.mean(np.abs(ret)) / (np.mean(amt) + EPS) * 1e6),
            "am_pm_ret": (close[n // 2 - 1] / day_open - 1) - (day_close / close[n // 2] - 1) if n > 2 else 0.0,
            "ushape_vol": (vol[:6].sum() + vol[-6:].sum()) / (vol[6:-6].sum() + EPS) if n > 12 else 1.0,
            # v2 新增 ~15 个
            "big_bar_vol_r": vol[big_mask].sum() / day_vol,
            "n_vol_spike": int(big_mask.sum()),
            "vol_autocorr": autocorr(vol),
            "ret_autocorr": autocorr(ret),
            "trend_persist": persist,
            "n_reversals": n_rev,
            "intraday_mdd": mdd,
            "path_efficiency": float(path_eff),
            "mean_bar_range": float(np.mean(bar_range)),
            "cs_spread": cs_spread,
            "vol_weighted_vol": vol_weighted_vol,
            "overnight_gap": day_open / (close[-2] if n > 1 else day_open) - 1 if False else 0.0,  # 用日线更准，占位
            "morning_mom": morning_mom,
            "tail_vs_body_ret": tail_vs_body,
            "vol_price_corr": float(np.corrcoef(close, vol)[0, 1]) if n > 2 and np.std(vol) > 0 else 0.0,
            "down_bar_vol_r": vol[1:][~up_mask].sum() / day_vol if n > 1 else 0.0,
            "obv_slope": float((obv[-1] - obv[0]) / (day_vol + EPS)) if n > 1 else 0.0,
        }

    rows = []
    for td, g in df.group_by("trade_date", maintain_order=True):
        if len(g) >= 10:
            td_val = td[0] if isinstance(td, (list, tuple)) else td
            rows.append((str(td_val), agg(g)))
    return pl.DataFrame(rows, schema=["trade_date", "f"], orient="row").unnest("f") if rows else pl.DataFrame()


def main():
    files = sorted(MIN_DIR.glob("*.parquet"))
    print(f"分钟数据文件: {len(files)} 只")
    parts = []
    for k, fp in enumerate(files):
        vt = fp.stem
        df = pl.read_parquet(fp)
        f = per_stock(df)
        if len(f):
            parts.append(f.with_columns(pl.lit(vt).alias("vt_symbol")))
        if (k + 1) % 300 == 0:
            print(f"  [{k+1}/{len(files)}] ...")
    out = pl.concat(parts)
    out = out.rename({"trade_date": "datetime"})
    out = out.with_columns(pl.col("datetime").str.strptime(pl.Date, format="%Y%m%d"))
    out.write_parquet(OUT)
    print(f"saved {len(out)} rows × {len(out.columns)-2} 因子 -> {OUT}")


if __name__ == "__main__":
    main()
