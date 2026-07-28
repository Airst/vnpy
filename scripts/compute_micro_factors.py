"""
日频微观结构因子计算（P0）— 从 5min 分钟线聚合，与日线因子正交

设计原则（专注日线看不到的"日内结构"）:
- 量能在日内的分布时点: 收盘竞价/尾盘30min/首小时 量比
- 日内价格路径: VWAP 偏离、收盘在日内区间位置、上下午收益分裂、尾盘收益
- 日内量价动态: 已实现波动、量能波动(vol of vol)、上涨量能占比、Kyle lambda、U型量能
- 输出: core/alpha_db/micro_factors.parquet (datetime, vt_symbol, 14 因子)

用法:
  /home/airst/Workspace/.venv/bin/python scripts/compute_micro_factors.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import numpy as np
import polars as pl
from pathlib import Path

MIN_DIR = Path("core/alpha_db/minute")
OUT = "core/alpha_db/micro_factors.parquet"
EPS = 1e-9


def per_stock(df: pl.DataFrame) -> pl.DataFrame:
    """对单只股票的 5min 数据按 trade_date 聚合出微观结构因子"""
    df = df.sort("trade_time")
    # 每日分组聚合
    def agg(g: pl.DataFrame) -> dict:
        n = len(g)
        close = g["close"].to_numpy()
        open_ = g["open"].to_numpy()
        high = g["high"].to_numpy()
        low = g["low"].to_numpy()
        vol = g["vol"].to_numpy().astype(float)
        amt = g["amount"].to_numpy().astype(float)
        ret = close[1:] / close[:-1] - 1 if n > 1 else np.array([0.0])
        day_vol = vol.sum() + EPS
        vwap = amt.sum() / day_vol
        hmax, lmin = high.max(), low.min()
        nb6 = min(6, n)
        nb12 = min(12, n)
        half = n // 2
        m_ret = close[half - 1] / open_[0] - 1 if half > 0 else 0.0
        a_ret = close[-1] / close[half] - 1 if half > 0 else 0.0
        up_mask = ret > 0
        return {
            "close_auct_vol_r": vol[-1] / day_vol,
            "tail30_vol_r": vol[-nb6:].sum() / day_vol,
            "tail30_ret": close[-1] / (close[-nb6 - 1] if n > nb6 else open_[0]) - 1,
            "first30_ret": close[nb6 - 1] / open_[0] - 1,
            "first60_vol_r": vol[:nb12].sum() / day_vol,
            "vwap_dev": close[-1] / (vwap + EPS) - 1,
            "close_pos_range": (close[-1] - lmin) / (hmax - lmin + EPS),
            "intraday_range": (hmax - lmin) / (close[-1] + EPS),
            "realized_vol_5m": float(np.std(ret)) if len(ret) > 1 else 0.0,
            "vol_of_vol": float(np.std(vol) / (np.mean(vol) + EPS)),
            "up_bar_vol_r": vol[1:][up_mask].sum() / day_vol if len(ret) else 0.0,
            "kyle_lambda": float(np.mean(np.abs(ret)) / (np.mean(amt) + EPS) * 1e6),
            "am_pm_ret": m_ret - a_ret,
            "ushape_vol": (vol[:nb6].sum() + vol[-nb6:].sum()) / (vol[nb6:-nb6].sum() + EPS) if n > 2 * nb6 else 1.0,
        }

    rows = []
    for td, g in df.group_by("trade_date", maintain_order=True):
        if len(g) >= 10:  # 至少 10 根 5min bar（过滤半日/异常）
            td_val = td[0] if isinstance(td, (list, tuple)) else td
            rows.append((str(td_val), agg(g)))
    return pl.DataFrame(rows, schema=["trade_date", "f"], orient="row").unnest("f") if rows else pl.DataFrame()


def main():
    files = sorted(MIN_DIR.glob("*.parquet"))
    print(f"分钟数据文件: {len(files)} 只")
    all_parts = []
    for k, fp in enumerate(files):
        vt = fp.stem
        df = pl.read_parquet(fp)
        f = per_stock(df)
        if len(f):
            f = f.with_columns(pl.lit(vt).alias("vt_symbol"))
            all_parts.append(f)
        if (k + 1) % 200 == 0:
            print(f"  [{k+1}/{len(files)}] ...")
    if not all_parts:
        print("无数据")
        return
    out = pl.concat(all_parts)
    out = out.rename({"trade_date": "datetime"})
    out = out.with_columns(pl.col("datetime").str.strptime(pl.Date, format="%Y%m%d"))
    out.write_parquet(OUT)
    print(f"saved {len(out)} rows × {len(out.columns)-2} 因子 -> {OUT}")
    print("因子:", [c for c in out.columns if c not in ("datetime", "vt_symbol")])


if __name__ == "__main__":
    main()
