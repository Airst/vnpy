"""
分钟线历史补下（向前扩展）— 2022-07-01 ~ 2024-07-01，合并进已有 parquet

已有数据从 2024-07 起，本脚本向前补 2022-07~2024-07（Tier-1 训练窗覆盖需要）。
对每只股: 下载缺失的 [START, 已有最早日期) 区间，concat+unique 合并。

用法:
  /home/airst/Workspace/.venv/bin/python scripts/download_minute_prepend.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import time
import polars as pl
import tushare as ts
from pathlib import Path
from datetime import datetime, timedelta
from vnpy.trader.setting import SETTINGS

START = sys.argv[1] if len(sys.argv) > 1 else "2022-07-01"
EXIST_START = sys.argv[2] if len(sys.argv) > 2 else "2024-07-01"
OUT_DIR = Path("core/alpha_db/minute")
CHUNK_DAYS = 150
SLEEP = 0.12


def to_ts(vt):
    code, ex = vt.split(".")
    return f"{code}.{'SH' if ex == 'SSE' else 'SZ'}"


def chunks(start, end):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=CHUNK_DAYS), end)
        yield cur, nxt
        cur = nxt


def main():
    files = sorted(OUT_DIR.glob("*.parquet"))
    start = datetime.strptime(START, "%Y-%m-%d")
    pro = ts.pro_api(SETTINGS["datafeed.password"])
    print(f"向前补 {len(files)} 只: {START} ~ {EXIST_START}")
    total_new = 0
    t0 = time.time()
    for k, fp in enumerate(files):
        vt = fp.stem
        ts_code = to_ts(vt)
        frames = []
        for cs, ce in chunks(start, datetime.strptime(EXIST_START, "%Y-%m-%d")):
            for attempt in range(3):
                try:
                    df = ts.pro_bar(ts_code=ts_code, asset="E", freq="5min",
                                    start_date=cs.strftime("%Y-%m-%d %H:%M:%S"),
                                    end_date=ce.strftime("%Y-%m-%d %H:%M:%S"))
                    if df is not None and len(df):
                        frames.append(pl.from_pandas(df))
                    break
                except Exception:
                    if attempt == 2:
                        print(f"  {vt} {cs.date()} 失败")
                    time.sleep(1.5 * (attempt + 1))
            time.sleep(SLEEP)
        if frames:
            new = pl.concat(frames)
            old = pl.read_parquet(fp)
            merged = pl.concat([old, new]).unique(subset=["ts_code", "trade_time"]).sort("trade_time")
            merged.write_parquet(fp)
            total_new += len(new)
        if (k + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  [{k+1}/{len(files)}] 新增 {total_new/1e6:.1f}M 行, 耗时 {el/60:.1f}min, ETA {(el/(k+1))*(len(files)-k-1)/60:.0f}min")
    print(f"完成: 新增 {total_new/1e6:.1f}M 行, 耗时 {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
