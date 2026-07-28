"""
分钟线下载（5min，P0 日内微观结构因子数据基础）

- 宇宙: 生产信号覆盖的 1179 只（双池）
- tushare pro_bar freq=5min, 单次 8000 行上限 → 按 ~5 个月分块翻页
- 区间: 2024-07-01 ~ 2026-07-17（约 2 年，匹配训练窗口）
- 存储: core/alpha_db/minute/{vt_symbol}.parquet（每股一份）
- 断点续传: 已有 parquet 且已到 end_date 的跳过
- 预计: ~7000 调用 × ~0.4s ≈ 50 min

用法:
  /home/airst/Workspace/.venv/bin/python scripts/download_minute.py [start] [end]
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

START = sys.argv[1] if len(sys.argv) > 1 else "2024-07-01"
END = sys.argv[2] if len(sys.argv) > 2 else "2026-07-17"
OUT_DIR = Path("core/alpha_db/minute")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DAYS = 150  # ~5 个月/块，控制在 8000 行内
SLEEP = 0.12


def to_ts(vt: str) -> str:
    code, ex = vt.split(".")
    return f"{code}.{ 'SH' if ex == 'SSE' else 'SZ'}"


def chunks(start: datetime, end: datetime):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=CHUNK_DAYS), end)
        yield cur, nxt
        cur = nxt


def download_stock(pro, vt: str, start: datetime, end: datetime) -> int:
    ts_code = to_ts(vt)
    out_path = OUT_DIR / f"{vt}.parquet"
    # 断点: 已到 end 则跳过
    if out_path.exists():
        try:
            existing = pl.read_parquet(out_path)
            if len(existing) and str(existing["trade_date"].max())[:10] >= END[:10]:
                return -1  # 已完成
        except Exception:
            pass
    frames = []
    for cs, ce in chunks(start, end):
        for attempt in range(3):
            try:
                df = ts.pro_bar(ts_code=ts_code, asset="E", freq="5min",
                                start_date=cs.strftime("%Y-%m-%d %H:%M:%S"),
                                end_date=ce.strftime("%Y-%m-%d %H:%M:%S"))
                if df is not None and len(df):
                    frames.append(pl.from_pandas(df))
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  {vt} {cs.date()} 失败: {str(e)[:80]}")
                time.sleep(1.5 * (attempt + 1))
        time.sleep(SLEEP)
    if not frames:
        return 0
    df_all = pl.concat(frames).unique(subset=["ts_code", "trade_time"]).sort("trade_time")
    df_all.write_parquet(out_path)
    return len(df_all)


def main():
    sig = pl.read_parquet("core/alpha_db/signal/ashare_mlp_signal_v15.parquet")
    syms = sorted(sig["vt_symbol"].unique().to_list())
    start = datetime.strptime(START, "%Y-%m-%d")
    end = datetime.strptime(END, "%Y-%m-%d")
    pro = ts.pro_api(SETTINGS["datafeed.password"])
    print(f"下载 {len(syms)} 只 5min 数据 {START} ~ {END}")

    done = skip = total_rows = 0
    t0 = time.time()
    for k, vt in enumerate(syms):
        n = download_stock(pro, vt, start, end)
        if n == -1:
            skip += 1
        else:
            done += 1
            total_rows += n
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  [{k+1}/{len(syms)}] 已下载 {done} 跳过 {skip} 累计 {total_rows/1e6:.1f}M 行, 耗时 {el/60:.1f}min, ETA {(el/(k+1))*(len(syms)-k-1)/60:.0f}min")
    print(f"完成: {done} 只下载, {skip} 只跳过, 共 {total_rows/1e6:.1f}M 行, 耗时 {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
