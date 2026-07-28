"""
SSL embedding 日级增量更新（盘后运行）

流程:
1. 增量下载缺失的分钟数据（复用 tushare 5min 接口，只补每股最新日期之后的部分，
   与已有 parquet 合并——注意 download_minute.py 是整段覆盖，这里必须合并防丢历史）
2. 加载冻结编码器（minute_ssl_encoder.pt，禁止重训），对 minute_ssl_emb.parquet
   中尚无 embedding 的 (vt_symbol, date) 推理并追加

用法（盘后，生成生产信号前）:
  /home/airst/Workspace/.venv/bin/python scripts/minute_ssl_update.py [--skip-download]
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from scripts.minute_ssl_pretrain import Day2Vec, day_features, load_stock_days

MIN_DIR = Path("core/alpha_db/minute")
EMB_PATH = "core/alpha_db/minute_ssl_emb.parquet"
ENC_PATH = "core/alpha_db/model/minute_ssl_encoder.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SLEEP = 0.12


def to_ts(vt: str) -> str:
    code, ex = vt.split(".")
    return f"{code}.{'SH' if ex == 'SSE' else 'SZ'}"


def incremental_download() -> int:
    """每股只补最新 trade_date 之后的分钟数据，合并写回（不覆盖历史）"""
    import tushare as ts
    from vnpy.trader.setting import SETTINGS

    pro_token = SETTINGS["datafeed.password"]
    ts.pro_api(pro_token)
    today = datetime.now().strftime("%Y%m%d")
    files = sorted(MIN_DIR.glob("*.parquet"))
    n_updated = 0
    for k, fp in enumerate(files):
        vt = fp.stem
        existing = pl.read_parquet(fp)
        last = str(existing["trade_date"].max())[:8]
        if last >= today:
            continue
        start = (datetime.strptime(last, "%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d 09:00:00")
        end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for attempt in range(3):
            try:
                df = ts.pro_bar(ts_code=to_ts(vt), asset="E", freq="5min",
                                start_date=start, end_date=end)
                if df is not None and len(df):
                    merged = pl.concat([existing, pl.from_pandas(df).select(existing.columns)])
                    merged = merged.unique(subset=["ts_code", "trade_time"]).sort("trade_time")
                    merged.write_parquet(fp)
                    n_updated += 1
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  {vt} 下载失败: {str(e)[:80]}")
                time.sleep(1.5 * (attempt + 1))
        time.sleep(SLEEP)
        if (k + 1) % 200 == 0:
            print(f"  [{k+1}/{len(files)}] 已更新 {n_updated}")
    print(f"[下载] 增量更新 {n_updated} 只")
    return n_updated


def update_embeddings() -> int:
    """对已有分钟数据但缺 embedding 的 (vt_symbol, date) 推理并追加"""
    model = Day2Vec().to(DEV)
    model.load_state_dict(torch.load(ENC_PATH, map_location=DEV))
    model.eval()

    emb_df = pl.read_parquet(EMB_PATH)
    z_cols = [c for c in emb_df.columns if c.startswith("ssl_emb_")]
    have = set(zip(emb_df["vt_symbol"].to_list(),
                   emb_df["datetime"].dt.strftime("%Y%m%d").to_list()))

    new_rows_dt, new_rows_vt, new_embs = [], [], []
    files = sorted(MIN_DIR.glob("*.parquet"))
    with torch.no_grad():
        for k, fp in enumerate(files):
            vt = fp.stem
            days = [(td, f) for td, f in load_stock_days(str(fp)) if (vt, td) not in have]
            if not days:
                continue
            feats = torch.from_numpy(np.stack([f for _, f in days]))
            z = model.encode(feats.to(DEV)).cpu().numpy()
            new_rows_dt.extend(td for td, _ in days)
            new_rows_vt.extend([vt] * len(days))
            new_embs.append(z)
            if (k + 1) % 300 == 0:
                print(f"  [{k+1}/{len(files)}] 新增 {len(new_rows_dt)}")
    if not new_embs:
        print("[推理] 无新增 (vt_symbol, date)")
        return 0
    E = np.concatenate(new_embs)
    add = pl.DataFrame({"trade_date": new_rows_dt, "vt_symbol": new_rows_vt,
                        **{c: E[:, i] for i, c in enumerate(z_cols)}})
    add = add.with_columns(
        pl.col("trade_date").str.strptime(pl.Datetime("us"), format="%Y%m%d").alias("datetime")
    ).drop("trade_date").select(emb_df.columns)
    merged = pl.concat([emb_df, add]).unique(subset=["datetime", "vt_symbol"]).sort(["datetime", "vt_symbol"])
    merged.write_parquet(EMB_PATH)
    print(f"[推理] 新增 {len(add)} 条 embedding, 总量 {len(merged):,}, 最新日 {merged['datetime'].max()}")
    return len(add)


def main():
    if "--skip-download" not in sys.argv:
        incremental_download()
    update_embeddings()


if __name__ == "__main__":
    main()
