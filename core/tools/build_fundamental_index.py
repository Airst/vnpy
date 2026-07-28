"""
基本面选股脚本：从当前股票池中选出优质股票，写入虚拟指数成分股 JSON。

选股逻辑：
  1. 候选池：数据库有日线 + 主板 + A股前缀过滤
  2. 硬过滤：PE>0(非亏损)、换手率>=1%、ln(市值)>=11.5(>10亿)
  3. 质量打分：ROE + 净利率 + 净利润增速 + 毛利率 z-score 等权
  4. 取前 N 名写入 JSON

Usage:
    python -m core.tools.build_fundamental_index
    python -m core.tools.build_fundamental_index --top 300 --code 888001.QD
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pymysql
import tushare as ts

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vnpy.trader.setting import SETTINGS
from core.selector.selector import FundamentalSelector
from data_manager.ts_downloader.stock_info_manager import StockInfoManager


ASHARE_PREFIXES = ('000', '002', '300', '600', '601', '603', '688')
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "core" / "alpha_db" / "index"


def get_db_config():
    return {
        "host": SETTINGS["database.host"],
        "port": SETTINGS["database.port"],
        "user": SETTINGS["database.user"],
        "password": SETTINGS["database.password"],
        "database": SETTINGS["database.database"],
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
    }


def vt_to_ts(vt_symbol: str) -> str:
    code, exchange = vt_symbol.split(".")
    suffix = "SH" if exchange == "SSE" else "SZ" if exchange == "SZSE" else "BJ"
    return f"{code}.{suffix}"


def ts_to_vt(ts_code: str) -> str:
    code, suffix = ts_code.split(".")
    exchange = "SSE" if suffix == "SH" else "SZSE" if suffix == "SZ" else "BSE"
    return f"{code}.{exchange}"


def get_latest_trade_date(conn) -> str:
    with conn.cursor() as cursor:
        cursor.execute("SELECT MAX(trade_date) as max_date FROM dailybasic")
        row = cursor.fetchone()
        return row["max_date"] if row and row["max_date"] else None


def load_daily_basic(conn, ts_codes: list, trade_date: str) -> pd.DataFrame:
    if not ts_codes:
        return pd.DataFrame()
    fmt = ",".join(["%s"] * len(ts_codes))
    sql = f"""
        SELECT ts_code, pe_ttm, total_mv, turnover_rate
        FROM dailybasic
        WHERE ts_code IN ({fmt}) AND trade_date = %s
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, ts_codes + [trade_date])
        rows = cursor.fetchall()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_latest_fina(conn, ts_codes: list) -> pd.DataFrame:
    if not ts_codes:
        return pd.DataFrame()
    fmt = ",".join(["%s"] * len(ts_codes))
    sql = f"""
        SELECT f.ts_code, f.roe, f.netprofit_margin, f.netprofit_yoy, f.gross_margin
        FROM fina_indicator f
        INNER JOIN (
            SELECT ts_code, MAX(ann_date) as max_ann, MAX(end_date) as max_end
            FROM fina_indicator
            WHERE ts_code IN ({fmt})
            GROUP BY ts_code
        ) latest ON f.ts_code = latest.ts_code AND f.ann_date = latest.max_ann AND f.end_date = latest.max_end
        WHERE f.ts_code IN ({fmt})
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, ts_codes + ts_codes)
        rows = cursor.fetchall()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def main():
    parser = argparse.ArgumentParser(description="基本面选股 → 虚拟指数")
    parser.add_argument("--top", type=int, default=300, help="选股数量")
    parser.add_argument("--code", default="888001.QD", help="虚拟指数代码")
    parser.add_argument("--name", default="基本面优质300", help="指数名称")
    args = parser.parse_args()

    print(f"=== 基本面选股: 选出 Top {args.top} → {args.code} ===\n")

    # Step 1: 获取候选池
    print("[1/4] 获取候选池...")
    selector = FundamentalSelector()
    all_symbols = selector.get_candidate_symbols()
    ashare_symbols = [s for s in all_symbols if any(s.startswith(p) for p in ASHARE_PREFIXES)]
    print(f"  候选池: {len(ashare_symbols)} 只 A股主板标的")

    if not ashare_symbols:
        print("ERROR: 候选池为空")
        return

    ts_codes = [vt_to_ts(s) for s in ashare_symbols]
    ts_to_vt_map = {vt_to_ts(s): s for s in ashare_symbols}

    # Step 2: 硬过滤 (dailybasic)
    print("\n[2/4] 硬过滤 (PE>0, 换手>=1%, 市值>10亿)...")
    conn = pymysql.connect(**get_db_config())
    try:
        trade_date = get_latest_trade_date(conn)
        if not trade_date:
            print("ERROR: dailybasic 表无数据")
            return
        print(f"  使用交易日: {trade_date}")

        basic_df = load_daily_basic(conn, ts_codes, trade_date)
    finally:
        conn.close()

    if basic_df.empty:
        print("ERROR: 无法获取 dailybasic 数据")
        return

    before = len(basic_df)
    basic_df = basic_df.dropna(subset=["pe_ttm", "total_mv", "turnover_rate"])
    basic_df = basic_df[basic_df["pe_ttm"] > 0]
    basic_df = basic_df[basic_df["turnover_rate"] >= 1.0]
    basic_df["ln_mv"] = basic_df["total_mv"].apply(lambda x: math.log(x) if x > 0 else 0)
    basic_df = basic_df[basic_df["ln_mv"] >= 11.5]
    print(f"  硬过滤: {before} → {len(basic_df)} 只")

    if basic_df.empty:
        print("ERROR: 硬过滤后无剩余标的")
        return

    filtered_ts_codes = basic_df["ts_code"].tolist()

    # Step 3: 质量打分 (fina_indicator)
    print("\n[3/4] 质量打分 (ROE + 净利率 + 净利润增速 + 毛利率)...")
    conn = pymysql.connect(**get_db_config())
    try:
        fina_df = load_latest_fina(conn, filtered_ts_codes)
    finally:
        conn.close()

    if fina_df.empty:
        print("WARNING: 无财务数据，将仅按市值排序")
        basic_df = basic_df.sort_values("total_mv", ascending=False)
        scored_codes = basic_df["ts_code"].head(args.top).tolist()
        scores = {code: 0.0 for code in scored_codes}
    else:
        merged = basic_df[["ts_code", "total_mv"]].merge(fina_df, on="ts_code", how="left")
        score_cols = ["roe", "netprofit_margin", "netprofit_yoy", "gross_margin"]
        for col in score_cols:
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = merged[col].fillna(0.0)

        for col in score_cols:
            merged[f"{col}_z"] = zscore(merged[col])

        merged["score"] = sum(merged[f"{col}_z"] for col in score_cols)
        merged = merged.sort_values("score", ascending=False)

        top_df = merged.head(args.top)
        scored_codes = top_df["ts_code"].tolist()
        scores = dict(zip(top_df["ts_code"], top_df["score"]))
        print(f"  打分完成, Top {args.top} score range: [{top_df['score'].min():.2f}, {top_df['score'].max():.2f}]")

    # Step 4: 获取股票名称并写入 JSON
    print(f"\n[4/4] 写入 {args.code} 成分股 JSON...")
    stock_info_mgr = StockInfoManager()
    vt_symbols_selected = [ts_to_vt_map.get(tc, ts_to_vt(tc)) for tc in scored_codes]
    info_df = stock_info_mgr.load_data(vt_symbols_selected)
    name_map = {}
    if info_df is not None and not info_df.empty:
        name_map = dict(zip(info_df["ts_code"], info_df["name"]))

    constituents = []
    for tc in scored_codes:
        constituents.append({
            "ts_code": tc,
            "vt_symbol": ts_to_vt_map.get(tc, ts_to_vt(tc)),
            "name": name_map.get(tc, ""),
            "score": round(scores.get(tc, 0.0), 4),
        })

    output = {
        "index_code": args.code,
        "name": args.name,
        "trade_date": trade_date,
        "count": len(constituents),
        "constituents": constituents,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.code}_constituents.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n=== 完成 ===")
    print(f"  输出: {output_path}")
    print(f"  成分股数量: {len(constituents)}")
    if constituents:
        print(f"  Top 5:")
        for c in constituents[:5]:
            print(f"    {c['ts_code']} {c['name']:8s} score={c['score']:.4f}")


if __name__ == "__main__":
    main()
