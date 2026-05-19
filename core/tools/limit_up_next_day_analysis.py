"""
涨停次日高开回落未破前日收盘买入策略统计

逻辑：
1. 找到涨停日（涨幅 >= 9.5%，排除ST的 ~5%涨停）
2. 涨停次日满足：
   - 高开：开盘价 > 前日收盘价
   - 下跌：收盘价 < 开盘价
   - 未破前日收盘价：收盘价 > 前日收盘价
   - 放量：成交量 > 前一日成交量 * 1.5
3. 以当日收盘价买入，统计未来3日收益率 > 1% 的概率

支持按指数成分股筛选：
  --index 000852.SH    中证1000
  --index 932000.CSI   中证2000
"""
import sys
sys.path.insert(0, '/home/airst/Workspace/vnpy')

import argparse
import polars as pl
import numpy as np
from pathlib import Path

DAILY_DIR = Path('/home/airst/Workspace/vnpy/core/alpha_db/daily')

INDEX_NAMES = {
    "000852.SH": "中证1000",
    "932000.CSI": "中证2000",
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
}


def get_index_constituents(index_code: str) -> set:
    """获取指数成分股，返回 vt_symbol 集合"""
    import tushare as ts
    from vnpy.trader.setting import SETTINGS

    pro = ts.pro_api(SETTINGS["datafeed.password"])
    df = pro.index_weight(index_code=index_code)
    if df is None or df.empty:
        print(f"警告: {index_code} 无成分股数据")
        return set()

    # 取最新一期
    latest_date = df["trade_date"].max()
    df = df[df["trade_date"] == latest_date]

    symbols = set()
    for code in df["con_code"].unique():
        parts = code.split(".")
        if len(parts) == 2:
            if parts[1] == "SZ":
                symbols.add(f"{parts[0]}.SZSE")
            elif parts[1] == "SH":
                symbols.add(f"{parts[0]}.SSE")
    return symbols


def print_stats(result: pl.DataFrame, title: str):
    """打印统计结果"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"总信号数: {len(result)}")
    print(f"时间范围: {result['datetime'].min()} ~ {result['datetime'].max()}")

    ret = result["ret_3d"].to_numpy()
    win_1pct = np.sum(ret > 0.01) / len(ret)
    win_0 = np.sum(ret > 0) / len(ret)

    print(f"\n--- 收益统计 ---")
    print(f"未来3日收益 > 1% 的概率: {win_1pct:.2%} ({np.sum(ret > 0.01)}/{len(ret)})")
    print(f"未来3日收益 > 0% 的概率: {win_0:.2%} ({np.sum(ret > 0)}/{len(ret)})")
    print(f"平均收益: {np.mean(ret):.2%}")
    print(f"中位数收益: {np.median(ret):.2%}")
    print(f"最大收益: {np.max(ret):.2%}")
    print(f"最大亏损: {np.min(ret):.2%}")
    print(f"收益标准差: {np.std(ret):.2%}")

    # 分年统计
    result_with_year = result.with_columns(
        pl.col("datetime").dt.year().alias("year")
    )
    print(f"\n--- 分年统计 ---")
    print(f"{'年份':<6} {'信号数':<8} {'3日>1%概率':<12} {'平均收益':<10} {'中位数收益':<10}")
    for year in sorted(result_with_year["year"].unique().to_list()):
        year_data = result_with_year.filter(pl.col("year") == year)
        yr = year_data["ret_3d"].to_numpy()
        prob = np.sum(yr > 0.01) / len(yr)
        print(f"{year:<6} {len(yr):<8} {prob:<12.2%} {np.mean(yr):<10.2%} {np.median(yr):<10.2%}")


def analyze(symbol_filter: set = None):
    files = sorted(DAILY_DIR.glob('*.parquet'))

    if symbol_filter:
        files = [f for f in files if f.stem in symbol_filter]

    print(f"共 {len(files)} 只股票")

    all_signals = []

    for f in files:
        vt_symbol = f.stem
        df = pl.read_parquet(f).sort("datetime")
        if len(df) < 10:
            continue

        # 计算涨跌幅、成交量变化
        df = df.with_columns([
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("pct_chg"),
            (pl.col("volume") / pl.col("volume").shift(1)).alias("vol_ratio"),
            pl.col("close").shift(1).alias("prev_close"),
            (pl.col("close").shift(-3) / pl.col("close") - 1).alias("ret_3d"),
        ])

        df = df.with_columns([
            pl.col("pct_chg").shift(1).alias("prev_pct_chg"),
        ])

        # 筛选条件
        signals = df.filter(
            (pl.col("prev_pct_chg") >= 0.095) &       # 前日涨停
            (pl.col("open") > pl.col("prev_close")) & # 高开
            (pl.col("close") < pl.col("open")) &      # 下跌（收阴）
            (pl.col("close") > pl.col("prev_close")) & # 未破前日收盘
            (pl.col("vol_ratio") > 1.5) &              # 放量
            (pl.col("ret_3d").is_not_null())            # 有未来数据
        ).select([
            pl.lit(vt_symbol).alias("symbol"),
            "datetime",
            "prev_pct_chg",
            "pct_chg",
            "vol_ratio",
            "ret_3d",
        ])

        if len(signals) > 0:
            all_signals.append(signals)

    if not all_signals:
        print("未找到符合条件的信号")
        return

    return pl.concat(all_signals)


def main():
    parser = argparse.ArgumentParser(description="涨停次日高开回落策略统计")
    parser.add_argument("--index", nargs="*", default=["000852.SH", "932000.CSI"],
                        help="指数代码，默认中证1000+中证2000")
    args = parser.parse_args()

    indices = args.index

    for index_code in indices:
        name = INDEX_NAMES.get(index_code, index_code)
        print(f"\n{'#'*60}")
        print(f"# 获取 {name} ({index_code}) 成分股...")
        print(f"{'#'*60}")

        constituents = get_index_constituents(index_code)
        if not constituents:
            print(f"跳过 {name}：无法获取成分股")
            continue
        print(f"成分股数量: {len(constituents)}")

        result = analyze(symbol_filter=constituents)
        if result is not None:
            print_stats(result, f"【{name}】涨停次日高开回落未破收盘+放量 → 收盘买入持有3日")


if __name__ == "__main__":
    main()
