"""
信号评测分析模块 — 训练输出信号 vs 未来真实走势的逐日递推评测

【定位】
指定一个训练产出的信号文件 (core/alpha_db/signal/{name}.parquet),
加载股票行情 (core/alpha_db/daily/{vt_symbol}.parquet, 与训练管线同源),
by 日递推: 每个交易日取当日信号 Top-K 个股, 对照未来 1/3/5 日真实收益,
收益率 > 0 即为准确, 输出整体准确率。

【主指标: 个股绝对收益准确率】
- 每个交易日, 按信号从高到低取 Top-K 只股票 (与策略实际持仓一致)
- 对每个持仓地平线 h ∈ {1, 3, 5}: 该股未来 h 日收益 > 0 记为一次"准确"
- 准确率 = 准确次数 / 总选股次数 (逐日等权平均)
- 同时给出全池基准 (当日全部股票上涨占比): 准确率高于基准才说明信号有效,
  否则只是搭了市场方向的便车

【口径说明】
- lag=0 (默认): 信号日 t 收盘 → t+h 收盘, 与训练标签 ts_delay(C,-h)/C-1
  完全同口径, 回答"模型学没学到"。
- lag=1: t+1 收盘 → t+1+h 收盘, 回答"信号可交易性" (T+1 才能建仓)。
- 前向收益按个股自身交易日序列 shift (停牌日自然跳过, 与标签构造一致);
  尾部不足 h 日的个股自动排除。

【辅助: 截面排序指标】
加 --cs 可附加输出全截面排序能力 (配对排序准确率 / Rank IC) 供对照,
默认不显示。

【用法】
  python core/tools/signal_evaluator.py --signal ashare_mlp_signal_v15
  python core/tools/signal_evaluator.py --signal ar_v15_baseline_s42 \
      --start 2026-04-01 --end 2026-07-27 --top 5 --lag 1
  python core/tools/signal_evaluator.py --signal xxx --dump log/xxx_daily.csv
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
SIGNAL_DIR = ROOT / "core" / "alpha_db" / "signal"
DAILY_DIR = ROOT / "core" / "alpha_db" / "daily"

HORIZONS = (1, 3, 5, 7, 10)
TOP_KS = tuple(range(1, 11))  # Top-1 ~ Top-10 多维视角
MAX_K = max(TOP_KS)


def load_signal(name: str) -> pl.DataFrame:
    path = SIGNAL_DIR / f"{name}.parquet"
    if not path.exists():
        available = sorted(p.stem for p in SIGNAL_DIR.glob("*.parquet"))
        sys.exit(f"信号文件不存在: {path}\n可用信号 ({len(available)}):\n  "
                 + "\n  ".join(available))
    df = pl.read_parquet(path).select(["datetime", "vt_symbol", "final_signal"])
    return df.filter(pl.col("final_signal").is_not_null())


def load_forward_returns(symbols: list, lag: int) -> pl.DataFrame:
    """逐股加载收盘价, 计算 t → t+lag+h 的前向收益 (个股交易日口径, h=1/3/5)。"""
    frames = []
    missing = 0
    for sym in symbols:
        path = DAILY_DIR / f"{sym}.parquet"
        if not path.exists():
            missing += 1
            continue
        frames.append(
            pl.scan_parquet(path)
            .select(["datetime", "close"])
            .sort("datetime")
            .with_columns([
                (pl.col("close").shift(-(lag + h))
                 / pl.col("close").shift(-lag) - 1).alias(f"fwd_{h}")
                for h in HORIZONS])
            .drop("close")
            .with_columns(pl.lit(sym).alias("vt_symbol"))
        )
    if missing:
        print(f"[warn] {missing}/{len(symbols)} 只股票无行情文件, 已跳过")
    return pl.concat(frames).collect()


def daily_metrics(day_df: pl.DataFrame, with_cs: bool) -> dict:
    """单日截面指标。day_df: [final_signal, fwd_1..fwd_10]。"""
    sig = day_df["final_signal"].to_numpy()
    n = len(sig)
    if n < max(MAX_K * 4, 20):  # 截面太小无统计意义
        return {}
    order = np.argsort(-sig)
    top_ret_all = {}

    m = {"n": n}
    for h in HORIZONS:
        ret = day_df[f"fwd_{h}"].to_numpy()
        univ = ret[np.isfinite(ret)]
        if len(univ) == 0:
            return {}  # 尾部截面, 整日排除
        m[f"base_{h}"] = float(np.mean(univ > 0))  # 全池上涨占比 (基准)
        m[f"univ_{h}"] = float(np.mean(univ))
        top_ret_all[h] = ret[order[:MAX_K]]

    for h in HORIZONS:
        top_ret = top_ret_all[h]
        for k in TOP_KS:
            pr = top_ret[:k]
            pr = pr[np.isfinite(pr)]
            if len(pr) == 0:
                return {}  # Top 股尾部无数据, 整日排除保证矩阵可比
            m[f"acc_{h}_k{k}"] = float(np.mean(pr > 0))
            m[f"ret_{h}_k{k}"] = float(np.mean(pr))

    if with_cs:
        ret5 = day_df["fwd_5"].to_numpy()
        ok = np.isfinite(ret5)
        m["pairwise_acc"] = (stats.kendalltau(sig[ok], ret5[ok]).statistic + 1) / 2
        m["rank_ic"] = stats.spearmanr(sig[ok], ret5[ok]).statistic
    return m


def main():
    ap = argparse.ArgumentParser(description="训练信号 vs 未来走势评测 (个股绝对收益口径)")
    ap.add_argument("--signal", required=True, help="信号名 (不含 .parquet)")
    ap.add_argument("--start", default=None, help="评测起始日 YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="评测结束日 YYYY-MM-DD")
    ap.add_argument("--top", type=int, default=5,
                    help="按月分解用的 Top-K (默认 5, 同策略持仓数, 矩阵固定输出 1~10)")
    ap.add_argument("--lag", type=int, default=0,
                    help="0=标签同口径 (默认), 1=T+1 可交易口径")
    ap.add_argument("--cs", action="store_true", help="附加输出全截面排序指标")
    ap.add_argument("--dump", default=None, help="导出逐日指标 CSV 路径")
    args = ap.parse_args()

    sig_df = load_signal(args.signal)
    if args.start:
        sig_df = sig_df.filter(pl.col("datetime") >= datetime.fromisoformat(args.start))
    if args.end:
        sig_df = sig_df.filter(pl.col("datetime") <= datetime.fromisoformat(args.end))
    if sig_df.is_empty():
        sys.exit("过滤后信号为空, 检查 --start/--end")

    symbols = sig_df["vt_symbol"].unique().to_list()
    d0, d1 = sig_df["datetime"].min(), sig_df["datetime"].max()
    print(f"信号: {args.signal}  区间: {d0:%Y-%m-%d} ~ {d1:%Y-%m-%d}  "
          f"股票数: {len(symbols)}  口径: lag={args.lag}, Top-{args.top}")

    fwd_df = load_forward_returns(symbols, args.lag)
    df = sig_df.join(fwd_df, on=["datetime", "vt_symbol"], how="inner")

    days = []
    for (dt,), day_df in sorted(df.group_by("datetime"), key=lambda x: x[0]):
        m = daily_metrics(day_df, args.cs)
        if m:
            m["date"] = dt
            days.append(m)
    if not days:
        sys.exit("无可评测截面 (数据不足)")
    daily = pl.DataFrame(days)

    # === 汇总 ===
    W = 72
    print("\n" + "=" * W)
    print(f"总评 — {len(daily)} 个交易日, 平均截面 {daily['n'].mean():.0f} 只")
    print("=" * W)
    hdr = "".join(f"{h}日".rjust(9) for h in HORIZONS)

    print(f"【准确率矩阵】(Top-K 个股未来 h 日收益 > 0 的比例, %)")
    print("  " + " " * 7 + hdr)
    for k in TOP_KS:
        row = "".join(f"{daily[f'acc_{h}_k{k}'].mean()*100:9.2f}" for h in HORIZONS)
        mark = " ←策略持仓数" if k == args.top else ""
        print(f"  Top-{k:<2}  {row}{mark}")
    base_row = "".join(f"{daily[f'base_{h}'].mean()*100:9.2f}" for h in HORIZONS)
    print(f"  全池基准 {base_row}")

    print(f"【平均超额收益矩阵】(Top-K 平均收益 − 全池平均, %/h日)")
    print("  " + " " * 7 + hdr)
    for k in TOP_KS:
        row = "".join(
            f"{(daily[f'ret_{h}_k{k}'].mean() - daily[f'univ_{h}'].mean())*100:+9.3f}"
            for h in HORIZONS)
        mark = " ←策略持仓数" if k == args.top else ""
        print(f"  Top-{k:<2}  {row}{mark}")
    univ_row = "".join(f"{daily[f'univ_{h}'].mean()*100:+9.3f}" for h in HORIZONS)
    print(f"  全池平均 {univ_row}")

    if args.cs:
        pa = daily["pairwise_acc"].to_numpy()
        ic = daily["rank_ic"].to_numpy()
        print(f"【全截面排序能力】(5日口径, 参考)")
        print(f"  配对排序准确率: {pa.mean()*100:.2f}%   Rank IC: mean={ic.mean():+.4f}"
              f"  ICIR={ic.mean()/max(ic.std(),1e-9):.3f}")

    # === 按月分解 (用 --top 指定的 K) ===
    kk = args.top
    monthly = daily.with_columns(
        pl.col("date").dt.strftime("%Y-%m").alias("month")
    ).group_by("month").agg(
        [pl.len().alias("days")]
        + [pl.col(f"acc_{h}_k{kk}").mean().alias(f"acc_{h}") for h in HORIZONS]
        + [pl.col(f"base_{h}").mean() for h in HORIZONS]
        + [pl.col(f"ret_5_k{kk}").mean().alias("ret_5")]
    ).sort("month")
    print("\n" + "=" * W)
    print(f"按月分解 Top-{kk}  (准确率1d/3d/5d/7d/10d | 5d基准 | 平均5d收益)")
    print("=" * W)
    for r in monthly.iter_rows(named=True):
        flag = " ←弱" if r["acc_5"] <= r["base_5"] else ""
        accs = "  ".join(f"{r[f'acc_{h}']*100:5.1f}%" for h in HORIZONS)
        print(f"  {r['month']}  {r['days']:>3}d   {accs}   "
              f"基准 {r['base_5']*100:5.1f}%   {r['ret_5']*100:+6.3f}%{flag}")

    if args.dump:
        daily.sort("date").write_csv(args.dump)
        print(f"\n逐日指标已导出: {args.dump}")


if __name__ == "__main__":
    main()
