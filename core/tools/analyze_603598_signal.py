"""
分析603598.SSE股票的V10信号为什么持续偏高
"""
import sys
sys.path.insert(0, '/home/airst/Workspace/vnpy')

import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def analyze_stock_603598():
    vt_symbol = "603598.SSE"
    
    print("=" * 80)
    print(f"分析股票: {vt_symbol}")
    print("=" * 80)
    
    # 1. 加载V10信号数据
    print("\n1. 加载V10信号数据...")
    try:
        signal_dir = PROJECT_ROOT / "core" / "alpha_db" / "signal"
        signal_files = list(signal_dir.glob("*v10*.parquet"))
        
        if signal_files:
            signal_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_signal_file = signal_files[0]
            print(f"使用信号文件: {latest_signal_file.name}")
            
            signal_df = pl.read_parquet(latest_signal_file)
            
            # 筛选该股票
            stock_signals = signal_df.filter(pl.col("vt_symbol") == vt_symbol)
            
            if stock_signals.is_empty():
                print(f"⚠️ 信号中未找到 {vt_symbol}")
            else:
                print(f"\n找到 {len(stock_signals)} 条信号记录")
                print("\n最近30天信号:")
                print(stock_signals.select(["datetime", "final_signal", "total_score"]).sort("datetime", descending=True).head(30))
                
                # 信号统计
                print("\n信号统计:")
                print(stock_signals.select([
                    pl.col("final_signal").mean().alias("mean_signal"),
                    pl.col("final_signal").min().alias("min_signal"),
                    pl.col("final_signal").max().alias("max_signal"),
                    pl.col("final_signal").std().alias("std_signal"),
                ]))
                
                # 计算信号分位数
                all_signals = signal_df.select("final_signal")
                latest_score = stock_signals.sort("datetime", descending=True).select("final_signal").row(0)[0]
                
                quantile = (all_signals.filter(pl.col("final_signal") <= latest_score).height / all_signals.height * 100)
                print(f"\n最新信号分数: {latest_score:.4f}")
                print(f"在全市场中的分位数: {quantile:.1f}%")
                print(f"全市场信号均值: {all_signals['final_signal'].mean():.4f}")
                print(f"全市场信号中位数: {all_signals['final_signal'].median():.4f}")
                
                # 高信号持续时间
                high_signal_threshold = 1.5
                high_signal_days = stock_signals.filter(pl.col("final_signal") > high_signal_threshold)
                print(f"\n信号 > {high_signal_threshold} 的天数: {len(high_signal_days)}")
                print(f"占总交易日的比例: {len(high_signal_days)/len(stock_signals)*100:.1f}%")
                
        else:
            print("⚠️ 未找到V10信号文件")
            
    except Exception as e:
        import traceback
        print(f"❌ 加载信号失败: {e}")
        traceback.print_exc()
    
    # 2. 加载日线数据查看价格走势
    print("\n\n2. 股票价格走势分析...")
    try:
        daily_dir = PROJECT_ROOT / "core" / "alpha_db" / "daily"
        daily_file = daily_dir / f"{vt_symbol}.parquet"
        
        if daily_file.exists():
            daily_df = pl.read_parquet(daily_file)
            
            # 最近6个月的数据
            six_months_ago = datetime.now() - timedelta(days=180)
            recent_daily = daily_df.filter(pl.col("datetime") >= six_months_ago).sort("datetime")
            
            if not recent_daily.is_empty():
                print(f"\n最近6个月交易日数: {len(recent_daily)}")
                print("\n最近20天价格:")
                print(recent_daily.select(["datetime", "open", "high", "low", "close", "volume"]).tail(20))
                
                # 计算波动率和收益率
                if len(recent_daily) > 1:
                    recent_daily = recent_daily.with_columns([
                        pl.col("close").pct_change().alias("daily_return"),
                    ])
                    
                    vol_20d = recent_daily.with_columns([
                        pl.col("daily_return").rolling_std(window_size=20).alias("vol_20d"),
                        pl.col("daily_return").rolling_mean(window_size=20).alias("mean_return_20d"),
                    ])
                    
                    print(f"\n波动率分析:")
                    print(f"  最新20日波动率: {vol_20d['vol_20d'].drop_nulls().tail(1)[0]:.4f}")
                    print(f"  6个月平均20日波动率: {vol_20d['vol_20d'].mean():.4f}")
                    print(f"  最新20日平均收益: {vol_20d['mean_return_20d'].drop_nulls().tail(1)[0]:.4f}")
                    
                    # 价格区间
                    print(f"\n价格区间:")
                    print(f"  最高价: {recent_daily['high'].max():.2f}")
                    print(f"  最低价: {recent_daily['low'].min():.2f}")
                    print(f"  最新收盘价: {recent_daily['close'].tail(1)[0]:.2f}")
                    print(f"  价格振幅: {(recent_daily['high'].max() - recent_daily['low'].min()) / recent_daily['low'].min() * 100:.1f}%")
                    
                    # 计算换手率
                    if 'turnover' in recent_daily.columns:
                        print(f"\n换手率分析:")
                        print(f"  平均换手率: {recent_daily['turnover'].mean():.2f}%")
                        print(f"  最大换手率: {recent_daily['turnover'].max():.2f}%")
                        print(f"  最新换手率: {recent_daily['turnover'].tail(1)[0]:.2f}%")
            else:
                print("⚠️ 无近期日线数据")
        else:
            print(f"⚠️ 日线文件不存在: {daily_file}")
            
    except Exception as e:
        import traceback
        print(f"❌ 价格走势分析失败: {e}")
        traceback.print_exc()
    
    # 3. 对比市场其他高信号股票
    print("\n\n3. 市场信号排名分析...")
    try:
        signal_dir = PROJECT_ROOT / "core" / "alpha_db" / "signal"
        signal_files = list(signal_dir.glob("*v10*.parquet"))
        
        if signal_files:
            signal_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_signal_file = signal_files[0]
            signal_df = pl.read_parquet(latest_signal_file)
            
            # 获取最新日期的信号
            latest_date = signal_df['datetime'].max()
            latest_signals = signal_df.filter(pl.col("datetime") == latest_date).sort("final_signal", descending=True)
            
            print(f"\n最新日期 ({latest_date}) 信号Top 20:")
            top_20 = latest_signals.head(20).select(["vt_symbol", "final_signal", "total_score"])
            print(top_20)
            
            # 查看603598的排名
            rank = latest_signals.filter(pl.col("vt_symbol") == vt_symbol).select(pl.col("final_signal").rank(descending=True))
            if not rank.is_empty():
                print(f"\n{vt_symbol} 在 {len(latest_signals)} 只股票中排名第 {int(rank.row(0)[0])}")
            
    except Exception as e:
        print(f"❌ 市场分析失败: {e}")

if __name__ == "__main__":
    analyze_stock_603598()
