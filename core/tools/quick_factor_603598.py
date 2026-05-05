"""
快速分析603598.SSE的关键因子 - 直接读取数据计算
"""
import sys
sys.path.insert(0, '/home/airst/Workspace/vnpy')

import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from main import PROJECT_ROOT

def quick_factor_analysis():
    vt_symbol = "603598.SSE"
    
    print("=" * 80)
    print(f"快速因子分析: {vt_symbol}")
    print("=" * 80)
    
    # 1. 读取日线数据
    print("\n1. 加载日线数据...")
    daily_dir = PROJECT_ROOT / "core" / "alpha_db" / "daily"
    daily_file = daily_dir / f"{vt_symbol}.parquet"
    
    if not daily_file.exists():
        print(f"❌ 日线文件不存在: {daily_file}")
        return
    
    daily_df = pl.read_parquet(daily_file).sort("datetime")
    print(f"✅ 加载 {len(daily_df)} 天数据")
    
    # 2. 计算关键因子
    print("\n2. 计算关键因子...")
    
    # 使用最近150天数据
    cutoff_date = daily_df['datetime'].max() - timedelta(days=200)
    recent_data = daily_df.filter(pl.col("datetime") >= cutoff_date).with_columns([
        # 收益率
        pl.col("close").pct_change().alias("ret_1d"),
        # 动量
        (pl.col("close") / pl.col("close").shift(5) - 1).alias("mom_5d"),
        (pl.col("close") / pl.col("close").shift(20) - 1).alias("mom_20d"),
        (pl.col("close") / pl.col("close").shift(60) - 1).alias("mom_60d"),
        (pl.col("close") / pl.col("close").shift(120) - 1).alias("mom_120d"),
        # 均线偏离
        (pl.col("close") / pl.col("close").rolling_mean(window_size=5) - 1).alias("bias_5"),
        (pl.col("close") / pl.col("close").rolling_mean(window_size=20) - 1).alias("bias_20"),
        (pl.col("close") / pl.col("close").rolling_mean(window_size=60) - 1).alias("bias_60"),
        # 波动率
        pl.col("close").pct_change().rolling_std(window_size=20).alias("vol_20d"),
        pl.col("close").pct_change().rolling_std(window_size=60).alias("vol_60d"),
        # 换手率
        pl.col("turnover").rolling_mean(window_size=20).alias("turnover_20d"),
        pl.col("turnover").rolling_mean(window_size=60).alias("turnover_60d"),
    ]).drop_nulls()
    
    if len(recent_data) == 0:
        print("❌ 无有效数据")
        return
    
    print(f"✅ 有效数据: {len(recent_data)} 天")
    
    # 3. 显示最新日期的因子
    latest = recent_data.tail(1)
    print(f"\n3. 最新日期 ({latest['datetime'].item(0)}) 因子值:")
    print("=" * 80)
    
    factors = {
        '动量因子': {
            'mom_5d': '5日动量',
            'mom_20d': '20日动量',
            'mom_60d': '60日动量',
            'mom_120d': '120日动量',
        },
        '均线偏离': {
            'bias_5': '5日均线偏离',
            'bias_20': '20日均线偏离',
            'bias_60': '60日均线偏离',
        },
        '波动率': {
            'vol_20d': '20日波动率',
            'vol_60d': '60日波动率',
        },
        '换手率': {
            'turnover_20d': '20日平均换手率',
            'turnover_60d': '60日平均换手率',
        },
    }
    
    for category, factor_dict in factors.items():
        print(f"\n{category}:")
        for col, desc in factor_dict.items():
            if col in recent_data.columns:
                val = latest.select(col).row(0)[0]
                if val is not None:
                    print(f"  {desc:20s}: {val:10.4f}")
    
    # 4. 时序分析
    print(f"\n\n4. 因子时序变化 (最近20天):")
    print("=" * 80)
    
    recent_20 = recent_data.tail(20)
    display_cols = ['datetime', 'close', 'mom_20d', 'mom_60d', 'vol_20d', 'turnover_20d']
    existing_cols = [c for c in display_cols if c in recent_data.columns]
    
    print(recent_20.select(existing_cols))
    
    # 5. 统计分析
    print(f"\n\n5. 因子统计特征:")
    print("=" * 80)
    
    stats_cols = ['mom_20d', 'mom_60d', 'mom_120d', 'vol_20d', 'turnover_20d']
    existing_stats = [c for c in stats_cols if c in recent_data.columns]
    
    if existing_stats:
        stats = recent_data.select(existing_stats).describe()
        print(stats)
    
    # 6. 与全市场对比 - 读取同一日期的其他股票
    print(f"\n\n6. 全市场对比分析...")
    print("=" * 80)
    
    latest_date = latest['datetime'].item(0)
    print(f"对比日期: {latest_date}")
    
    # 加载所有股票的最近数据
    all_files = list(daily_dir.glob("*.parquet"))
    
    if len(all_files) > 0:
        print(f"扫描 {len(all_files)} 个股票文件...")
        
        # 随机采样100只股票做对比
        import random
        sample_files = random.sample(all_files, min(100, len(all_files)))
        
        market_factors = []
        
        for f in sample_files:
            try:
                df = pl.read_parquet(f).sort("datetime")
                if len(df) < 120:
                    continue
                
                # 获取最新日期附近的数据
                stock_latest = df.tail(1)
                stock_date = stock_latest['datetime'].item(0)
                
                # 如果日期相近（30天内）
                if abs((stock_date - latest_date).days) < 30:
                    close_val = stock_latest['close'].item(0)
                    if close_val and close_val > 0:
                        # 计算简单动量
                        if len(df) >= 60:
                            mom_60 = df.tail(60).with_columns([
                                (pl.col("close") / pl.col("close").shift(60) - 1).alias("mom_60d")
                            ]).drop_nulls()
                            
                            if len(mom_60) > 0:
                                m60 = mom_60.tail(1).select("mom_60d").row(0)[0]
                                if m60 is not None:
                                    market_factors.append({
                                        'vt_symbol': f.stem,
                                        'mom_60d': m60,
                                    })
            except:
                continue
        
        if len(market_factors) > 0:
            market_df = pl.DataFrame(market_factors)
            
            # 该股票的动量
            stock_mom_60 = latest.select("mom_60d").row(0)[0]
            
            print(f"\n60日动量市场对比 (样本: {len(market_df)} 只股票):")
            print(f"  603598.SSE: {stock_mom_60:.4f}")
            print(f"  市场均值: {market_df['mom_60d'].mean():.4f}")
            print(f"  市场中位数: {market_df['mom_60d'].median():.4f}")
            print(f"  市场标准差: {market_df['mom_60d'].std():.4f}")
            
            # 分位数
            percentile = (market_df.filter(pl.col("mom_60d") <= stock_mom_60).height / len(market_df) * 100)
            print(f"  分位数: {percentile:.1f}%")
    
    # 7. 结论
    print(f"\n\n7. 初步结论:")
    print("=" * 80)
    
    mom_60 = latest.select("mom_60d").row(0)[0]
    mom_120 = latest.select("mom_120d").row(0)[0]
    vol_20 = latest.select("vol_20d").row(0)[0]
    
    print(f"\n动量特征:")
    if mom_60 and mom_60 > 0.2:
        print(f"  ✓ 60日动量强劲: {mom_60:.2%}")
    if mom_120 and mom_120 > 0.3:
        print(f"  ✓ 120日动量强劲: {mom_120:.2%}")
    
    print(f"\n波动率特征:")
    if vol_20:
        if vol_20 < 0.03:
            print(f"  ✓ 低波动率: {vol_20:.2%}")
        elif vol_20 > 0.05:
            print(f"  ⚠️ 高波动率: {vol_20:.2%}")
        else:
            print(f"  - 中等波动率: {vol_20:.2%}")
    
    print(f"\n综合判断:")
    if mom_60 and mom_60 > 0.2 and vol_20 and vol_20 < 0.04:
        print("  → 该股票表现为'低波动+强动量'，这是MLP模型偏好的模式")
        print("  → 即使价格在波动，但只要整体趋势向上且波动可控，模型就会给高分")
    elif mom_60 and mom_60 > 0.3:
        print("  → 强动量驱动高分")
    else:
        print("  → 需要更多因子数据来解释")

if __name__ == "__main__":
    quick_factor_analysis()
