"""
对比V10和V11下603598.SSE的信号表现
"""
import sys
sys.path.insert(0, '/home/airst/Workspace/vnpy')

import polars as pl
from pathlib import Path

def analyze_603598_signals():
    vt_symbol = "603598.SSE"
    
    # 加载V10信号
    v10_file = Path("core/alpha_db/signal/ashare_mlp_signal_v10.parquet")
    if not v10_file.exists():
        print(f"❌ V10信号文件不存在: {v10_file}")
        return
    
    v10_df = pl.read_parquet(v10_file).filter(pl.col("vt_symbol") == vt_symbol).sort("datetime")
    print(f"V10信号: {len(v10_df)} 条记录")
    print(f"  时间范围: {v10_df['datetime'].min()} ~ {v10_df['datetime'].max()}")
    print(f"  final_signal 均值: {v10_df['final_signal'].mean():.4f}")
    print(f"  final_signal 最大值: {v10_df['final_signal'].max():.4f}")
    print(f"  final_signal 最小值: {v10_df['final_signal'].min():.4f}")
    
    # 加载V11信号（从训练日志中提取）
    print("\nV11信号（从训练日志提取的关键时段）:")
    print("  2025-03-05: score=1.728 (买入)")
    print("  2025-03-24: score=1.545 (止损)")
    print("  注: V11信号文件已删除，仅从日志提取")
    
    # 分析V10信号分布
    print("\n\nV10信号分布分析:")
    print("=" * 80)
    
    v10_df = v10_df.with_columns([
        pl.col("datetime").dt.year().alias("year"),
        pl.col("datetime").dt.month().alias("month")
    ])
    
    # 按年统计
    print("\n按年统计:")
    yearly_stats = v10_df.group_by("year").agg([
        pl.col("final_signal").mean().alias("mean_signal"),
        pl.col("final_signal").max().alias("max_signal"),
        pl.col("final_signal").min().alias("min_signal"),
        pl.col("final_signal").std().alias("std_signal"),
        pl.count().alias("count")
    ]).sort("year")
    
    print(yearly_stats)
    
    # 高分时段分析
    print("\n\n高分时段 (final_signal > 2.0):")
    high_signal = v10_df.filter(pl.col("final_signal") > 2.0)
    if len(high_signal) > 0:
        print(f"  高分天数: {len(high_signal)}")
        print(f"  最早: {high_signal['datetime'].min()}")
        print(f"  最晚: {high_signal['datetime'].max()}")
        
        # 显示前10个高分记录
        print("\n  前10个高分记录:")
        print(high_signal.sort("final_signal", descending=True).head(10).select([
            "datetime", "final_signal", "total_score"
        ]))
    
    # 低分时段分析
    print("\n\n低分时段 (final_signal < -1.0):")
    low_signal = v10_df.filter(pl.col("final_signal") < -1.0)
    if len(low_signal) > 0:
        print(f"  低分天数: {len(low_signal)}")
        print(f"  最早: {low_signal['datetime'].min()}")
        print(f"  最晚: {low_signal['datetime'].max()}")

if __name__ == "__main__":
    analyze_603598_signals()
