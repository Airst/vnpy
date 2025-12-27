import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import polars as pl

# Ensure project root is in path
# core/main.py -> core -> root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.alpha.engine import AlphaEngine
from core.alpha.ashare_factors import AShareFactorCalculator
from core.alpha.ashare_factors_v2 import AShareFactorCalculator as AShareFactorCalculatorV2

def run():
    print("Initializing Alpha Engine...")
    engine = AlphaEngine()

    # engine.sync_data()
    
    # Run the strategy
    # 创建因子计算器
    # calculator = AShareFactorCalculator(engine)
    calculator = AShareFactorCalculatorV2(engine)
    
    # 计算所有因子
    start_time = datetime.now()
    
    signal_df = calculator.calculate_all_factors(
        start_date="2020-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d")
    )
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"\n计算完成，耗时: {elapsed:.2f}秒")
    
    if signal_df is not None:
        # 显示信号统计
        stats = signal_df.group_by("datetime").agg([
            pl.col("final_signal").mean().alias("mean_signal"),
            pl.col("final_signal").std().alias("std_signal"),
            pl.count().alias("stock_count")
        ]).sort("datetime")
        
        print(f"\n信号统计:")
        print(f"时间范围: {stats['datetime'].min()} 到 {stats['datetime'].max()}")
        print(f"平均每日股票数量: {stats['stock_count'].mean():.0f}")
        print(f"信号均值: {stats['mean_signal'].mean():.4f}")
        print(f"信号标准差: {stats['std_signal'].mean():.4f}")
        
        # 保存结果到CSV
        signal_df.write_csv("./ashare_factor_signals.csv")
        print("\n信号已保存到: ./ashare_factor_signals.csv")

if __name__ == "__main__":
    run()