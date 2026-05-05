"""
深度分析603598.SSE的V10因子值
"""
import sys
sys.path.insert(0, '/home/airst/Workspace/vnpy')

import torch
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from core.alpha.v10_factor_calculator import V10FactorCalculator
from core.alpha.engine import AlphaEngine
from main import PROJECT_ROOT

def analyze_factors():
    vt_symbol = "603598.SSE"
    
    print("=" * 80)
    print(f"深度因子分析: {vt_symbol}")
    print("=" * 80)
    
    # 1. 使用AlphaEngine计算因子
    print("\n1. 计算V10因子...")
    try:
        engine = AlphaEngine()
        factor_calc = V10FactorCalculator()
        
        # 获取最近120个交易日的数据
        end_date = "20260430"
        start_date = "20251101"
        
        # 准备数据 - 使用engine的方法
        symbols = [vt_symbol]
        data = engine.lab.load_daily_data(symbols, start_date, end_date)
        
        if data is None or data.is_empty():
            print("❌ 无法加载日线数据")
            return
        
        print(f"加载数据: {len(data)} 行")
        
        # 计算因子
        factor_data = factor_calc.calculate_single_stock_factors(data, symbols)
        
        if factor_data is None:
            print("❌ 因子计算失败")
            return
        
        print(f"✅ 成功计算 {len(factor_data)} 天的因子")
        
        # 2. 分析最新日期的因子
        latest_date = factor_data['datetime'].max()
        latest_factors = factor_data.filter(pl.col("datetime") == latest_date)
        
        print(f"\n2. 最新日期 ({latest_date}) 的因子值:")
        print("=" * 80)
        
        # 获取所有因子列
        factor_cols = [col for col in factor_data.columns 
                      if col not in ['vt_symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover']]
        
        # 分类分析关键因子
        factor_groups = {
            '动量/反转': ['mom_5d', 'mom_20d', 'mom_60d', 'mom_120d', 'rev_5d', 'ma_bias_120'],
            '波动率': ['volatility_20d', 'volatility_60d'],
            '换手率': ['turnover_20d', 'turnover_60d', 'turnover_x_bull'],
            '量价关系': ['volume_ratio_5d', 'volume_ratio_20d', 'amount_ratio_5d'],
            '估值': ['pe', 'pb', 'ps', 'dv_ratio'],
            '市值': ['ln_cap', 'total_mv'],
        }
        
        for group_name, factor_names in factor_groups.items():
            print(f"\n{group_name}:")
            for fname in factor_names:
                if fname in factor_cols:
                    val = latest_factors.select(fname).row(0)[0]
                    if val is not None and not np.isnan(val):
                        # 获取该因子的横截面分位数
                        all_values = factor_data.select(fname).drop_nulls()
                        if len(all_values) > 0:
                            percentile = (all_values.filter(pl.col(fname) <= val).height / all_values.height * 100)
                            print(f"  {fname:30s}: {val:10.4f}  (分位数: {percentile:.1f}%)")
                        else:
                            print(f"  {fname:30s}: {val:10.4f}")
        
        # 3. 时序分析 - 看因子的变化趋势
        print(f"\n\n3. 关键因子时序分析 (最近30天):")
        print("=" * 80)
        
        recent_30 = factor_data.sort("datetime", descending=True).head(30)
        
        key_factors_timeline = ['mom_20d', 'mom_60d', 'volatility_20d', 'turnover_20d']
        existing_timeline = [f for f in key_factors_timeline if f in factor_cols]
        
        if existing_timeline:
            display_df = recent_30.select(['datetime'] + existing_timeline)
            print(display_df)
        
        # 4. 统计分析
        print(f"\n\n4. 因子统计特征 (全区间):")
        print("=" * 80)
        
        stats_factors = ['mom_20d', 'mom_60d', 'mom_120d', 'volatility_20d', 'turnover_20d']
        existing_stats = [f for f in stats_factors if f in factor_cols]
        
        if existing_stats:
            stats = factor_data.select(existing_stats).describe()
            print(stats)
        
        # 5. 分析为什么信号高
        print(f"\n\n5. 信号高的可能原因分析:")
        print("=" * 80)
        
        # 检查动量因子
        if 'mom_60d' in factor_cols:
            mom_60d_latest = latest_factors.select('mom_60d').row(0)[0]
            mom_60d_mean = factor_data.select('mom_60d').mean().row(0)[0]
            mom_60d_std = factor_data.select('mom_60d').std().row(0)[0]
            
            if mom_60d_latest > mom_60d_mean + 2 * mom_60d_std:
                print("⚠️ 60日动量因子显著高于平均水平 (超过2个标准差)")
                print(f"   最新值: {mom_60d_latest:.4f}, 均值: {mom_60d_mean:.4f}, 标准差: {mom_60d_std:.4f}")
            elif mom_60d_latest > mom_60d_mean + mom_60d_std:
                print("ℹ️ 60日动量因子高于平均水平 (超过1个标准差)")
        
        # 检查波动率
        if 'volatility_20d' in factor_cols:
            vol_latest = latest_factors.select('volatility_20d').row(0)[0]
            vol_mean = factor_data.select('volatility_20d').mean().row(0)[0]
            print(f"\n波动率特征:")
            print(f"   最新20日波动率: {vol_latest:.4f}")
            print(f"   平均20日波动率: {vol_mean:.4f}")
            if vol_latest < vol_mean * 0.5:
                print("   ⚠️ 波动率显著低于平均，可能被模型认为是'稳定上涨'")
        
        # 检查换手率
        if 'turnover_20d' in factor_cols:
            to_latest = latest_factors.select('turnover_20d').row(0)[0]
            to_mean = factor_data.select('turnover_20d').mean().row(0)[0]
            print(f"\n换手率特征:")
            print(f"   最新20日换手率: {to_latest:.4f}")
            print(f"   平均20日换手率: {to_mean:.4f}")
        
    except Exception as e:
        import traceback
        print(f"❌ 因子分析失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    analyze_factors()
