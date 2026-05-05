"""
深入分析603598.SSE的V10因子值 - 横截面对比
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

def deep_factor_analysis():
    vt_symbol = "603598.SSE"
    
    print("=" * 80)
    print(f"深度因子分析: {vt_symbol}")
    print("=" * 80)
    
    # 1. 使用AlphaEngine计算因子
    print("\n1. 计算V10因子...")
    try:
        engine = AlphaEngine()
        factor_calc = V10FactorCalculator()
        
        # 获取最近150个交易日的数据
        end_date = "20260430"
        start_date = "20251001"
        
        # 加载全市场数据用于横截面对比
        print("加载全市场日线数据...")
        all_daily = engine.lab.load_all_daily_data(start_date, end_date)
        
        if all_daily is None or all_daily.is_empty():
            print("❌ 无法加载日线数据")
            return
        
        print(f"加载全市场数据: {len(all_daily)} 行, {all_daily['vt_symbol'].n_unique()} 只股票")
        
        # 计算全市场因子
        print("计算全市场因子...")
        factor_data = factor_calc.calculate_factors_from_dataframe(all_daily)
        
        if factor_data is None:
            print("❌ 因子计算失败")
            return
        
        print(f"✅ 成功计算因子，共 {len(factor_data)} 行")
        
        # 2. 分析最新日期的因子
        latest_date = factor_data['datetime'].max()
        print(f"\n2. 最新日期: {latest_date}")
        print("=" * 80)
        
        # 全市场横截面
        cross_section = factor_data.filter(pl.col("datetime") == latest_date)
        print(f"横截面股票数: {len(cross_section)}")
        
        # 目标股票
        target_stock = cross_section.filter(pl.col("vt_symbol") == vt_symbol)
        
        if target_stock.is_empty():
            print(f"❌ 横截面中未找到 {vt_symbol}")
            return
        
        print(f"\n✅ 找到 {vt_symbol}")
        
        # 3. 获取所有因子并分析
        print(f"\n3. 因子横截面对比分析:")
        print("=" * 80)
        
        # 排除非因子列
        exclude_cols = {'vt_symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover', 
                       'turnover_rate', 'pe', 'pb', 'ps', 'dv_ratio', 'total_mv', 'industry',
                       'open_interest', 'amount'}
        
        factor_cols = [col for col in factor_data.columns if col not in exclude_cols]
        print(f"因子总数: {len(factor_cols)}")
        
        # 分析每个因子
        factor_analysis = []
        
        for fname in factor_cols:
            if fname not in cross_section.columns:
                continue
                
            # 获取目标股票的值
            target_val = target_stock.select(fname).row(0)[0]
            
            if target_val is None or np.isnan(target_val):
                continue
            
            # 获取横截面统计
            cs_data = cross_section.select(fname).drop_nulls()
            
            if len(cs_data) == 0:
                continue
            
            cs_mean = cs_data.mean().row(0)[0]
            cs_std = cs_data.std().row(0)[0]
            cs_median = cs_data.median().row(0)[0]
            cs_min = cs_data.min().row(0)[0]
            cs_max = cs_data.max().row(0)[0]
            
            # 计算分位数
            percentile = (cs_data.filter(pl.col(fname) <= target_val).height / len(cs_data) * 100)
            
            # 计算Z-score
            if cs_std > 1e-8:
                z_score = (target_val - cs_mean) / cs_std
            else:
                z_score = 0
            
            factor_analysis.append({
                'factor': fname,
                'target_value': target_val,
                'cs_mean': cs_mean,
                'cs_std': cs_std,
                'cs_median': cs_median,
                'percentile': percentile,
                'z_score': z_score,
            })
        
        # 转换为DataFrame
        analysis_df = pl.DataFrame(factor_analysis)
        
        # 4. 按Z-score绝对值排序，找出最突出的因子
        print(f"\n4. 最突出的因子 (|Z-score| > 1.5):")
        print("=" * 80)
        
        extreme_factors = analysis_df.filter(pl.col("z_score").abs() > 1.5).sort("z_score", descending=True)
        
        if len(extreme_factors) > 0:
            print(f"\n{'因子':30s} {'目标值':>12s} {'市场均值':>12s} {'Z-score':>10s} {'分位数':>10s}")
            print("-" * 80)
            
            for row in extreme_factors.iter_rows(named=True):
                print(f"{row['factor']:30s} {row['target_value']:12.4f} {row['cs_mean']:12.4f} {row['z_score']:10.2f} {row['percentile']:9.1f}%")
        else:
            print("没有|Z-score| > 1.5的因子")
        
        # 5. 分类分析
        print(f"\n\n5. 按因子类别分析:")
        print("=" * 80)
        
        factor_categories = {
            '动量/反转因子': ['mom_5d', 'mom_20d', 'mom_60d', 'mom_120d', 'rev_5d', 'ma_bias_120', 
                           'price_zscore_20d', 'bias_5', 'bias_10', 'bias_20', 'bias_60'],
            '波动率因子': ['volatility_20d', 'volatility_60d', 'volatility_120d'],
            '换手率因子': ['turnover_20d', 'turnover_60d', 'turnover_x_bull'],
            '量价关系': ['volume_ratio_5d', 'volume_ratio_20d', 'amount_ratio_5d', 'amount_ratio_20d'],
            'Beta因子': ['beta_20d', 'beta_60d'],
            '相关性因子': ['corr_mv_20d', 'corr_vol_20d'],
            '技术指标': ['rsi_14d', 'atr_14d', 'kdj_k', 'kdj_d'],
        }
        
        for category, factor_list in factor_categories.items():
            print(f"\n{category}:")
            print("-" * 80)
            
            for fname in factor_list:
                match = analysis_df.filter(pl.col("factor") == fname)
                if len(match) > 0:
                    row = match.row(0, named=True)
                    flag = "⚠️ " if abs(row['z_score']) > 1.5 else "  "
                    print(f"{flag}{fname:28s}: {row['target_value']:10.4f}  (Z:{row['z_score']:7.2f}, 分位:{row['percentile']:6.1f}%)")
        
        # 6. 时序稳定性分析
        print(f"\n\n6. 因子时序稳定性分析 (最近60天):")
        print("=" * 80)
        
        recent_60 = factor_data.filter(
            pl.col("datetime") >= (latest_date - timedelta(days=90))
        ).filter(
            pl.col("vt_symbol") == vt_symbol
        ).sort("datetime", descending=True)
        
        if len(recent_60) > 0:
            print(f"\n{vt_symbol} 最近60天因子均值:")
            
            key_factors = ['mom_20d', 'mom_60d', 'mom_120d', 'volatility_20d', 'turnover_20d']
            
            for fname in key_factors:
                if fname in recent_60.columns:
                    mean_val = recent_60.select(fname).mean().row(0)[0]
                    std_val = recent_60.select(fname).std().row(0)[0]
                    latest_val = recent_60.select(fname).head(1).row(0)[0]
                    
                    if mean_val is not None:
                        print(f"  {fname:25s}: 均值={mean_val:10.4f}, 标准差={std_val:10.4f}, 最新={latest_val:10.4f}")
        
        # 7. 总结
        print(f"\n\n7. 分析总结:")
        print("=" * 80)
        
        # 找出贡献最大的因子
        top_positive = extreme_factors.head(5)
        top_negative = extreme_factors.tail(5)
        
        print("\n推高信号的主要因素 (Z-score最高的5个):")
        for row in top_positive.iter_rows(named=True):
            print(f"  {row['factor']:30s}: Z={row['z_score']:.2f}, 分位={row['percentile']:.1f}%")
        
        print("\n拉低信号的主要因素 (Z-score最低的5个):")
        for row in top_negative.iter_rows(named=True):
            print(f"  {row['factor']:30s}: Z={row['z_score']:.2f}, 分位={row['percentile']:.1f}%")
        
        # 判断信号高的主要原因
        mom_factors = [f for f in extreme_factors['factor'] if 'mom' in f or 'bias' in f]
        vol_factors = [f for f in extreme_factors['factor'] if 'vol' in f]
        turnover_factors = [f for f in extreme_factors['factor'] if 'turnover' in f]
        
        print(f"\n主要原因判断:")
        if mom_factors:
            print(f"  ✓ 动量因子驱动: {', '.join(mom_factors[:3])}")
        if vol_factors:
            print(f"  ✓ 波动率特征: {', '.join(vol_factors[:3])}")
        if turnover_factors:
            print(f"  ✓ 换手率特征: {', '.join(turnover_factors[:3])}")
        
    except Exception as e:
        import traceback
        print(f"❌ 因子分析失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    deep_factor_analysis()
