import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import polars as pl
from vnpy.trader.database import get_database
from vnpy.trader.constant import Interval, Exchange
from vnpy.alpha.lab import AlphaLab

from core.alpha.factor_calculator import FactorCalculator
from core.alpha.mlp_signals import MLPSignals
from core.selector import FundamentalSelector
from core.alpha.data_loader import DataLoader
from data_manager.daily_basic_manager import DailyBasicManager

ALPHA_DB_PATH = "core/alpha_db"

class AlphaEngine:
    def __init__(self, factor_calculator: FactorCalculator, mlp_signals: MLPSignals, selector: FundamentalSelector, signal_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.project_root = Path(os.getcwd())
        self.lab_path = self.project_root / ALPHA_DB_PATH
        self.lab = AlphaLab(str(self.lab_path))
        self.selector = selector
        self.factor_calculator = factor_calculator
        self.mlp_signals = mlp_signals
        self.signal_name = signal_name
        self.database = get_database()
        self.data_loader = DataLoader(self.lab)
        # 1. Configuration & Scope
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            # Load enough history for training (e.g., 3 years)
            start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
        
        self.start_date = start_date
        self.end_date = end_date

    def sync_data(self, start_date: Optional[datetime] = None, end_date: datetime = datetime.now()):
        """
        Sync data from vnpy database to AlphaLab parquet files.
        """
        symbols = self.selector.get_candidate_symbols()
        if not symbols:
            print("[AlphaEngine] No symbols found in selector.")
            return

        # Determine overall range if not provided
        if not start_date:
            s, _ = self.selector.get_data_range()
            start_date = s if s else datetime(2020, 12, 24)
        
        print(f"[AlphaEngine] Syncing data for {len(symbols)} symbols from {start_date} to {end_date}...")

        for vt_symbol in symbols:
            symbol, exchange_str = vt_symbol.split(".")
            exchange = Exchange(exchange_str)
            
            bars = self.database.load_bar_data(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.DAILY,
                start=start_date,
                end=end_date
            )
            
            if bars:
                self.lab.save_bar_data(bars)
        
        print("[AlphaEngine] Data sync complete.")

    def calculate_factors(self) -> pl.DataFrame:
        """
        Calculate factors and save them as signals/datasets.
        This is a placeholder for the actual research workflow.
        """

        print(f"[AlphaEngine] Range: {self.start_date} to {self.end_date}")

        # 2. Get Symbols
        symbols = self._get_ashare_symbols()
        if not symbols:
            print("[AlphaEngine] No symbols found.")
            raise ValueError("No A-share symbols found.")
        print(f"[AlphaEngine] Symbols: {len(symbols)}")
        
        # 3. Load Data
        df = self.data_loader.load_ashare_data(symbols, self.start_date, self.end_date)
        if df.is_empty():
            print("[AlphaEngine] No data loaded.")
            raise ValueError("No data loaded.")


        return self.factor_calculator.calculate_features(df)

    def analyze_factor_performance(self, factors_df: pl.DataFrame, threshold: float = 0.02) -> pl.DataFrame:
        """
        因子绩效分析（仅展示）
        
        Args:
            factors_df: 包含因子和label的DataFrame
            threshold: (仅用于展示高亮) IC绝对值阈值
            
        Returns:
            pl.DataFrame: 返回包含所有因子的原始DataFrame（不进行剔除，防止Look-ahead Bias）
        """
        print("\n=== 因子绩效分析 (仅供参考，不进行预筛选) ===")
        
        if factors_df.is_empty():
            print("无因子数据可分析")
            return factors_df
            
        # 1. Deep Copy for Analysis (防止污染原始数据)
        df_analysis = factors_df.clone()
        
        # 准备数据：计算未来收益率 (5日)用于IC计算
        print("计算未来5日收益率作为基准...")
        if "label" in df_analysis.columns:
             # Label is already calculated (normalized future return)
             df_calc = df_analysis.with_columns(pl.col("label").alias("next_ret"))
        elif "close" in df_analysis.columns:
            df_calc = df_analysis.with_columns([
                ((pl.col("close").shift(-5).over("vt_symbol") / pl.col("close")) - 1).alias("next_ret")
            ])
        else:
            print("无法计算未来收益率：缺少 'close' 或 'label' 列")
            return factors_df
        
        # 去除无效数据用于统计
        df_calc = df_calc.filter(pl.col("next_ret").is_not_null())
        
        if df_calc.is_empty():
            print("有效数据不足进行IC分析")
            return factors_df

        # 分析各因子的IC（信息系数）
        # 排除非因子列
        exclude_cols = ["datetime", "vt_symbol", "close", "open", "high", "low", "volume", "next_ret", "label"]
        # 确保只分析原始factors_df中存在的列
        factor_cols = [col for col in factors_df.columns if col not in exclude_cols]
        
        print(f"正在分析 {len(factor_cols)} 个因子的全局IC (Mean Rank IC & ICIR)...")
        print("注意：此分析基于全样本数据，仅供观察因子整体质量，不用于模型特征筛选。")
        
        ic_results = []
        
        # 计算每日截面 Rank IC
        ic_exprs = [
            pl.corr(pl.col(f).rank(), pl.col("next_ret").rank()).alias(f) 
            for f in factor_cols
        ]
        
        try:
            # 1. Calculate Daily ICs
            daily_ics = df_calc.group_by("datetime").agg(ic_exprs)
            
            # 2. Aggregate Results (Mean, Std -> IR)
            stats = daily_ics.select([
                pl.col(f).fill_nan(pl.lit(None)).mean().alias(f"{f}_mean") for f in factor_cols
            ] + [
                pl.col(f).fill_nan(pl.lit(None)).std().alias(f"{f}_std") for f in factor_cols
            ])
            
            stats_row = stats.row(0)
            cols = stats.columns
            
            for f in factor_cols:
                mean_ic = stats_row[cols.index(f"{f}_mean")]
                std_ic = stats_row[cols.index(f"{f}_std")]
                
                if mean_ic is None:
                    continue
                    
                icir = mean_ic / (std_ic + 1e-9)
                
                ic_results.append({"factor": f, "ic": mean_ic, "icir": icir})

        except Exception as e:
            print(f"IC Analysis Failed: {e}")
            # 出错也不影响主流程，返回原数据
            return factors_df 
        
        # 展示结果
        if ic_results:
            # 区分正负因子
            pos_ics = [r for r in ic_results if r["ic"] > 0]
            neg_ics = [r for r in ic_results if r["ic"] < 0]

            # 正向因子：从大到小排序 (Strong -> Weak)
            pos_ics.sort(key=lambda x: x["ic"], reverse=True)
            # 负向因子：从小到大排序 (Strong Negative -> Weak Negative)
            neg_ics.sort(key=lambda x: x["ic"], reverse=False)
            
            print(f"\n因子表现概览:")
            
            # 1. Top 5 正向强相关
            print(f"\n[Top 5 正向强相关 (IC > 0, Descending)]:")
            for r in pos_ics[:5]:
                print(f"  {r['factor']}: IC {r['ic']:.4f}, ICIR {r['icir']:.4f}")
                
            # 2. Top 5 负向强相关
            print(f"\n[Top 5 负向强相关 (IC < 0, Ascending)]:")
            for r in neg_ics[:5]:
                print(f"  {r['factor']}: IC {r['ic']:.4f}, ICIR {r['icir']:.4f}")
                
            # 3. Top 5 正向最弱 (Closest to 0)
            print(f"\n[Top 5 正向最弱 (Close to 0)]:")
            # 取最后5个，并按IC从小到大(接近0到远离0)排序展示
            for r in sorted(pos_ics[-5:], key=lambda x: x["ic"]):
                print(f"  {r['factor']}: IC {r['ic']:.4f}, ICIR {r['icir']:.4f}")

            # 4. Top 5 负向最弱 (Closest to 0)
            print(f"\n[Top 5 负向最弱 (Close to 0)]:")
            # 取最后5个(它们是接近0的)，按绝对值从小到大排序展示
            for r in sorted(neg_ics[-5:], key=lambda x: abs(x["ic"])):
                print(f"  {r['factor']}: IC {r['ic']:.4f}, ICIR {r['icir']:.4f}")
            
            # 统计达标数量
            qualified = sum(1 for r in ic_results if abs(r["ic"]) >= threshold)
            print(f"\n统计: {qualified} / {len(factor_cols)} 个因子 |IC| >= {threshold}")
        else:
             print("无有效IC结果。")

        # 关键修改：返回原始 factors_df (包含所有列)，不进行剔除
        # 让后续的 MLP 模型在滚动训练中自己决定如何使用这些特征
        return factors_df

    def calculate_signals(self, factor_df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate signals using ML models based on factor DataFrame.
        """
        signal_df = self.mlp_signals.generate_signals(factor_df, self.start_date)
        return signal_df

    def get_signal_df(self, name: str) -> Optional[pl.DataFrame]:
        return self.lab.load_signal(name)


    def _get_ashare_symbols(self) -> List[str]:
        """获取A股标的（过滤ST、次新股等）"""
        all_symbols = self.selector.get_candidate_symbols()
        
        # A股代码过滤规则
        ashare_symbols = []
        for symbol in all_symbols:
            # 只保留沪深A股（代码以特定前缀开头）
            if any(symbol.startswith(prefix) for prefix in ['000', '002', '300', '600', '601', '603', '688']):
                ashare_symbols.append(symbol)
        
        return ashare_symbols

    def save_factors(self, signal_df):
        if signal_df is not None:
            print(f"[AlphaEngine] Saving signals to '{self.signal_name}'...")
            self.lab.save_signal(self.signal_name, signal_df)
            print("[AlphaEngine] Saved.")