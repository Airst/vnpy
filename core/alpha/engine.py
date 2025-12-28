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
        # 1. Configuration & Scope
        if not end_date:
            self.end_date = datetime.now().strftime("%Y-%m-%d")
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
        df = self._load_ashare_data(symbols, self.start_date, self.end_date)
        if df.is_empty():
            print("[AlphaEngine] No data loaded.")
            raise ValueError("No data loaded.")


        return self.factor_calculator.calculate_features(df)

    def analyze_factor_performance(self, factors_df):
        """因子绩效分析（简化版）"""
        print("\n=== 因子绩效分析 ===")
        
        if factors_df.is_empty():
            print("无因子数据可分析")
            return
        
        # 准备数据：计算未来收益率 (5日)
        print("计算未来5日收益率作为基准...")
        if "label" in factors_df.columns:
            df = factors_df.with_columns(pl.col("label").alias("next_ret"))
        elif "close" in factors_df.columns:
            df = factors_df.with_columns([
                ((pl.col("close").shift(-5).over("vt_symbol") / pl.col("close")) - 1).alias("next_ret")
            ])
        else:
            print("无法计算未来收益率：缺少 'close' 或 'label' 列")
            return
        
        # 去除无效数据
        df = df.filter(pl.col("next_ret").is_not_null())
        
        if df.is_empty():
            print("有效数据不足")
            return

        # 分析各因子的IC（信息系数）
        # 排除非因子列
        exclude_cols = ["datetime", "vt_symbol", "close", "open", "high", "low", "volume", "next_ret", "label"]
        factor_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"分析 {len(factor_cols)} 个因子的IC (Rank IC)...")
        
        ic_results = []
        
        for factor in factor_cols:
            try:
                # 计算 Rank IC (Spearman Correlation)
                # Polars corr 默认为 Pearson，先 rank 再 corr 近似 Spearman
                ic = df.select(
                    pl.corr(
                        pl.col(factor).rank(), 
                        pl.col("next_ret").rank()
                    )
                ).item()
                
                ic_results.append({"factor": factor, "ic": ic})
                print(f"  - {factor}: IC = {ic:.4f}")
            except Exception as e:
                print(f"  - {factor}: 计算失败 ({e})")
        
        # 简单总结
        if ic_results:
            valid_results = [r for r in ic_results if r["ic"] is not None]
            avg_ic = sum(r["ic"] for r in valid_results) / len(valid_results)
            print(f"\n平均 IC: {avg_ic:.4f}")

            # Sort by IC descending
            valid_results.sort(key=lambda x: x["ic"], reverse=True)
            
            print("\nTop 5 正向因子:")
            for r in valid_results[:5]:
                print(f"  {r['factor']}: {r['ic']:.4f}")
                
            print("\nTop 5 负向因子:")
            # Last 5, reversed to show most negative first
            for r in reversed(valid_results[-5:]):
                print(f"  {r['factor']}: {r['ic']:.4f}")
        
        print("分析完成")

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

    def _load_ashare_data(self, symbols, start_date, end_date) -> pl.DataFrame:
        """加载A股数据，包含财务数据和行情数据"""
        # 扩展天数考虑A股交易特点
        extended_days = 250  # 考虑A股年线计算
        
        print(f"加载数据，扩展天数: {extended_days}")
        
        # 1. 加载行情数据
        price_df = self.lab.load_bar_df(
            vt_symbols=symbols,
            interval="d",
            start=start_date,
            end=end_date,
            extended_days=extended_days
        )
        
        if price_df is None or price_df.is_empty():
            return pl.DataFrame()
        
        # 2. 加载财务数据（如果有的话）
        financial_df = self._load_financial_data(symbols, end_date)
        
        # 3. 合并数据
        if not financial_df.is_empty():
            # 对财务数据进行前向填充（季度数据填充到日频）
            financial_df = financial_df.sort(["vt_symbol", "report_date"])
            price_df = price_df.join(
                financial_df,
                on=["vt_symbol", "datetime"],
                how="left"
            )
            
            # 前向填充财务数据
            price_df = price_df.sort(["vt_symbol", "datetime"])
            price_df = price_df.with_columns([
                pl.col(col).forward_fill().over("vt_symbol")
                for col in financial_df.columns if col not in ["vt_symbol", "datetime"]
            ])
        
        return price_df

    def _load_financial_data(self, symbols, end_date) -> pl.DataFrame:
        """模拟加载财务数据（实际使用时需要接入财务数据库）"""
        # 这里返回一个空DataFrame，实际使用时需要实现
        return pl.DataFrame()


    def save_factors(self, signal_df):
        if signal_df is not None:
            print(f"[AlphaEngine] Saving signals to '{self.signal_name}'...")
            self.lab.save_signal(self.signal_name, signal_df)
            print("[AlphaEngine] Saved.")