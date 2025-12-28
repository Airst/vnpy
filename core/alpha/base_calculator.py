import polars as pl
from datetime import datetime, timedelta
from typing import List

class BaseAShareFactorCalculator:
    """
    Base class for A-Share Factor Calculators.
    Provides common methods for symbol retrieval and data loading.
    """
    
    def __init__(self, engine):
        self.engine = engine

    def get_ashare_symbols(self) -> List[str]:
        """获取A股标的（过滤ST、次新股等）"""
        all_symbols = self.engine.selector.get_candidate_symbols()
        
        # A股代码过滤规则
        ashare_symbols = []
        for symbol in all_symbols:
            # 只保留沪深A股（代码以特定前缀开头）
            if any(symbol.startswith(prefix) for prefix in ['000', '002', '300', '600', '601', '603', '688']):
                ashare_symbols.append(symbol)
        
        return ashare_symbols

    def load_ashare_data(self, symbols, start_date, end_date) -> pl.DataFrame:
        """加载A股数据，包含财务数据和行情数据"""
        # 扩展天数考虑A股交易特点
        extended_days = 250  # 考虑A股年线计算
        
        print(f"加载数据，扩展天数: {extended_days}")
        
        # 1. 加载行情数据
        price_df = self.engine.lab.load_bar_df(
            vt_symbols=symbols,
            interval="d",
            start=start_date,
            end=end_date,
            extended_days=extended_days
        )
        
        if price_df is None or price_df.is_empty():
            return pl.DataFrame()
        
        # 2. 加载财务数据（如果有的话）
        financial_df = self.load_financial_data(symbols, end_date)
        
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

    def load_financial_data(self, symbols, end_date) -> pl.DataFrame:
        """模拟加载财务数据（实际使用时需要接入财务数据库）"""
        # 这里返回一个空DataFrame，实际使用时需要实现
        return pl.DataFrame()
