import polars as pl
from typing import List, Optional
from datetime import datetime
from vnpy.alpha.lab import AlphaLab
from data_manager.ts_downloader.daily_basic_manager import DailyBasicManager
from data_manager.ts_downloader.stock_info_manager import StockInfoManager
from core.alpha.concept_embedding import ConceptEmbedding

class DataLoader:
    def __init__(self, lab: AlphaLab):
        self.lab = lab

    def load_ashare_data(self, symbols: List[str], start_date: str, end_date: str) -> pl.DataFrame:
        """
        加载A股数据，包含财务数据和行情数据
        
        Returns:
            pl.DataFrame: 包含以下列:
                - vt_symbol: str
                - datetime: datetime
                - open, high, low, close, volume: float
                - turnover, open_interest: float
                - turnover_rate: float (换手率)
                - turnover_rate_f: float (换手率-自由流通)
                - volume_ratio: float (量比)
                - pe: float (市盈率)
                - pe_ttm: float (市盈率TTM)
                - pb: float (市净率)
                - ps: float (市销率)
                - ps_ttm: float (市销率TTM)
                - dv_ratio: float (股息率)
                - dv_ttm: float (股息率TTM)
                - total_share: float (总股本)
                - float_share: float (流通股本)
                - free_share: float (自由流通股本)
                - total_mv: float (总市值)
                - circ_mv: float (流通市值)
                - industry: str (所属行业)
        """
        # 扩展天数考虑A股交易特点
        extended_days = 250  
        
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
        
        # 2. 加载每日指标数据（Daily Basic）
        print("加载每日指标数据(Daily Basic)...")
        try:
            db_manager = DailyBasicManager()
            # 格式化日期为 YYYYMMDD
            s_date = start_date.replace("-", "")
            e_date = end_date.replace("-", "")
            
            # 由于需要计算前向填充，开始时间也尽量往前推一点，或直接使用price_df的最小时间
            if not price_df.is_empty():
                min_date = price_df["datetime"].min()
                if min_date:
                     s_date = min_date.strftime("%Y%m%d") #type: ignore

            basic_df_pd = db_manager.load_data(symbols, s_date, e_date)
            
            if not basic_df_pd.empty:
                # 转换为 Polars
                basic_df = pl.from_pandas(basic_df_pd)
                
                # Align datetime precision to microseconds (us) to match price_df
                # Fix: Explicitly check and cast. Pandas usually results in ns.
                if "datetime" in basic_df.columns:
                     basic_df = basic_df.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
                
                # 处理列名冲突，移除多余列
                # basic_df 包含: vt_symbol, datetime, close, turnover_rate...
                # 移除 close, ts_code, trade_date
                cols_to_drop = ["close", "ts_code", "trade_date"]
                basic_df = basic_df.drop([c for c in cols_to_drop if c in basic_df.columns])
                
                # 合并
                price_df = price_df.join(
                    basic_df,
                    on=["vt_symbol", "datetime"],
                    how="left"
                )
                
                # 前向填充
                price_df = price_df.sort(["vt_symbol", "datetime"])
                # 需要填充的列是 basic_df 的列
                fill_cols = [c for c in basic_df.columns if c not in ["vt_symbol", "datetime"]]
                
                price_df = price_df.with_columns([
                    pl.col(col).forward_fill().over("vt_symbol")
                    for col in fill_cols
                ])
                print(f"每日指标数据加载完成，合并后维度: {price_df.shape}")
            else:
                print("未查询到每日指标数据")
                
        except Exception as e:
            print(f"加载每日指标数据失败: {e}")
            import traceback
            traceback.print_exc()

        # 3. 加载股票基础信息（Stock Info，含行业信息）
        print("加载股票基础信息(Stock Info)...")
        try:
            stock_manager = StockInfoManager()
            stock_info_pd = stock_manager.load_data(symbols)
            
            if not stock_info_pd.empty:
                stock_info_df = pl.from_pandas(stock_info_pd)
                
                # 只保留 vt_symbol 和 industry
                if "industry" in stock_info_df.columns:
                    stock_info_df = stock_info_df.select(["vt_symbol", "industry"])
                    
                    # 合并行业信息
                    price_df = price_df.join(
                        stock_info_df,
                        on="vt_symbol",
                        how="left"
                    )
                    print(f"股票基础信息加载完成 (含Industry)")
                else:
                    print("股票基础信息中未找到 industry 字段")
            else:
                print("未查询到股票基础信息")
                
        except Exception as e:
            print(f"加载股票基础信息失败: {e}")
            import traceback
            traceback.print_exc()

        # 4. 加载其他财务数据（如果有的话）
        financial_df = self._load_financial_data(symbols, end_date)
        
        # 5. 合并其他财务数据
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

        # 6. 加载概念因子数据 (Concept Embedding)
        print("加载概念因子数据 (Concept Embedding)...")
        try:
            ce = ConceptEmbedding()
            concept_df = ce.get_concept_features(start_date, end_date)
            
            if not concept_df.is_empty():
                # Align datetime precision
                if "datetime" in concept_df.columns:
                     concept_df = concept_df.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
                
                price_df = price_df.join(
                    concept_df,
                    on=["vt_symbol", "datetime"],
                    how="left"
                )
                
                # Fill nulls (stocks with no concepts or missing concept data) with 0
                cols_to_fill = [
                    "concept_mom_5d", "concept_mom_10d", "concept_mom_20d", "concept_turnover_20d", "concept_vol_20d", "concept_count",
                    "concept_mom_20d_max", "concept_mom_20d_min", "concept_mom_20d_std"
                ]
                # Only fill columns that exist
                cols_to_fill = [c for c in cols_to_fill if c in price_df.columns]
                
                price_df = price_df.with_columns([
                    pl.col(c).fill_null(0).fill_nan(0) for c in cols_to_fill
                ])
                print("概念因子数据加载完成")
            else:
                print("未生成概念因子数据")
        except Exception as e:
            print(f"加载概念因子数据失败: {e}")
        
        return price_df

    def _load_financial_data(self, symbols, end_date) -> pl.DataFrame:
        """模拟加载财务数据（实际使用时需要接入财务数据库）"""
        # 这里返回一个空DataFrame，实际使用时需要实现
        return pl.DataFrame()
