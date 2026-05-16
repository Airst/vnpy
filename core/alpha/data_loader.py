from typing import List, Optional
from datetime import datetime
import polars as pl
from vnpy.alpha.lab import AlphaLab
from data_manager.ts_downloader.daily_basic_manager import DailyBasicManager
from data_manager.ts_downloader.stock_info_manager import StockInfoManager
from data_manager.ts_downloader.fina_indicator_manager import FinaIndicatorManager
from data_manager.ts_downloader.moneyflow_manager import MoneyFlowManager
from core.alpha.concept_embedding import ConceptEmbedding

class DataLoader:
    def __init__(self, lab: AlphaLab):
        self.lab = lab

    def load_ashare_data(self, symbols: List[str], start_date: str, end_date: str) -> pl.DataFrame:
        """
        加载A股数据，包含财务数据和行情数据
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
        
        # 格式化日期为 YYYYMMDD
        s_date_str = start_date.replace("-", "")
        e_date_str = end_date.replace("-", "")
        if not price_df.is_empty():
            min_date = price_df["datetime"].min()
            if min_date:
                 s_date_str = min_date.strftime("%Y%m%d") #type: ignore

        # 2. 加载每日指标数据（Daily Basic）
        print("加载每日指标数据(Daily Basic)...")
        try:
            db_manager = DailyBasicManager()
            basic_df_pd = db_manager.load_data(symbols, s_date_str, e_date_str)
            
            if not basic_df_pd.empty:
                basic_df = pl.from_pandas(basic_df_pd)
                if "datetime" in basic_df.columns:
                     basic_df = basic_df.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
                
                cols_to_drop = ["close", "ts_code", "trade_date"]
                basic_df = basic_df.drop([c for c in cols_to_drop if c in basic_df.columns])
                
                price_df = price_df.join(basic_df, on=["vt_symbol", "datetime"], how="left")
                
                price_df = price_df.sort(["vt_symbol", "datetime"])
                fill_cols = [c for c in basic_df.columns if c not in ["vt_symbol", "datetime"]]
                price_df = price_df.with_columns([
                    pl.col(col).forward_fill().over("vt_symbol")
                    for col in fill_cols
                ])
                print(f"每日指标数据加载完成，维度: {price_df.shape}")
        except Exception as e:
            print(f"加载每日指标数据失败: {e}")

        # 3. 加载财务指标数据 (Fina Indicator)
        print("加载财务指标数据(Fina Indicator)...")
        try:
            fina_df = self._load_financial_data(symbols, s_date_str, e_date_str)
            if not fina_df.is_empty():
                # 使用 join_asof 按公告日期(ann_date)取最近已发布数据
                # 显式按 [分组列, 时间列] 排序以消除警告并确保匹配准确
                price_df = price_df.sort(["vt_symbol", "datetime"])
                fina_df = fina_df.sort(["vt_symbol", "datetime"])
                
                price_df = price_df.join_asof(
                    fina_df,
                    on="datetime",
                    by="vt_symbol",
                    strategy="backward"
                )
                print(f"财务指标数据加载完成，维度: {price_df.shape}")
        except Exception as e:
            print(f"加载财务指标数据失败: {e}")

        # 4. 加载股票基础信息（Stock Info，含行业信息）
        print("加载股票基础信息(Stock Info)...")
        try:
            stock_manager = StockInfoManager()
            stock_info_pd = stock_manager.load_data(symbols)
            if not stock_info_pd.empty:
                stock_info_df = pl.from_pandas(stock_info_pd)
                if "industry" in stock_info_df.columns:
                    stock_info_df = stock_info_df.select(["vt_symbol", "industry"])
                    price_df = price_df.join(stock_info_df, on="vt_symbol", how="left")
                    print(f"股票基础信息加载完成 (含Industry)")

                    print("构建行业因子(Industry Factors)...")
                    price_df = price_df.sort(["vt_symbol", "datetime"])
                    
                    price_df = price_df.with_columns([
                        (pl.col("close") / pl.col("close").shift(5).over("vt_symbol") - 1).alias("_tmp_mom_5d"),
                        (pl.col("close") / pl.col("close").shift(20).over("vt_symbol") - 1).alias("_tmp_mom_20d"),
                        (pl.col("close") / pl.col("close").shift(60).over("vt_symbol") - 1).alias("_tmp_mom_60d"),
                        (pl.col("close") / pl.col("close").rolling_mean(20).over("vt_symbol") - 1).alias("_tmp_bias_20"),
                        pl.col("turnover_rate").rolling_mean(20).over("vt_symbol").alias("_tmp_turnover_20d"),
                        (pl.col("close") / pl.col("close").shift(1).over("vt_symbol") - 1).alias("_tmp_ret_1")
                    ])
                    price_df = price_df.with_columns(
                        pl.col("_tmp_ret_1").rolling_std(20).over("vt_symbol").alias("_tmp_vol_20d")
                    )

                    ind_group = ["datetime", "industry"]
                    price_df = price_df.with_columns([
                        pl.col("_tmp_mom_5d").mean().over(ind_group).alias("ind_mom_5d"),
                        pl.col("_tmp_mom_20d").mean().over(ind_group).alias("ind_mom_20d"),
                        pl.col("_tmp_mom_60d").mean().over(ind_group).alias("ind_mom_60d"),
                        pl.col("pe").mean().over(ind_group).alias("ind_pe"),
                        pl.col("_tmp_turnover_20d").mean().over(ind_group).alias("ind_turnover_20d"),
                        pl.col("_tmp_vol_20d").mean().over(ind_group).alias("ind_vol_20d"),
                        pl.col("_tmp_bias_20").mean().over(ind_group).alias("ind_bias_20"),
                    ])
                    
                    price_df = price_df.with_columns([
                        (pl.col("_tmp_mom_60d") - pl.col("ind_mom_60d")).alias("ind_rel_mom_60d"),
                        (pl.col("_tmp_mom_20d") - pl.col("ind_mom_20d")).alias("ind_rel_mom_20d"),
                        (pl.col("pe") / (pl.col("ind_pe") + 1e-8)).alias("ind_rel_pe"),
                        (pl.col("_tmp_turnover_20d") / (pl.col("ind_turnover_20d") + 1e-8)).alias("ind_rel_turnover_20d"),
                        (pl.col("_tmp_vol_20d") / (pl.col("ind_vol_20d") + 1e-8)).alias("ind_rel_vol_20d"),
                        (pl.col("_tmp_bias_20") - pl.col("ind_bias_20")).alias("ind_rel_bias_20"),
                    ])
                    
                    tmp_cols = [c for c in price_df.columns if c.startswith("_tmp_")]
                    price_df = price_df.drop(tmp_cols)
                    print("行业因子构建完成")

        except Exception as e:
            print(f"加载股票基础信息失败: {e}")

        # 5. 加载概念因子数据 (Concept Embedding)
        print("加载概念因子数据 (Concept Embedding)...")
        try:
            ce = ConceptEmbedding()
            concept_df = ce.get_concept_features(start_date, end_date)
            if not concept_df.is_empty():
                if "datetime" in concept_df.columns:
                     concept_df = concept_df.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
                
                price_df = price_df.join(concept_df, on=["vt_symbol", "datetime"], how="left")
                # Ensure all required concept columns exist and handle defaults
                concept_cols = [
                    "concept_mom_5d", "concept_mom_10d", "concept_mom_20d", "concept_mom_20d_max", 
                    "concept_mom_20d_min", "concept_mom_20d_std",
                    "concept_turnover_20d", "concept_vol_20d", "concept_count", "concept_daily_ret",
                    "concept_hot_ratio", "concept_top3_mean", "concept_cohesion",
                    "concept_acc_5_mean", "concept_rank_score_mean"
                ]
                
                # Add missing concept columns as 0.0
                for c in concept_cols:
                    if c not in price_df.columns:
                        price_df = price_df.with_columns(pl.lit(0.0).alias(c))
                
                # Fill nulls in existing concept columns
                cols_to_fill = [c for c in concept_cols if c in price_df.columns]
                price_df = price_df.with_columns([pl.col(c).fill_null(0.0).fill_nan(0.0) for c in cols_to_fill])
                
                print("概念因子数据加载完成")
        except Exception as e:
            print(f"加载概念因子数据失败: {e}")
        
        # 6. 加载资金流向数据 (MoneyFlow - Alpha因子)
        print("加载资金流向数据 (MoneyFlow Alpha)...")
        try:
            mf_manager = MoneyFlowManager()
            mf_df_pd = mf_manager.load_data(symbols, s_date_str, e_date_str)
            
            if not mf_df_pd.empty:
                mf_df = pl.from_pandas(mf_df_pd)
                if "datetime" in mf_df.columns:
                    mf_df = mf_df.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
                
                # 移除不需要的列
                cols_to_drop = ["ts_code", "trade_date"]
                mf_df = mf_df.drop([c for c in cols_to_drop if c in mf_df.columns])
                
                price_df = price_df.join(mf_df, on=["vt_symbol", "datetime"], how="left")
                
                # 资金流向列
                mf_cols = [
                    "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount",
                    "buy_md_vol", "buy_md_amount", "sell_md_vol", "sell_md_amount",
                    "buy_lg_vol", "buy_lg_amount", "sell_lg_vol", "sell_lg_amount",
                    "buy_elg_vol", "buy_elg_amount", "sell_elg_vol", "sell_elg_amount",
                    "net_mf_vol", "net_mf_amount"
                ]
                
                # 确保所有资金流向列存在
                for c in mf_cols:
                    if c not in price_df.columns:
                        price_df = price_df.with_columns(pl.lit(0.0).alias(c))
                
                # Fill nulls
                price_df = price_df.with_columns([
                    pl.col(c).fill_null(0.0).fill_nan(0.0) for c in mf_cols if c in price_df.columns
                ])
                
                print(f"资金流向数据加载完成，维度: {price_df.shape}")
        except Exception as e:
            print(f"加载资金流向数据失败: {e}")
        

        # 7. 加载筹码分布数据 (Cyq Perf - Alpha因子)
        print("加载筹码分布数据 (Cyq Perf)...")
        try:
            from data_manager.ts_downloader.cyq_manager import CyqPerfManager
            cyq_manager = CyqPerfManager()
            cyq_df_pd = cyq_manager.load_data(symbols, s_date_str, e_date_str)

            if not cyq_df_pd.empty:
                cyq_df = pl.from_pandas(cyq_df_pd)
                if "datetime" in cyq_df.columns:
                    cyq_df = cyq_df.with_columns(pl.col("datetime").cast(pl.Datetime("us")))

                cols_to_drop = ["ts_code", "trade_date"]
                cyq_df = cyq_df.drop([c for c in cols_to_drop if c in cyq_df.columns])

                price_df = price_df.join(cyq_df, on=["vt_symbol", "datetime"], how="left")

                cyq_cols = ["his_low", "his_high", "cost_5pct", "cost_15pct",
                            "cost_50pct", "cost_85pct", "weight_avg"]
                price_df = price_df.sort(["vt_symbol", "datetime"])
                price_df = price_df.with_columns([
                    pl.col(col).forward_fill().over("vt_symbol")
                    for col in cyq_cols if col in price_df.columns
                ])

                print(f"筹码分布数据加载完成，维度: {price_df.shape}")
        except Exception as e:
            print(f"加载筹码分布数据失败: {e}")

        # 8. 加载融资融券数据 (Margin Detail - 知情交易者信号)
        print("加载融资融券数据 (Margin Detail)...")
        try:
            from data_manager.ts_downloader.margin_manager import MarginManager
            margin_manager = MarginManager()
            margin_df_pd = margin_manager.load_data(symbols, s_date_str, e_date_str)

            if not margin_df_pd.empty:
                margin_df = pl.from_pandas(margin_df_pd)
                if "datetime" in margin_df.columns:
                    margin_df = margin_df.with_columns(pl.col("datetime").cast(pl.Datetime("us")))

                cols_to_drop = ["ts_code", "trade_date"]
                margin_df = margin_df.drop([c for c in cols_to_drop if c in margin_df.columns])

                price_df = price_df.join(margin_df, on=["vt_symbol", "datetime"], how="left")

                margin_cols = ["rzye", "rzmre", "rzche", "rqye", "rqyl", "rqmcl", "rqchl", "rzrqye"]
                price_df = price_df.sort(["vt_symbol", "datetime"])
                price_df = price_df.with_columns([
                    pl.col(col).forward_fill().over("vt_symbol")
                    for col in margin_cols if col in price_df.columns
                ])

                print(f"融资融券数据加载完成，维度: {price_df.shape}")
        except Exception as e:
            print(f"加载融资融券数据失败: {e}")

        return price_df

    def _load_financial_data(self, symbols: List[str], start_date: str, end_date: str) -> pl.DataFrame:
        """加载财务指标数据"""
        try:
            fina_manager = FinaIndicatorManager()
            fina_pd = fina_manager.load_data(symbols, start_date, end_date)
            if fina_pd.empty:
                return pl.DataFrame()
            
            fina_df = pl.from_pandas(fina_pd)
            if "datetime" in fina_df.columns:
                fina_df = fina_df.with_columns(pl.col("datetime").cast(pl.Datetime("us")))
            
            # 移除 ts_code, ann_date, end_date
            cols_to_drop = ["ts_code", "ann_date", "end_date"]
            fina_df = fina_df.drop([c for c in cols_to_drop if c in fina_df.columns])
            
            # 聚合：同一个 datetime + vt_symbol 可能有多个公告（如修正公告），取最后一条
            fina_df = fina_df.unique(subset=["vt_symbol", "datetime"], keep="last")
            
            return fina_df
        except Exception as e:
            print(f"[_load_financial_data] 错误: {e}")
            return pl.DataFrame()
