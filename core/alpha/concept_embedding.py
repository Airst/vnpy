import polars as pl
from datetime import datetime, timedelta
from typing import List
import pandas as pd
from data_manager.ts_downloader.concept_manager import ConceptManager

class ConceptEmbedding:
    def __init__(self):
        self.manager = ConceptManager()
        
    def get_concept_features(self, start_date: str, end_date: str) -> pl.DataFrame:
        """
        Calculate Concept-based features for stocks.
        Returns DataFrame with columns: 
        [datetime, vt_symbol, concept_mom_5d, concept_mom_20d, concept_turnover_20d, concept_vol_20d, concept_count]
        """
        print("[ConceptEmbedding] Loading concept data...")
        
        # 1. Load Data
        # Ensure we have enough history for momentum calc (e.g. 60 days)
        try:
            s_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=100)
            s_date_str = s_dt.strftime("%Y%m%d")
            e_date_str = end_date.replace("-", "")
            
            daily_pd = self.manager.load_daily_data(s_date_str, e_date_str)
            member_pd = self.manager.load_member_data()
            
            if daily_pd.empty or member_pd.empty:
                print("[ConceptEmbedding] No concept data found.")
                return pl.DataFrame()
        except Exception as e:
            print(f"[ConceptEmbedding] Error loading data: {e}")
            return pl.DataFrame()

        # 2. Process Concept Daily Data (Calculate Factors per Concept)
        # daily_pd: ts_code (ConceptID), trade_date, close, pct_change, turnover_rate
        
        print("[ConceptEmbedding] Calculating concept factors...")
        
        # Clean Pandas Data First
        # 1. Clean Numeric Columns
        daily_pd['close'] = pd.to_numeric(daily_pd['close'], errors='coerce')
        daily_pd['pct_change'] = pd.to_numeric(daily_pd['pct_change'], errors='coerce')
        daily_pd['turnover_rate'] = pd.to_numeric(daily_pd['turnover_rate'], errors='coerce')
        
        # 2. Clean Date Columns
        daily_pd['trade_date'] = pd.to_datetime(daily_pd['trade_date'], format='%Y%m%d', errors='coerce')
        
        # 3. Drop Invalid Rows
        daily_pd.dropna(subset=['close', 'pct_change', 'turnover_rate', 'trade_date'], inplace=True)
        
        # Convert to Polars
        concept_df = pl.from_pandas(daily_pd)
        concept_df = concept_df.with_columns([
            pl.col("trade_date").alias("datetime"),
            pl.col("close").cast(pl.Float32),
            pl.col("pct_change").cast(pl.Float32),
            pl.col("turnover_rate").cast(pl.Float32)
        ])
        
        # Sort
        concept_df = concept_df.sort(["ts_code", "datetime"])
        
        # Calculate Concept Factors
        # mom_N = close / delay(close, N) - 1
        concept_df = concept_df.with_columns([
            (pl.col("close") / pl.col("close").shift(5).over("ts_code") - 1).alias("con_mom_5"),
            (pl.col("close") / pl.col("close").shift(10).over("ts_code") - 1).alias("con_mom_10"),
            (pl.col("close") / pl.col("close").shift(20).over("ts_code") - 1).alias("con_mom_20"),
            # Normalized Turnover (MA20)
            pl.col("turnover_rate").rolling_mean(20).over("ts_code").alias("con_turnover_20"),
            # Volatility (of returns)
             pl.col("pct_change").rolling_std(20).over("ts_code").alias("con_vol_20")
        ])
        
        # Filter range back to requested start_date
        req_start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        concept_df = concept_df.filter(pl.col("datetime") >= req_start_dt)
        concept_df = concept_df.rename({"ts_code": "concept_code"})
        
        # Add Month Column for Joining
        concept_df = concept_df.with_columns([
            pl.col("datetime").dt.strftime("%Y%m").alias("month")
        ])
        
        # 3. Process Member Data
        # member_pd: ts_code (Concept), con_code (Stock), trade_date
        
        print(f"[ConceptEmbedding] Member PD Shape (Raw): {member_pd.shape}")
        if not member_pd.empty:
             print(f"[ConceptEmbedding] Member PD Date Sample: {member_pd['trade_date'].head().tolist()}")

        # Clean Member Data Date
        # Try converting with coerce to handle potential issues
        member_pd['trade_date'] = pd.to_datetime(member_pd['trade_date'], format='%Y%m%d', errors='coerce')
        
        # Debug: Check na counts
        na_count = member_pd['trade_date'].isna().sum()
        if na_count > 0:
            print(f"[ConceptEmbedding] Warning: {na_count} rows have invalid dates in member_pd and will be dropped.")
            
        member_pd.dropna(subset=['trade_date'], inplace=True)
        print(f"[ConceptEmbedding] Member PD Shape (Cleaned): {member_pd.shape}")
        
        mem_df = pl.from_pandas(member_pd)
        mem_df = mem_df.with_columns([
            pl.col("trade_date").alias("datetime")
        ])

        # Tushare Code to VNPY Symbol Conversion
        # ts_code is Concept, con_code is Stock
        mem_df = mem_df.with_columns(
            pl.col("con_code").str.replace(".SZ", ".SZSE", literal=True)
            .str.replace(".SH", ".SSE", literal=True)
            .str.replace(".BJ", ".BSE", literal=True)
            .alias("vt_symbol")
        )
        mem_df = mem_df.rename({"ts_code": "concept_code"})
        mem_df = mem_df.select(["vt_symbol", "concept_code", "datetime"])
        
        # Add Month Column & Deduplicate
        # We want one snapshot per month. If multiple exist, take the first one (sorted by date).
        mem_df = mem_df.with_columns([
            pl.col("datetime").dt.strftime("%Y%m").alias("month")
        ])
        mem_df = mem_df.sort("datetime")
        mem_df = mem_df.unique(subset=["month", "concept_code", "vt_symbol"], keep="first")
        
        if mem_df.is_empty():
             print("[ConceptEmbedding] Member dataframe is empty after processing.")
             return pl.DataFrame()

        min_mem_date = mem_df["datetime"].min()
        print(f"[ConceptEmbedding] Earliest member date: {min_mem_date}")
        
        # 4. Join & Aggregate Separately to save memory
        print("[ConceptEmbedding] Joining and Aggregating concept data (Chunked)...")
        
        features_list = []

        # Part A: History (Before Member Data Start)
        concept_hist = concept_df.filter(pl.col("datetime") < min_mem_date)
        if not concept_hist.is_empty():
            # Create Snapshot from the earliest available data
            mem_snapshot = mem_df.filter(pl.col("datetime") == min_mem_date).drop(["datetime", "month"])
            # Inner Join
            merged_hist = concept_hist.join(mem_snapshot, on="concept_code", how="inner")
            print(f"[ConceptEmbedding] Backfilled {merged_hist.shape[0]} rows for history.")
            
            if not merged_hist.is_empty():
                print("[ConceptEmbedding] Aggregating history chunk...")
                # Lazy aggregation
                feat_hist = merged_hist.lazy().group_by(["datetime", "vt_symbol"]).agg([
                    pl.col("con_mom_5").mean().alias("concept_mom_5d"),
                    pl.col("con_mom_10").mean().alias("concept_mom_10d"),
                    pl.col("con_mom_20").mean().alias("concept_mom_20d"),
                    pl.col("con_mom_20").max().alias("concept_mom_20d_max"),
                    pl.col("con_mom_20").min().alias("concept_mom_20d_min"),
                    pl.col("con_mom_20").std().alias("concept_mom_20d_std"),
                    pl.col("con_turnover_20").mean().alias("concept_turnover_20d"),
                    pl.col("con_turnover_20").max().alias("concept_turnover_20d_max"),
                    pl.col("con_vol_20").mean().alias("concept_vol_20d"),
                    pl.len().alias("concept_count"),
                    # New Features
                    (pl.col("pct_change") / 100.0).mean().alias("concept_daily_ret"),
                    (pl.col("pct_change") > 3.0).mean().alias("concept_hot_ratio"),
                    pl.col("pct_change").top_k(3).mean().alias("concept_top3_mean"),
                    pl.col("pct_change").std().alias("concept_cohesion")
                ]).collect()
                features_list.append(feat_hist)
                del merged_hist # Free memory
        
        del concept_hist # Free memory

        # Part B: Recent (On or After Member Data Start)
        concept_recent = concept_df.filter(pl.col("datetime") >= min_mem_date)
        if not concept_recent.is_empty():
            merged_recent = concept_recent.join(
                mem_df.drop("datetime"), 
                on=["month", "concept_code"], 
                how="inner"
            )
            print(f"[ConceptEmbedding] Joined {merged_recent.shape[0]} rows for recent data.")
            
            if not merged_recent.is_empty():
                print("[ConceptEmbedding] Aggregating recent chunk...")
                feat_recent = merged_recent.lazy().group_by(["datetime", "vt_symbol"]).agg([
                    pl.col("con_mom_5").mean().alias("concept_mom_5d"),
                    pl.col("con_mom_10").mean().alias("concept_mom_10d"),
                    pl.col("con_mom_20").mean().alias("concept_mom_20d"),
                    pl.col("con_mom_20").max().alias("concept_mom_20d_max"),
                    pl.col("con_mom_20").min().alias("concept_mom_20d_min"),
                    pl.col("con_mom_20").std().alias("concept_mom_20d_std"),
                    pl.col("con_turnover_20").mean().alias("concept_turnover_20d"),
                    pl.col("con_turnover_20").max().alias("concept_turnover_20d_max"),
                    pl.col("con_vol_20").mean().alias("concept_vol_20d"),
                    pl.len().alias("concept_count"),
                    # New Features
                    (pl.col("pct_change") / 100.0).mean().alias("concept_daily_ret"),
                    (pl.col("pct_change") > 3.0).mean().alias("concept_hot_ratio"),
                    pl.col("pct_change").top_k(3).mean().alias("concept_top3_mean"),
                    pl.col("pct_change").std().alias("concept_cohesion")
                ]).collect()
                features_list.append(feat_recent)
                del merged_recent

        del concept_recent
        
        if not features_list:
             print("[ConceptEmbedding] Merged data is empty.")
             return pl.DataFrame()
             
        # Concatenate results
        print("[ConceptEmbedding] Concatenating result chunks...")
        stock_features = pl.concat(features_list)
        
        # Fill nulls (if some concepts had NaNs)
        stock_features = stock_features.fill_nan(0).fill_null(0)
        
        print(f"[ConceptEmbedding] Generated features for {stock_features.shape[0]} rows.")
        return stock_features
