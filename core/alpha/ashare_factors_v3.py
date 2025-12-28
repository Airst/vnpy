import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import polars as pl
import numpy as np
from typing import Optional, List, Dict
from tqdm import tqdm

# Import vnpy alpha components
# from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.model.models.mlp_model import MlpModel
from vnpy.alpha import Segment, AlphaDataset

from core.alpha.features import AshareGPUDataset
from core.alpha.base_calculator import BaseAShareFactorCalculator

class AShareFactorCalculatorV3(BaseAShareFactorCalculator):
    """
    A-Share Factor Calculator V3
    Based on GPU-calculated features and MLP Model.
    """
    
    def __init__(self, engine):
        super().__init__(engine)
        self.model_settings = {
            "hidden_sizes": (256, 128, 64),
            "n_epochs": 300,  # Adjustable based on needs
            "batch_size": 4096,
            "lr": 0.001,
            "early_stop_rounds": 20,
            "device": "auto"  # Will detect GPU
        }

    def calculate_all_factors(self, start_date: str = None, end_date: str = None): # type: ignore
        """
        Main entry point to calculate factors and predict signals using Rolling Window.
        """
        # 1. Configuration & Scope
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            # Load enough history for training (e.g., 3 years)
            start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
            
        print(f"[V3] Range: {start_date} to {end_date}")
        
        # 2. Get Symbols
        symbols = self.get_ashare_symbols()
        if not symbols:
            print("[V3] No symbols found.")
            return None
        print(f"[V3] Symbols: {len(symbols)}")
        
        # 3. Load Data
        df = self.load_ashare_data(symbols, start_date, end_date)
        if df.is_empty():
            print("[V3] No data loaded.")
            return None

        # 4. Calculate Features (GPU)
        print("[V3] Calculating Features on GPU...")
        try:
            calculator = AshareGPUDataset()
            df_features = calculator.calculate_features(df)
        except Exception as e:
            print(f"[V3] Feature calculation error: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        # 5. Pre-process Data (Global Normalization)
        print("[V3] Pre-processing data (Global Cross-Sectional Normalization)...")
        try:
            # Identify feature columns
            exclude_cols = {"datetime", "vt_symbol", "label"}
            raw_cols = ["open", "high", "low", "close", "volume", "turnover", "open_interest"]
            existing_raw = [c for c in raw_cols if c in df_features.columns]
            
            # Keep features and label, drop raw columns
            dataset_df = df_features.drop(existing_raw)
            
            if "label" not in dataset_df.columns:
                 print("[V3] Error: 'label' column missing in features.")
                 return None

            feature_cols = [c for c in dataset_df.columns if c not in exclude_cols]
            feature_cols.sort()
            
            # Final column order
            final_cols = ["datetime", "vt_symbol"] + feature_cols + ["label"]
            dataset_df = dataset_df.select(final_cols)
            
            # Apply Normalization Globally
            dataset_df = self._normalize_data(dataset_df, feature_cols)
            
        except Exception as e:
            print(f"[V3] Data pre-processing error: {e}")
            import traceback
            traceback.print_exc()
            return None

        # 6. Rolling Window Loop
        print("[V3] Starting Rolling Window Training & Prediction...")
        
        dates = dataset_df["datetime"].unique().sort()
        if len(dates) < 150: # Need at least 120 + some buffer
             print(f"[V3] Not enough dates for rolling window: {len(dates)}")
             return None
             
        # Determine start index for prediction
        # We need to align with user requested start_date, but ensure we have 120 days history
        target_start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Find index of first date >= target_start_dt
        start_idx = 0
        for i, d in enumerate(dates):
            if d >= target_start_dt:
                start_idx = i
                break
        
        # Ensure we have 120 days before start_idx
        if start_idx < 120:
            print(f"[V3] Warning: Not enough history before {start_date} for 120-day training.")
            print(f"[V3] Adjusting start index to 120 (Date: {dates[120]})")
            start_idx = 120
            
        all_predictions = []
        
        # Rolling Loop
        # Step: 1 month (approx 20 trading days or just calendar month check)
        # We iterate by index, but we need to jump by month. 
        # Easier: Iterate current_date from start to end by month, find indices.
        
        curr_idx = start_idx
        total_dates = len(dates)
        
        while curr_idx < total_dates:
            # Define Prediction Window
            pred_start_date = dates[curr_idx]
            
            # Next month date
            next_month_date = pred_start_date + timedelta(days=30)
            
            # Find index for next month (end of this prediction window)
            next_idx = total_dates # Default to end
            for i in range(curr_idx, total_dates):
                if dates[i] >= next_month_date:
                    next_idx = i
                    break
            
            pred_end_idx = next_idx - 1
            if pred_end_idx < curr_idx:
                pred_end_idx = curr_idx # At least one day
                
            pred_end_date = dates[pred_end_idx]
            
            print(f"[V3] Window: Train [120 days pre {pred_start_date.date()}] -> Predict [{pred_start_date.date()} to {pred_end_date.date()}]")
            
            # Define Training Window (Previous 120 indices)
            train_end_idx = curr_idx - 1
            train_start_idx = train_end_idx - 119 # 120 days total (0 to 119)
            
            train_period = (dates[train_start_idx].strftime("%Y-%m-%d"), dates[train_end_idx - 20].strftime("%Y-%m-%d"))
            valid_period = (dates[train_end_idx - 19].strftime("%Y-%m-%d"), dates[train_end_idx].strftime("%Y-%m-%d"))
            test_period = (dates[curr_idx].strftime("%Y-%m-%d"), dates[pred_end_idx].strftime("%Y-%m-%d"))
            
            # Construct Dataset for this window
            # Note: passing the WHOLE dataset_df is fine, AlphaDataset filters by period
            dataset = AlphaDataset(
                df=dataset_df,
                train_period=train_period,
                valid_period=valid_period,
                test_period=test_period
            )

            # Manual initialization since we skip prepare_data()
            dataset.raw_df = dataset_df
            dataset.infer_df = dataset_df
            
            # Add label cleaner (only for learning)
            dataset.add_processor("learn", self._clean_label)
            dataset.process_data()
            
            # Train Model
            model = self._train_model(dataset)
            
            if model:
                # Predict
                try:
                    preds = model.predict(dataset, Segment.TEST)
                    meta = dataset.fetch_infer(Segment.TEST).select(["datetime", "vt_symbol"])
                    
                    if len(preds) == len(meta):
                        meta = meta.with_columns(pl.Series(preds).alias("raw_score"))
                        all_predictions.append(meta)
                    else:
                         print(f"[V3] Mismatch in prediction length: {len(preds)} vs {len(meta)}")
                except Exception as e:
                    print(f"[V3] Prediction failed for window: {e}")
            
            # Move to next window
            curr_idx = next_idx

        # 7. Concatenate and Post-process
        if not all_predictions:
            print("[V3] No predictions generated.")
            return None
            
        print("[V3] Aggregating results...")
        full_result = pl.concat(all_predictions)
        
        # Sort by date
        full_result = full_result.sort(["datetime", "vt_symbol"])
        
        # Post-process
        final_df = self._post_process_signals(full_result)
        
        # 8. Analyze (Optional, on the last model or aggregate? Aggregate analysis is better)
        # We can't easily run the old analyze_factor_performance because it expects a dataset with train data.
        # We can analyze the IC of the Final Output if we merge labels back.
        # For now, skip analysis or do a simple IC check if labels align.
        
        # 9. Save & Return
        self.save_factors(final_df)
        
        return final_df

    def _normalize_data(self, df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
        """Apply Cross-sectional Z-Score Normalization"""
        # 1. Replace Inf and Fill NaN
        fill_exprs = [
            pl.when(pl.col(c).is_infinite())
            .then(0)
            .otherwise(pl.col(c))
            .fill_nan(0)
            .fill_null(0)
            .alias(c)
            for c in feature_cols
        ]
        
        # 2. Cross-sectional Standardization (Z-Score)
        norm_exprs = []
        for col in feature_cols:
            mean = pl.col(col).mean().over("datetime")
            std = pl.col(col).std().over("datetime")
            expr = ((pl.col(col) - mean) / (std + 1e-8)).clip(-3, 3).fill_nan(0).fill_null(0).alias(col)
            norm_exprs.append(expr)
        
        return (
            df.lazy()
            .with_columns(fill_exprs)
            .with_columns(norm_exprs)
            .collect()
        )

    def _clean_label(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop_nulls(subset=["label"])


    def _train_model(self, dataset: AlphaDataset) -> Optional[MlpModel]:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[V3] Device: {device}")
        
        sample_df = dataset.fetch_learn(Segment.TRAIN)
        if sample_df.is_empty():
            print("[V3] Training data empty!")
            return None
            
        input_size = len(sample_df.columns) - 3 # datetime, vt_symbol, label
        print(f"[V3] Input Feature Size: {input_size}")
        
        model = MlpModel(
            input_size=input_size,
            hidden_sizes=self.model_settings["hidden_sizes"],
            n_epochs=self.model_settings["n_epochs"],
            batch_size=self.model_settings["batch_size"],
            lr=self.model_settings["lr"],
            early_stop_rounds=self.model_settings["early_stop_rounds"],
            device=device,
            seed=42
        )
        
        try:
            model.fit(dataset)
            return model
        except Exception as e:
            print(f"[V3] Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _post_process_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns([
            pl.col("raw_score").rank(method="average").over("datetime").alias("rank"),
            pl.col("raw_score").count().over("datetime").alias("count")
        ])
        
        df = df.with_columns([
            (((pl.col("rank") / pl.col("count")) - 0.5) * 3.46)
            .clip(-3, 3)
            .alias("final_signal")
        ])
        
        return df.select(["datetime", "vt_symbol", "raw_score", "final_signal"])

    def save_factors(self, signal_df):
        if signal_df is not None:
            print("[V3] Saving signals to 'ashare_mlp_v3'...")
            self.engine.lab.save_signal("ashare_mlp_v3", signal_df)
            print("[V3] Saved.")
