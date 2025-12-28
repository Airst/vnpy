from datetime import datetime, timedelta
import polars as pl
import numpy as np
from typing import Optional, List, Dict
from tqdm import tqdm

# Import vnpy alpha components
# from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.model.models.mlp_model import MlpModel
from vnpy.alpha import Segment, AlphaDataset

class MLPSignals:

    def __init__(self):
        self.model_settings = {
            "hidden_sizes": (256, 128, 64),
            "n_epochs": 300,  # Adjustable based on needs
            "batch_size": 4096,
            "lr": 0.001,
            "early_stop_rounds": 20,
            "device": "auto"  # Will detect GPU
        }


    def generate_signals(self, dataset_df: pl.DataFrame, start_date: str) -> pl.DataFrame:
        # 6. Rolling Window Loop
        print("[MLPSignals] Starting Rolling Window Training & Prediction...")
        
        dates = dataset_df["datetime"].unique().sort()
        # Increased requirement for 500-day window
        if len(dates) < 550: 
             print(f"[MLPSignals] Not enough dates for rolling window: {len(dates)} (Need ~550)")
             raise ValueError("Insufficient date range for rolling window.")
             
        # Determine start index for prediction
        # We need to align with user requested start_date, but ensure we have 500 days history
        target_start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Find index of first date >= target_start_dt
        start_idx = 0
        for i, d in enumerate(dates):
            if d >= target_start_dt:
                start_idx = i
                break
        
        # Ensure we have 500 days before start_idx
        if start_idx < 500:
            print(f"[MLPSignals] Warning: Not enough history before {start_date} for 500-day training.")
            print(f"[MLPSignals] Adjusting start index to 500 (Date: {dates[500]})")
            start_idx = 500
            
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
            
            print(f"[MLPSignals] Window: Train [500 days pre {pred_start_date.date()}] -> Predict [{pred_start_date.date()} to {pred_end_date.date()}]")
            
            # Define Training Window (Previous 500 indices)
            train_end_idx = curr_idx - 1
            # 500 days total (0 to 499)
            train_start_idx = max(0, train_end_idx - 499) 
            
            # Need at least some valid/test split inside training if we want early stopping?
            # AlphaDataset splits by Train/Valid/Test periods provided.
            # We use Train for Training, Valid for Early Stopping.
            # Let's split the 500 days: 450 Train, 50 Valid?
            
            valid_len = 50
            train_period_end_idx = train_end_idx - valid_len
            
            train_period = (dates[train_start_idx].strftime("%Y-%m-%d"), dates[train_period_end_idx].strftime("%Y-%m-%d"))
            valid_period = (dates[train_period_end_idx + 1].strftime("%Y-%m-%d"), dates[train_end_idx].strftime("%Y-%m-%d"))
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
                        meta = meta.with_columns(pl.Series(preds).alias("total_score"))
                        all_predictions.append(meta)
                    else:
                         print(f"[MLPSignals] Mismatch in prediction length: {len(preds)} vs {len(meta)}")
                except Exception as e:
                    print(f"[MLPSignals] Prediction failed for window: {e}")
            
            # Move to next window
            curr_idx = next_idx

        # 7. Concatenate and Post-process
        if not all_predictions:
            print("[MLPSignals] No predictions generated.")
            raise ValueError("No predictions generated.")
            
        print("[MLPSignals] Aggregating results...")
        full_result = pl.concat(all_predictions)
        
        # Sort by date
        full_result = full_result.sort(["datetime", "vt_symbol"])
        
        # Post-process
        return self._post_process_signals(full_result)
    
    def _clean_label(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop_nulls(subset=["label"])


    def _train_model(self, dataset: AlphaDataset) -> Optional[MlpModel]:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[MLPSignals] Device: {device}")
        
        sample_df = dataset.fetch_learn(Segment.TRAIN)
        if sample_df.is_empty():
            print("[MLPSignals] Training data empty!")
            return None
            
        input_size = len(sample_df.columns) - 3 # datetime, vt_symbol, label
        print(f"[MLPSignals] Input Feature Size: {input_size}")
        
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
            print(f"[MLPSignals] Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _post_process_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns([
            pl.col("total_score").rank(method="average").over("datetime").alias("rank"),
            pl.col("total_score").count().over("datetime").alias("count")
        ])
        
        df = df.with_columns([
            (((pl.col("rank") / pl.col("count")) - 0.5) * 3.46)
            .clip(-3, 3)
            .alias("final_signal")
        ])
        
        return df.select(["datetime", "vt_symbol", "total_score", "final_signal"])