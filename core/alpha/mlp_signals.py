from datetime import datetime, timedelta
import polars as pl
import numpy as np
from typing import Optional, List, Dict
from tqdm import tqdm
import concurrent.futures
import os
import glob
import re
import torch
from pathlib import Path

# Import vnpy alpha components
# from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.model.models.mlp_model import MlpModel
from vnpy.alpha import Segment, AlphaDataset
from vnpy.alpha.lab import AlphaLab

def set_seed(seed: int = 42):
    import random
    import numpy as np
    import torch
    import os
    
    # Required for deterministic algorithms in CuBLAS
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"[MLPSignals] Warning: Could not enable deterministic algorithms: {e}")
        
    print(f"[MLPSignals] Random seed set to {seed} (Deterministic Mode)")

class MLPSignals:

    def __init__(self, signal_name: str = "mlp_signal", force_retrain: bool = False):
        self.signal_name = signal_name
        self.force_retrain = force_retrain
        # self.model_dir = Path("core/alpha_db/model") # Managed by AlphaLab
        
        #self.model_settings = {
        #    "hidden_sizes": (256, 128, 64),
        #    "n_epochs": 400,  # Adjustable based on needs
        #    "batch_size": 4096,
        #    "lr": 0.001,
        #    "early_stop_rounds": 20,
        #    "device": "auto"  # Will detect GPU
        #}
        self.model_settings = {
            "hidden_sizes": (256, 128, 64),  # 保持结构
            "n_epochs": 800,                  # 增加训练轮数
            "batch_size": 2048,              # 减小批量大小
            "lr": 0.0005,                    # 降低学习率（微调）
            "early_stop_rounds": 30,         # 增加早停耐心
            "weight_decay": 0.0001,          # 添加轻微正则化
            "optimizer": "adam"             # 如果有的话
        }
        self.n_jobs = 1  # 改为单线程以保证结果可复现 (多线程下全局Seed会被频繁重置导致随机性)


    def generate_signals(self, dataset_df: pl.DataFrame, start_date: str, lab: AlphaLab) -> pl.DataFrame:
        set_seed(42)  # Ensure reproducibility
        
        if self.force_retrain:
            print("[MLPSignals] force_retrain, remove old signals.")
            lab.remove_signal(self.signal_name)
        
        # Drop 'industry' if present (MlpModel only supports numeric features)
        if "industry" in dataset_df.columns:
            print("[MLPSignals] Dropping 'industry' column for model training.")
            dataset_df = dataset_df.drop("industry")

        # 6. Rolling Window Loop
        print("[MLPSignals] Starting Rolling Window Training & Prediction...")
        
        dates = dataset_df["datetime"].unique().sort().to_list()
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
        tasks = []
        
        # Pre-calculate tasks
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
            
            # Format dates for logging
            ps_str = pred_start_date.strftime("%Y-%m-%d") if hasattr(pred_start_date, "strftime") else str(pred_start_date)
            pe_str = pred_end_date.strftime("%Y-%m-%d") if hasattr(pred_end_date, "strftime") else str(pred_end_date)
            
            # Define Training Window (Previous 500 indices)
            train_end_idx = curr_idx - 1
            # 500 days total (0 to 499)
            train_start_idx = max(0, train_end_idx - 499) 
            
            valid_len = 50
            train_period_end_idx = train_end_idx - valid_len
            
            # Use raw datetime objects to avoid string ambiguity and ensure precision
            train_period = (dates[train_start_idx], dates[train_period_end_idx])
            valid_period = (dates[train_period_end_idx + 1], dates[train_end_idx])
            test_period = (dates[curr_idx], dates[pred_end_idx])
            
            task_info = {
                "train_period": train_period,
                "valid_period": valid_period,
                "test_period": test_period,
                "ps_str": ps_str,
                "pe_str": pe_str,
                "save_model": False
            }
            tasks.append(task_info)
            
            # Move to next window
            curr_idx = next_idx

        # Mark the last task to save model
        if tasks:
            tasks[-1]["save_model"] = True

        if not self.force_retrain:
            # Incremental mode: Only process the last window
            if not tasks:
                 print("[MLPSignals] No tasks generated.")
                 raise ValueError("No tasks generated.")
            
            last_task = tasks[-1]
            print(f"[MLPSignals] Incremental Mode: Processing only latest window ({last_task['ps_str']} - {last_task['pe_str']})")
            
            # Check for existing model
            model_name = f"{self.signal_name}_{last_task['ps_str']}"
            existing_models = lab.list_all_models()
            
            if model_name in existing_models:
                print(f"[MLPSignals] Found existing model: {model_name}. Loading...")
                result = self._predict_with_existing_model(dataset_df, last_task, lab)
            else:
                print(f"[MLPSignals] Model not found ({model_name}). Training new model...")
                result = self._train_and_predict_window(dataset_df, last_task, lab)
            
            if result is not None:
                all_predictions.append(result)
                
        else:
            # Full Rolling Mode
            print(f"[MLPSignals] Force Retrain Mode: Executing {len(tasks)} windows with {self.n_jobs} threads...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self._train_and_predict_window, dataset_df, t, lab): t 
                    for t in tasks
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks)):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        if result is not None:
                            all_predictions.append(result)
                    except Exception as exc:
                        print(f"[MLPSignals] Task {task['ps_str']} generated an exception: {exc}")

        # 7. Concatenate and Post-process
        if not all_predictions:
            print("[MLPSignals] No predictions generated.")
            # raise ValueError("No predictions generated.") 
            # Allow empty if really nothing new
            return pl.DataFrame()
            
        print("[MLPSignals] Aggregating results...")
        full_result = pl.concat(all_predictions)
        
        # Sort by date
        full_result = full_result.sort(["datetime", "vt_symbol"])
        
        # Post-process
        return self._post_process_signals(full_result)

    def _save_model(self, lab: AlphaLab, model: MlpModel, start_date_str: str, end_date_str: str):
        model_name = f"{self.signal_name}_{start_date_str}"
        print(f"[MLPSignals] Saving model {model_name}...")
        try:
            lab.save_model(model_name, model)
        except Exception as e:
            print(f"[MLPSignals] Failed to save model: {e}")

    def _load_model(self, lab: AlphaLab, start_date_str: str, end_date_str: str) -> Optional[MlpModel]:
        model_name = f"{self.signal_name}_{start_date_str}"
        try:
            model = lab.load_model(model_name)
            return model
        except Exception as e:
            print(f"[MLPSignals] Failed to load model {model_name}: {e}")
            return None

    def _predict_with_existing_model(self, dataset_df: pl.DataFrame, task_info: Dict, lab: AlphaLab) -> Optional[pl.DataFrame]:
        # Load model
        ps_str = task_info["ps_str"]
        pe_str = task_info["pe_str"]
        
        model = self._load_model(lab, ps_str, pe_str)
        if not model:
            return None
            
        test_period = task_info["test_period"]

        print(f"[MLPSignals] Predicting with loaded model for [{ps_str} to {pe_str}]")
        
        # Construct Dataset (Only need test part really, but AlphaDataset needs periods)
        # We can pass dummy train/valid periods if we don't call fit(), 
        # but to be safe and use same structure:
        dataset = AlphaDataset(
            df=dataset_df,
            train_period=task_info["train_period"], # Not used for predict but required by init
            valid_period=task_info["valid_period"],
            test_period=test_period
        )
        dataset.raw_df = dataset_df
        dataset.infer_df = dataset_df
        dataset.process_data() # Mainly to setup features if needed, though we manually set input_size in load
        
        result_df = None
        try:
            preds = model.predict(dataset, Segment.TEST)
            meta = dataset.fetch_infer(Segment.TEST).select(["datetime", "vt_symbol"])
            
            if len(preds) == len(meta):
                meta = meta.with_columns(pl.Series(preds).alias("total_score"))
                result_df = meta
            else:
                print(f"[MLPSignals] Mismatch in prediction length: {len(preds)} vs {len(meta)}")
        except Exception as e:
            print(f"[MLPSignals] Prediction failed: {e}")
            
        return result_df

    def _train_and_predict_window(self, dataset_df: pl.DataFrame, task_info: Dict, lab: AlphaLab) -> Optional[pl.DataFrame]:
        set_seed(42)  # Reset seed for this window (Deterministic)
        
        train_period = task_info["train_period"]
        valid_period = task_info["valid_period"]
        test_period = task_info["test_period"]
        ps_str = task_info["ps_str"]
        pe_str = task_info["pe_str"]
        save_model = task_info.get("save_model", False)
        
        print(f"[MLPSignals] Window: Train [500 days pre {ps_str}] -> Predict [{ps_str} to {pe_str}]")
        
        # Construct Dataset for this window
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
        
        if model and save_model:
            self._save_model(lab, model, ps_str, pe_str)
        
        result_df = None
        if model:
            # Predict
            try:
                preds = model.predict(dataset, Segment.TEST)
                meta = dataset.fetch_infer(Segment.TEST).select(["datetime", "vt_symbol"])
                
                if len(preds) == len(meta):
                    meta = meta.with_columns(pl.Series(preds).alias("total_score"))
                    result_df = meta
                else:
                    print(f"[MLPSignals] Mismatch in prediction length: {len(preds)} vs {len(meta)}")
            except Exception as e:
                print(f"[MLPSignals] Prediction failed for window {ps_str}: {e}")
                
        return result_df
    
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
            eval_steps=self.model_settings.get("eval_steps", 20),
            weight_decay=self.model_settings.get("weight_decay", 0.0),
            optimizer=self.model_settings.get("optimizer", "adam"),
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