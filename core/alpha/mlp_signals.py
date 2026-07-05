"""
MLP 滚动训练与信号生成模块

== 版本演进 ==
初始: MLP 三层隐藏层(64/32/16), 500天窗口, 90天重训
V9 Step 3a: MLP → FactorAttentionNetwork (d_token=64, 4heads, 1层)
V10: 训练窗口 500天 → 700天（更多 regime 多样性）
V15.2: 验证集 50天 → 100天, 3-seed ensemble 降低variance
V15.3-revert: 回退为单次训练（ensemble 多次重训仍不稳定，OOS variance 改善有限）
V15.4-exp: 重训周期 90天 → 45天（加快风格切换适应，保留700天窗口）
V15.6-fail: 重训周期 45天 → 30天（Sharpe 1.36，下降，回退为45天）

== 当前状态 ==
模型: factor_attention (d_token=64, n_heads=4, n_attn_layers=1, d_ffn=128)
训练窗口: 700 天 (600训练 + 100验证)
重训周期: 45 天
批量大小: 2048, 学习率: 0.001, weight_decay: 0.002
早停: 40 轮无改善
单次训练 (seed=42)，n_jobs=1 保证可复现性

== 设计决策 ==
- Factor Attention 选择: 模型结构改变 > 损失函数改造（连续3次损失函数实验失败）
- d_token=64: 核心超参数，32不足，64最优，128无额外收益
- n_attn_layers=1: 2层过拟合（500~700天训练数据量有限）
- 700天窗口: 提供充足 regime 多样性，是隐式正则化（400天窗口实验确认：丧失多样性导致Q4暴跌）
- 45天重训: 更频繁的更新，验证集更贴近当前市场（30天实验失败已回退）
- 100天验证集: 覆盖~4个月行情，避免early stopping被短期市场偏差误导
- 单次训练 + 固定 seed: 多次重训发现 ensemble 仍不稳定，回退为单次保证可复现
- n_jobs=1: 单线程保证可复现性

== 失败记录 ==
- IC-Loss: 改善非牛市但损害牛市(Sharpe 1.10→0.70)，梯度方向与MSE冲突
- 混合损失(MSE+IC): 梯度冲突，效果最差
- 多任务学习(1d/10d/20d预测头): Sharpe 1.20→0.86，辅助损失梯度冲突
- 时间衰减采样(decay=0.995): Sharpe 0.68→0.24，丧失 regime 多样性
- 2层Attention: 过拟合，不如1层
- 2层Attention+强dropout(0.30): Sharpe 1.36→1.13，MaxDD持续64天，依然不如1层
- d_ffn 128→256: Sharpe 1.36→1.01，MaxDD持续519天，FFN加大导致过拟合
- Input feature dropout(0.15): Sharpe 1.36→0.96，MaxDD -43.6%/526天，因子遮蔽过激损害学习
- 3-seed ensemble: 单窗口降variance有效，但跨多次重训整体OOS仍不稳定，
  且训练成本x3，收益不匹配，已回退为单次训练
- 400天训练窗口(300训练+100验证): Q2从-5.42%改善到-2.75%，但丧失regime多样性
  导致Q4 2025从+22%暴跌到-6%，Sharpe从1.17降到0.98。短窗口不可取
- 30天重训周期: Sharpe 1.79→1.36，Q1从+8~10%降到-1.51%。更频繁重训引入预测
  不稳定性，45天是当前最优重训频率
"""
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

    def __init__(self, signal_name: str = "mlp_signal", force_retrain: bool = False, retrain_days: int = 45, ensemble_size: int = 1):
        self.signal_name = signal_name
        self.force_retrain = force_retrain
        self.retrain_days = retrain_days
        self.ensemble_size = ensemble_size
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
            # V10 Step 3a: Factor Self-Attention (best config)
            "model_type": "factor_attention",
            "d_token": 64,
            "n_heads": 4,
            "n_attn_layers": 1,
            "d_ffn": 128,
            "head_hidden": 128,
            "attn_dropout": 0.15,
            "ffn_dropout": 0.15,
            "head_dropout": 0.10,
            "attn_activation": "softmax",  # "entmax15" available but ~40x slower
            "input_dropout": 0.0,  # B5 failed: 0.15 caused Sharpe 1.36→0.96, MaxDD -43.6%
            # Training hyperparameters (unchanged from Step 1)
            "n_epochs": 1000,
            "batch_size": 2048,
            "lr": 0.001,
            "early_stop_rounds": 40,
            "weight_decay": 0.002,
            "optimizer": "adam",
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
        # Increased requirement for 700-day window
        if len(dates) < 750:
             print(f"[MLPSignals] Not enough dates for rolling window: {len(dates)} (Need ~750)")
             raise ValueError("Insufficient date range for rolling window.")

        # Determine start index for prediction
        # We need to align with user requested start_date, but ensure we have 700 days history
        target_start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        # Find index of first date >= target_start_dt
        start_idx = 0
        for i, d in enumerate(dates):
            if d >= target_start_dt:
                start_idx = i
                break

        # Ensure we have 700 days before start_idx
        if start_idx < 700:
            print(f"[MLPSignals] Warning: Not enough history before {start_date} for 700-day training.")
            print(f"[MLPSignals] Adjusting start index to 700 (Date: {dates[700]})")
            start_idx = 700
            
        all_predictions = []
        tasks = []
        
        # Pre-calculate tasks
        curr_idx = start_idx
        total_dates = len(dates)
        
        while curr_idx < total_dates:
            # Define Prediction Window
            pred_start_date = dates[curr_idx]
            
            # Next window date
            next_window_date = pred_start_date + timedelta(days=self.retrain_days)
            
            # Find index for next window (end of this prediction window)
            next_idx = total_dates # Default to end
            for i in range(curr_idx, total_dates):
                if dates[i] >= next_window_date:
                    next_idx = i
                    break
            
            pred_end_idx = next_idx - 1
            if pred_end_idx < curr_idx:
                pred_end_idx = curr_idx # At least one day
                
            pred_end_date = dates[pred_end_idx]
            
            # Format dates for logging
            ps_str = pred_start_date.strftime("%Y-%m-%d") if hasattr(pred_start_date, "strftime") else str(pred_start_date)
            pe_str = pred_end_date.strftime("%Y-%m-%d") if hasattr(pred_end_date, "strftime") else str(pred_end_date)
            
            # Define Training Window (Previous 700 indices)
            train_end_idx = curr_idx - 1
            # 700 days total (0 to 699)
            train_start_idx = max(0, train_end_idx - 699) 
            
            valid_len = 100
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
            print(f"[MLPSignals] Force Retrain Mode: Executing {len(tasks)} windows sequentially...")
            for task in tqdm(tasks, total=len(tasks)):
                try:
                    result = self._train_and_predict_window(dataset_df, task, lab)
                    if result is not None:
                        all_predictions.append(result)
                except Exception as exc:
                    print(f"[MLPSignals] Task {task['ps_str']} generated an exception: {exc}")
                    import traceback
                    traceback.print_exc()

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

    # Seeds for ensemble training (diverse primes for independence)
    ENSEMBLE_SEEDS = [42, 123, 2024]

    def _train_and_predict_window(self, dataset_df: pl.DataFrame, task_info: Dict, lab: AlphaLab) -> Optional[pl.DataFrame]:
        train_period = task_info["train_period"]
        valid_period = task_info["valid_period"]
        test_period = task_info["test_period"]
        ps_str = task_info["ps_str"]
        pe_str = task_info["pe_str"]
        save_model = task_info.get("save_model", False)
        
        n_models = self.ensemble_size
        seeds = self.ENSEMBLE_SEEDS[:n_models]
        print(f"[MLPSignals] Window: Train [700 days pre {ps_str}] -> Predict [{ps_str} to {pe_str}] (ensemble={n_models})")
        
        # Construct Dataset for this window (shared across ensemble members)
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
        
        result_df = None
        all_preds = []
        
        try:
            for i, seed in enumerate(seeds):
                set_seed(seed)
                if n_models > 1:
                    print(f"[MLPSignals]   Ensemble member {i+1}/{n_models} (seed={seed})")
                
                model = self._train_model(dataset, seed=seed)
                
                if model:
                    if save_model and i == 0:
                        self._save_model(lab, model, ps_str, pe_str)
                    
                    preds = model.predict(dataset, Segment.TEST)
                    all_preds.append(preds)
                    
                    del model
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            if all_preds:
                # Average predictions across ensemble members
                avg_preds = np.mean(all_preds, axis=0)
                meta = dataset.fetch_infer(Segment.TEST).select(["datetime", "vt_symbol"])
                
                if len(avg_preds) == len(meta):
                    meta = meta.with_columns(pl.Series(avg_preds).alias("total_score"))
                    result_df = meta
                else:
                    print(f"[MLPSignals] Mismatch in prediction length: {len(avg_preds)} vs {len(meta)}")
                    
        except Exception as e:
            print(f"[MLPSignals] Prediction/Training failed for window {ps_str}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # === Memory Cleanup ===
            if 'dataset' in locals() and dataset is not None:
                del dataset
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return result_df
    
    def _clean_label(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop_nulls(subset=["label"])


    def _train_model(self, dataset: AlphaDataset, seed: int = 42) -> Optional[MlpModel]:
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
            hidden_sizes=self.model_settings.get("hidden_sizes", (256, 128, 64)),
            n_epochs=self.model_settings["n_epochs"],
            batch_size=self.model_settings["batch_size"],
            lr=self.model_settings["lr"],
            early_stop_rounds=self.model_settings["early_stop_rounds"],
            eval_steps=self.model_settings.get("eval_steps", 20),
            weight_decay=self.model_settings.get("weight_decay", 0.0),
            optimizer=self.model_settings.get("optimizer", "adam"),
            device=device,
            seed=seed,
            model_type=self.model_settings.get("model_type", "mlp"),
            d_token=self.model_settings.get("d_token", 32),
            n_heads=self.model_settings.get("n_heads", 4),
            n_attn_layers=self.model_settings.get("n_attn_layers", 1),
            d_ffn=self.model_settings.get("d_ffn", 64),
            head_hidden=self.model_settings.get("head_hidden", 64),
            attn_dropout=self.model_settings.get("attn_dropout", 0.15),
            ffn_dropout=self.model_settings.get("ffn_dropout", 0.15),
            head_dropout=self.model_settings.get("head_dropout", 0.10),
            attn_activation=self.model_settings.get("attn_activation", "softmax"),
            input_dropout=self.model_settings.get("input_dropout", 0.0),
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