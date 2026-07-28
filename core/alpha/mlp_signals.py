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
- Vintage ensemble（exp_051, 2026-07-17）: 当前窗口模型+过去2窗口模型逐日截面rank平均。
  Tier-1 (3 seeds×8窗×lgb) 配对 median delta +0.044 ≤ margin 0.05 → REVERT。
  seed spread 压缩 41%（1.91→1.12）但中位数门槛未过——降方差有效、提均值不足。
  seed 稳健性问题由准则 #28（持仓广度 N=10）以更强效应解决。代码保留默认关闭。
- valid_len 50→100 回退（2026-07-17）: exp_050 在 Tier-1 (lgb/8窗/in-sample) keep 了 50，
  但配对 Tier-3 (3 seeds×35窗×attention/含OOS) 推翻：median delta -0.023、1/3 为正、
  seed42 delta -3.2（RDD 5.42 vs 2.22），OOS median -0.08 vs -0.49。准则 #17 被再次证实。
  教训：Tier-1 keep 必须经 Tier-3 确认才能进生产。
- SWA 检查点权重平均（exp_053, 2026-07-18）: top-3 valid 检查点权重平均（greedy model soup）。
  3 seeds×8窗×attention 配对：deltas {42:+0.36, 123:+1.72, 2024:+1.94}，median +1.71，
  3/3 为正 → KEEP 并设为默认。机制：34/35 窗口训满 1000 epoch 后 best checkpoint 是噪声
  高原上的随机点，top-3 平均落到更平更稳的区域。topk_pred（预测平均）median +0.04 边界
  revert——同一次训练的检查点相关性太高，预测平均无增量；ema（全轨迹）median -2.27
  失败——decay=0.999 记忆过长，混入高原前的劣质权重。
"""
from collections import deque
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
from vnpy.alpha.model.models.lgb_model import LgbModel
from vnpy.alpha.model.models.tabnet_model import TabNetModel
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

    def __init__(self, signal_name: str = "mlp_signal", force_retrain: bool = False, retrain_days: int = 45, ensemble_size: int = 1, max_windows: int = 0, model_backend: str = "attention", seed: int = 42, vintage_ensemble: int = 0, valid_len: int = 100, run_manager=None, run_id: str = None):
        self.signal_name = signal_name
        self.force_retrain = force_retrain
        self.retrain_days = retrain_days
        self.ensemble_size = ensemble_size
        self.max_windows = max_windows
        self.model_backend = model_backend
        # Auto-research: parameterized seed so research_runner can sweep
        # ensemble_size=1 × 3 distinct seeds for a variance-threshold keep/discard.
        # ensemble_size>1 still uses ENSEMBLE_SEEDS (averaged ensemble) — see below.
        self.seed = seed
        # Vintage ensemble: number of PAST retrain-window models to rank-average with the
        # current window's model at prediction time. 0 = off (baseline behavior).
        # Diversity comes from different training windows (not seeds) — targets regime-switch
        # instability (principle #18) without seed-ensemble's rank-collapse problem.
        self.vintage_ensemble = vintage_ensemble
        self._vintage_models = deque(maxlen=vintage_ensemble) if vintage_ensemble > 0 else None
        self._tasks: List[Dict] = []  # incremental mode looks up past window models by name
        # Validation window length (days) for early stopping. exp_050 kept 50 at Tier-1,
        # but paired Tier-3 (3 seeds × 35w × attention, 2026-07-17) overturned: median delta
        # -0.023, 1/3 positive, seed42 -3.2 → REVERTED to 100 (principle #17 vindicated).
        self.valid_len = valid_len
        # Run 产物管理 (可选): 传入 run_manager + run_id 时模型读写走 runs/{run_id}/models/,
        # 增量模式变为 run 级补全 (从 run 信号最后日期起逐窗口补全)。
        # 不传则完全走现有 lab 路径 — research_runner / auto-research 行为零变化。
        self.run_manager = run_manager
        self.run_id = run_id
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
            # Checkpoint averaging (2026-07-18 验证, exp_053): swa 3/3 seeds 配对 median +1.71 → KEEP
            # topk_pred median +0.04 边界 revert; ema median -2.27 明确失败
            "checkpoint_mode": "swa",
            "topk": 3,
            "ema_decay": 0.999,
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
        set_seed(self.seed)  # Ensure reproducibility (auto-research: per-instance seed)
        
        if self.force_retrain and not self._run_mode():
            # Run 模式下不清理生产信号: run 信号独立存储, 生产信号只经 set_active 同步覆盖
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
            
            valid_len = self.valid_len
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

        # Mark the last task to save model.
        # Run mode: EVERY window's model is persisted (runs/{run_id}/models/{ps_str}.pkl),
        # so any window can later be re-inferred without retraining.
        if tasks:
            if self._run_mode():
                for t in tasks:
                    t["save_model"] = True
            else:
                tasks[-1]["save_model"] = True
            tasks[-1]["show_importance"] = True  # permutation importance 只算最后窗口 (昂贵)

        self._tasks = tasks  # vintage ensemble: incremental mode finds past windows by ps_str

        if not self.force_retrain:
            if not tasks:
                 print("[MLPSignals] No tasks generated.")
                 raise ValueError("No tasks generated.")

            if self._run_mode():
                # Run 补全模式: 从 run 信号最后日期起逐窗口补全
                # (窗口模型已存在则纯推理, 跨过窗口边界则训练新窗口模型并存入该 run)
                run_signal = self.run_manager.load_signal(self.run_id)
                last_dt = None
                if run_signal is not None and not run_signal.is_empty():
                    last_dt = run_signal["datetime"].max()

                pending = [t for t in tasks if last_dt is None or t["test_period"][1] > last_dt]
                print(f"[MLPSignals] Run Completion Mode ({self.run_id}): signal last date = {last_dt}, {len(pending)}/{len(tasks)} windows to fill")

                for task in pending:
                    try:
                        if self.run_manager.has_model(self.run_id, task["ps_str"]):
                            print(f"[MLPSignals] Window {task['ps_str']}: model exists in run. Predicting...")
                            result = self._predict_with_existing_model(dataset_df, task, lab)
                        else:
                            print(f"[MLPSignals] Window {task['ps_str']}: model missing in run. Training new window model...")
                            result = self._train_and_predict_window(dataset_df, task, lab)
                        if result is not None:
                            all_predictions.append(result)
                    except Exception as exc:
                        print(f"[MLPSignals] Run completion task {task['ps_str']} failed: {exc}")
                        import traceback
                        traceback.print_exc()
            else:
                # Legacy incremental mode: Only process the last window
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
            if self.max_windows and self.max_windows < len(tasks):
                total = len(tasks)
                tasks = tasks[-self.max_windows:]
                print(f"[MLPSignals] Quick mode: training last {len(tasks)}/{total} windows only")
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

    def _run_mode(self) -> bool:
        """是否启用 run 产物管理 (模型读写走 runs/{run_id}/ 目录)"""
        return self.run_manager is not None and self.run_id is not None

    def _save_model(self, lab: AlphaLab, model: MlpModel, start_date_str: str, end_date_str: str):
        if self._run_mode():
            print(f"[MLPSignals] Saving window model {start_date_str} to run {self.run_id}...")
            try:
                self.run_manager.save_model(self.run_id, start_date_str, end_date_str, model)
            except Exception as e:
                print(f"[MLPSignals] Failed to save model to run: {e}")
            return
        model_name = f"{self.signal_name}_{start_date_str}"
        print(f"[MLPSignals] Saving model {model_name}...")
        try:
            lab.save_model(model_name, model)
        except Exception as e:
            print(f"[MLPSignals] Failed to save model: {e}")

    def _load_model(self, lab: AlphaLab, start_date_str: str, end_date_str: str) -> Optional[MlpModel]:
        if self._run_mode():
            try:
                return self.run_manager.load_model(self.run_id, start_date_str)
            except Exception as e:
                print(f"[MLPSignals] Failed to load model {start_date_str} from run {self.run_id}: {e}")
                return None
        model_name = f"{self.signal_name}_{start_date_str}"
        try:
            model = lab.load_model(model_name)
            return model
        except Exception as e:
            print(f"[MLPSignals] Failed to load model {model_name}: {e}")
            return None

    def _get_past_window_models(self, lab: AlphaLab, current_task: Dict) -> list:
        """Return up to `vintage_ensemble` past retrain-window models for rank-averaging.

        Full-retrain mode: in-memory deque (models trained earlier in this run).
        Incremental mode: load from disk by window-start name (saved by previous runs).
        """
        if self.vintage_ensemble <= 0:
            return []
        if self.force_retrain:
            return list(self._vintage_models) if self._vintage_models else []
        idx = None
        for i, t in enumerate(self._tasks):
            if t["ps_str"] == current_task["ps_str"]:
                idx = i
                break
        if idx is None:
            return []
        past = []
        for t in self._tasks[max(0, idx - self.vintage_ensemble):idx]:
            m = self._load_model(lab, t["ps_str"], t["pe_str"])
            if m is not None:
                past.append(m)
        return past

    @staticmethod
    def _rank_average_preds(meta: pl.DataFrame, preds_list: list) -> np.ndarray:
        """Per-day cross-sectional rank-average of prediction arrays (scale-immune)."""
        df = meta.select("datetime").with_columns([
            pl.Series(f"_p{i}", p) for i, p in enumerate(preds_list)
        ])
        df = df.with_columns([
            pl.col(f"_p{i}").rank(method="average").over("datetime").alias(f"_r{i}")
            for i in range(len(preds_list))
        ])
        return df.select([f"_r{i}" for i in range(len(preds_list))]).mean_horizontal().to_numpy()

    def _model_predict(self, model, dataset) -> np.ndarray:
        """Predict with checkpoint averaging when the model uses topk_pred mode."""
        if getattr(model, "checkpoint_mode", "best") == "topk_pred":
            meta = dataset.fetch_infer(Segment.TEST).select(["datetime", "vt_symbol"])
            ckpt_preds = model.predict_checkpoints(dataset, Segment.TEST)
            if len(ckpt_preds) > 1:
                return self._rank_average_preds(meta, ckpt_preds)
            return ckpt_preds[0]
        return model.predict(dataset, Segment.TEST)

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
        dataset = AlphaDataset(
            df=dataset_df,
            train_period=task_info["train_period"],
            valid_period=task_info["valid_period"],
            test_period=test_period
        )
        dataset.raw_df = dataset_df
        dataset.infer_df = dataset_df
        dataset.process_data()
        
        result_df = None
        try:
            preds = self._model_predict(model, dataset)
            meta = dataset.fetch_infer(Segment.TEST).select(["datetime", "vt_symbol"])

            # Vintage ensemble: rank-average with past window models loaded from disk
            past_models = self._get_past_window_models(lab, task_info)
            if past_models:
                preds_list = [preds]
                for vm in past_models:
                    try:
                        preds_list.append(vm.predict(dataset, Segment.TEST))
                    except Exception as e:
                        print(f"[MLPSignals] Vintage model predict failed (skipped): {e}")
                if len(preds_list) > 1:
                    preds = self._rank_average_preds(meta, preds_list)
                    print(f"[MLPSignals] Vintage ensemble: rank-averaged {len(preds_list)} models (1 current + {len(preds_list) - 1} past)")

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
        # ensemble_size>1: averaged ensemble over ENSEMBLE_SEEDS (one averaged signal, no spread).
        # ensemble_size==1: single run at self.seed — research_runner sweeps this 3× for variance.
        seeds = [self.seed] if n_models == 1 else self.ENSEMBLE_SEEDS[:n_models]
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
        # Vintage ensemble: capture past-window models BEFORE this window's model
        # is appended to the deque (full mode) / looked up from disk (incremental).
        past_models = self._get_past_window_models(lab, task_info)

        try:
            for i, seed in enumerate(seeds):
                set_seed(seed)
                if n_models > 1:
                    print(f"[MLPSignals]   Ensemble member {i+1}/{n_models} (seed={seed})")
                
                model = self._train_model(dataset, seed=seed)
                
                if model:
                    if save_model and i == 0:
                        self._save_model(lab, model, ps_str, pe_str)
                    # permutation importance 昂贵, 只在最后窗口计算 (run 模式每窗口都存模型)
                    if task_info.get("show_importance", False) and i == 0:
                        try:
                            importance_df = model.detail()
                            if importance_df is not None:
                                print(f"[MLPSignals] === Factor Importance (Permutation) ===")
                                print(importance_df.to_string())
                        except Exception as e:
                            print(f"[MLPSignals] Feature importance extraction failed: {e}")
                    
                    preds = self._model_predict(model, dataset)
                    all_preds.append(preds)

                    # Vintage ensemble: retain first member for future windows' rank-averaging
                    # (full-retrain mode only; incremental mode reloads past models from disk)
                    if i == 0 and self.vintage_ensemble > 0 and self.force_retrain:
                        self._vintage_models.append(model)
                    else:
                        del model
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            
            if all_preds:
                # Average predictions across ensemble members
                avg_preds = np.mean(all_preds, axis=0)
                meta = dataset.fetch_infer(Segment.TEST).select(["datetime", "vt_symbol"])

                # Vintage ensemble: per-day cross-sectional rank-average with past window models
                if past_models:
                    preds_list = [avg_preds]
                    for vm in past_models:
                        try:
                            preds_list.append(vm.predict(dataset, Segment.TEST))
                        except Exception as e:
                            print(f"[MLPSignals] Vintage model predict failed (skipped): {e}")
                    if len(preds_list) > 1:
                        avg_preds = self._rank_average_preds(meta, preds_list)
                        print(f"[MLPSignals] Vintage ensemble: rank-averaged {len(preds_list)} models (1 current + {len(preds_list) - 1} past)")

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
        print(f"[MLPSignals] Model Backend: {self.model_backend}")

        if self.model_backend == "lgb":
            model = LgbModel(
                learning_rate=0.05,
                num_leaves=63,
                num_boost_round=2000,
                early_stopping_rounds=50,
                log_evaluation_period=100,
                seed=seed,
            )
            try:
                model.fit(dataset)
                return model
            except Exception as e:
                print(f"[MLPSignals] LGB training failed: {e}")
                import traceback
                traceback.print_exc()
                return None

        if self.model_backend == "tabnet":
            model = TabNetModel(
                n_d=64,
                n_a=64,
                n_steps=5,
                gamma=1.5,
                n_independent=2,
                n_shared=2,
                lambda_sparse=1e-4,
                learning_rate=0.02,
                batch_size=4096,
                max_epochs=300,
                patience=30,
                seed=seed,
                device="auto",
            )
            try:
                model.fit(dataset)
                return model
            except Exception as e:
                print(f"[MLPSignals] TabNet training failed: {e}")
                import traceback
                traceback.print_exc()
                return None

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
            checkpoint_mode=self.model_settings.get("checkpoint_mode", "best"),
            topk=self.model_settings.get("topk", 3),
            ema_decay=self.model_settings.get("ema_decay", 0.999),
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