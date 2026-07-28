"""
LSTM Baseline 实验 — 验证时序信号是否有独立 alpha

== 实验设计 ==
目标: 用 LSTM 从单只股票的历史价格序列预测 5 日收益率，
      转为截面排序信号后评估 IC/Rank IC，判断时序信号的增量贡献。

输入: 每只股票过去 lookback 天的日收益率序列 (单变量)
输出: 未来 5 日收益率预测值
评估: 截面 Rank IC, IC, 与现有 Factor Attention 信号的相关性

== 使用方法 ==
# 快速验证 (最近3个窗口)
python scripts/lstm_baseline.py --max-windows 3 --index 399303.SZ

# 完整回测
python scripts/lstm_baseline.py --index 000852.SH,399303.SZ

== 设计决策 ==
- lookback=60: A 股风格切换约 60 天，更长序列信噪比下降
- hidden_size=64: 轻量级，避免过拟合
- 2 层 LSTM: 1 层表达力不够，3 层过拟合
- 日收益率输入 (非价格): 消除尺度差异，平稳化序列
- 批量训练所有股票: GPU 效率高，共享时序模式
- MSE loss on 截面 rank label: 与主模型标签一致，可公平比较
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from datetime import datetime
from typing import Optional
from tqdm import tqdm
from scipy import stats

# ============================================================
# LSTM Model
# ============================================================

class LSTMPriceModel(nn.Module):
    """轻量级 LSTM 时序预测模型"""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size) 日收益率序列
        Returns:
            (batch,) 预测值
        """
        # LSTM output: (batch, seq_len, hidden)
        lstm_out, _ = self.lstm(x)
        # 取最后一步的隐状态
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)
        out = self.head(last_hidden).squeeze(-1)  # (batch,)
        return out


# ============================================================
# Data Preparation
# ============================================================

def prepare_sequences(
    df: pl.DataFrame,
    lookback: int = 60,
    horizon: int = 5,
    multivariate: bool = False,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """
    从 DataFrame 构建 LSTM 训练序列

    Args:
        df: 含 datetime, vt_symbol, close (+ OHLCV) 的 DataFrame (已排序)
        lookback: 回看天数
        horizon: 预测天数
        multivariate: 是否使用 OHLCV 多通道

    Returns:
        X: (N, lookback, C) 特征序列 (C=1 单变量, C=5 多变量)
        y: (N,) 未来 horizon 日收益率 (原始值，非 rank)
        meta: (N, 2) datetime, vt_symbol 用于后续对齐
    """
    # 按 vt_symbol 排序保证序列连续
    df = df.sort(["vt_symbol", "datetime"])

    # 计算日收益率
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("vt_symbol") - 1)
        .alias("daily_ret")
    )

    # 多变量特征: OHLC 相对 close 的比率 + volume 变化率
    if multivariate:
        df = df.with_columns([
            (pl.col("high") / pl.col("close") - 1).alias("high_ret"),
            (pl.col("low") / pl.col("close") - 1).alias("low_ret"),
            (pl.col("open") / pl.col("close") - 1).alias("open_ret"),
            (pl.col("volume") / pl.col("volume").rolling_mean(20).over("vt_symbol"))
            .alias("vol_ratio"),
        ])
        # clip volume ratio to avoid extreme outliers
        df = df.with_columns(
            pl.col("vol_ratio").clip(0, 5).fill_null(1.0).fill_nan(1.0)
        )
        feature_cols = ["daily_ret", "high_ret", "low_ret", "open_ret", "vol_ratio"]
    else:
        feature_cols = ["daily_ret"]

    # 计算未来 horizon 日收益率 (原始收益率，不做截面 rank)
    df = df.with_columns(
        (pl.col("close").shift(-horizon).over("vt_symbol") / pl.col("close") - 1)
        .alias("fwd_ret")
    )

    # 构建序列: 对每只股票，滑动窗口提取 lookback 天的特征
    symbols = df["vt_symbol"].unique().sort().to_list()
    n_features = len(feature_cols)

    all_X = []
    all_y = []
    all_meta = []

    for sym in symbols:
        sym_df = df.filter(pl.col("vt_symbol") == sym).sort("datetime")
        features = sym_df.select(feature_cols).to_numpy()  # (T, n_features)
        fwd_rets = sym_df["fwd_ret"].to_numpy()
        sym_dates = sym_df["datetime"].to_list()

        n = len(features)
        for i in range(lookback, n):
            seq = features[i - lookback: i]  # (lookback, n_features)
            label = fwd_rets[i]

            # 检查序列完整性
            if np.isnan(seq).sum() > lookback * n_features * 0.1:
                continue
            if np.isnan(label):
                continue

            # 填充 NaN
            seq = np.nan_to_num(seq, nan=0.0)

            all_X.append(seq)
            all_y.append(label)
            all_meta.append((sym_dates[i], sym))

    if not all_X:
        return np.array([]), np.array([]), pl.DataFrame()

    X = np.array(all_X, dtype=np.float32)  # (N, lookback, n_features)
    y = np.array(all_y, dtype=np.float32)  # (N,)
    meta = pl.DataFrame(
        {"datetime": [m[0] for m in all_meta], "vt_symbol": [m[1] for m in all_meta]}
    )

    return X, y, meta


# ============================================================
# Training & Evaluation
# ============================================================

def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    hidden_size: int = 64,
    num_layers: int = 2,
    lr: float = 0.001,
    batch_size: int = 2048,
    n_epochs: int = 100,
    early_stop: int = 15,
    device: str = "cuda",
) -> LSTMPriceModel:
    """训练 LSTM 模型"""
    input_size = X_train.shape[2]
    model = LSTMPriceModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = nn.MSELoss()

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_valid_t = torch.from_numpy(X_valid).to(device)
    y_valid_t = torch.from_numpy(y_valid).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    n_train = len(X_train_t)

    for epoch in range(n_epochs):
        model.train()
        # Mini-batch training
        indices = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = indices[start:end]
            batch_x = X_train_t[batch_idx]
            batch_y = y_train_t[batch_idx]

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_valid_t)
            val_loss = criterion(val_pred, y_valid_t).item()

        scheduler.step(val_loss)
        avg_train_loss = epoch_loss / max(n_batches, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if (epoch + 1) % 10 == 0:
            print(f"  [Epoch {epoch+1:3d}] train_loss={avg_train_loss:.6f} val_loss={val_loss:.6f} best={best_val_loss:.6f}")

        if patience_count >= early_stop:
            print(f"  Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_lstm(
    model: LSTMPriceModel,
    X: np.ndarray,
    device: str = "cuda",
    batch_size: int = 8192,
) -> np.ndarray:
    """LSTM 批量预测"""
    model.eval()
    preds = []
    X_t = torch.from_numpy(X).to(device)

    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            batch = X_t[start: start + batch_size]
            pred = model(batch).cpu().numpy()
            preds.append(pred)

    return np.concatenate(preds)


def evaluate_ic(predictions: pl.DataFrame) -> dict:
    """
    计算截面 IC 和 Rank IC

    Args:
        predictions: DataFrame with datetime, vt_symbol, pred, fwd_ret
    """
    dates = predictions["datetime"].unique().sort().to_list()

    ic_list = []
    rank_ic_list = []

    for dt in dates:
        day_df = predictions.filter(pl.col("datetime") == dt)
        if len(day_df) < 30:  # 至少 30 只股票
            continue

        pred = day_df["pred"].to_numpy()
        actual = day_df["fwd_ret"].to_numpy()

        # 去掉 NaN 和 Inf
        mask = np.isfinite(pred) & np.isfinite(actual)
        if mask.sum() < 30:
            continue

        pred_clean = pred[mask]
        actual_clean = actual[mask]

        # 跳过方差为零的情况 (模型输出常数)
        if np.std(pred_clean) < 1e-8:
            continue

        # Pearson IC
        ic, _ = stats.pearsonr(pred_clean, actual_clean)
        # Spearman Rank IC
        rank_ic, _ = stats.spearmanr(pred_clean, actual_clean)

        if np.isfinite(ic):
            ic_list.append(ic)
        if np.isfinite(rank_ic):
            rank_ic_list.append(rank_ic)

    if not ic_list:
        return {"ic_mean": 0, "ic_std": 0, "icir": 0, "rank_ic_mean": 0,
                "rank_ic_std": 0, "rank_icir": 0, "ic_positive_ratio": 0, "n_days": 0}

    ic_arr = np.array(ic_list)
    rank_ic_arr = np.array(rank_ic_list) if rank_ic_list else np.array([0.0])

    return {
        "ic_mean": float(np.mean(ic_arr)),
        "ic_std": float(np.std(ic_arr)),
        "icir": float(np.mean(ic_arr) / (np.std(ic_arr) + 1e-8)),
        "rank_ic_mean": float(np.mean(rank_ic_arr)),
        "rank_ic_std": float(np.std(rank_ic_arr)),
        "rank_icir": float(np.mean(rank_ic_arr) / (np.std(rank_ic_arr) + 1e-8)),
        "ic_positive_ratio": float((ic_arr > 0).mean()),
        "n_days": len(ic_list),
    }


# ============================================================
# Main Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LSTM Baseline for Time-Series Alpha Signal")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window (trading days)")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon (trading days)")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layers")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=200, help="Max epochs per window")
    parser.add_argument("--retrain-days", type=int, default=45, help="Retrain cycle")
    parser.add_argument("--max-windows", type=int, default=0, help="Quick mode: last N windows only")
    parser.add_argument("--index", type=str, default="399303.SZ", help="Index filter")
    parser.add_argument("--start-date", type=str, default="2019-12-28", help="Data start")
    parser.add_argument("--multivariate", action="store_true", help="Use OHLCV (5 channels) instead of close only")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== LSTM Baseline Experiment ===")
    print(f"Device: {device}")
    print(f"Config: lookback={args.lookback}, horizon={args.horizon}, hidden={args.hidden_size}")
    print(f"        layers={args.num_layers}, lr={args.lr}, retrain_days={args.retrain_days}")
    print(f"        index={args.index}, multivariate={args.multivariate}")
    print()

    # ---- 1. Load Data ----
    print("[1/5] Loading data...")
    from core.alpha.engine import AlphaEngine
    from core.alpha.mlp_signals import MLPSignals
    from core.selector.selector import FundamentalSelector

    # Use v158 factor calculator for label alignment
    from core.alpha.v158_factor_calculator import V158FactorCalculator

    selector = FundamentalSelector()
    _, _ = selector.get_data_range()
    last_trading_date = selector.get_last_trading_day() or datetime.now()

    engine = AlphaEngine(
        factor_calculator=V158FactorCalculator(gp_status_filter=[]),
        mlp_signals=MLPSignals(signal_name="lstm_baseline_tmp", force_retrain=False),
        selector=selector,
        signal_name="lstm_baseline_tmp",
        start_date=args.start_date,
        end_date=last_trading_date.strftime("%Y-%m-%d"),
        index_filter=args.index,
    )

    data_df = engine.load_data()
    print(f"  Loaded {data_df.shape[0]} rows, {data_df['vt_symbol'].n_unique()} stocks")

    # 只需要 close 价格 (或 OHLCV)
    needed_cols = ["datetime", "vt_symbol", "close"]
    if args.multivariate:
        needed_cols = ["datetime", "vt_symbol", "open", "high", "low", "close", "volume"]

    price_df = data_df.select([c for c in needed_cols if c in data_df.columns])
    price_df = price_df.sort(["vt_symbol", "datetime"])

    # ---- 2. Build Sequences & Rolling Windows ----
    print("[2/5] Building rolling windows...")
    dates = price_df["datetime"].unique().sort().to_list()
    total_dates = len(dates)

    # 训练窗口: 700 天 (同主模型)，验证 100 天
    train_window = 700
    valid_window = 100
    lookback = args.lookback
    horizon = args.horizon

    # 最小需要 train_window + lookback 天历史
    min_start_idx = train_window + lookback
    if min_start_idx >= total_dates:
        print(f"ERROR: Not enough data. Have {total_dates} dates, need {min_start_idx}")
        return

    # 构建预测窗口列表
    windows = []
    curr_idx = min_start_idx
    while curr_idx < total_dates - horizon:
        pred_start = dates[curr_idx]
        # 找 retrain_days 后的日期
        next_idx = min(curr_idx + args.retrain_days, total_dates - horizon)
        pred_end = dates[next_idx - 1]

        train_start = dates[curr_idx - train_window - lookback]
        train_end = dates[curr_idx - valid_window - 1]
        valid_start = dates[curr_idx - valid_window]
        valid_end = dates[curr_idx - 1]

        windows.append({
            "train_start": train_start,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "pred_start": pred_start,
            "pred_end": pred_end,
        })
        curr_idx = next_idx

    if args.max_windows and args.max_windows < len(windows):
        total = len(windows)
        windows = windows[-args.max_windows:]
        print(f"  Quick mode: using last {len(windows)}/{total} windows")
    else:
        print(f"  Total windows: {len(windows)}")

    # ---- 3. Rolling Train & Predict ----
    print("[3/5] Rolling LSTM training...")
    all_predictions = []

    for wi, win in enumerate(tqdm(windows, desc="Windows")):
        print(f"\n  Window {wi+1}/{len(windows)}: "
              f"Train [{win['train_start'].strftime('%Y-%m-%d')} ~ {win['train_end'].strftime('%Y-%m-%d')}] "
              f"Valid [{win['valid_start'].strftime('%Y-%m-%d')} ~ {win['valid_end'].strftime('%Y-%m-%d')}] "
              f"Pred [{win['pred_start'].strftime('%Y-%m-%d')} ~ {win['pred_end'].strftime('%Y-%m-%d')}]")

        # 提取训练数据
        train_df = price_df.filter(
            (pl.col("datetime") >= win["train_start"]) &
            (pl.col("datetime") <= win["train_end"])
        )
        valid_df = price_df.filter(
            (pl.col("datetime") >= win["valid_start"]) &
            (pl.col("datetime") <= win["valid_end"])
        )
        # 预测期需要包含 lookback 天历史
        pred_history_start = dates[max(0, dates.index(win["pred_start"]) - lookback)]
        pred_df = price_df.filter(
            (pl.col("datetime") >= pred_history_start) &
            (pl.col("datetime") <= win["pred_end"])
        )

        # 准备训练序列
        X_train, y_train, _ = prepare_sequences(train_df, lookback, horizon, args.multivariate)
        X_valid, y_valid, _ = prepare_sequences(valid_df, lookback, horizon, args.multivariate)
        X_pred, _, meta_pred = prepare_sequences(pred_df, lookback, horizon, args.multivariate)

        if len(X_train) == 0 or len(X_valid) == 0:
            print(f"  Skipping window {wi+1}: insufficient data")
            continue

        print(f"  Train samples: {len(X_train)}, Valid: {len(X_valid)}, Pred: {len(X_pred)}")

        # 训练
        model = train_lstm(
            X_train, y_train, X_valid, y_valid,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            lr=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            device=device,
        )

        # 预测
        if len(X_pred) > 0:
            preds = predict_lstm(model, X_pred, device=device)

            # 只保留预测期内的结果
            result = meta_pred.with_columns(pl.Series("pred", preds))
            result = result.filter(
                (pl.col("datetime") >= win["pred_start"]) &
                (pl.col("datetime") <= win["pred_end"])
            )
            all_predictions.append(result)

        # 清理 GPU
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not all_predictions:
        print("ERROR: No predictions generated!")
        return

    # ---- 4. Merge & Evaluate ----
    print("\n[4/5] Evaluating LSTM signal...")
    predictions = pl.concat(all_predictions)

    # 获取实际未来收益率用于 IC 计算
    price_with_fwd = price_df.sort(["vt_symbol", "datetime"]).with_columns(
        (pl.col("close").shift(-horizon).over("vt_symbol") / pl.col("close") - 1)
        .alias("fwd_ret")
    )

    # Join predictions with actual returns
    eval_df = predictions.join(
        price_with_fwd.select(["datetime", "vt_symbol", "fwd_ret"]),
        on=["datetime", "vt_symbol"],
        how="left",
    )

    # IC evaluation
    metrics = evaluate_ic(eval_df)

    print("\n" + "=" * 60)
    print("LSTM Baseline Results")
    print("=" * 60)
    print(f"  Prediction days:     {metrics['n_days']}")
    print(f"  IC mean:             {metrics['ic_mean']:.4f}")
    print(f"  IC std:              {metrics['ic_std']:.4f}")
    print(f"  ICIR:                {metrics['icir']:.4f}")
    print(f"  Rank IC mean:        {metrics['rank_ic_mean']:.4f}")
    print(f"  Rank IC std:         {metrics['rank_ic_std']:.4f}")
    print(f"  Rank ICIR:           {metrics['rank_icir']:.4f}")
    print(f"  IC positive ratio:   {metrics['ic_positive_ratio']:.2%}")
    print("=" * 60)

    # ---- 5. Orthogonality Check ----
    print("\n[5/5] Checking orthogonality with existing signals...")
    try:
        from vnpy.alpha.lab import AlphaLab
        lab = AlphaLab()
        # Try to load existing Factor Attention signal
        existing_signal = lab.load_signal("ashare_mlp_signal_v158")
        if existing_signal is not None and not existing_signal.is_empty():
            # Merge on datetime + vt_symbol
            merged = eval_df.join(
                existing_signal.select(["datetime", "vt_symbol", "final_signal"]),
                on=["datetime", "vt_symbol"],
                how="inner",
            )
            if len(merged) > 100:
                lstm_pred = merged["pred"].to_numpy()
                fa_signal = merged["final_signal"].to_numpy()
                mask = ~(np.isnan(lstm_pred) | np.isnan(fa_signal))
                if mask.sum() > 100:
                    corr, _ = stats.spearmanr(lstm_pred[mask], fa_signal[mask])
                    print(f"\n  Spearman correlation with Factor Attention signal: {corr:.4f}")
                    print(f"  (Low correlation = high orthogonality = potential alpha)")
                    if abs(corr) < 0.3:
                        print("  ✓ LSTM signal appears ORTHOGONAL to existing model")
                    elif abs(corr) < 0.5:
                        print("  ~ Moderate correlation, some incremental value possible")
                    else:
                        print("  ✗ High correlation, limited incremental alpha expected")
        else:
            print("  (No existing signal found for comparison)")
    except Exception as e:
        print(f"  (Could not load existing signal: {e})")

    # Save results
    output_path = "core/alpha_db/lstm_baseline_results.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    eval_df.write_parquet(output_path)
    print(f"\n  Results saved to {output_path}")

    # Summary verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if metrics["rank_ic_mean"] > 0.03:
        print("  ★ STRONG signal — worth integrating as a factor")
    elif metrics["rank_ic_mean"] > 0.02:
        print("  ● MODERATE signal — may provide incremental alpha")
    elif metrics["rank_ic_mean"] > 0.01:
        print("  ○ WEAK signal — marginal, needs further investigation")
    else:
        print("  ✗ NO signal — LSTM time-series approach not effective here")


if __name__ == "__main__":
    main()
