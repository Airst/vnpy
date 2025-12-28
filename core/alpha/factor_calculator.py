import polars as pl
from datetime import datetime, timedelta
from typing import List
import torch.nn.functional as F
import numpy as np

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class FactorCalculator:
    """
    Base class for A-Share Factor Calculators.
    Provides common methods for symbol retrieval and data loading.
    """

    def __init__(self) -> None:
        print(f"[FactorCalculator] Using device: {device}")

    def calculate_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Main entry point. Takes raw Polars DataFrame, calculates features on GPU,
        returns Polars DataFrame with features added.
        """
        print("[FactorCalculator] Preparing data for GPU...")
        
        # 1. Encode Symbols and Sort
        # We need to process by symbol.
        # Ideally, we pad sequences to the max length.
        
        # Ensure sorted
        df = df.sort(["vt_symbol", "datetime"])
        
        # Convert to numpy/torch
        # We need to handle the grouping.
        # Strategy:
        # 1. Get unique symbols and their counts/indices.
        # 2. Create padded tensor (Batch, Time, Features).
        
        # Extract columns
        dates = df["datetime"].to_numpy()
        symbols = df["vt_symbol"].to_numpy()
        
        # Numerical columns needed
        cols = ["open", "high", "low", "close", "volume"]
        raw_data = df.select(cols).to_numpy().astype(np.float32)
        
        # Group info
        unique_symbols, inverse_indices, counts = np.unique(symbols, return_inverse=True, return_counts=True)
        num_stocks = len(unique_symbols)
        max_len = counts.max()
        
        print(f"[FactorCalculator] Stocks: {num_stocks}, Max Len: {max_len}")
        
        # Prepare Tensors
        # Shape: (Batch, Time) for each feature
        # We will pad with NaNs.
        
        # Create a mask for valid data
        # To do this efficiently in numpy/torch:
        # We can construct the padded array.
        
        # Fast construction of padded array:
        # We know the start index of each group (cumsum of counts)
        # But `np.unique` returns counts in sorted order of unique_symbols.
        # Since `df` is sorted by `vt_symbol`, `unique_symbols` should match the order in `df` (if sorted).
        # Let's verify `df` sort order matches `np.unique` output order. Yes, if strings.
        
        # Calculate split indices
        # split_indices = np.cumsum(counts)[:-1]
        # split_arrays = np.split(raw_data, split_indices)
        # This is slow if 5000 stocks.
        
        # Optimized Padded Construction:
        # Create a flat index array that maps to (Batch, Time)
        # 1. Create 'time_idx' for each row: 0, 1, 2... for each stock
        #    We can do this by `df.with_columns(pl.int_range(0, pl.len()).over("vt_symbol").alias("t_idx"))`
        #    Then `t_idx` and `symbol_idx` (from inverse_indices) give coordinates.
        
        print("[FactorCalculator] Creating padded tensors...")
        # Add indexers in Polars (fast)
        df_idx = df.select(["vt_symbol"]).with_columns([
            pl.int_range(0, pl.len()).over("vt_symbol").alias("t_idx")
        ])
        t_indices = df_idx["t_idx"].to_numpy()
        s_indices = inverse_indices # Already 0..N-1 maps to symbols
        
        # Create Empty Tensor (Batch, MaxLen, NumCols) filled with NaN
        padded_raw = torch.full((num_stocks, max_len, len(cols)), float('nan'), device=device, dtype=torch.float32)
        
        # Fill data
        # Use simple indexing: padded[s_idx, t_idx] = raw_val
        # Move raw_data to GPU first if VRAM allows, else loop?
        # A-share daily data ~3000 days * 5000 stocks * 5 cols * 4 bytes ~ 300MB. Fits easily.
        
        raw_tensor = torch.from_numpy(raw_data.copy()).to(device)
        t_indices_t = torch.from_numpy(t_indices.copy()).to(device)
        s_indices_t = torch.from_numpy(s_indices.copy()).to(device)
        
        padded_raw[s_indices_t, t_indices_t, :] = raw_tensor
        
        # --- Feature Calculation ---
        print("[FactorCalculator] Calculating features...")
        features = self.build_features(padded_raw)
        
        # --- Flatten and Merge ---
        print("[FactorCalculator] reconstructing dataframe...")
        
        # We have tensors (Batch, Time).
        # We need to extract values corresponding to valid data points (not padding).
        # We can use `s_indices_t` and `t_indices_t` to gather results if we kept the mapping?
        # Actually `s_indices_t` and `t_indices_t` map from Raw 1D to Padded 2D.
        # So `padded[s_indices_t, t_indices_t]` extracts the values back to 1D aligned with `df`.
        
        feature_cols = []
        feature_names = []
        
        for name, tensor in features.items():
            # Extract
            flat_vals = tensor[s_indices_t, t_indices_t]
            # To CPU numpy
            feature_cols.append(flat_vals.cpu().numpy())
            feature_names.append(name)
            
        # Add to original DF
        # df is already sorted by vt_symbol, datetime
        
        # Create polars series
        new_cols = [
            pl.Series(name, vals).fill_nan(None) 
            for name, vals in zip(feature_names, feature_cols)
        ]
        
        df_features = df.with_columns(new_cols)

        print("[FactorCalculator] Pre-processing data (Global Cross-Sectional Normalization)...")
        try:
            # Identify feature columns
            exclude_cols = {"datetime", "vt_symbol", "label"}
            raw_cols = ["open", "high", "low", "close", "volume", "turnover", "open_interest"]
            existing_raw = [c for c in raw_cols if c in df_features.columns]
            
            # Keep features and label, drop raw columns
            dataset_df = df_features.drop(existing_raw)
            
            if "label" not in dataset_df.columns:
                 print("[FactorCalculator] Error: 'label' column missing in features.")
                 raise ValueError("'label' column missing in features.")

            feature_cols = [c for c in dataset_df.columns if c not in exclude_cols]
            feature_cols.sort()
            
            # Final column order
            final_cols = ["datetime", "vt_symbol"] + feature_cols + ["label"]
            dataset_df = dataset_df.select(final_cols)
            
            # Apply Normalization Globally
            dataset_df = self._normalize_data(dataset_df, feature_cols)
            
            return dataset_df
        except Exception as e:
            print(f"[FactorCalculator] Data pre-processing error: {e}")
            import traceback
            traceback.print_exc()
            raise e


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

    def build_features(self, padded_raw):
        # Unpack
        # 0:open, 1:high, 2:low, 3:close, 4:volume
        # Transpose to (Batch, Features, Time) might be better for conv1d?
        # Conv1d expects (Batch, In_Channels, Time).
        # So (Batch, Time) -> (Batch, 1, Time)
        
        # Let's keep (Batch, Time) for basic ops
        O = padded_raw[:, :, 0]
        H = padded_raw[:, :, 1]
        L = padded_raw[:, :, 2]
        C = padded_raw[:, :, 3]
        V = padded_raw[:, :, 4]
        
        # Helper for mask (where C is not NaN)
        mask = ~torch.isnan(C)

        features = {}

        # 1. Momentum / Reversal
        features["rev_5d"] = (C / ts_delay(C, 5) - 1) * -1
        features["mom_5d"] = C / ts_delay(C, 5) - 1
        features["bias_6"] = (C / ts_mean(C, 6)) - 1
        features["mom_20d"] = C / ts_delay(C, 20) - 1
        features["mom_60d"] = C / ts_delay(C, 60) - 1
        features["mom_120d"] = C / ts_delay(C, 120) - 1
        features["ma_bias_60"] = C / ts_mean(C, 60) - 1
        features["ma_bias_120"] = C / ts_mean(C, 120) - 1
        features["price_zscore_20d"] = (C - ts_mean(C, 20)) / (ts_std(C, 20) + 1e-8)
        
        # 2. Volatility
        ret_1 = C / ts_delay(C, 1) - 1
        features["volatility_20d"] = ts_std(ret_1, 20)
        features["volatility_60d"] = ts_std(ret_1, 60)
        features["std_20"] = ts_std(C, 20) / C
        features["atr_ratio_14"] = ta_atr(H, L, C, 14) / C
        features["drawdown_20d"] = (C / ts_max(C, 20)) - 1
        features["daily_range"] = H / L - 1
        
        # New Positive Factors
        # Inverse Volatility (Low Vol Anomaly)
        features["inv_std_20"] = 1.0 / (features["std_20"] + 1e-4)
        
        # Trend Efficiency (Net Move / Total Path)
        net_move_20 = (C - ts_delay(C, 20)).abs()
        total_path_20 = ts_sum((C - ts_delay(C, 1)).abs(), 20)
        features["trend_efficiency_20"] = net_move_20 / (total_path_20 + 1e-8)
        
        # 3. Technical
        ma_20 = ts_mean(C, 20)
        std_20 = ts_std(C, 20)
        features["bollinger_position"] = (C - ma_20) / (std_20 * 2 + 1e-8)
        features["boll_width_20"] = (std_20 * 4) / ma_20
        
        features["rsi_14"] = ta_rsi(C, 14)
        features["rsi_6"] = ta_rsi(C, 6)
        
        # KDJ RSV
        ll_9 = ts_min(L, 9)
        hh_9 = ts_max(H, 9)
        features["kdj_rsv_9"] = (C - ll_9) / (hh_9 - ll_9 + 0.0001)
        
        # PSY: Mean of sign(return) > 0? No, sign of delta.
        # sign(ts_delta(close, 1)) -> 1 if >0, -1 if <0, 0. 
        # PSY is percentage of up days. (sign > 0).
        # We can implement:
        delta_c = ts_delta(C, 1)
        is_up = (delta_c > 0).float()
        features["psy_12"] = ts_mean(is_up, 12)
        
        # MA Alignment
        ma_5 = ts_mean(C, 5)
        ma_10 = ts_mean(C, 10)
        # ma_20 defined above
        # ((ma_5 > ma_10) & (ma_10 > ma_20)) * 1
        features["ma_alignment"] = ((ma_5 > ma_10) & (ma_10 > ma_20)).float()
        
        # 4. Volume
        features["volume_ratio"] = V / ts_mean(V, 20)
        features["vol_roc_5"] = V / ts_delay(V, 5) - 1
        features["vol_cv_20"] = ts_std(V, 20) / ts_mean(V, 20)
        features["vol_stability_20"] = 1.0 / (features["vol_cv_20"] + 1e-4)
        
        # Price Volume Divergence
        # (close > prev_close) & (volume < prev_volume)
        c_prev = ts_delay(C, 1)
        v_prev = ts_delay(V, 1)
        price_up = C > c_prev
        vol_down = V < v_prev
        features["price_volume_divergence"] = (price_up & vol_down).float()
        
        # 5. Money Flow
        # (((close - open) / (high - low + 0.0001)) * volume)
        mf_val = ((C - O) / (H - L + 0.0001)) * V
        features["money_flow_20d"] = mf_val / ts_mean(mf_val, 20)
        
        # VWAP Dev
        # ts_sum(close * volume, 20) / ts_sum(volume, 20)
        vwap_20 = ts_sum(C * V, 20) / ts_sum(V, 20)
        features["vwap_dev_20"] = C / vwap_20 - 1
        
        # Label: Next 5 days return
        # ts_delay(close, -5) / close - 1
        # Shift LEFT by 5.
        features["label"] = ts_delay(C, -5) / C - 1

        return features

# Helper: Rolling Ops
def ts_delay(x, d):
    # shift right by d. Fill with NaN.
    # x: (Batch, Time)
    res = torch.roll(x, shifts=d, dims=1)
    # Mask the rolled-in elements (first d)
    if d > 0:
        res[:, :d] = float('nan')
    else:
        res[:, d:] = float('nan')
    return res

def ts_mean(x, d):
    # Moving Average
    # Use conv1d.
    # x: (Batch, Time) -> (Batch, 1, Time)
    x_u = x.unsqueeze(1)
    # Filter NaN for conv? Conv with NaN results in NaN, which is correct (if any NaN in window).
    # But usually we want 'ignore NaN' only if data is missing inside history. 
    # With padded structure, we have NaNs at end.
    # Standard conv1d is fine.
    kernel = torch.ones(1, 1, d, device=device) / d
    # Pad input to keep size same (aligned to right)
    # We want result[t] to be mean(t-d+1 ... t)
    # So we pad left by d-1.
    x_pad = F.pad(x_u, (d-1, 0), value=float('nan'))
    res = F.conv1d(x_pad, kernel)
    return res.squeeze(1)

def ts_sum(x, d):
    x_u = x.unsqueeze(1)
    kernel = torch.ones(1, 1, d, device=device)
    x_pad = F.pad(x_u, (d-1, 0), value=float('nan'))
    res = F.conv1d(x_pad, kernel)
    return res.squeeze(1)

def ts_std(x, d):
    # E[x^2] - (E[x])^2
    mean_x = ts_mean(x, d)
    mean_x2 = ts_mean(x**2, d)
    var = mean_x2 - mean_x**2
    # Numerical stability clamping
    var = torch.clamp(var, min=0)
    return torch.sqrt(var)

def ts_max(x, d):
    # MaxPool1d
    x_u = x.unsqueeze(1)
    # Pad left
    x_pad = F.pad(x_u, (d-1, 0), value=float('nan')) 
    # Note: pad with -inf for max
    res = F.max_pool1d(x_pad, kernel_size=d, stride=1)
    return res.squeeze(1)

def ts_min(x, d):
    x_u = x.unsqueeze(1)
    x_pad = F.pad(x_u, (d-1, 0), value=float('nan'))
    res = -F.max_pool1d(-x_pad, kernel_size=d, stride=1)
    return res.squeeze(1)
    
def ts_delta(x, d):
    return x - ts_delay(x, d)
    
def ta_atr(h, l, c, d):
    # TR = max(h-l, abs(h-c_prev), abs(l-c_prev))
    c_prev = ts_delay(c, 1)
    tr1 = h - l
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    tr = torch.maximum(tr1, torch.maximum(tr2, tr3))
    # ATR is usually SMA(TR) or RMA(TR). Use SMA for simplicity or match vnpy?
    # vnpy ta_atr usually uses TA-Lib which is Wilder's. 
    # Approximating with SMA for now as it's cleaner.
    return ts_mean(tr, d)

def ta_rsi(c, d):
    delta = ts_delta(c, 1)
    up = torch.clamp(delta, min=0)
    down = torch.clamp(-delta, min=0)
    # Use SMA for RS (or use EMA if needed). 
    # Using SMA matching ts_mean style for speed/simplicity here.
    avg_up = ts_mean(up, d)
    avg_down = ts_mean(down, d)
    rs = avg_up / (avg_down + 1e-8)
    return 100 - 100 / (1 + rs)
    
def sequence_mean(x, d):
    # for 'ma_alignment' logic: ts_mean(x, d)
    return ts_mean(x, d)