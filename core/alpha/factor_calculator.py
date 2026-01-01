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
        cols = ["open", "high", "low", "close", "volume", "turnover", "turnover_rate", "pe", "pb", "ps", "dv_ratio", "total_mv"]
        
        # Check for industry
        if "industry" in df.columns:
            print("[FactorCalculator] Found 'industry' column. Encoding...")
            # Encode industry to integer
            # Cast to Categorical and then to Physical (Integer ID)
            # Handle nulls
            df = df.with_columns(
                pl.col("industry").fill_null("Unknown").cast(pl.Categorical).to_physical().alias("industry_code")
            )
            cols.append("industry_code")

        # Check if columns exist, if not, fill with NaN
        # Ideally AlphaEngine provides them. If not, we might crash or should handle gracefully.
        # Assuming they exist for now as per previous step.
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
            exclude_cols = {"datetime", "vt_symbol", "label", "industry"}
            raw_cols = ["open", "high", "low", "close", "volume", "turnover", "open_interest", "turnover_rate", "pe"]
            existing_raw = [c for c in raw_cols if c in df_features.columns]
            
            # Keep features and label, drop raw columns
            dataset_df = df_features.drop(existing_raw)
            
            if "label" not in dataset_df.columns:
                 print("[FactorCalculator] Error: 'label' column missing in features.")
                 raise ValueError("'label' column missing in features.")

            feature_cols = [c for c in dataset_df.columns if c not in exclude_cols]
            feature_cols.sort()
            
            # Final column order
            base_cols = ["datetime", "vt_symbol"]
            if "industry" in dataset_df.columns:
                base_cols.append("industry")
                
            final_cols = base_cols + feature_cols + ["label"]
            dataset_df = dataset_df.select(final_cols)
            
            # Apply Normalization Globally
            # Normalize label as well to remove Market Beta and focus on Alpha (ranking)
            cols_to_norm = feature_cols + ["label"]
            dataset_df = self._normalize_data(dataset_df, cols_to_norm)
            
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

    def build_features(self, padded_raw) -> dict[str, torch.Tensor]: #type: ignore
        pass

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
    # Robust Moving Average (ignoring NaNs)
    # x: (Batch, Time)
    x_u = x.unsqueeze(1)
    
    # 1. Mask valid values
    mask = (~torch.isnan(x_u)).type(x.dtype)
    x_zero = torch.nan_to_num(x_u, nan=0.0)
    
    # 2. Kernel for summation
    kernel = torch.ones(1, 1, d, device=x.device, dtype=x.dtype)
    
    # 3. Pad with 0
    x_pad = F.pad(x_zero, (d-1, 0), value=0.0)
    mask_pad = F.pad(mask, (d-1, 0), value=0.0)
    
    # 4. Convolve (Sum)
    sum_res = F.conv1d(x_pad, kernel)
    count_res = F.conv1d(mask_pad, kernel)
    
    # 5. Calculate Mean
    # Avoid div by zero without epsilon error (which kills precision for large numbers)
    count_safe = torch.where(count_res == 0, torch.ones_like(count_res), count_res)
    out = sum_res / count_safe
    
    # Restore NaNs where no data was available (count == 0)
    out[count_res == 0] = float('nan')
    
    return out.squeeze(1)

def ts_sum(x, d):
    # Robust Rolling Sum
    x_u = x.unsqueeze(1)
    
    # Handle NaNs as 0
    mask = (~torch.isnan(x_u)).type(x.dtype)
    x_zero = torch.nan_to_num(x_u, nan=0.0)
    
    kernel = torch.ones(1, 1, d, device=x.device, dtype=x.dtype)
    x_pad = F.pad(x_zero, (d-1, 0), value=0.0)
    mask_pad = F.pad(mask, (d-1, 0), value=0.0)
    
    sum_res = F.conv1d(x_pad, kernel)
    count_res = F.conv1d(mask_pad, kernel)
    
    res = sum_res.squeeze(1)
    
    # If no valid values in window, return NaN
    res[count_res.squeeze(1) == 0] = float('nan')
    return res

def ts_std(x, d):
    # Robust Moving Std (Sample Standard Deviation)
    x_64 = x.double()
    
    # We need count for Bessel's correction
    # Re-implement mean calculation to get counts
    x_u = x_64.unsqueeze(1)
    mask = (~torch.isnan(x_u)).type(x_64.dtype)
    x_zero = torch.nan_to_num(x_u, nan=0.0)
    
    # Kernel must match dtype of input (x_64 is double)
    kernel = torch.ones(1, 1, d, device=x.device, dtype=x_64.dtype)
    
    x_pad = F.pad(x_zero, (d-1, 0), value=0.0)
    mask_pad = F.pad(mask, (d-1, 0), value=0.0)
    x2_pad = F.pad(x_zero**2, (d-1, 0), value=0.0)
    
    sum_x = F.conv1d(x_pad, kernel)
    sum_x2 = F.conv1d(x2_pad, kernel)
    count = F.conv1d(mask_pad, kernel)
    
    # Avoid division by zero
    count_safe = torch.where(count == 0, torch.ones_like(count), count)
    
    mean_x = sum_x / count_safe
    mean_x2 = sum_x2 / count_safe
    
    # Population Variance = E[x^2] - (E[x])^2
    var_pop = mean_x2 - mean_x**2
    var_pop = torch.clamp(var_pop, min=0)
    
    # Sample Variance = N / (N-1) * Pop_Var
    # Only valid if count > 1
    
    # Correction factor
    # If count <= 1, correction is undefined (or inf), we should mask later
    bessel_correction = count / (count - 1)
    
    var_sample = var_pop * bessel_correction
    
    std = torch.sqrt(var_sample)
    
    # Mask where count < 2
    std[count < 2] = float('nan')
    
    return std.squeeze(1).to(x.dtype)

def ts_max(x, d):
    # Robust Rolling Max (Ignore NaNs)
    x_u = x.unsqueeze(1)
    
    # Fill NaN with -inf
    x_filled = torch.nan_to_num(x_u, nan=-float('inf'))
    
    # Pad with -inf
    x_pad = F.pad(x_filled, (d-1, 0), value=-float('inf')) 
    
    res = F.max_pool1d(x_pad, kernel_size=d, stride=1)
    res = res.squeeze(1)
    
    # Restore NaNs if all were -inf (no data)
    res[res == -float('inf')] = float('nan')
    return res

def ts_min(x, d):
    # Robust Rolling Min (Ignore NaNs)
    x_u = x.unsqueeze(1)
    
    # Fill NaN with inf
    x_filled = torch.nan_to_num(x_u, nan=float('inf'))
    
    # Pad with inf
    x_pad = F.pad(x_filled, (d-1, 0), value=float('inf'))
    
    # Min pooling = - Max pooling of negative
    res = -F.max_pool1d(-x_pad, kernel_size=d, stride=1)
    res = res.squeeze(1)
    
    # Restore NaNs if all were inf
    res[res == float('inf')] = float('nan')
    return res
    
def ts_delta(x, d):
    return x - ts_delay(x, d)
    
def ta_atr(h, l, c, d):
    # TR = max(h-l, abs(h-c_prev), abs(l-c_prev))
    c_prev = ts_delay(c, 1)
    
    # Handle NaNs in c_prev (first day)
    # If c_prev is NaN, abs(h-c_prev) is NaN.
    # We want robust max.
    # But usually ATR needs previous close.
    # We can rely on robust ts_mean to handle NaNs in TR stream.
    
    tr1 = h - l
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    
    # torch.maximum propagates NaN.
    # We can use nan_to_num or fmax? 
    # torch.fmax ignores NaN!
    tr = torch.fmax(tr1, torch.fmax(tr2, tr3))
    
    return ts_mean(tr, d)

def ta_rsi(c, d):
    delta = ts_delta(c, 1)
    
    # delta is NaN for first element
    up = torch.clamp(delta, min=0)
    down = torch.clamp(-delta, min=0)
    
    # ts_mean handles NaNs in up/down
    avg_up = ts_mean(up, d)
    avg_down = ts_mean(down, d)
    
    rs = avg_up / (avg_down + 1e-8)
    return 100 - 100 / (1 + rs)
    
def sequence_mean(x, d):
    # for 'ma_alignment' logic: ts_mean(x, d)
    return ts_mean(x, d)

def ts_corr(x, y, d):
    # Rolling Correlation
    mean_x = ts_mean(x, d)
    mean_y = ts_mean(y, d)
    mean_xy = ts_mean(x * y, d)
    
    cov_xy = mean_xy - mean_x * mean_y
    
    std_x = ts_std(x, d)
    std_y = ts_std(y, d)
    
    corr = cov_xy / (std_x * std_y + 1e-8)
    return corr

def cs_rank(x):
    # x: (Batch, Time)
    # Rank over Batch dim.
    
    # 1. Count valid
    mask = ~torch.isnan(x)
    valid_count = mask.sum(dim=0, keepdim=True)
    
    # 2. Fill NaN with -inf for ranking
    x_filled = torch.nan_to_num(x, nan=-float('inf'))
    
    # 3. Rank
    # argsort twice gives 0-based rank indices
    ranks = x_filled.argsort(dim=0).argsort(dim=0).float()
    
    # 4. Adjust ranks to ignore NaNs (which are at the bottom)
    nan_count = (~mask).sum(dim=0, keepdim=True)
    ranks_adj = ranks - nan_count
    
    # 5. Normalize to [0, 1]
    denom = valid_count - 1
    denom = torch.clamp(denom, min=1)
    
    res = ranks_adj / denom
    
    # Restore NaNs
    res[~mask] = float('nan')
    return res

def cs_scale(x):
    # Rescale such that sum(abs(x)) = 1 (across batch)
    # x: (Batch, Time)
    
    # 1. Calculate sum of abs
    # Treat NaN as 0 for sum
    x_zero = torch.nan_to_num(x, nan=0.0)
    sum_abs = torch.sum(torch.abs(x_zero), dim=0, keepdim=True)
    
    # 2. Scale
    res = x / (sum_abs + 1e-8)
    return res

def ts_rank(x, d):
    # Rolling Rank over past d days
    # Returns normalized rank [0, 1]
    
    count_valid = torch.zeros_like(x)
    rank = torch.zeros_like(x)
    
    current = x
    
    for i in range(d):
        past = ts_delay(x, i)
        mask_valid = ~torch.isnan(past)
        count_valid += mask_valid.float()
        
        # current >= past
        # Propagate NaNs from current
        is_ge = (current >= past) & mask_valid
        rank += is_ge.float()
        
    rank[torch.isnan(current)] = float('nan')
    
    denom = count_valid
    denom = torch.clamp(denom, min=1) 
    
    out = rank / denom
    
    # If no valid data, rank is 0, out is 0. But should be NaN.
    # count_valid == 0 implies all were NaNs.
    out[count_valid == 0] = float('nan')
    
    return out

def quesval(threshold, x, true_val, false_val):
    # if x > threshold: true_val else: false_val
    # threshold can be scalar or tensor
    # x is tensor
    condition = x > threshold
    
    # Broadcast if necessary (helper to handle scalar/tensor mix)
    # torch.where supports broadcasting
    # But if true_val/false_val are scalars, we need to ensure they match type/device if x is tensor?
    # torch.where handles scalar scalars.
    
    # Ensure float dtype if inputs are scalars to avoid int64 issues with NaNs later
    target_dtype = x.dtype if x.is_floating_point() else torch.float32

    if isinstance(true_val, (float, int)):
        true_val = torch.tensor(true_val, device=x.device, dtype=target_dtype)
    if isinstance(false_val, (float, int)):
        false_val = torch.tensor(false_val, device=x.device, dtype=target_dtype)
        
    return torch.where(condition, true_val, false_val)

def quesval2(threshold, x, true_val, false_val):
    # if x < threshold: true_val else: false_val
    # (Based on vnpy implementation: feature1 < threshold)
    condition = x < threshold
    
    # Ensure float dtype if inputs are scalars to avoid int64 issues with NaNs later
    target_dtype = x.dtype if x.is_floating_point() else torch.float32

    if isinstance(true_val, (float, int)):
        true_val = torch.tensor(true_val, device=x.device, dtype=target_dtype)
    if isinstance(false_val, (float, int)):
        false_val = torch.tensor(false_val, device=x.device, dtype=target_dtype)
        
    return torch.where(condition, true_val, false_val)

def ts_argmax(x, d):
    # Returns 1-based index (1=Oldest, d=Newest) where max occurred
    # Consistent with vnpy (Oldest=1)
    # My lag: 0=Newest, d-1=Oldest.
    # Result = d - lag.
    
    max_val = torch.full_like(x, -float('inf'))
    argmax_lag = torch.zeros_like(x)
    
    for i in range(d):
        cur = ts_delay(x, i)
        mask = (cur > max_val) & (~torch.isnan(cur))
        max_val = torch.where(mask, cur, max_val)
        argmax_lag = torch.where(mask, torch.tensor(i, device=x.device, dtype=x.dtype), argmax_lag)
        
    argmax_lag[max_val == -float('inf')] = float('nan')
    return d - argmax_lag

def ts_argmin(x, d):
    # Returns 1-based index (1=Oldest, d=Newest) where min occurred
    min_val = torch.full_like(x, float('inf'))
    argmin_lag = torch.zeros_like(x)
    
    for i in range(d):
        cur = ts_delay(x, i)
        mask = (cur < min_val) & (~torch.isnan(cur))
        min_val = torch.where(mask, cur, min_val)
        argmin_lag = torch.where(mask, torch.tensor(i, device=x.device, dtype=x.dtype), argmin_lag)
        
    argmin_lag[min_val == float('inf')] = float('nan')
    return d - argmin_lag

def ts_product(x, d):
    # Rolling Product.
    res = torch.ones_like(x)
    for i in range(d):
        cur = ts_delay(x, i)
        cur_filled = torch.nan_to_num(cur, nan=1.0)
        res = res * cur_filled
    return res

def ts_decay_linear(x, d):
    # Weighted average with weights d, d-1, ..., 1
    sum_w_x = torch.zeros_like(x)
    sum_w = torch.zeros_like(x)
    
    for i in range(d):
        w = d - i
        cur = ts_delay(x, i)
        mask = ~torch.isnan(cur)
        
        cur_zero = torch.nan_to_num(cur, nan=0.0)
        sum_w_x += cur_zero * w
        sum_w += mask.float() * w
        
    return sum_w_x / (sum_w + 1e-8)

def ts_cov(x, y, d):
    # Consistent covariance
    mask = (~torch.isnan(x)) & (~torch.isnan(y))
    x_m = torch.where(mask, x, torch.tensor(float('nan'), device=x.device))
    y_m = torch.where(mask, y, torch.tensor(float('nan'), device=y.device))
    
    mean_x = ts_mean(x_m, d)
    mean_y = ts_mean(y_m, d)
    mean_xy = ts_mean(x_m * y_m, d)
    
    return mean_xy - mean_x * mean_y

def pow1(x, e):
    # Signed power: sign(x) * abs(x)^e
    # Matches vnpy pow1
    return torch.sign(x) * torch.pow(torch.abs(x), e)

def pow2(x, e):
    # vnpy pow2: x^e. 
    # If x < 0 and e is integer -> -1 * |x|^e
    # Else if x < 0 -> 0 (or NaN/Null in vnpy)
    # Here we can approximate.
    # Usually e is tensor.
    
    # Condition: x > 0
    res = torch.zeros_like(x)
    pos_mask = x > 0
    res[pos_mask] = torch.pow(x[pos_mask], e[pos_mask])
    
    # Condition: x < 0
    neg_mask = x < 0
    # Check if e is integer
    # float equality is tricky. 
    # Assume if abs(e - round(e)) < epsilon
    e_round = torch.round(e)
    is_int = torch.abs(e - e_round) < 1e-5
    
    neg_valid = neg_mask & is_int
    res[neg_valid] = -1 * torch.pow(torch.abs(x[neg_valid]), e[neg_valid])
    
    # Otherwise 0 (or NaN). vnpy returns None/NaN.
    # We leave as 0 or set to NaN? 
    # vnpy: otherwise(pl.lit(None)).fill_null(0).
    # So it returns 0.
    
    return res

def ts_greater(x, y):
    # max(x, y) ignoring NaNs
    return torch.fmax(x, y)

def ts_less(x, y):
    # min(x, y) ignoring NaNs
    return torch.fmin(x, y)

def ts_log(x):
    return torch.log(x)

def ts_abs(x):
    return torch.abs(x)

def _rolling_window(x, d):
    # Helper to unfold tensor for rolling operations
    # x: (Batch, Time) -> (Batch, Time, d)
    # We pad the time dimension with NaNs at the beginning
    # Pad (d-1) to the left
    pad_size = d - 1
    # unsqueeze to (Batch, 1, Time) for padding? No, F.pad works on last dim
    x_pad = F.pad(x, (pad_size, 0), value=float('nan'))
    # unfold: dimension, size, step
    return x_pad.unfold(dimension=1, size=d, step=1)

def ts_quantile(x, d, q):
    # x: (Batch, Time)
    # q: scalar float 0..1
    
    # 1. Unfold to (Batch, Time, d)
    x_unfolded = _rolling_window(x, d)
    
    # 2. Quantile
    # torch.quantile requires value to be computed.
    # It might be slow on large data.
    # nanquantile is available in newer torch versions?
    # torch.nanquantile is available since 1.8.
    
    # We operate on the last dimension
    return torch.nanquantile(x_unfolded, q, dim=2)

def ts_slope(y, d):
    # Simple rolling slope of y against x=0..d-1
    # beta = Cov(x, y) / Var(x)
    # Var(x) is constant for window size d.
    # Cov(x, y) = E[xy] - E[x]E[y]
    
    # x constants
    x = torch.arange(d, device=y.device, dtype=y.dtype)
    mean_x = (d - 1) / 2.0
    var_x = (d**2 - 1) / 12.0
    
    # E[y] (rolling mean) - simplified, not ignoring NaNs to match convolution
    # If we want to match ts_mean robust logic, we can't efficiently mix with convolution for xy.
    # So we stick to convolution that propagates NaNs.
    
    kernel_sum = torch.ones(1, 1, d, device=y.device, dtype=y.dtype)
    kernel_xy = torch.arange(d, device=y.device, dtype=y.dtype).view(1, 1, d)
    
    y_u = y.unsqueeze(1)
    y_pad = F.pad(y_u, (d-1, 0), value=float('nan')) # Pad with NaN, convolution will return NaN if any NaN
    # Actually conv1d with NaN input returns NaN? Yes.
    
    sum_y = F.conv1d(y_pad, kernel_sum).squeeze(1)
    sum_xy = F.conv1d(y_pad, kernel_xy).squeeze(1)
    
    mean_y = sum_y / d
    mean_xy = sum_xy / d
    
    cov_xy = mean_xy - mean_x * mean_y
    beta = cov_xy / var_x
    return beta

def ts_rsquare(y, d):
    # R^2 = beta^2 * Var(x) / Var(y)
    # We need beta and Var(y)
    
    # Re-calculate basics (could optimize if we had a shared context)
    beta = ts_slope(y, d)
    
    # Var(y) = E[y^2] - E[y]^2
    # Standard rolling variance (population or sample? R^2 usually uses same basis)
    # Let's use population variance for consistency with Var(x) derivation
    
    kernel_sum = torch.ones(1, 1, d, device=y.device, dtype=y.dtype)
    y_u = y.unsqueeze(1)
    y_pad = F.pad(y_u, (d-1, 0), value=float('nan'))
    
    sum_y = F.conv1d(y_pad, kernel_sum).squeeze(1)
    sum_yy = F.conv1d(y_pad**2, kernel_sum).squeeze(1)
    
    mean_y = sum_y / d
    mean_yy = sum_yy / d
    
    var_y = mean_yy - mean_y**2
    
    var_x = (d**2 - 1) / 12.0
    
    r2 = (beta**2 * var_x) / (var_y + 1e-8)
    # Clip 0-1
    return torch.clamp(r2, 0, 1)

def ts_resi(y, d):
    # Residual at the last point
    # resi = y_t - (alpha + beta * x_t)
    # x_t = d - 1
    # alpha = mean_y - beta * mean_x
    # y_pred_t = mean_y - beta * mean_x + beta * (d - 1)
    #          = mean_y + beta * (d - 1 - (d-1)/2)
    #          = mean_y + beta * (d-1)/2
    
    beta = ts_slope(y, d)
    
    kernel_sum = torch.ones(1, 1, d, device=y.device, dtype=y.dtype)
    y_u = y.unsqueeze(1)
    y_pad = F.pad(y_u, (d-1, 0), value=float('nan'))
    sum_y = F.conv1d(y_pad, kernel_sum).squeeze(1)
    mean_y = sum_y / d
    
    mean_x = (d - 1) / 2.0
    
    # y_t is just y (the current value)
    y_pred = mean_y + beta * ((d - 1) - mean_x)
    
    return y - y_pred

def cs_group_mean(x, groups, num_groups=None):
    """
    Calculate Cross-Sectional Mean per Group.
    x: (Batch, Time) Values
    groups: (Batch, Time) Integer Group IDs
    
    Returns:
    out: (Batch, Time) where each element is the mean of its group at that time step.
    """
    # 1. Handle NaNs in groups
    # Create mask for valid groups
    mask_groups = ~torch.isnan(groups)
    
    # Fill NaN groups with 0 for safe conversion to long
    # (These 0s will be masked out during scatter so they don't affect Group 0)
    groups_filled = torch.nan_to_num(groups, nan=0.0)
    groups_long = groups_filled.long()
    
    if num_groups is None:
        # Infer max group ID from valid groups only
        if mask_groups.any():
            # Masked select is 1D, which is fine for max
            valid_groups = torch.masked_select(groups_long, mask_groups)
            num_groups = int(torch.max(valid_groups).item()) + 1
        else:
            num_groups = 1 # Fallback if no valid groups

    B, T = x.shape
    
    # Prepare scatter target
    # Shape: (NumGroups, Time)
    sums = torch.zeros(num_groups, T, device=x.device, dtype=x.dtype)
    counts = torch.zeros(num_groups, T, device=x.device, dtype=x.dtype)
    
    # 2. Handle NaNs in x
    mask_x = ~torch.isnan(x)
    
    # Combined mask: Valid Group AND Valid Data
    valid_mask = mask_groups & mask_x
    
    x_zero = torch.nan_to_num(x, nan=0.0)
    
    # Scatter Add
    # We scatter 'x' into 'sums' at 'groups_long' indices.
    # We only want to add where 'valid_mask' is True.
    # Where valid_mask is False, we add 0.0 (which does nothing).
    # Note: If a position was NaN group (mapped to 0), valid_mask is False.
    # So we add 0.0 to sum[0] and 0.0 to count[0]. This is safe.
    
    # sums[group, t] += x[b, t] * mask
    sums.scatter_add_(0, groups_long, x_zero * valid_mask.float())
    counts.scatter_add_(0, groups_long, valid_mask.float())
    
    # Calculate Means
    means = sums / (counts + 1e-8)
    
    # Map back to (Batch, Time)
    out = means.gather(0, groups_long)
    
    # Restore NaNs where groups were invalid or result is invalid (count=0)
    # If count was 0, means is 0. 
    # But for an original position (b, t):
    # If groups[b,t] was NaN -> we want NaN.
    # If groups[b,t] was valid G, but count[G] was 0 (no valid x in that group) -> we want NaN.
    
    # Check count for the assigned group
    assigned_counts = counts.gather(0, groups_long)
    
    # Final mask
    final_mask = mask_groups & (assigned_counts > 0)
    
    out[~final_mask] = float('nan')
    
    return out

def ts_kdj(C, H, L, n=9):
    # C, H, L: (Batch, Time)
    # 1. RSV
    low_n = ts_min(L, n)
    high_n = ts_max(H, n)
    rsv = (C - low_n) / (high_n - low_n + 1e-8) * 100
    
    # Fill NaN with 50
    rsv = torch.nan_to_num(rsv, nan=50.0)
    
    # 2. Iterate for K, D
    # K = 2/3 PrevK + 1/3 RSV
    # D = 2/3 PrevD + 1/3 K
    
    B, T = C.shape
    device = C.device
    dtype = C.dtype
    
    k = torch.zeros_like(C)
    d = torch.zeros_like(C)
    
    k_val = torch.full((B,), 50.0, device=device, dtype=dtype)
    d_val = torch.full((B,), 50.0, device=device, dtype=dtype)
    
    # Loop over Time
    for t in range(T):
        curr_rsv = rsv[:, t]
        k_val = (2.0/3.0) * k_val + (1.0/3.0) * curr_rsv
        d_val = (2.0/3.0) * d_val + (1.0/3.0) * k_val
        k[:, t] = k_val
        d[:, t] = d_val
        
    j = 3 * k - 2 * d
    return k, d, j

