from core.alpha.factor_calculator import (
    FactorCalculator, 
    ts_mean, ts_std, ts_max, ts_min, ts_delay, ts_delta, 
    ts_rank, ts_argmax, ts_argmin, ts_corr, 
    ts_greater, ts_less, ts_slope, ts_rsquare, ts_resi, 
    ts_quantile, ts_log, ts_abs, ts_sum,
    torch
)

class V5FactorCalculator(FactorCalculator): 
    """
    Alpha158 Factor Calculator
    Migrated from vnpy/alpha/dataset/datasets/alpha_158.py
    """
    def __init__(self):
        super().__init__()

    def build_features(self, padded_raw) -> dict[str, torch.Tensor]:
        # Unpack
        # 0:open, 1:high, 2:low, 3:close, 4:volume, 5:turnover, 6:turnover_rate, 7:pe
        
        O = padded_raw[:, :, 0]
        H = padded_raw[:, :, 1]
        L = padded_raw[:, :, 2]
        C = padded_raw[:, :, 3]
        V = padded_raw[:, :, 4]
        # Turnover (Amount) is 5, but Alpha158 uses VWAP from Qlib which is Amount/Volume usually.
        # We can calculate VWAP from Turnover / Volume.
        # However, Alpha158 source code (qlib) often provides VWAP as a column.
        # In our padded_raw, we have Turnover (Amount).
        # vwap = Turnover / (Volume + 1e-8)
        # Note: if Volume is 0, VWAP is usually Close or NaN.
        
        T = padded_raw[:, :, 5] 
        vwap = T / (V + 1e-8)
        vwap = torch.where(V < 1e-5, C, vwap)

        features = {}
        
        # --- Candlestick pattern features ---
        # kmid = (close - open) / open
        features["kmid"] = (C - O) / O
        
        # klen = (high - low) / open
        features["klen"] = (H - L) / O
        
        # kmid_2 = (close - open) / (high - low + 1e-12)
        features["kmid_2"] = (C - O) / (H - L + 1e-12)
        
        # kup = (high - ts_greater(open, close)) / open
        features["kup"] = (H - ts_greater(O, C)) / O
        
        # kup_2 = (high - ts_greater(open, close)) / (high - low + 1e-12)
        features["kup_2"] = (H - ts_greater(O, C)) / (H - L + 1e-12)
        
        # klow = (ts_less(open, close) - low) / open
        features["klow"] = (ts_less(O, C) - L) / O
        
        # klow_2 = ((ts_less(open, close) - low) / (high - low + 1e-12))
        features["klow_2"] = (ts_less(O, C) - L) / (H - L + 1e-12)
        
        # ksft = (close * 2 - high - low) / open
        features["ksft"] = (C * 2 - H - L) / O
        
        # ksft_2 = (close * 2 - high - low) / (high - low + 1e-12)
        features["ksft_2"] = (C * 2 - H - L) / (H - L + 1e-12)
        
        # --- Price change features ---
        # {field}_0 = {field} / close
        features["open_0"] = O / C
        features["high_0"] = H / C
        features["low_0"] = L / C
        features["vwap_0"] = vwap / C
        
        # --- Time series features ---
        windows = [5, 10, 20, 30, 60]
        
        for w in windows:
            # roc_{w} = ts_delay(close, w) / close  (Note: Qlib definition might be different? 
            # In vnpy alpha_158.py: "ts_delay(close, {w}) / close".
            # Usually ROC is Close / Delay(Close) - 1. But we follow the string formula exactly.
            # "ts_delay(close, {w}) / close" -> Price(t-w) / Price(t)
            features[f"roc_{w}"] = ts_delay(C, w) / C
            
            # ma_{w} = ts_mean(close, w) / close
            features[f"ma_{w}"] = ts_mean(C, w) / C
            
            # std_{w} = ts_std(close, w) / close
            features[f"std_{w}"] = ts_std(C, w) / C
            
            # beta_{w} = ts_slope(close, w) / close
            features[f"beta_{w}"] = ts_slope(C, w) / C
            
            # rsqr_{w} = ts_rsquare(close, w)
            features[f"rsqr_{w}"] = ts_rsquare(C, w)
            
            # resi_{w} = ts_resi(close, w) / close
            features[f"resi_{w}"] = ts_resi(C, w) / C
            
            # max_{w} = ts_max(high, w) / close
            features[f"max_{w}"] = ts_max(H, w) / C
            
            # min_{w} = ts_min(low, w) / close
            features[f"min_{w}"] = ts_min(L, w) / C
            
            # qtlu_{w} = ts_quantile(close, w, 0.8) / close
            features[f"qtlu_{w}"] = ts_quantile(C, w, 0.8) / C
            
            # qtld_{w} = ts_quantile(close, w, 0.2) / close
            features[f"qtld_{w}"] = ts_quantile(C, w, 0.2) / C
            
            # rank_{w} = ts_rank(close, w)
            features[f"rank_{w}"] = ts_rank(C, w)
            
            # rsv_{w} = (close - ts_min(low, w)) / (ts_max(high, w) - ts_min(low, w) + 1e-12)
            min_l = ts_min(L, w)
            max_h = ts_max(H, w)
            features[f"rsv_{w}"] = (C - min_l) / (max_h - min_l + 1e-12)
            
            # imax_{w} = ts_argmax(high, w) / w
            features[f"imax_{w}"] = ts_argmax(H, w) / w
            
            # imin_{w} = ts_argmin(low, w) / w
            features[f"imin_{w}"] = ts_argmin(L, w) / w
            
            # imxd_{w} = (ts_argmax(high, w) - ts_argmin(low, w)) / w
            features[f"imxd_{w}"] = (ts_argmax(H, w) - ts_argmin(L, w)) / w
            
            # corr_{w} = ts_corr(close, ts_log(volume + 1), w)
            log_vol = ts_log(V + 1)
            features[f"corr_{w}"] = ts_corr(C, log_vol, w)
            
            # cord_{w} = ts_corr(close / ts_delay(close, 1), ts_log(volume / ts_delay(volume, 1) + 1), w)
            ret = C / ts_delay(C, 1)
            vol_ret_log = ts_log(V / ts_delay(V, 1) + 1)
            features[f"cord_{w}"] = ts_corr(ret, vol_ret_log, w)
            
            # cntp_{w} = ts_mean(close > ts_delay(close, 1), w)
            # Boolean to float
            is_pos = (C > ts_delay(C, 1)).float()
            features[f"cntp_{w}"] = ts_mean(is_pos, w)
            
            # cntn_{w} = ts_mean(close < ts_delay(close, 1), w)
            is_neg = (C < ts_delay(C, 1)).float()
            features[f"cntn_{w}"] = ts_mean(is_neg, w)
            
            # cntd_{w} = cntp - cntn
            features[f"cntd_{w}"] = features[f"cntp_{w}"] - features[f"cntn_{w}"]
            
            # sump_{w} = ts_sum(ts_greater(close - ts_delay(close, 1), 0), w) / (ts_sum(ts_abs(close - ts_delay(close, 1)), w) + 1e-12)
            delta_c = C - ts_delay(C, 1)
            pos_delta = ts_greater(delta_c, torch.tensor(0.0, device=C.device))
            abs_delta = ts_abs(delta_c)
            features[f"sump_{w}"] = ts_mean(pos_delta, w) * w / (ts_mean(abs_delta, w) * w + 1e-12) # ts_sum implemented as conv, but robust ts_sum uses mean * count?
            # My ts_sum(x, d) returns sum directly.
            features[f"sump_{w}"] = ts_mean(pos_delta, w) * w / (ts_mean(abs_delta, w) * w + 1e-12) # Wait, ts_sum uses ts_mean logic? No, factor_calculator has explicit ts_sum
            # Let's use ts_sum directly
            features[f"sump_{w}"] = ts_mean(pos_delta, w) * w / (ts_mean(abs_delta, w) * w + 1e-12)
            # Actually better to use ts_sum directly if available and robust
            
            # Re-check ts_sum in factor_calculator:
            # def ts_sum(x, d):
            #     ...
            #     res = sum_res.squeeze(1)
            #     res[count_res.squeeze(1) == 0] = float('nan')
            #     return res
            # It sums 0-padded values.
            
            features[f"sump_{w}"] = ts_sum(pos_delta, w) / (ts_sum(abs_delta, w) + 1e-12)

            # sumn_{w} = ts_sum(ts_greater(ts_delay(close, 1) - close, 0), w) / ...
            neg_delta_val = ts_delay(C, 1) - C
            neg_delta = ts_greater(neg_delta_val, torch.tensor(0.0, device=C.device))
            features[f"sumn_{w}"] = ts_sum(neg_delta, w) / (ts_sum(abs_delta, w) + 1e-12)
            
            # sumd_{w} = (sump - sumn)
            # Formula: (ts_sum(pos) - ts_sum(neg)) / ts_sum(abs)
            # This is equivalent to (sump_{w} - sumn_{w}) if denominators are same.
            features[f"sumd_{w}"] = (ts_sum(pos_delta, w) - ts_sum(neg_delta, w)) / (ts_sum(abs_delta, w) + 1e-12)
            
            # vma_{w} = ts_mean(volume, w) / (volume + 1e-12)
            features[f"vma_{w}"] = ts_mean(V, w) / (V + 1e-12)
            
            # vstd_{w} = ts_std(volume, w) / (volume + 1e-12)
            features[f"vstd_{w}"] = ts_std(V, w) / (V + 1e-12)
            
            # wvma_{w} = ts_std(ts_abs(close / ts_delay(close, 1) - 1) * volume, w) / (ts_mean(ts_abs(close / ts_delay(close, 1) - 1) * volume, w) + 1e-12)
            abs_ret_vol = ts_abs(C / ts_delay(C, 1) - 1) * V
            features[f"wvma_{w}"] = ts_std(abs_ret_vol, w) / (ts_mean(abs_ret_vol, w) + 1e-12)
            
            # vsump_{w} = ts_sum(ts_greater(volume - ts_delay(volume, 1), 0), w) / (ts_sum(ts_abs(volume - ts_delay(volume, 1)), w) + 1e-12)
            delta_v = V - ts_delay(V, 1)
            pos_delta_v = ts_greater(delta_v, torch.tensor(0.0, device=V.device))
            abs_delta_v = ts_abs(delta_v)
            features[f"vsump_{w}"] = ts_sum(pos_delta_v, w) / (ts_sum(abs_delta_v, w) + 1e-12)
            
            # vsumn_{w} = ts_sum(ts_greater(ts_delay(volume, 1) - volume, 0), w) / ...
            neg_delta_v_val = ts_delay(V, 1) - V
            neg_delta_v = ts_greater(neg_delta_v_val, torch.tensor(0.0, device=V.device))
            features[f"vsumn_{w}"] = ts_sum(neg_delta_v, w) / (ts_sum(abs_delta_v, w) + 1e-12)
            
            # vsumd_{w} = ...
            features[f"vsumd_{w}"] = (ts_sum(pos_delta_v, w) - ts_sum(neg_delta_v, w)) / (ts_sum(abs_delta_v, w) + 1e-12)

        # Label: ts_delay(close, -3) / ts_delay(close, -1) - 1
        # Shift -1 means shift left by 1 (future).
        # We need to use negative delay.
        # Check ts_delay implementation: 
        # res = torch.roll(x, shifts=d, dims=1)
        # if d > 0: res[:, :d] = nan
        # else: res[:, d:] = nan
        # So it supports negative d.
        
        features["label"] = ts_delay(C, -3) / ts_delay(C, -1) - 1
        
        return features
