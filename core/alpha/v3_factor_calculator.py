from core.alpha.factor_calculator import FactorCalculator, cs_rank, ts_corr, ts_cov, ts_delay, ts_mean, ts_min, ts_max, ts_quantile, ts_std, ts_sum, ts_rsquare, ts_slope, ta_atr, ta_rsi, cs_group_mean, ts_kdj, torch


class V3FactorCalculator(FactorCalculator): 
    def __init__(self):
        super().__init__()

    def build_features(self, padded_raw) -> dict[str, torch.Tensor]:
        # Unpack
        # 0:open, 1:high, 2:low, 3:close, 4:volume, 5:turnover, 6:turnover_rate, 7:pe
        # 8:pb, 9:ps, 10:dv_ratio, 11:total_mv
        
        # Let's keep (Batch, Time) for basic ops
        O = padded_raw[:, :, 0]
        H = padded_raw[:, :, 1]
        L = padded_raw[:, :, 2]
        C = padded_raw[:, :, 3]
        V = padded_raw[:, :, 4]
        T = padded_raw[:, :, 5] # Turnover (Amount)
        TR = padded_raw[:, :, 6] # Turnover Rate
        PE = padded_raw[:, :, 7] # PE Ratio
        PB = padded_raw[:, :, 8] # PB Ratio
        PS = padded_raw[:, :, 9] # PS Ratio
        DV = padded_raw[:, :, 10] # Dividend Ratio
        MV = padded_raw[:, :, 11] # Total Market Value
        
        # Industry Code (if available, index 12)
        IND = None
        if padded_raw.shape[2] > 12:
            IND = padded_raw[:, :, 12]
        
        # Helper vars
        # Helper for mask (where C is not NaN)
        mask = ~torch.isnan(C)
        # VWAP = Turnover / Volume. 
        # Handle cases where Volume is 0 or NaN.
        vwap = T / (V + 1e-8)
        vwap = torch.where(torch.isnan(vwap), C, vwap) 

        features = {}

        # 1. Momentum / Reversal
        features["rev_5d"] = (C / ts_delay(C, 5) - 1) * -1
        features["mom_5d"] = C / ts_delay(C, 5) - 1
        # features["bias_6"] = (C / ts_mean(C, 6)) - 1
        features["mom_20d"] = C / ts_delay(C, 20) - 1
        features["mom_60d"] = C / ts_delay(C, 60) - 1
        features["mom_120d"] = C / ts_delay(C, 120) - 1
        # features["ma_bias_60"] = C / ts_mean(C, 60) - 1
        features["ma_bias_120"] = C / ts_mean(C, 120) - 1
        features["price_zscore_20d"] = (C - ts_mean(C, 20)) / (ts_std(C, 20) + 1e-8)

        # A-share specific: Overnight vs Intraday
        features["ret_overnight"] = O / ts_delay(C, 1) - 1
        features["ret_intraday"] = C / O - 1
        
        # Bias (Distance from MA) - Mean Reversion signals
        features["bias_5"] = C / ts_mean(C, 5) - 1
        features["bias_10"] = C / ts_mean(C, 10) - 1
        features["bias_20"] = C / ts_mean(C, 20) - 1
        features["bias_60"] = C / ts_mean(C, 60) - 1

        # Industry Factors
        if IND is not None:
             # Industry Momentum (20d, 5d)
             # Group Mean of individual stock momentums
             ind_mom_60d = cs_group_mean(features["mom_60d"], IND)
             ind_mom_20d = cs_group_mean(features["mom_20d"], IND)
             ind_mom_5d = cs_group_mean(features["mom_5d"], IND)
             
             features["ind_mom_60d"] = ind_mom_60d
             features["ind_mom_20d"] = ind_mom_20d
             features["ind_mom_5d"] = ind_mom_5d
             
             # Relative Momentum (Stock Mom - Ind Mom)
             features["ind_rel_mom_60d"] = features["mom_60d"] - ind_mom_60d
             features["ind_rel_mom_20d"] = features["mom_20d"] - ind_mom_20d
             
             # Industry PE
             ind_pe = cs_group_mean(PE, IND)
             features["ind_pe"] = ind_pe
             # Relative PE (Stock PE / Ind PE)
             features["ind_rel_pe"] = PE / (ind_pe + 1e-8)
        
        # 2. Volatility
        ret_1 = C / ts_delay(C, 1) - 1
        features["volatility_20d"] = ts_std(ret_1, 20)
        
        # Trend Quality (Bull Market Helpers)
        # High R^2 = Smooth Trend. Low R^2 = Choppy.
        features["trend_rsquare_20"] = ts_rsquare(C, 20)
        
        # Linear Slope (Normalized)
        # measures the steepness of the trend
        slope_20 = ts_slope(C, 20)
        features["trend_slope_20"] = slope_20 / (C + 1e-8)
        
        # Modified Sharpe (Slope / Volatility)
        features["trend_sharpe_20"] = features["trend_slope_20"] / (features["volatility_20d"] + 1e-8)
        
        features["volatility_60d"] = ts_std(ret_1, 60)
        features["volatility_120d"] = ts_std(ret_1, 120) # Long term risk
        # features["std_20"] = ts_std(C, 20) / C
        features["atr_ratio_14"] = ta_atr(H, L, C, 14) / C
        # MAX factor (Lottery ticket effect - typically negative alpha in A-share)
        features["max_ret_20d"] = ts_max(ret_1, 20)
        features["min_ret_20d"] = ts_min(ret_1, 20) # Tail risk

        # features["drawdown_20d"] = (C / ts_max(C, 20)) - 1
        features["daily_range"] = H / L - 1
        
        # Downside Volatility (Bear Market Defense)
        # sqrt( sum(min(r, 0)^2) / N )
        neg_ret = torch.clamp(ret_1, max=0)
        features["downside_vol_20d"] = torch.sqrt(ts_mean(neg_ret ** 2, 20))

        # New Positive Factors
        # Inverse Volatility (Low Vol Anomaly)
        # features["inv_std_20"] = 1.0 / (features["std_20"] + 1e-4)
        
        # Trend Efficiency (Net Move / Total Path)
        # High efficiency = strong trend (less noise)
        # net_move_20 = (C - ts_delay(C, 20)).abs()
        # total_path_20 = ts_sum((C - ts_delay(C, 1)).abs(), 20)
        # features["trend_efficiency_20"] = net_move_20 / (total_path_20 + 1e-8)

        # Price-Volume Correlation (20d)
        # Correlation between Close and Volume. 
        # Positive corr: Price up/Vol up or Price down/Vol down (Trend confirmation).
        features["price_vol_corr_20"] = ts_corr(C, V, 20)
        
        # Intraday Strength (Close Location Value)
        # (C - L) / (H - L). Closer to 1 means closing strong (buying pressure).
        # features["close_loc_range"] = (C - L) / (H - L + 1e-8)
        
        # Alpha 13
        # -1 * cs_rank(ts_cov(cs_rank(close), cs_rank(volume), 5))
        #features["alpha013"] = -1 * cs_rank(ts_cov(cs_rank(C), cs_rank(V), 5))

        # Alpha 40
        # ((-1) * cs_rank(ts_std(high, 10))) * ts_corr(high, volume, 10)
        features["alpha040"] = -1 * cs_rank(ts_std(H, 10)) * ts_corr(H, V, 10)

        # Alpha 42
        # cs_rank((vwap - close)) / cs_rank((vwap + close))
        #features["alpha042"] = cs_rank(vwap - C) / (cs_rank(vwap + C) + 1e-8)
        
        # Inverse Volatility (Longer term - 60d)
        # Low beta/volatility stocks tend to outperform in bear/stable markets.
        features["inv_vol_60"] = 1.0 / (features["volatility_60d"] + 1e-4)

        # Return Skewness Proxy (Upside Vol / Downside Vol)
        # If upside vol > downside vol -> Positive Skew potential
        ret_pos = torch.clamp(ret_1, min=0)
        ret_neg_abs = torch.clamp(ret_1, max=0).abs()
        vol_pos = ts_sum(ret_pos**2, 20).sqrt()
        vol_neg = ts_sum(ret_neg_abs**2, 20).sqrt()
        features["vol_skew_20"] = vol_pos / (vol_neg + 1e-8)
        
        # 3. Technical
        ma_20 = ts_mean(C, 20)
        std_20 = ts_std(C, 20)
        features["bollinger_position"] = (C - ma_20) / (std_20 * 2 + 1e-8)
        features["boll_width_20"] = (std_20 * 4) / ma_20
        
        features["rsi_14"] = ta_rsi(C, 14)
        # features["rsi_6"] = ta_rsi(C, 6)

        # PSY (Psychological Line) - Sentiment
        delta = C - ts_delay(C, 1)
        is_up = (delta > 0).float()
        features["psy_12"] = ts_mean(is_up, 12)
        
        # Drawdown from peak (20d)
        features["drawdown_20d"] = C / ts_max(C, 20) - 1
        
        # Rebound from trough (20d)
        features["rebound_20d"] = C / ts_min(L, 20) - 1
        
        # KDJ
        # k, d, j = ts_kdj(C, H, L)
        # features["kdj_k"] = k
        # features["kdj_d"] = d
        # features["kdj_j"] = j
        
        # KDJ Relationships (Interaction Factors)
        # features["kdj_kd_diff"] = k - d
        # features["kdj_j_k_diff"] = j - k
        # features["kdj_j_d_diff"] = j - d
        
        # J Slope
        # features["kdj_j_slope_5"] = ts_slope(j, 5)
        
        # features["kdj_rsv_9"] = (C - ll_9) / (hh_9 - ll_9 + 0.0001)
        
        # PSY: Mean of sign(return) > 0? No, sign of delta.
        # sign(ts_delta(close, 1)) -> 1 if >0, -1 if <0, 0. 
        # PSY is percentage of up days. (sign > 0).
        # We can implement:
        # delta_c = ts_delta(C, 1)
        # is_up = (delta_c > 0).float()
        # features["psy_12"] = ts_mean(is_up, 12)
        
        # MA Alignment
        # ma_5 = ts_mean(C, 5)
        # ma_10 = ts_mean(C, 10)
        # ma_20 defined above
        # ((ma_5 > ma_10) & (ma_10 > ma_20)) * 1
        # features["ma_alignment"] = ((ma_5 > ma_10) & (ma_10 > ma_20)).float()

        # CCI 14 (Commodity Channel Index) - Good for oscillating markets
        # TP = (H + L + C) / 3
        # CCI = (TP - SMA(TP)) / (0.015 * MeanDev(TP))
        tp = (H + L + C) / 3.0
        sma_tp = ts_mean(tp, 14)
        mad_tp = ts_mean(torch.abs(tp - sma_tp), 14)
        features["tech_cci_14"] = (tp - sma_tp) / (0.015 * mad_tp + 1e-8)
        
        # 4. Volume
        features["volume_ratio"] = V / ts_mean(V, 20)
        # features["vol_roc_5"] = V / ts_delay(V, 5) - 1
        features["vol_cv_20"] = ts_std(V, 20) / ts_mean(V, 20)
        features["vol_stability_20"] = 1.0 / (features["vol_cv_20"] + 1e-4)

        # Coefficient of Variation of Turnover (Instability)
        features["turnover_cv_20d"] = ts_std(TR, 20) / (ts_mean(TR, 20) + 1e-8)
        
        # Amihud Illiquidity (Price Impact)
        # |Ret| / (Price * Volume) => |Ret| / Turnover
        # High Illiquidity -> Low Volume for big move.
        abs_ret = torch.abs(ret_1)
        # Add epsilon to turnover to avoid div by zero
        illiq = abs_ret / (T + 1e-1) * 1e8 # Scale up
        features["illiquidity_20d"] = ts_mean(illiq, 20)

        # Price Volume Divergence
        # (close > prev_close) & (volume < prev_volume)
        # c_prev = ts_delay(C, 1)
        # v_prev = ts_delay(V, 1)
        # price_up = C > c_prev
        # vol_down = V < v_prev
        # features["price_volume_divergence"] = (price_up & vol_down).float()
        
        # 5. Money Flow
        # (((close - open) / (high - low + 0.0001)) * volume)
        # mf_val = ((C - O) / (H - L + 0.0001)) * V
        # features["money_flow_20d"] = mf_val / ts_mean(mf_val, 20)
        
        # VWAP Dev
        # ts_sum(close * volume, 20) / ts_sum(volume, 20)
        vwap_20 = ts_sum(C * V, 20) / ts_sum(V, 20)
        features["vwap_dev_20"] = C / vwap_20 - 1
        
        # 6. Fundamental / Daily Basic
        # Turnover Rate
        features["turnover_mean_5d"] = ts_mean(TR, 5)
        features["turnover_mean_20d"] = ts_mean(TR, 20)
        features["turnover_std_20d"] = ts_std(TR, 20)

        # Turnover Growth (Activity Change)
        # TR / delay(TR, 20) - 1
        features["fund_turnover_growth"] = TR / (ts_delay(TR, 20) + 1e-8) - 1
        
        # PE / Valuation
        # EP Ratio (Earnings Yield) = 1 / PE
        # Handle division by zero or near zero if PE is 0.
        features["ep_ratio"] = 1.0 / (PE + 1e-4)
        
        # Value Factors (PB, PS, Dividend) - Defensive
        features["val_pb"] = 1.0 / (PB + 1e-4)
        features["val_ps"] = 1.0 / (PS + 1e-4)
        features["val_dv"] = DV # Dividend Yield
        
        # Size Factor (Log Market Cap)
        # Use log to normalize the distribution
        features["size_ln_cap"] = torch.log(MV + 1.0)
        
        # PE Z-Score (Time-series)
        # (PE - Mean_PE) / Std_PE
        pe_mean_60 = ts_mean(PE, 60)
        pe_std_60 = ts_std(PE, 60)
        features["pe_zscore_60d"] = (PE - pe_mean_60) / (pe_std_60 + 1e-8)
        
        # PE Rank Change (Relative Valuation)
        # Current PE / Avg PE(20d) - 1
        pe_mean_20 = ts_mean(PE, 20)
        features["pe_rank_change_20d"] = PE / (pe_mean_20 + 1e-8) - 1

        
        # qtld_{w} = ts_quantile(close, w, 0.2) / close
        features[f"qtld_60"] = ts_quantile(C, 60, 0.2) / C
        
        # klen = (high - low) / close
        features["klen"] = (H - L) / C

        for w in [10, 20, 30]:
            # min_{w} = ts_min(low, w) / close
            features[f"min_{w}"] = ts_min(L, w) / C
        
        for w in [5, 10, 20]:
            # std_{w} = ts_std(ret_1, w)
            features[f"std_{w}"] = ts_std(ret_1, w)
        
        # Label: Next 5 days return (Market Neutral Rank)
        # Using cs_rank on the future return ensures we are learning to rank stocks,
        # which is regime-independent (works in both Bull and Bear markets).
        raw_ret_5 = ts_delay(C, -5) / C - 1
        features["label"] = cs_rank(raw_ret_5)

        return features