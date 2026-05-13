from core.alpha.factor_calculator import FactorCalculator, device, torch, np, pl, cs_rank, ts_corr, cs_zscore, ts_delay, ts_mean, ts_min, ts_max, ts_quantile, ts_std, ts_sum, ts_rsquare, ts_slope, ta_atr, ta_rsi, cs_group_mean, ts_kdj, ts_cov

class V8FactorCalculator(FactorCalculator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # col_map结构可参考 core/alpha/data_columns_info.txt
    def build_features(self, padded_raw, col_map) -> dict[str, torch.Tensor]:
        # Let's keep (Batch, Time) for basic ops
        print(f"[DEBUG] padded_raw.shape: {padded_raw.shape}")
        
        O = padded_raw[:, :, col_map['open']]
        H = padded_raw[:, :, col_map['high']]
        L = padded_raw[:, :, col_map['low']]
        C = padded_raw[:, :, col_map['close']]
        V = padded_raw[:, :, col_map['volume']]
        T = padded_raw[:, :, col_map['turnover']] # Turnover (Amount)
        TR = padded_raw[:, :, col_map['turnover_rate']] # Turnover Rate
        PE = padded_raw[:, :, col_map['pe']] # PE Ratio
        PB = padded_raw[:, :, col_map['pb']] # PB Ratio
        PS = padded_raw[:, :, col_map['ps']] # PS Ratio
        DV = padded_raw[:, :, col_map['dv_ratio']] # Dividend Ratio
        MV = padded_raw[:, :, col_map['total_mv']] # Total Market Value
        
        # Industry Code
        IND = None
        if 'industry' in col_map and col_map['industry'] < padded_raw.shape[2]:
            IND = padded_raw[:, :, col_map['industry']]
            print(f"[DEBUG] IND extracted. Shape: {IND.shape}")
            # Debug IND values
            mask_ind = ~torch.isnan(IND)
            if mask_ind.any():
                print(f"[DEBUG] IND stats - Min: {torch.min(IND[mask_ind])}, Max: {torch.max(IND[mask_ind])}")
                print(f"[DEBUG] IND NaNs: {(~mask_ind).sum()}, Infs: {torch.isinf(IND).sum()}")
            else:
                 print("[DEBUG] IND all NaNs!")
        
        # 保留V6因子
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
        
        # --- Market Style Factors (New) ---
        # 1. Market Return (Cross-Sectional Mean of Returns)
        # (Time,) -> (Batch, Time)
        ret_1_clean = torch.nan_to_num(ret_1, nan=0.0)
        ret_1_mask = ~torch.isnan(ret_1)
        valid_cnt = ret_1_mask.sum(dim=0)
        # Avoid division by zero
        mkt_ret_1d = ret_1_clean.sum(dim=0) / (valid_cnt + 1e-8)
        mkt_ret_broad = mkt_ret_1d.unsqueeze(0).expand_as(ret_1)
        
        # 2. Beta (Sensitivity to Market)
        # Beta = Cov(R_i, R_m) / Var(R_m)
        cov_im = ts_cov(ret_1, mkt_ret_broad, 20)
        var_m = ts_std(mkt_ret_broad, 20) ** 2
        features["beta_20d"] = cov_im / (var_m + 1e-8)
        
        # 3. Residual Volatility (Idiosyncratic Risk)
        # epsilon = R_i - (alpha + beta * R_m)
        # alpha = E[R_i] - beta * E[R_m]
        # We can calculate residual directly from realized values?
        # Standard approach: Resid = ret - beta * mkt_ret (assuming alpha is small or using rolling alpha)
        # Let's use rolling alpha for correctness.
        mean_ret = ts_mean(ret_1, 20)
        mean_mkt = ts_mean(mkt_ret_broad, 20)
        alpha = mean_ret - features["beta_20d"] * mean_mkt
        exp_ret = alpha + features["beta_20d"] * mkt_ret_broad
        resid = ret_1 - exp_ret
        features["resid_vol_20d"] = ts_std(resid, 20)
        
        # 4. Non-Linear Size (Size Cube)
        # Commonly used in Barra models (NLSIZE)
        # Here we just use the cube of log cap to capture tails
        #features["size_nl_cap"] = torch.pow(torch.log(MV + 1.0), 3)

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
        net_move_20 = (C - ts_delay(C, 20)).abs()
        total_path_20 = ts_sum((C - ts_delay(C, 1)).abs(), 20)
        #features["trend_efficiency_20"] = net_move_20 / (total_path_20 + 1e-8)

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
        inv_vol_60 = 1.0 / (features["volatility_60d"] + 1e-4)
        features["inv_vol_60"] = inv_vol_60


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
        k, d, j = ts_kdj(C, H, L)
        # Normalize KDJ to 0-1 range for better NN stability
        kdj_k = k / 100.0
        kdj_d = d / 100.0
        kdj_j = j / 100.0
        
        # KDJ Auxiliary Trends (Predicting Future Crosses)
        # Distance between K and D. Near 0 = Potential Cross.
        features["kdj_kd_diff"] = kdj_k - kdj_d
        
        # Velocity of convergence (Change in KD diff)
        # If Diff is negative (K below D) and Velocity is positive, it means K is approaching D (Pre-Golden Cross).
        # Smooth velocity over 3 days to reduce noise
        raw_velocity = features["kdj_kd_diff"] - ts_delay(features["kdj_kd_diff"], 1)
        features["kdj_kd_velocity"] = ts_mean(raw_velocity, 3)
        
        
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
        
        # --- Consolidation / Plateau Detectors ---
        # Volatility Ratio: Short-term vol / Long-term vol. 
        # Low ratio (<1) indicates volatility is compressing (Consolidation/Plateau).
        features["vol_ratio_5_20"] = features["std_5"] / (features["std_20"] + 1e-8)
        
        # Turnover Ratio: Activity change.
        features["turnover_ratio_5_20"] = features["turnover_mean_5d"] / (features["turnover_mean_20d"] + 1e-8)
        
        # Short term trend slope (5d)
        slope_5 = ts_slope(C, 5)
        features["trend_slope_5"] = slope_5 / (C + 1e-8)
        
        # Slope Divergence: Short term slope - Long term slope
        # If Long term is + (Up) and Short term is 0 (Flat) -> Negative divergence (Plateauing)
        features["slope_div_5_20"] = features["trend_slope_5"] - features["trend_slope_20"]
        
        # --- V5 Optimization: Relative Strength & Interaction Factors ---
        # 1. Relative Turnover (Market Adjusted)
        mkt_turnover_20d = torch.nanmean(features["turnover_mean_20d"], dim=0, keepdim=True)
        features["rel_turnover_20d"] = features["turnover_mean_20d"] / (mkt_turnover_20d + 1e-8)
        
        # 2. Market Momentum & Volatility (Base Environment Vars)
        mkt_mom_20d = torch.nanmean(features["mom_20d"], dim=0, keepdim=True)
        mkt_mom_60d = torch.nanmean(features["mom_60d"], dim=0, keepdim=True)
        mkt_vol_20d = torch.nanmean(features["volatility_20d"], dim=0, keepdim=True)
        
        # === V6: Improved Regime Detection (Faster Response) ===
        # Optimized: Use blended momentum (20d only for speed) AND Market Breadth (Bias > 0).
        # Removed mkt_mom_60d to avoid lag in bull market detection.
        mkt_breadth = torch.nanmean(features["bias_20"], dim=0, keepdim=True)
        bull_prob = torch.sigmoid(((mkt_mom_20d + mkt_breadth) / 2.0) * 15.0)

        # 1. Momentum x Market (Bull Feature)
        features["mom_x_mkt"] = features["mom_20d"] * bull_prob

        # 3. Technical Reversal (Bear Feature)
        # Deep Value: Price far below MA60 (Bias 60)
        ma_60 = ts_mean(C, 60)
        deep_value = (C - ma_60) / (ma_60 + 1e-8)
        
        # Base Reversal Score: Low RSI + Deep Value
        features["tech_reversal"] = cs_rank(features["rsi_14"] * -1) + cs_rank(deep_value * -1)
        
        # Bear Reversal Interaction
        # Active mainly in Bear Markets (1 - bull_prob).
        features["bear_reversal"] = features["tech_reversal"] * (1.0 - bull_prob)

        # === Direction 2: Industry Factors (Balanced) ===
        if IND is not None:
            # 1. Industry Relative Turnover (Activity vs Peers)
            ind_turnover_20d = cs_group_mean(features["turnover_mean_20d"], IND)
            features["ind_rel_turnover_20d"] = features["turnover_mean_20d"] / (ind_turnover_20d + 1e-8)
            
            # 2. Industry Relative Volatility (Risk vs Peers)
            ind_vol_20d = cs_group_mean(features["volatility_20d"], IND)
            features["ind_rel_vol_20d"] = features["volatility_20d"] / (ind_vol_20d + 1e-8)
            
            # 3. Relative Bias 
            ind_bias_20 = cs_group_mean(features["bias_20"], IND)
            features["ind_rel_bias_20"] = features["bias_20"] - ind_bias_20

        # === V6: Fusion Dragon Logic (Attack + Defense) ===
        # Revert to V4's Tanh interaction (Turnover * Tanh(Mom)) to filter toxic turnover.
        # But blend Mom_20d (70%) and Mom_60d (30%) for stability.
        combined_mom = cs_rank(features["mom_20d"]) * 0.7 + cs_rank(features["mom_60d"]) * 0.3
        
        features["dragon_score"] = combined_mom + cs_rank(features["turnover_mean_20d"]) * torch.tanh(features["mom_20d"] * 5.0)
        
        # Low Volatility Anomaly (still useful for filtering garbage)
        features["inv_vol_20"] = 1.0 / (features["volatility_20d"] + 1e-4)

        # === V6: Bear Market Defense (Vol Penalty) ===
        # In Bear Markets, punish high volatility.
        vol_rank = cs_rank(features["volatility_20d"])
        features["vol_penalty"] = vol_rank * (1.0 - bull_prob) * -0.5
        
        
        # Now add V7 Concept Factors
        con_mom_5 = padded_raw[:, :, col_map['concept_mom_5d']]
        con_mom_10 = padded_raw[:, :, col_map['concept_mom_10d']]
        con_mom_20 = padded_raw[:, :, col_map['concept_mom_20d']]
        con_mom_20_max = padded_raw[:, :, col_map['concept_mom_20d_max']]
        con_mom_20_min = padded_raw[:, :, col_map['concept_mom_20d_min']]
        con_mom_20_std = padded_raw[:, :, col_map['concept_mom_20d_std']]
        con_turnover_20 = padded_raw[:, :, col_map['concept_turnover_20d']]
        con_vol_20 = padded_raw[:, :, col_map['concept_vol_20d']]
        con_count = padded_raw[:, :, col_map['concept_count']]
        # con_daily_ret = padded_raw[:, :, col_map['concept_daily_ret']]
        con_hot_ratio = padded_raw[:, :, col_map['concept_hot_ratio']]
        # con_top3_mean = padded_raw[:, :, col_map['concept_top3_mean']]
        # con_cohesion = padded_raw[:, :, col_map['concept_cohesion']]
        con_acc_5 = padded_raw[:, :, col_map['concept_acc_5_mean']]
        con_rank_score = padded_raw[:, :, col_map['concept_rank_score_mean']]
        
        # Add to features
        features["con_mom_5d"] = con_mom_5
        features["con_mom_20d"] = con_mom_20
        features["con_mom_20d_max"] = con_mom_20_max
        features["con_turnover_20d"] = con_turnover_20
        # features["con_acc_5_mean"] = con_acc_5
        # features["con_rank_score_mean"] = con_rank_score
        
        # === Concept Relative Strength (Alpha vs Concept) ===
        # 1. Relative Momentum (Mean): Is the stock stronger than its average concept?
        features["rel_con_mom_20d"] = features["mom_20d"] - con_mom_20
        
        # 2. Leader Distance (Max): How far is the stock from its strongest concept's performance?
        # If close to 0 (or positive), it's likely a leader/beneficiary of the hot theme.
        # If very negative, it's lagging its best story.
        features["rel_con_mom_max_20d"] = features["mom_20d"] - con_mom_20_max
        
        # 3. Concept Divergence
        # High std means the stock belongs to some hot and some cold concepts. 
        # Market might be confused about which logic to trade.
        features["con_divergence_20d"] = con_mom_20_std

        # === V8.8: Reconstructed Rebound Strategy (User Request: "Fast Rise Fast Fall" + "Support" + "Head-Lift") ===
        
        # 1. Characteristic: "Fast Rise Fast Fall" (Elasticity)
        # Stocks that have history of big moves (High Vol Range) but are currently oversold.
        # Vol Range 20d: (Max - Min) / Min
        # Re-calculate Price Base/Peak for 20d first as they are needed
        price_base_20 = ts_min(L, 20)
        price_peak_20 = ts_max(H, 20)
        price_base_60 = ts_min(L, 60)
        price_peak_60 = ts_max(H, 60)
        
        features["vol_range_20d"] = (price_peak_20 - price_base_20) / (price_base_20 + 1e-8)
        elasticity_rank = cs_rank(features["vol_range_20d"])
        
        # 2. Status: "Deep Oversold" (Fast Fall)
        # Bias 10d: Distance from 10-day MA. Negative = Oversold.
        # RSI 14: < 30 = Oversold.
        # We want NEGATIVE bias and LOW RSI.
        oversold_score = cs_rank(features["bias_10"] * -1) + cs_rank(features["rsi_14"] * -1)
        
        # 3. Location: "Near Support"
        # Explicit Support/Resistance Distance
        features["dist_support_20"] = (C - price_base_20) / (price_base_20 + 1e-8)
        features["dist_support_60"] = (C - price_base_60) / (price_base_60 + 1e-8)
        
        features["dist_pressure_20"] = (price_peak_20 - C) / (C + 1e-8)
        features["dist_pressure_60"] = (price_peak_60 - C) / (C + 1e-8)
        
        features["rr_ratio_20"] = features["dist_pressure_20"] / (features["dist_support_20"] + 1e-4)
        features["rr_ratio_60"] = features["dist_pressure_60"] / (features["dist_support_60"] + 1e-4)
        
        # We want dist_support to be SMALL (Near 0).
        support_score = 1.0 - torch.clamp(features["dist_support_20"] / 0.15, 0, 1) # Linear decay from 0% to 15% distance
        
        # 4. Trigger: "Head-Lifting Signal" (Momentum Ignition)
        # A. Price Action: Close > MA5 AND Price Crossed MA5 Upwards (or just solid red bar from low)
        ma_5 = ts_mean(C, 5)
        ma_5_prev = ts_delay(ma_5, 1)
        c_prev = ts_delay(C, 1)
        
        # Condition 1: Cross Up MA5 (Classic Reversal)
        cross_up_ma5 = (C > ma_5) & (c_prev < ma_5_prev)
        
        # Condition 2: Big Yang Candle (Strong buying today)
        is_big_red = (C / O - 1) > 0.025
        
        # Condition 3: Reversal from Low (Intraday or Day-to-Day)
        # Close is significantly higher than recent low
        reversal_strength = (C - ts_min(L, 5)) / (ts_min(L, 5) + 1e-8)
        
        # Condition 4: Volume Pickup (Ignition)
        vol_ma_5 = ts_mean(V, 5)
        vol_ignition = V > vol_ma_5 * 1.2
        
        # Composite Head-Lift Signal
        # Strongest: Cross MA5 + Big Candle + Volume
        head_lift_signal = (cross_up_ma5.float() * 0.4 + is_big_red.float() * 0.4) * vol_ignition.float()
        features["head_lift_signal"] = head_lift_signal
        
        # === Redefine Camel Hump: Now "Elastic Oversold Support" ===
        # It's no longer just a "hump", it's a "Loaded Spring" (Compressed at support).
        features["camel_hump_score"] = (
            elasticity_rank * 0.3 +       # Must be active stock
            oversold_score * 0.3 +        # Must be beaten down
            support_score * 0.4           # Must be at support
        )
        
        # === V8.2: Resonance Trigger (Group Ignition) ===
        # Logic: A stock at the bottom (Camel) is only a buy if its SECTOR/THEME is moving.
        
        # 1. Concept Momentum (5d) - Theme Trigger
        rank_con_mom = cs_rank(features["con_mom_5d"])
        
        # 2. Industry Momentum (5d) - Sector Trigger
        if IND is not None:
             rank_ind_mom = cs_rank(features["ind_mom_5d"])
        else:
             rank_ind_mom = rank_con_mom # Fallback if no industry data
             
        # Resonance Score: Blend of Concept and Industry Momentum
        features["resonance_signal"] = (rank_con_mom + rank_ind_mom) / 2.0
        
        # === V8.3: Meta-Features for MLP Learning ===
        # Explicitly export Regime Indicators so the MLP can learn "When to use what".
        features["meta_bull_prob"] = bull_prob.expand_as(C)
        
        # Explicit Interactions
        # 1. Bull Attack: Resonance works best when Market is Bullish.
        features["inter_res_bull"] = features["resonance_signal"] * bull_prob
        
        # 2. Bear Defense: Camel/Reversion works best when Market is Bearish/Oscillating.
        bear_prob = 1.0 - bull_prob
        features["inter_camel_bear"] = features["camel_hump_score"] * bear_prob

        # === Helper: ZT Count (Needed for Dragon Score) ===
        is_zt = (ret_1 > 0.095).float()
        features["zt_count_20d"] = ts_sum(is_zt, 20)

        # === V9: Adaptive Dragon Score (Hybrid Logic) ===
        # Rationale: Dynamically shift focus based on Market Regime.
        # Bull Market: Chase Leaders & Trend (Attack)
        # Bear Market: Buy Structure & Reversion (Defense)
        
        rank_turnover = cs_rank(features["turnover_mean_20d"])
        rank_pv_corr = cs_rank(features["price_vol_corr_20"])
        rank_mom_20d = cs_rank(features["mom_20d"])
        rank_rel_con = cs_rank(features["rel_con_mom_20d"])
        
        rank_camel = cs_rank(features["camel_hump_score"])
        rank_resonance = cs_rank(features["resonance_signal"])
        rank_rr = cs_rank(features["rr_ratio_20"])
        
        # New Ranks
        rank_zt = cs_rank(features["zt_count_20d"])
        # rank_low_price = cs_rank(features["low_price_factor"])
        rank_elasticity = cs_rank(features["vol_range_20d"])
        
        # === V9.3: Triple-Regime Adaptive Logic (Bull, Bear, Chaos) ===
        # Backtest Analysis: In 2023-2024 "Choppy Bear", both Momentum and Reversion failed.
        # Only Turnover (Liquidity) worked. We need to detect "Chaos" and hide in Liquidity.
        
        # 1. Define Regimes
        # Bull: Momentum is Up.
        # Bear: Momentum is Down.
        # Chaos: Volatility is High (Panic/Choppy).
        
        # Volatility Rank (Market Level)
        # If Market Volatility is in the top 80%, we are in Chaos Mode.
        # Normalized Volatility: 0.0 = Low, 1.0 = High
        vol_regime = torch.sigmoid((mkt_vol_20d - 0.015) * 100.0)
        
        # Regime Probability Matrix
        # P(Bull) = bull_prob
        # P(Bear) = (1 - bull_prob) * (1 - vol_regime)  -> Grinding Down (Safe for Reversion)
        # P(Chaos) = (1 - bull_prob) * vol_regime       -> Crashing/Choppy (Unsafe for Reversion)
        
        prob_bull = bull_prob
        prob_bear = (1.0 - bull_prob) * (1.0 - vol_regime)
        prob_chaos = (1.0 - bull_prob) * vol_regime
        
        # 2. Strategy Compositions
        
        # A. Bull Strategy: Momentum + Resonance (Attack)
        score_bull = (
            rank_mom_20d * 0.2 + 
            rank_rel_con * 0.2 + 
            rank_resonance * 0.2 +
            features["head_lift_signal"] * 0.2 + 
            rank_zt * 0.1 +              
            rank_elasticity * 0.1
        )
        
        # B. Bear Strategy: Reversion + Structure (Defense)
        # Works well in low-volatility downtrends (Grinding bear)
        score_bear = (
            rank_camel * 0.4 +       
            rank_rel_con * 0.2 +     
            rank_rr * 0.1 +          
            rank_resonance * 0.3     
        )
        
        # C. Chaos Strategy: Liquidity + Low Volatility (Survival)
        # In High Vol crashes, Alpha fails. Stick to high liquidity.
        score_chaos = rank_turnover # Pure Liquidity
        
        # 3. Dynamic Blending
        # Base Score is always anchored by Turnover (The only all-weather factor)
        # We blend the "Alpha Component" based on regime.
        
        alpha_component = (
            score_bull * prob_bull + 
            score_bear * prob_bear + 
            score_chaos * prob_chaos
        )
        
        # Final Dragon Score
        # In Chaos, we boost the weight of the Base Turnover significantly.
        # Normal: 50% Turnover + 50% Alpha
        # Chaos:  80% Turnover + 20% Alpha (Survival Mode)
        
        turnover_weight = 0.5 + prob_chaos * 0.3 # 0.5 -> 0.8
        alpha_weight = 1.0 - turnover_weight
        
        features["dragon_score"] = rank_turnover * turnover_weight + alpha_component * alpha_weight
        
        # Add Inflection Bonus (Aggressive booster for early entry)
        # Only apply in Bull or Bear regimes, NOT in Chaos (Too risky)
        inflection_bonus = (ret_1 > 0.0).float() * (features["mom_5d"] < -0.03).float() * (features["dist_support_20"] < 0.08).float()
        features["dragon_score"] = features["dragon_score"] + inflection_bonus * 0.3 * (1.0 - prob_chaos)
        
        # Penalty 1: Short term overheat (Velocity too fast)
        # Relax penalty if ZT count is high (Dragons are allowed to be overheated)
        overheat_threshold = 0.2 + bull_prob * 0.2
        is_dragon_mode = (features["zt_count_20d"] > 0).float()
        
        # Only penalize if NOT in Dragon Mode (Normal overheat is bad, Dragon overheat is good)
        features["dragon_score"] = features["dragon_score"] - (features["mom_5d"] > overheat_threshold).float() * (1.0 - is_dragon_mode) * 1.0

        # Penalty 2: "At Ceiling" Penalty
        # In Bear Markets, hitting resistance is a sell signal.
        at_ceiling = (features["dist_pressure_20"] < 0.01).float()
        features["dragon_score"] = features["dragon_score"] - at_ceiling * bear_prob * 0.5
        
        # Penalty 3: Bear Trap / False Stabilization (Downtrend Trap)
        # Identifies "Falling Knife Pause": Downtrend + Consolidation + Not at Bottom.
        features["bear_trap_score"] = (features["mom_20d"] * -1).clamp(min=0) * (1.0 - features["vol_ratio_5_20"]).clamp(min=0) * features["dist_support_20"] * 2.0
        
        # FIX: If Head Lift is present, this is NOT a bear trap, it's a reversal.
        features["bear_trap_score"] = features["bear_trap_score"] * (1.0 - features["head_lift_signal"]).clamp(min=0)

        # Penalty 4: Pullback Trap (Uptrend Trap)
        # Identifies "Incomplete Pullback": Uptrend (Mom20>0) + Pullback (Mom5<0) + Not at Support.
        # User request: "If it's a pullback to halfway I don't want to select it."
        # We penalize if dist_support > 0.10 (still 10% room to fall to support).
        pullback_trap = (features["mom_20d"] > 0).float() * (features["mom_5d"] < -0.02).float() * (features["dist_support_20"] > 0.10).float()
        
        # === V8.6: ST and Dead Stock Filter (User Request) ===
        # Filter 1: ST Stock Proxy
        # ST stocks usually have 5% limit. If max daily range over 20 days is < 5.5%, it's likely ST or very low elasticity.
        max_range_20 = ts_max(features["daily_range"], 20)
        is_likely_st = (max_range_20 < 0.055).float()
        
        # Filter 2: Dead/Flat Phase ("One-character" distribution)
        # If daily range is < 1.2%, it's too flat for a volatility-based strategy.
        # This prevents buying during "suspiciously stable" periods in a downtrend.
        # FIX: If we are AT SUPPORT, low volatility is good (Base Building).
        is_dead_flat = (features["daily_range"] < 0.012).float() * (features["dist_support_20"] > 0.05).float()
        
        # Penalize Dragon Score
        # Reduced bear_trap penalty weight from 0.5 to 0.3
        features["dragon_score"] = features["dragon_score"] - cs_rank(features["bear_trap_score"]) * 0.3 - pullback_trap * 0.5
        
        # Apply ST/Flat Penalties (Severe)
        features["dragon_score"] = features["dragon_score"] - is_likely_st * 2.0 - is_dead_flat * 2.0

        # Label: Next 5 days return (Market Neutral Rank)
        raw_ret_5 = ts_delay(C, -5) / C - 1
        
        # Penalize low liquidity stocks (Turnover < 1%)
        low_liq_penalty = (features["turnover_mean_20d"] < 1.0).float() * 0.05
        raw_ret_5 = raw_ret_5 - low_liq_penalty

        features["label"] = cs_rank(raw_ret_5)

        return features
