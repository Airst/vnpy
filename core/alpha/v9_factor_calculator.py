from core.alpha.factor_calculator import FactorCalculator, device, torch, np, pl, cs_rank, ts_corr, cs_zscore, ts_delay, ts_mean, ts_min, ts_max, ts_quantile, ts_std, ts_sum, ts_rsquare, ts_slope, ta_atr, ta_rsi, cs_group_mean, ts_kdj, ts_cov

class V9FactorCalculator(FactorCalculator):
    """
    V9 Baseline Factor Calculator
    Base: V8 complete factor set (~100 factors)
    Changes from V8:
      - Phase 1: Beta-neutral label (excess return ranking)
      - Phase 3: Simplified dragon_score (remove triple-regime hardcoding)
      - Phase 4: Add turnover_x_bull interaction factor
    """
    def __init__(self):
        super().__init__()

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
        
        # Helper vars
        mask = ~torch.isnan(C)
        vwap = T / (V + 1e-8)
        vwap = torch.where(torch.isnan(vwap), C, vwap) 

        features = {}

        # 1. Momentum / Reversal
        features["rev_5d"] = (C / ts_delay(C, 5) - 1) * -1
        features["mom_5d"] = C / ts_delay(C, 5) - 1
        features["mom_20d"] = C / ts_delay(C, 20) - 1
        features["mom_60d"] = C / ts_delay(C, 60) - 1
        features["mom_120d"] = C / ts_delay(C, 120) - 1
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
             ind_mom_60d = cs_group_mean(features["mom_60d"], IND)
             ind_mom_20d = cs_group_mean(features["mom_20d"], IND)
             ind_mom_5d = cs_group_mean(features["mom_5d"], IND)
             
             features["ind_mom_60d"] = ind_mom_60d
             features["ind_mom_20d"] = ind_mom_20d
             features["ind_mom_5d"] = ind_mom_5d
             
             features["ind_rel_mom_60d"] = features["mom_60d"] - ind_mom_60d
             features["ind_rel_mom_20d"] = features["mom_20d"] - ind_mom_20d
             
             ind_pe = cs_group_mean(PE, IND)
             features["ind_pe"] = ind_pe
             features["ind_rel_pe"] = PE / (ind_pe + 1e-8)
        
        # 2. Volatility
        ret_1 = C / ts_delay(C, 1) - 1
        features["volatility_20d"] = ts_std(ret_1, 20)
        
        # Market Return (Cross-Sectional Mean of Returns)
        ret_1_clean = torch.nan_to_num(ret_1, nan=0.0)
        ret_1_mask = ~torch.isnan(ret_1)
        valid_cnt = ret_1_mask.sum(dim=0)
        mkt_ret_1d = ret_1_clean.sum(dim=0) / (valid_cnt + 1e-8)
        mkt_ret_broad = mkt_ret_1d.unsqueeze(0).expand_as(ret_1)
        
        # Beta (Sensitivity to Market)
        cov_im = ts_cov(ret_1, mkt_ret_broad, 20)
        var_m = ts_std(mkt_ret_broad, 20) ** 2
        features["beta_20d"] = cov_im / (var_m + 1e-8)
        
        # Residual Volatility (Idiosyncratic Risk)
        mean_ret = ts_mean(ret_1, 20)
        mean_mkt = ts_mean(mkt_ret_broad, 20)
        alpha = mean_ret - features["beta_20d"] * mean_mkt
        exp_ret = alpha + features["beta_20d"] * mkt_ret_broad
        resid = ret_1 - exp_ret
        features["resid_vol_20d"] = ts_std(resid, 20)

        # Trend Quality
        features["trend_rsquare_20"] = ts_rsquare(C, 20)
        slope_20 = ts_slope(C, 20)
        features["trend_slope_20"] = slope_20 / (C + 1e-8)
        features["trend_sharpe_20"] = features["trend_slope_20"] / (features["volatility_20d"] + 1e-8)
        
        features["volatility_60d"] = ts_std(ret_1, 60)
        features["volatility_120d"] = ts_std(ret_1, 120)
        features["atr_ratio_14"] = ta_atr(H, L, C, 14) / C
        features["max_ret_20d"] = ts_max(ret_1, 20)
        features["min_ret_20d"] = ts_min(ret_1, 20)
        features["daily_range"] = H / L - 1
        
        # Downside Volatility
        neg_ret = torch.clamp(ret_1, max=0)
        features["downside_vol_20d"] = torch.sqrt(ts_mean(neg_ret ** 2, 20))

        # Trend Efficiency (intermediate, not exported)
        net_move_20 = (C - ts_delay(C, 20)).abs()
        total_path_20 = ts_sum((C - ts_delay(C, 1)).abs(), 20)

        # Price-Volume Correlation (20d)
        features["price_vol_corr_20"] = ts_corr(C, V, 20)

        # Alpha 40
        features["alpha040"] = -1 * cs_rank(ts_std(H, 10)) * ts_corr(H, V, 10)
        
        # Inverse Volatility (60d)
        features["inv_vol_60"] = 1.0 / (features["volatility_60d"] + 1e-4)

        # Return Skewness Proxy
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

        # PSY (Psychological Line)
        delta = C - ts_delay(C, 1)
        is_up = (delta > 0).float()
        features["psy_12"] = ts_mean(is_up, 12)
        
        features["drawdown_20d"] = C / ts_max(C, 20) - 1
        features["rebound_20d"] = C / ts_min(L, 20) - 1
        
        # KDJ
        k, d, j = ts_kdj(C, H, L)
        kdj_k = k / 100.0
        kdj_d = d / 100.0
        kdj_j = j / 100.0
        features["kdj_kd_diff"] = kdj_k - kdj_d
        raw_velocity = features["kdj_kd_diff"] - ts_delay(features["kdj_kd_diff"], 1)
        features["kdj_kd_velocity"] = ts_mean(raw_velocity, 3)

        # CCI 14
        tp = (H + L + C) / 3.0
        sma_tp = ts_mean(tp, 14)
        mad_tp = ts_mean(torch.abs(tp - sma_tp), 14)
        features["tech_cci_14"] = (tp - sma_tp) / (0.015 * mad_tp + 1e-8)
        
        # 4. Volume
        features["volume_ratio"] = V / ts_mean(V, 20)
        features["vol_cv_20"] = ts_std(V, 20) / ts_mean(V, 20)
        features["vol_stability_20"] = 1.0 / (features["vol_cv_20"] + 1e-4)
        features["turnover_cv_20d"] = ts_std(TR, 20) / (ts_mean(TR, 20) + 1e-8)
        
        # Amihud Illiquidity
        abs_ret = torch.abs(ret_1)
        illiq = abs_ret / (T + 1e-1) * 1e8
        features["illiquidity_20d"] = ts_mean(illiq, 20)
        
        # VWAP Dev
        vwap_20 = ts_sum(C * V, 20) / ts_sum(V, 20)
        features["vwap_dev_20"] = C / vwap_20 - 1
        
        # 6. Fundamental / Daily Basic
        features["turnover_mean_5d"] = ts_mean(TR, 5)
        features["turnover_mean_20d"] = ts_mean(TR, 20)
        features["turnover_std_20d"] = ts_std(TR, 20)
        features["fund_turnover_growth"] = TR / (ts_delay(TR, 20) + 1e-8) - 1
        features["ep_ratio"] = 1.0 / (PE + 1e-4)
        features["val_pb"] = 1.0 / (PB + 1e-4)
        features["val_ps"] = 1.0 / (PS + 1e-4)
        features["val_dv"] = DV
        features["size_ln_cap"] = torch.log(MV + 1.0)
        
        pe_mean_60 = ts_mean(PE, 60)
        pe_std_60 = ts_std(PE, 60)
        features["pe_zscore_60d"] = (PE - pe_mean_60) / (pe_std_60 + 1e-8)
        pe_mean_20 = ts_mean(PE, 20)
        features["pe_rank_change_20d"] = PE / (pe_mean_20 + 1e-8) - 1

        features["qtld_60"] = ts_quantile(C, 60, 0.2) / C
        features["klen"] = (H - L) / C

        for w in [10, 20, 30]:
            features[f"min_{w}"] = ts_min(L, w) / C
        
        for w in [5, 10, 20]:
            features[f"std_{w}"] = ts_std(ret_1, w)
        
        # Consolidation / Plateau Detectors
        features["vol_ratio_5_20"] = features["std_5"] / (features["std_20"] + 1e-8)
        features["turnover_ratio_5_20"] = features["turnover_mean_5d"] / (features["turnover_mean_20d"] + 1e-8)
        
        slope_5 = ts_slope(C, 5)
        features["trend_slope_5"] = slope_5 / (C + 1e-8)
        features["slope_div_5_20"] = features["trend_slope_5"] - features["trend_slope_20"]
        
        # Relative Strength & Interaction Factors
        mkt_turnover_20d = torch.nanmean(features["turnover_mean_20d"], dim=0, keepdim=True)
        features["rel_turnover_20d"] = features["turnover_mean_20d"] / (mkt_turnover_20d + 1e-8)
        
        mkt_mom_20d = torch.nanmean(features["mom_20d"], dim=0, keepdim=True)
        mkt_mom_60d = torch.nanmean(features["mom_60d"], dim=0, keepdim=True)
        mkt_vol_20d = torch.nanmean(features["volatility_20d"], dim=0, keepdim=True)
        
        # V6: Regime Detection
        mkt_breadth = torch.nanmean(features["bias_20"], dim=0, keepdim=True)
        bull_prob = torch.sigmoid(((mkt_mom_20d + mkt_breadth) / 2.0) * 15.0)

        features["mom_x_mkt"] = features["mom_20d"] * bull_prob

        # Technical Reversal (Bear Feature)
        ma_60 = ts_mean(C, 60)
        deep_value = (C - ma_60) / (ma_60 + 1e-8)
        features["tech_reversal"] = cs_rank(features["rsi_14"] * -1) + cs_rank(deep_value * -1)
        features["bear_reversal"] = features["tech_reversal"] * (1.0 - bull_prob)

        # Industry Relative Factors
        if IND is not None:
            ind_turnover_20d = cs_group_mean(features["turnover_mean_20d"], IND)
            features["ind_rel_turnover_20d"] = features["turnover_mean_20d"] / (ind_turnover_20d + 1e-8)
            ind_vol_20d = cs_group_mean(features["volatility_20d"], IND)
            features["ind_rel_vol_20d"] = features["volatility_20d"] / (ind_vol_20d + 1e-8)
            ind_bias_20 = cs_group_mean(features["bias_20"], IND)
            features["ind_rel_bias_20"] = features["bias_20"] - ind_bias_20

        # === V9 Phase 3: Simplified Dragon Score (no regime hardcoding) ===
        # V6 base logic only. Let MLP learn regime interactions from atomic factors.
        combined_mom = cs_rank(features["mom_20d"]) * 0.7 + cs_rank(features["mom_60d"]) * 0.3
        rank_turnover = cs_rank(features["turnover_mean_20d"])
        features["dragon_score"] = combined_mom + rank_turnover * torch.tanh(features["mom_20d"] * 5.0)
        
        # Low Volatility Anomaly
        features["inv_vol_20"] = 1.0 / (features["volatility_20d"] + 1e-4)

        # Bear Market Defense (Vol Penalty)
        vol_rank = cs_rank(features["volatility_20d"])
        features["vol_penalty"] = vol_rank * (1.0 - bull_prob) * -0.5
        
        # V7 Concept Factors
        con_mom_5 = padded_raw[:, :, col_map['concept_mom_5d']]
        con_mom_10 = padded_raw[:, :, col_map['concept_mom_10d']]
        con_mom_20 = padded_raw[:, :, col_map['concept_mom_20d']]
        con_mom_20_max = padded_raw[:, :, col_map['concept_mom_20d_max']]
        con_mom_20_min = padded_raw[:, :, col_map['concept_mom_20d_min']]
        con_mom_20_std = padded_raw[:, :, col_map['concept_mom_20d_std']]
        con_turnover_20 = padded_raw[:, :, col_map['concept_turnover_20d']]
        con_vol_20 = padded_raw[:, :, col_map['concept_vol_20d']]
        con_count = padded_raw[:, :, col_map['concept_count']]
        con_hot_ratio = padded_raw[:, :, col_map['concept_hot_ratio']]
        con_acc_5 = padded_raw[:, :, col_map['concept_acc_5_mean']]
        con_rank_score = padded_raw[:, :, col_map['concept_rank_score_mean']]
        
        features["con_mom_5d"] = con_mom_5
        features["con_mom_20d"] = con_mom_20
        features["con_mom_20d_max"] = con_mom_20_max
        features["con_turnover_20d"] = con_turnover_20
        
        # Concept Relative Strength
        features["rel_con_mom_20d"] = features["mom_20d"] - con_mom_20
        features["rel_con_mom_max_20d"] = features["mom_20d"] - con_mom_20_max
        features["con_divergence_20d"] = con_mom_20_std

        # V8.8: Rebound Strategy
        price_base_20 = ts_min(L, 20)
        price_peak_20 = ts_max(H, 20)
        price_base_60 = ts_min(L, 60)
        price_peak_60 = ts_max(H, 60)
        
        features["vol_range_20d"] = (price_peak_20 - price_base_20) / (price_base_20 + 1e-8)
        elasticity_rank = cs_rank(features["vol_range_20d"])
        oversold_score = cs_rank(features["bias_10"] * -1) + cs_rank(features["rsi_14"] * -1)
        
        features["dist_support_20"] = (C - price_base_20) / (price_base_20 + 1e-8)
        features["dist_support_60"] = (C - price_base_60) / (price_base_60 + 1e-8)
        features["dist_pressure_20"] = (price_peak_20 - C) / (C + 1e-8)
        features["dist_pressure_60"] = (price_peak_60 - C) / (C + 1e-8)
        features["rr_ratio_20"] = features["dist_pressure_20"] / (features["dist_support_20"] + 1e-4)
        features["rr_ratio_60"] = features["dist_pressure_60"] / (features["dist_support_60"] + 1e-4)
        
        support_score = 1.0 - torch.clamp(features["dist_support_20"] / 0.15, 0, 1)
        
        # Head-Lifting Signal
        ma_5 = ts_mean(C, 5)
        ma_5_prev = ts_delay(ma_5, 1)
        c_prev = ts_delay(C, 1)
        cross_up_ma5 = (C > ma_5) & (c_prev < ma_5_prev)
        is_big_red = (C / O - 1) > 0.025
        reversal_strength = (C - ts_min(L, 5)) / (ts_min(L, 5) + 1e-8)
        vol_ma_5 = ts_mean(V, 5)
        vol_ignition = V > vol_ma_5 * 1.2
        head_lift_signal = (cross_up_ma5.float() * 0.4 + is_big_red.float() * 0.4) * vol_ignition.float()
        features["head_lift_signal"] = head_lift_signal
        
        # Camel Hump Score
        features["camel_hump_score"] = (
            elasticity_rank * 0.3 +
            oversold_score * 0.3 +
            support_score * 0.4
        )
        
        # Resonance Trigger
        rank_con_mom = cs_rank(features["con_mom_5d"])
        if IND is not None:
             rank_ind_mom = cs_rank(features["ind_mom_5d"])
        else:
             rank_ind_mom = rank_con_mom
        features["resonance_signal"] = (rank_con_mom + rank_ind_mom) / 2.0
        
        # Meta-Features for MLP Learning
        features["meta_bull_prob"] = bull_prob.expand_as(C)
        features["inter_res_bull"] = features["resonance_signal"] * bull_prob
        bear_prob = 1.0 - bull_prob
        features["inter_camel_bear"] = features["camel_hump_score"] * bear_prob

        # ZT Count
        is_zt = (ret_1 > 0.095).float()
        features["zt_count_20d"] = ts_sum(is_zt, 20)

        # bear_trap_score as atomic factor (Phase 3: no penalty applied to dragon_score)
        features["bear_trap_score"] = (features["mom_20d"] * -1).clamp(min=0) * (1.0 - features["vol_ratio_5_20"]).clamp(min=0) * features["dist_support_20"] * 2.0
        features["bear_trap_score"] = features["bear_trap_score"] * (1.0 - features["head_lift_signal"]).clamp(min=0)

        # === V9 Phase 4: Turnover x Bull interaction factor ===
        features["turnover_x_bull"] = features["rel_turnover_20d"] * bull_prob

        # === V9 Phase 1: Beta-Neutral Label ===
        raw_ret_5 = ts_delay(C, -5) / C - 1
        mkt_ret_5 = torch.nanmean(raw_ret_5, dim=0, keepdim=True)
        excess_ret_5 = raw_ret_5 - features["beta_20d"] * mkt_ret_5

        low_liq_penalty = (features["turnover_mean_20d"] < 1.0).float() * 0.05
        excess_ret_5 = excess_ret_5 - low_liq_penalty

        features["label"] = cs_rank(excess_ret_5)

        return features
