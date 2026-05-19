from core.alpha.factor_calculator import FactorCalculator, device, torch, np, pl, cs_rank, ts_corr, cs_zscore, ts_delay, ts_mean, ts_min, ts_max, ts_quantile, ts_std, ts_sum, ts_rsquare, ts_slope, ta_atr, ta_rsi, cs_group_mean, ts_kdj, ts_cov, cs_neutralize

class V15FactorCalculator(FactorCalculator):
    """
    V15 因子计算器 — 单模型双股票池（Style Regime 因子 + V14 标签）

    == 版本演进 ==
    V14: 10日beta-neutral标签, 中证2000 Sharpe 1.42 / 中证1000 Sharpe 1.19
    V15-初版: 5个regime特征+全截面排名, 2024年-15.5%失败
    V15-Size-Neutral: Size-Neutral 标签 + Size相对因子, Sharpe 0.31 失败
    V15.0: 恢复V14标签 + 5个regime因子, Sharpe 1.33, 收益回撤比6.52, 但每年都不如单池最优
    V15.1-当前: 升级 regime 信号到 13 个，Sharpe 1.51 / 年化 89.7% / MaxDD -23.5% （超越两个单池）

    == 设计决策 ==
    在双股票池（CSI 1000 + CSI 2000）混合训练下，让模型识别当前风格偏向：
    - pool_size_rank: 截面市值排名（个股身份特征）
    - style_regime_5/20/60: 多窗口大盘组 vs 小盘组动量差
    - style_regime_change: 短-长期差，捕捉风格切换加速度
    - style_regime_strength: 风格分化强度（无方向）
    - vol_regime / turnover_regime / amihud_regime: 波动/换手/流动性维度的风格分化
    - mom_x_regime_align/contrarian: 个股动量与当前风格的交互
    - pool_size_x_regime: 个股市值排名 × 风格方向（核心 timing 交互）
    - pool_size_x_regime_change: 个股市值排名 × 风格切换方向
    标签仍用 V14 全截面 beta-neutral 10日标签。

    == 失败记录 ==
    - V15-初版 5 regime+全截面排名: 2024年 -15.5%, 风格轮动信号滞后被甩
    - V15 Size-Neutral 标签 + Size相对因子: Sharpe 0.31 (CSI 1000+2000 混池),
      远不如单池 V14（Sharpe 1.42 / 1.19）。剥离 size premium 等于剥离主 alpha 信号。
    - V15.0 弱 regime 因子（5个）: Sharpe 1.33 但每年都不如单池最优年份，timing 信号太弱。
    """
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

        # === V10 Step 4: Factors re-tested under Factor Attention ===
        # cord_20: 量价同步性 (ret change vs volume change correlation, 20d)
        # IC=0.061, previously failed under MLP (Sharpe -0.28), re-test under Attention
        ret_1 = C / ts_delay(C, 1) - 1
        vol_change = torch.log(V / (ts_delay(V, 1) + 1e-8) + 1e-8)
        features["cord_20"] = ts_corr(ret_1, vol_change, 20)


        # === V10: Conditional Reversal Confirmation Factor ===
        # oversold_vol_confirm - 超跌×缩量交互
        # 超跌深度 × 缩量程度，组合底部形态信号
        # IC=0.079 (t=0.85), 下跌市IC更高(0.109/0.112)
        vol_shrink_ratio = ts_mean(V, 5) / (ts_mean(V, 20) + 1e-8)
        drawdown_60d = C / (ts_max(C, 60) + 1e-8) - 1
        oversold_depth = (-drawdown_60d).clamp(min=0, max=0.30)
        vol_shrink = (1.0 - vol_shrink_ratio).clamp(min=0)
        features["oversold_vol_confirm"] = oversold_depth * vol_shrink

        # === V10 Validated New Factors ===

        # amihud_20d: Amihud流动性冲击因子
        # |daily_return| / daily_volume, averaged over 20 days
        # IC improving over time: -0.047(early) -> 0.075(2025) -> 0.059(recent)
        abs_ret = torch.abs(ret_1)
        daily_amihud = abs_ret / (V + 1e-8)
        features["amihud_20d"] = ts_mean(daily_amihud, 20)

        # price_impact_asym: 非对称价格冲击
        # 下跌日冲击 / 上涨日冲击, IC improving (0.041 recent)
        down_mask = (ret_1 < 0).float()
        up_mask = (ret_1 > 0).float()
        down_impact = ts_mean(daily_amihud * down_mask, 20) / (ts_mean(down_mask, 20) + 1e-8)
        up_impact = ts_mean(daily_amihud * up_mask, 20) / (ts_mean(up_mask, 20) + 1e-8)
        features["price_impact_asym"] = down_impact / (up_impact + 1e-8)

        # vol_price_div: 量价背离信号 (IC=0.049 recent, stable)
        price_trend = (C / ts_mean(C, 20) - 1)
        vol_trend = 1.0 - vol_shrink_ratio
        features["vol_price_div"] = (-price_trend).clamp(min=0) * vol_trend.clamp(min=0)

        # === V10 Batch 2: Amihud/Liquidity Variants for IC Screening ===

        # 1. amihud_5d: 短窗口Amihud (捕捉近期流动性突变)
        features["amihud_5d"] = ts_mean(daily_amihud, 5)

        # 2. amihud_60d: 长窗口Amihud (流动性结构性水平)
        features["amihud_60d"] = ts_mean(daily_amihud, 60)

        # 3. log_amihud_20d: 对数Amihud (压缩极端值，改善分布)
        features["log_amihud_20d"] = torch.log1p(features["amihud_20d"] * 1e6)

        # 4. amihud_trend: 流动性恶化趋势 (短/长比值)
        # 高值=近期流动性比长期差=流动性在恶化
        features["amihud_trend"] = features["amihud_5d"] / (features["amihud_60d"] + 1e-8)

        # 5. liquidity_shock: 流动性冲击z-score
        # 当前amihud vs 自身60日历史的偏离程度
        amihud_60d_mean = ts_mean(daily_amihud, 60)
        amihud_60d_std = ts_std(daily_amihud, 60)
        features["liquidity_shock"] = (features["amihud_5d"] - amihud_60d_mean) / (amihud_60d_std + 1e-8)

        # 6. kyle_lambda_20d: Kyle's Lambda代理 (价格冲击斜率)
        # 用 |ret| / sqrt(volume) 近似，捕捉知情交易者冲击
        kyle_daily = abs_ret / (torch.sqrt(V) + 1e-8)
        features["kyle_lambda_20d"] = ts_mean(kyle_daily, 20)

        # === V10 Batch 8: Chip Distribution (筹码分布) Factors ===
        # 筹码分布反映散户/主力持仓结构，是超跌反转时机的重要信号
        # 数据来源: tushare cyq_perf接口 (his_low, his_high, cost_5pct, cost_15pct, cost_50pct, cost_85pct, weight_avg)
        # 数据从2018年开始，按日更新

        if False :
        #if all(k in col_map for k in ['his_low', 'his_high', 'cost_5pct', 'cost_50pct', 'cost_85pct', 'weight_avg']):
            his_low = padded_raw[:, :, col_map['his_low']]
            his_high = padded_raw[:, :, col_map['his_high']]
            cost_5pct = padded_raw[:, :, col_map['cost_5pct']]
            cost_50pct = padded_raw[:, :, col_map['cost_50pct']]
            cost_85pct = padded_raw[:, :, col_map['cost_85pct']]
            weight_avg = padded_raw[:, :, col_map['weight_avg']]

            # 1. chip_cost_deviation: 筹码成本偏离度
            # (price - avg_chip_cost) / avg_chip_cost
            # 负值=当前价格低于筹码平均成本=超跌
            # Expected IC: -0.03 ~ -0.05 (负IC: 价格低于成本越低，反弹概率越高)
            features["chip_cost_deviation"] = (C - weight_avg) / (weight_avg + 1e-8)

            # 2. chip_concentration: 筹码集中度
            # (cost_85pct - cost_5pct) / weight_avg
            # 低值=筹码集中（散户割肉后筹码向低位集中）
            # Expected IC: -0.02 ~ -0.04
            features["chip_concentration"] = (cost_85pct - cost_5pct) / (weight_avg + 1e-8)

            # 3. chip_profit_ratio_proxy: 获利盘比例代理
            # (C - cost_50pct) / cost_50pct
            # 正值=多数筹码盈利；负值=多数筹码套牢
            # Expected IC: 0.02 ~ 0.04
            features["chip_profit_ratio"] = (C - cost_50pct) / (cost_50pct + 1e-8)

            # 4. chip_peak_distance: 筹码峰距离
            # 当前价格距离筹码密集峰(cost_50pct)的偏离
            # 负值=价格在筹码峰下方=超跌
            # Expected IC: -0.02 ~ -0.03
            features["chip_peak_distance"] = (C - cost_50pct) / (cost_50pct + 1e-8)

            # 5. chip_range: 筹码分布范围
            # (his_high - his_low) / weight_avg
            # 高值=筹码分布分散；低值=筹码集中
            features["chip_range"] = (his_high - his_low) / (weight_avg + 1e-8)

            # 6. retail_capitulation: 散户割肉信号
            # 筹码集中度低(筹码向低位集中) + 价格低于筹码成本
            # 这是"散户割肉、主力吸筹"的底部信号
            # Expected IC: 0.03 ~ 0.05
            low_concentration = (features["chip_concentration"] < torch.nanmean(features["chip_concentration"], dim=0, keepdim=True)).float()
            below_cost = (features["chip_cost_deviation"] < 0).float()
            features["retail_capitulation"] = low_concentration * below_cost

        else:
            print("[WARNING] Cyq Perf columns not available, skipping chip distribution factors")

        # === V10 Batch 7: Informed Trading Screening Results ===
        # All informed trading factors failed:
        # - vpin_proxy: calculation error (all NaN) -> REMOVED
        # - order_flow_toxicity: IC=-0.199 (future function bug fixed, still negative) -> REMOVED
        # - informed_herding: calculation error (all NaN) -> REMOVED
        # - adverse_selection: IC=0.082 but redundant with amihud/volatility -> REMOVED
        # - informed_buy_pressure: IC=0.019 (noise) -> REMOVED
        # - information_asymmetry: IC=0.017 (noise, Sharpe dropped from 1.37 to 0.95) -> REMOVED
        #
        # Conclusion: Informed trading dimensions cannot be captured from daily OHLCV data.
        # These microstructure indicators require tick-level or minute-level data.
        # Restored to V10 stable state (110 factors: V9 + 10 Amihud liquidity factors)

        # === GP-Mined Short-Period Factors ===
        gp_factors = self.gp_miner.compute_factors(padded_raw, col_map)
        if gp_factors:
            print(f"[V15] Adding {len(gp_factors)} GP factors")
            features.update(gp_factors)

        # === Style Regime Features (V15.1: 升级 timing 信号) ===
        # 用市值均值划"大/小组"，从多维度捕捉风格轮动 timing
        size_ln = features["size_ln_cap"]
        size_clean = torch.nan_to_num(size_ln, nan=0.0)
        size_valid = ~torch.isnan(size_ln)
        size_sum = (size_clean * size_valid.float()).sum(dim=0, keepdim=True)
        size_cnt = size_valid.float().sum(dim=0, keepdim=True)
        size_mean = size_sum / (size_cnt + 1e-8)
        is_large_cap = (size_ln >= size_mean).float()
        is_small_cap = 1.0 - is_large_cap

        nan_t = torch.tensor(float('nan'), device=C.device)

        def group_spread(factor_tensor):
            """Compute large_cap_mean - small_cap_mean (broadcastable scalar across batch)."""
            large_masked = torch.where(is_large_cap.bool(), factor_tensor, nan_t)
            small_masked = torch.where(is_small_cap.bool(), factor_tensor, nan_t)
            return torch.nanmean(large_masked, dim=0, keepdim=True) - torch.nanmean(small_masked, dim=0, keepdim=True)

        # 1. pool_size_rank: 截面市值排名（身份特征）
        features["pool_size_rank"] = cs_zscore(size_ln)

        # 2-4. 多窗口动量风格（捕捉短/中/长期风格强度）
        style_regime_5 = group_spread(features["mom_5d"])
        style_regime_20 = group_spread(features["mom_20d"])
        style_regime_60 = group_spread(features["mom_60d"])
        features["style_regime_5"] = style_regime_5.expand_as(C)
        features["style_regime"] = style_regime_20.expand_as(C)
        features["style_regime_60"] = style_regime_60.expand_as(C)

        # 5. style_regime_change: 短期 vs 长期 → 风格切换加速度
        # 正值=大盘最近5日突然占优(短期>长期)，负值=大盘走弱
        features["style_regime_change"] = (style_regime_5 - style_regime_60).expand_as(C)

        # 6. style_regime_strength: 当前风格强度的绝对值（无方向）
        # 高值=风格分化剧烈，低值=风格混沌
        features["style_regime_strength"] = style_regime_20.abs().expand_as(C)

        # 7. vol_regime: 波动率风格（大盘组 vs 小盘组的 20d 波动差）
        # 通常熊市/紧张期大盘波动相对低 → vol_regime 负 → 利大盘
        features["vol_regime"] = group_spread(features["volatility_20d"]).expand_as(C)

        # 8. turnover_regime: 换手风格
        # 正值=大盘换手相对高（罕见，多为牛市末期）
        # 负值=小盘换手高（情绪期，利小盘）
        features["turnover_regime"] = group_spread(features["turnover_mean_20d"]).expand_as(C)

        # 9. amihud_regime: 流动性环境
        # 正值=大盘流动性比小盘差（紧张期，反而利小盘 mean reversion）
        features["amihud_regime"] = group_spread(features["amihud_20d"]).expand_as(C)

        # 10-11. 个股动量 × 当前风格（顺势/逆势分离）
        regime_direction = torch.tanh(style_regime_20.expand_as(C) * 10.0)
        features["mom_x_regime_align"] = features["mom_20d"] * regime_direction
        features["mom_x_regime_contrarian"] = features["mom_20d"] * (1.0 - regime_direction.abs())

        # 12. pool_size_x_regime: 个股市值排名 × 风格方向（关键 timing 交互）
        # 大盘股(rank高) × 大盘占优(regime正) → 强正
        # 小盘股(rank低) × 小盘占优(regime负) → 强正
        # 让模型直接学到"当前 size 段是否顺风"
        features["pool_size_x_regime"] = features["pool_size_rank"] * regime_direction

        # 13. pool_size_x_regime_change: 个股市值排名 × 风格切换信号
        # 风格刚切换 + 个股在被切到的池里 → 正信号（早期入场）
        regime_change_dir = torch.tanh(features["style_regime_change"] * 10.0)
        features["pool_size_x_regime_change"] = features["pool_size_rank"] * regime_change_dir

        # === Beta-Neutral Label (V14: 10日, 全截面排名) ===
        raw_ret_10 = ts_delay(C, -10) / C - 1
        mkt_ret_10 = torch.nanmean(raw_ret_10, dim=0, keepdim=True)
        excess_ret_10 = raw_ret_10 - features["beta_20d"] * mkt_ret_10

        # 低流动性惩罚
        low_liq_penalty = (features["turnover_mean_20d"] < 1.0).float() * 0.05
        excess_ret_10 = excess_ret_10 - low_liq_penalty

        features["label"] = cs_rank(excess_ret_10)

        return features
