from core.alpha.factor_calculator import FactorCalculator, device, torch, np, pl, cs_rank, ts_corr, cs_zscore, ts_delay, ts_mean, ts_min, ts_max, ts_quantile, ts_std, ts_sum, ts_rsquare, ts_slope, ta_atr, ta_rsi, cs_group_mean, ts_kdj, ts_cov

class V12FactorCalculator(FactorCalculator):
    """
    V12 因子计算器 — 精简基本面 + GP因子挖掘

    == 版本演进 ==
    V11: 110+因子, GP挖掘, Sharpe 1.17~1.39
    V11 → V12: 精简为基本技术因子, 回归标签(future_ret_5 - 5%), GP因子挖掘

    == 当前状态 ==
    手工因子: ~30 基本技术因子 (动量/波动率/量价/技术指标)
    标签: future_ret_5 - 0.05 (回归任务, 预测能否跑赢5%)
    GP因子: 由注册表管理

    == 设计决策 ==
    - 精简因子体系，让GP在干净的基础上挖掘
    - 回归标签直接预测收益，不做截面排名
    - 5%阈值作为"有意义收益"的分界线
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_features(self, padded_raw, col_map) -> dict[str, torch.Tensor]:
        print(f"[DEBUG] padded_raw.shape: {padded_raw.shape}")

        O = padded_raw[:, :, col_map['open']]
        H = padded_raw[:, :, col_map['high']]
        L = padded_raw[:, :, col_map['low']]
        C = padded_raw[:, :, col_map['close']]
        V = padded_raw[:, :, col_map['volume']]
        T = padded_raw[:, :, col_map['turnover']]
        TR = padded_raw[:, :, col_map['turnover_rate']]

        features = {}
        ret_1 = C / ts_delay(C, 1) - 1

        # === Momentum ===
        features["mom_5d"] = C / ts_delay(C, 5) - 1
        features["mom_20d"] = C / ts_delay(C, 20) - 1
        features["mom_60d"] = C / ts_delay(C, 60) - 1
        features["bias_5"] = C / ts_mean(C, 5) - 1
        features["bias_20"] = C / ts_mean(C, 20) - 1
        features["bias_60"] = C / ts_mean(C, 60) - 1

        # === Volatility ===
        features["volatility_20d"] = ts_std(ret_1, 20)
        features["volatility_60d"] = ts_std(ret_1, 60)
        features["atr_ratio_14"] = ta_atr(H, L, C, 14) / C
        features["daily_range"] = H / L - 1

        # === Volume/Turnover ===
        features["volume_ratio"] = V / (ts_mean(V, 20) + 1e-8)
        features["turnover_mean_5d"] = ts_mean(TR, 5)
        features["turnover_mean_20d"] = ts_mean(TR, 20)
        features["vol_cv_20"] = ts_std(V, 20) / (ts_mean(V, 20) + 1e-8)

        # === Technical ===
        features["rsi_14"] = ta_rsi(C, 14)
        ma_20 = ts_mean(C, 20)
        std_20 = ts_std(C, 20)
        features["bollinger_position"] = (C - ma_20) / (std_20 * 2 + 1e-8)
        features["drawdown_20d"] = C / ts_max(C, 20) - 1
        features["rebound_20d"] = C / ts_min(L, 20) - 1

        # === Trend ===
        features["trend_rsquare_20"] = ts_rsquare(C, 20)
        slope_20 = ts_slope(C, 20)
        features["trend_slope_20"] = slope_20 / (C + 1e-8)

        # === Price-Volume ===
        features["price_vol_corr_20"] = ts_corr(C, V, 20)
        vol_change = torch.log(V / (ts_delay(V, 1) + 1e-8) + 1e-8)
        features["cord_20"] = ts_corr(ret_1, vol_change, 20)

        # === Liquidity ===
        abs_ret = torch.abs(ret_1)
        daily_amihud = abs_ret / (V + 1e-8)
        features["amihud_20d"] = ts_mean(daily_amihud, 20)

        # === GP-Mined Factors ===
        gp_factors = self.gp_miner.compute_factors(padded_raw, col_map)
        if gp_factors:
            print(f"[V12] Adding {len(gp_factors)} GP factors")
            features.update(gp_factors)

        # === Label: 回归标签 ===
        # 未来5日收益率 - 5%, 正值=跑赢阈值, 负值=跑输
        future_ret_5 = ts_delay(C, -5) / C - 1
        features["label"] = future_ret_5 - 0.05

        return features
