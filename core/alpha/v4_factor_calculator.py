from core.alpha.factor_calculator import (
    FactorCalculator, 
    ts_delay, ts_mean, ts_sum, ts_std, ts_max, ts_min, ts_delta,
    ta_atr, ta_rsi, ts_corr, cs_rank, cs_scale, ts_rank,
    quesval, quesval2, ts_argmax, ts_argmin, ts_product,
    ts_decay_linear, ts_cov, pow1, pow2, ts_greater, ts_less,
    torch
)

class V4FactorCalculator(FactorCalculator): 
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
        T = padded_raw[:, :, 5] # Turnover (Amount)
        TR = padded_raw[:, :, 6] # Turnover Rate
        PE = padded_raw[:, :, 7] # PE Ratio
        
        # Helper vars
        # VWAP = Turnover / Volume. 
        # Handle cases where Volume is 0 or NaN.
        vwap = T / (V + 1e-8)
        vwap = torch.where(torch.isnan(vwap), C, vwap) 

        returns = C / ts_delay(C, 1) - 1
        # Avoid infinite returns if prev close is 0/nan
        returns = torch.nan_to_num(returns, nan=0.0)

        features = {}

        # Alpha 1
        # (cs_rank(ts_argmax(pow1(quesval(0, returns, close, ts_std(returns, 20)), 2.0), 5)) - 0.5)
        cond_1 = quesval(0, returns, C, ts_std(returns, 20))
        features["alpha001"] = cs_rank(ts_argmax(pow1(cond_1, 2.0), 5)) - 0.5

        # Alpha 2
        # (-1) * ts_corr(cs_rank(ts_delta(log(volume), 2)), cs_rank((close - open) / open), 6)
        # Add epsilon to log
        log_v = torch.log(V + 1e-8)
        a2_part1 = cs_rank(ts_delta(log_v, 2))
        a2_part2 = cs_rank((C - O) / (O + 1e-8))
        features["alpha002"] = -1 * ts_corr(a2_part1, a2_part2, 6)

        # Alpha 3
        # ts_corr(cs_rank(open), cs_rank(volume), 10) * -1
        features["alpha003"] = -1 * ts_corr(cs_rank(O), cs_rank(V), 10)

        # Alpha 4
        # -1 * ts_rank(cs_rank(low), 9)
        features["alpha004"] = -1 * ts_rank(cs_rank(L), 9)

        # Alpha 5
        # cs_rank((open - (ts_sum(vwap, 10) / 10))) * (-1 * abs(cs_rank((close - vwap))))
        a5_p1 = cs_rank(O - (ts_sum(vwap, 10) / 10))
        a5_p2 = -1 * torch.abs(cs_rank(C - vwap))
        features["alpha005"] = a5_p1 * a5_p2

        # Alpha 6
        # (-1) * ts_corr(open, volume, 10)
        features["alpha006"] = -1 * ts_corr(O, V, 10)

        # Alpha 7
        # quesval2(ts_mean(volume, 20), volume, (-1 * ts_rank(abs(close - ts_delay(close, 7)), 60)) * sign(ts_delta(close, 7)), -1)
        # if V < mean(V,20) ...
        a7_cond = ts_mean(V, 20)
        a7_true = (-1 * ts_rank(torch.abs(C - ts_delay(C, 7)), 60)) * torch.sign(ts_delta(C, 7))
        features["alpha007"] = quesval2(a7_cond, V, a7_true, -1)

        # Alpha 8
        # -1 * cs_rank(((ts_sum(open, 5) * ts_sum(returns, 5)) - ts_delay((ts_sum(open, 5) * ts_sum(returns, 5)), 10)))
        a8_term = ts_sum(O, 5) * ts_sum(returns, 5)
        features["alpha008"] = -1 * cs_rank(a8_term - ts_delay(a8_term, 10))

        # Alpha 9
        # quesval(0, ts_min(ts_delta(close, 1), 5), ts_delta(close, 1), quesval(0, ts_max(ts_delta(close, 1), 5), (-1 * ts_delta(close, 1)), ts_delta(close, 1)))
        delta_c = ts_delta(C, 1)
        features["alpha009"] = quesval(
            0, 
            ts_min(delta_c, 5), 
            delta_c, 
            quesval(0, ts_max(delta_c, 5), -1 * delta_c, delta_c)
        )

        # Alpha 10
        # cs_rank(quesval(0, ts_min(ts_delta(close, 1), 4), ts_delta(close, 1), quesval(0, ts_max(ts_delta(close, 1), 4), (-1 * ts_delta(close, 1)), ts_delta(close, 1))))
        delta_c = ts_delta(C, 1)
        a10_val = quesval(
            0,
            ts_min(delta_c, 4),
            delta_c,
            quesval(0, ts_max(delta_c, 4), -1 * delta_c, delta_c)
        )
        features["alpha010"] = cs_rank(a10_val)

        # Alpha 11
        # (cs_rank(ts_max(vwap - close, 3)) + cs_rank(ts_min(vwap - close, 3))) * cs_rank(ts_delta(volume, 3))
        features["alpha011"] = (cs_rank(ts_max(vwap - C, 3)) + cs_rank(ts_min(vwap - C, 3))) * cs_rank(ts_delta(V, 3))

        # Alpha 12
        # sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1))
        features["alpha012"] = torch.sign(ts_delta(V, 1)) * (-1 * ts_delta(C, 1))

        # Alpha 13
        # -1 * cs_rank(ts_cov(cs_rank(close), cs_rank(volume), 5))
        features["alpha013"] = -1 * cs_rank(ts_cov(cs_rank(C), cs_rank(V), 5))

        # Alpha 14
        # (-1 * cs_rank(returns - ts_delay(returns, 3))) * ts_corr(open, volume, 10)
        features["alpha014"] = (-1 * cs_rank(returns - ts_delay(returns, 3))) * ts_corr(O, V, 10)

        # Alpha 15
        # -1 * ts_sum(cs_rank(ts_corr(cs_rank(high), cs_rank(volume), 3)), 3)
        features["alpha015"] = -1 * ts_sum(cs_rank(ts_corr(cs_rank(H), cs_rank(V), 3)), 3)

        # Alpha 16
        # -1 * cs_rank(ts_cov(cs_rank(high), cs_rank(volume), 5))
        features["alpha016"] = -1 * cs_rank(ts_cov(cs_rank(H), cs_rank(V), 5))

        # Alpha 17
        # (-1 * cs_rank(ts_rank(close, 10))) * cs_rank(close - 2 * ts_delay(close, 1) + ts_delay(close, 2)) * cs_rank(ts_rank(volume / ts_mean(volume, 20), 5))
        term_17_a = -1 * cs_rank(ts_rank(C, 10))
        term_17_b = cs_rank(C - 2 * ts_delay(C, 1) + ts_delay(C, 2))
        term_17_c = cs_rank(ts_rank(V / (ts_mean(V, 20) + 1e-8), 5))
        features["alpha017"] = term_17_a * term_17_b * term_17_c

        # Alpha 18
        # -1 * cs_rank((ts_std(abs(close - open), 5) + (close - open)) + ts_corr(close, open, 10))
        features["alpha018"] = -1 * cs_rank(ts_std(torch.abs(C - O), 5) + (C - O) + ts_corr(C, O, 10))

        # Alpha 19
        # (-1 * sign(ts_delta(close, 7) + (close - ts_delay(close, 7)))) * (cs_rank(ts_sum(returns, 250) + 1) + 1)
        term_19_a = -1 * torch.sign(ts_delta(C, 7) + (C - ts_delay(C, 7)))
        term_19_b = cs_rank(ts_sum(returns, 250) + 1) + 1
        features["alpha019"] = term_19_a * term_19_b

        # Alpha 20
        # (-1 * cs_rank(open - ts_delay(high, 1))) * cs_rank(open - ts_delay(close, 1)) * cs_rank(open - ts_delay(low, 1))
        features["alpha020"] = (-1 * cs_rank(O - ts_delay(H, 1))) * cs_rank(O - ts_delay(C, 1)) * cs_rank(O - ts_delay(L, 1))

        # Alpha 21
        # quesval2((ts_mean(close, 8) + ts_std(close, 8)), ts_mean(close, 2), -1, quesval2(ts_mean(close, 2), (ts_mean(close, 8) - ts_std(close, 8)), 1, quesval(1, (volume / ts_mean(volume, 20)), 1, -1)))
        cond1 = ts_mean(C, 8) + ts_std(C, 8)
        cond2 = ts_mean(C, 8) - ts_std(C, 8)
        mc2 = ts_mean(C, 2)
        val_inner = quesval(1, V / (ts_mean(V, 20) + 1e-8), 1, -1)
        features["alpha021"] = quesval2(
            cond1, 
            mc2, 
            -1, 
            quesval2(mc2, cond2, 1, val_inner)
        )

        # Alpha 22
        # -1 * ts_delta(ts_corr(high, volume, 5), 5) * cs_rank(ts_std(close, 20))
        features["alpha022"] = -1 * ts_delta(ts_corr(H, V, 5), 5) * cs_rank(ts_std(C, 20))

        # Alpha 23
        # quesval2(ts_mean(high, 20), high, -1 * ts_delta(high, 2), 0)
        features["alpha023"] = quesval2(ts_mean(H, 20), H, -1 * ts_delta(H, 2), 0)

        # Alpha 24
        # quesval(0.05, ts_delta(ts_sum(close, 100) / 100, 100) / ts_delay(close, 100), (-1 * ts_delta(close, 3)), (-1 * (close - ts_min(close, 100))))
        mean_c_100 = ts_mean(C, 100)
        term_24 = ts_delta(mean_c_100, 100) / (ts_delay(C, 100) + 1e-8)
        features["alpha024"] = quesval(
            0.05, 
            term_24, 
            -1 * ts_delta(C, 3), 
            -1 * (C - ts_min(C, 100))
        )

        # Alpha 25
        # cs_rank( (-1 * returns) * ts_mean(volume, 20) * vwap * (high - close) )
        features["alpha025"] = cs_rank((-1 * returns) * ts_mean(V, 20) * vwap * (H - C))

        # Alpha 26
        # -1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)
        features["alpha026"] = -1 * ts_max(ts_corr(ts_rank(V, 5), ts_rank(H, 5), 5), 3)

        # Alpha 27
        # quesval(0.5, cs_rank(ts_mean(ts_corr(cs_rank(volume), cs_rank(vwap), 6), 2)), -1, 1)
        term_27 = cs_rank(ts_mean(ts_corr(cs_rank(V), cs_rank(vwap), 6), 2))
        features["alpha027"] = quesval(0.5, term_27, -1, 1)

        # Alpha 28
        # cs_scale(ts_corr(ts_mean(volume, 20), low, 5) + (high + low) / 2 - close)
        features["alpha028"] = cs_scale(ts_corr(ts_mean(V, 20), L, 5) + (H + L) / 2 - C)

        # Alpha 29
        # ts_min(ts_product(cs_rank(cs_rank(cs_scale(log(ts_sum(ts_min(cs_rank(cs_rank((-1 * cs_rank(ts_delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(ts_delay((-1 * returns), 6), 5)
        inner_29 = -1 * cs_rank(ts_delta(C, 5))
        inner_29 = cs_rank(cs_rank(inner_29))
        inner_29 = ts_min(inner_29, 2)
        inner_29 = torch.log(inner_29 + 1e-8) 
        inner_29 = cs_rank(cs_rank(cs_scale(inner_29)))
        term_29_a = ts_min(inner_29, 5)
        term_29_b = ts_rank(ts_delay(-1 * returns, 6), 5)
        features["alpha029"] = term_29_a + term_29_b

        # Alpha 30
        # ((cs_rank(sign(close - ts_delay(close, 1)) + sign(ts_delay(close, 1) - ts_delay(close, 2)) + sign(ts_delay(close, 2) - ts_delay(close, 3))) * -1 + 1) * ts_sum(volume, 5)) / ts_sum(volume, 20)
        d1 = ts_delta(C, 1)
        s_sum = torch.sign(d1) + torch.sign(ts_delay(d1, 1)) + torch.sign(ts_delay(d1, 2))
        features["alpha030"] = ((cs_rank(s_sum) * -1 + 1) * ts_sum(V, 5)) / (ts_sum(V, 20) + 1e-8)

        # Alpha 31
        # (cs_rank(cs_rank(cs_rank(ts_decay_linear((-1) * cs_rank(cs_rank(ts_delta(close, 10))), 10)))) + cs_rank((-1) * ts_delta(close, 3))) + sign(cs_scale(ts_corr(ts_mean(volume, 20), low, 12)))
        term_31_a = ts_decay_linear(-1 * cs_rank(cs_rank(ts_delta(C, 10))), 10)
        term_31_a = cs_rank(cs_rank(cs_rank(term_31_a)))
        term_31_b = cs_rank(-1 * ts_delta(C, 3))
        term_31_c = torch.sign(cs_scale(ts_corr(ts_mean(V, 20), L, 12)))
        features["alpha031"] = term_31_a + term_31_b + term_31_c

        # Alpha 32
        # cs_scale((ts_sum(close, 7) / 7 - close)) + (20 * cs_scale(ts_corr(vwap, ts_delay(close, 5), 230)))
        features["alpha032"] = cs_scale(ts_mean(C, 7) - C) + 20 * cs_scale(ts_corr(vwap, ts_delay(C, 5), 230))

        # Alpha 33
        # cs_rank((-1) * (open / close * -1 + 1))
        features["alpha033"] = cs_rank(-1 * (1 - O / (C + 1e-8)))

        # Alpha 34
        # cs_rank((cs_rank(ts_std(returns, 2) / ts_std(returns, 5)) * -1 + 1) + (cs_rank(ts_delta(close, 1)) * -1 + 1))
        term_34_a = cs_rank(ts_std(returns, 2) / (ts_std(returns, 5) + 1e-8)) * -1 + 1
        term_34_b = cs_rank(ts_delta(C, 1)) * -1 + 1
        features["alpha034"] = cs_rank(term_34_a + term_34_b)

        # Alpha 35
        # (ts_rank(volume, 32) * (ts_rank((close + high - low), 16) * -1 + 1)) * (ts_rank(returns, 32) * -1 + 1)
        features["alpha035"] = (ts_rank(V, 32) * (ts_rank(C + H - L, 16) * -1 + 1)) * (ts_rank(returns, 32) * -1 + 1)

        # Alpha 36
        # ((((2.21 * cs_rank(ts_corr((close - open), ts_delay(volume, 1), 15))) + (0.7 * cs_rank((open - close)))) + (0.73 * cs_rank(ts_rank(ts_delay((-1) * returns, 6), 5)))) + cs_rank(abs(ts_corr(vwap, ts_mean(volume, 20), 6)))) + (0.6 * cs_rank(((ts_sum(close, 200) / 200 - open) * (close - open))))
        a36_1 = 2.21 * cs_rank(ts_corr(C - O, ts_delay(V, 1), 15))
        a36_2 = 0.7 * cs_rank(O - C)
        a36_3 = 0.73 * cs_rank(ts_rank(ts_delay(-1 * returns, 6), 5))
        a36_4 = cs_rank(torch.abs(ts_corr(vwap, ts_mean(V, 20), 6)))
        a36_5 = 0.6 * cs_rank((ts_mean(C, 200) - O) * (C - O))
        features["alpha036"] = a36_1 + a36_2 + a36_3 + a36_4 + a36_5

        # Alpha 37
        # cs_rank(ts_corr(ts_delay((open - close), 1), close, 200)) + cs_rank((open - close))
        features["alpha037"] = cs_rank(ts_corr(ts_delay(O - C, 1), C, 200)) + cs_rank(O - C)

        # Alpha 38
        # ((-1) * cs_rank(ts_rank(close, 10))) * cs_rank((close / open))
        features["alpha038"] = (-1 * cs_rank(ts_rank(C, 10))) * cs_rank(C / (O + 1e-8))

        # Alpha 39
        # ((-1) * cs_rank((ts_delta(close, 7) * (cs_rank(ts_decay_linear((volume / ts_mean(volume, 20)), 9)) * -1 + 1)))) * (cs_rank(ts_sum(returns, 250)) + 1)
        a39_decay = ts_decay_linear(V / (ts_mean(V, 20) + 1e-8), 9)
        a39_p1 = -1 * cs_rank(ts_delta(C, 7) * (cs_rank(a39_decay) * -1 + 1))
        a39_p2 = cs_rank(ts_sum(returns, 250)) + 1
        features["alpha039"] = a39_p1 * a39_p2

        # Alpha 40
        # ((-1) * cs_rank(ts_std(high, 10))) * ts_corr(high, volume, 10)
        features["alpha040"] = -1 * cs_rank(ts_std(H, 10)) * ts_corr(H, V, 10)

        # Alpha 41
        # pow1((high * low), 0.5) - vwap
        features["alpha041"] = pow1(H * L, 0.5) - vwap

        # Alpha 42
        # cs_rank((vwap - close)) / cs_rank((vwap + close))
        features["alpha042"] = cs_rank(vwap - C) / (cs_rank(vwap + C) + 1e-8)

        # Alpha 43
        # ts_rank((volume / ts_mean(volume, 20)), 20) * ts_rank((-1) * ts_delta(close, 7), 8)
        features["alpha043"] = ts_rank(V / (ts_mean(V, 20) + 1e-8), 20) * ts_rank(-1 * ts_delta(C, 7), 8)

        # Alpha 44
        # (-1) * ts_corr(high, cs_rank(volume), 5)
        features["alpha044"] = -1 * ts_corr(H, cs_rank(V), 5)

        # Alpha 45
        # (-1) * cs_rank(ts_sum(ts_delay(close, 5), 20) / 20) * ts_corr(close, volume, 2) * cs_rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))
        a45_p1 = -1 * cs_rank(ts_mean(ts_delay(C, 5), 20))
        a45_p2 = ts_corr(C, V, 2)
        a45_p3 = cs_rank(ts_corr(ts_sum(C, 5), ts_sum(C, 20), 2))
        features["alpha045"] = a45_p1 * a45_p2 * a45_p3

        # Alpha 46
        # quesval(0.25, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), -1, quesval(0, ..., (-1) * (close - ts_delay(close, 1)), 1))
        val_x = (ts_delay(C, 20) - ts_delay(C, 10)) / 10.0 - (ts_delay(C, 10) - C) / 10.0
        features["alpha046"] = quesval(
            0.25, 
            val_x, 
            -1, 
            quesval(0, val_x, -1 * (C - ts_delay(C, 1)), 1)
        )

        # Alpha 47
        # ((cs_rank(pow1(close, -1)) * volume / ts_mean(volume, 20)) * (high * cs_rank(high - close)) / (ts_sum(high, 5) / 5)) - cs_rank(vwap - ts_delay(vwap, 5))
        a47_p1 = cs_rank(pow1(C, -1)) * V / (ts_mean(V, 20) + 1e-8)
        a47_p2 = (H * cs_rank(H - C)) / (ts_mean(H, 5) + 1e-8)
        features["alpha047"] = a47_p1 * a47_p2 - cs_rank(vwap - ts_delay(vwap, 5))

        # Alpha 48: IndNeutralize, Skip

        # Alpha 49
        # quesval(-0.1, val_x, (-1) * (close - ts_delay(close, 1)), 1)
        features["alpha049"] = quesval(
            -0.1, 
            val_x, 
            -1 * (C - ts_delay(C, 1)), 
            1
        )

        # Alpha 50
        # (-1) * ts_max(cs_rank(ts_corr(cs_rank(volume), cs_rank(vwap), 5)), 5)
        features["alpha050"] = -1 * ts_max(cs_rank(ts_corr(cs_rank(V), cs_rank(vwap), 5)), 5)

        # Alpha 51
        # quesval(-0.05, val_x, (-1) * (close - ts_delay(close, 1)), 1)
        features["alpha051"] = quesval(
            -0.05, 
            val_x, 
            -1 * (C - ts_delay(C, 1)), 
            1
        )

        # Alpha 52
        # (((-1) * ts_min(low, 5)) + ts_delay(ts_min(low, 5), 5)) * cs_rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) * ts_rank(volume, 5)
        a52_p1 = (-1 * ts_min(L, 5)) + ts_delay(ts_min(L, 5), 5)
        a52_p2 = cs_rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220.0)
        features["alpha052"] = a52_p1 * a52_p2 * ts_rank(V, 5)

        # Alpha 53
        # (-1) * ts_delta(((close - low) - (high - close)) / (close - low), 9)
        denom_53 = C - L
        term_53 = ((C - L) - (H - C)) / (denom_53 + 1e-8)
        features["alpha053"] = -1 * ts_delta(term_53, 9)

        # Alpha 54
        # ((-1) * ((low - close) * pow1(open, 5))) / ((low - high) * pow1(close, 5))
        num_54 = -1 * (L - C) * pow1(O, 5)
        den_54 = (L - H) * pow1(C, 5)
        features["alpha054"] = num_54 / (den_54 + 1e-8)

        # Alpha 55
        # (-1) * ts_corr(cs_rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), cs_rank(volume), 6)
        min_l_12 = ts_min(L, 12)
        max_h_12 = ts_max(H, 12)
        term_55 = (C - min_l_12) / (max_h_12 - min_l_12 + 1e-8)
        features["alpha055"] = -1 * ts_corr(cs_rank(term_55), cs_rank(V), 6)

        # Alpha 56: Skip (cap field missing)

        # Alpha 57
        # -1 * ((close - vwap) / ts_decay_linear(cs_rank(ts_argmax(close, 30)), 2))
        features["alpha057"] = -1 * ((C - vwap) / (ts_decay_linear(cs_rank(ts_argmax(C, 30)), 2) + 1e-8))

        # Alpha 58, 59: IndNeutralize, Skip

        # Alpha 60
        # - 1 * ((2 * cs_scale(cs_rank((((close - low) - (high - close)) / (high - low)) * volume))) - cs_scale(cs_rank(ts_argmax(close, 10))))
        term_60_a = (((C - L) - (H - C)) / (H - L + 1e-8)) * V
        term_60_b = ts_argmax(C, 10)
        features["alpha060"] = -1 * ((2 * cs_scale(cs_rank(term_60_a))) - cs_scale(cs_rank(term_60_b)))

        # Alpha 61
        # quesval2(cs_rank(vwap - ts_min(vwap, 16)), cs_rank(ts_corr(vwap, ts_mean(volume, 180), 18)), 1, 0)
        a61_th = cs_rank(vwap - ts_min(vwap, 16))
        a61_x = cs_rank(ts_corr(vwap, ts_mean(V, 180), 18))
        features["alpha061"] = quesval2(a61_th, a61_x, 1, 0)

        # Alpha 62
        # (cs_rank(ts_corr(vwap, ts_sum(ts_mean(volume, 20), 22), 10)) < cs_rank((cs_rank(open) + cs_rank(open)) < (cs_rank((high + low) / 2) + cs_rank(high)))) * -1
        a62_rhs_inner = (cs_rank(O) * 2) < (cs_rank((H + L)/2) + cs_rank(H))
        a62_rhs = cs_rank(a62_rhs_inner.float())
        a62_lhs = cs_rank(ts_corr(vwap, ts_sum(ts_mean(V, 20), 22), 10))
        features["alpha062"] = (a62_lhs < a62_rhs).float() * -1

        # Alpha 63: IndNeutralize, Skip

        # Alpha 64
        # (cs_rank(ts_corr(ts_sum(((open * 0.178404) + (low * (1 - 0.178404))), 13), ts_sum(ts_mean(volume, 120), 13), 17)) < cs_rank(ts_delta((((high + low) / 2 * 0.178404) + (vwap * (1 - 0.178404))), 4))) * -1
        w = 0.178404
        a64_term1 = ts_sum(O * w + L * (1-w), 13)
        a64_term2 = ts_sum(ts_mean(V, 120), 13)
        a64_lhs = cs_rank(ts_corr(a64_term1, a64_term2, 17))
        a64_term3 = ((H + L) / 2) * w + vwap * (1-w)
        a64_rhs = cs_rank(ts_delta(a64_term3, 4))
        features["alpha064"] = (a64_lhs < a64_rhs).float() * -1

        # Alpha 65
        # (cs_rank(ts_corr(((open * 0.00817205) + (vwap * (1 - 0.00817205))), ts_sum(ts_mean(volume, 60), 9), 6)) < cs_rank(open - ts_min(open, 14))) * -1
        w65 = 0.00817205
        a65_lhs = cs_rank(ts_corr(O * w65 + vwap * (1-w65), ts_sum(ts_mean(V, 60), 9), 6))
        a65_rhs = cs_rank(O - ts_min(O, 14))
        features["alpha065"] = (a65_lhs < a65_rhs).float() * -1

        # Alpha 66
        # (cs_rank(ts_decay_linear(ts_delta(vwap, 4), 7)) + ts_rank(ts_decay_linear((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2)), 11), 7)) * -1
        a66_term = (L - vwap) / (O - (H+L)/2 + 1e-8)
        features["alpha066"] = (cs_rank(ts_decay_linear(ts_delta(vwap, 4), 7)) + ts_rank(ts_decay_linear(a66_term, 11), 7)) * -1

        # Alpha 67: IndNeutralize, Skip

        # Alpha 68
        # (ts_rank(ts_corr(cs_rank(high), cs_rank(ts_mean(volume, 15)), 9), 14) < cs_rank(ts_delta((close * 0.518371 + low * (1 - 0.518371)), 1))) * -1
        w68 = 0.518371
        a68_lhs = ts_rank(ts_corr(cs_rank(H), cs_rank(ts_mean(V, 15)), 9), 14)
        a68_rhs = cs_rank(ts_delta(C * w68 + L * (1-w68), 1))
        features["alpha068"] = (a68_lhs < a68_rhs).float() * -1

        # Alpha 69, 70: IndNeutralize, Skip

        # Alpha 71
        # ts_greater(ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 3), ts_rank(ts_mean(volume, 180), 12), 18), 4), 16), ts_rank(ts_decay_linear(pow1(cs_rank((low + open) - (vwap + vwap)), 2), 16), 4))
        a71_lhs = ts_rank(ts_decay_linear(ts_corr(ts_rank(C, 3), ts_rank(ts_mean(V, 180), 12), 18), 4), 16)
        a71_rhs = ts_rank(ts_decay_linear(pow1(cs_rank((L+O) - 2*vwap), 2), 16), 4)
        features["alpha071"] = ts_greater(a71_lhs, a71_rhs)

        # Alpha 72
        # cs_rank(ts_decay_linear(ts_corr((high + low) / 2, ts_mean(volume, 40), 9), 10)) / cs_rank(ts_decay_linear(ts_corr(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))
        a72_num = cs_rank(ts_decay_linear(ts_corr((H+L)/2, ts_mean(V, 40), 9), 10))
        a72_den = cs_rank(ts_decay_linear(ts_corr(ts_rank(vwap, 4), ts_rank(V, 19), 7), 3))
        features["alpha072"] = a72_num / (a72_den + 1e-8)

        # Alpha 73
        # ts_greater(cs_rank(ts_decay_linear(ts_delta(vwap, 5), 3)), ts_rank(ts_decay_linear((ts_delta(open * 0.147155 + low * 0.852845, 2) / (open * 0.147155 + low * 0.852845)) * -1, 3), 17)) * -1
        w73 = 0.147155
        term73 = O * w73 + L * (1-w73)
        a73_lhs = cs_rank(ts_decay_linear(ts_delta(vwap, 5), 3))
        a73_rhs = ts_rank(ts_decay_linear((ts_delta(term73, 2) / (term73 + 1e-8)) * -1, 3), 17)
        features["alpha073"] = ts_greater(a73_lhs, a73_rhs) * -1

        # Alpha 74
        # quesval2(cs_rank(ts_corr(close, ts_sum(ts_mean(volume, 30), 37), 15)), cs_rank(ts_corr(cs_rank(high * 0.0261661 + vwap * 0.9738339), cs_rank(volume), 11)), 1, 0) * -1
        w74 = 0.0261661
        a74_th = cs_rank(ts_corr(C, ts_sum(ts_mean(V, 30), 37), 15))
        a74_x = cs_rank(ts_corr(cs_rank(H * w74 + vwap * (1-w74)), cs_rank(V), 11))
        features["alpha074"] = quesval2(a74_th, a74_x, 1, 0) * -1

        # Alpha 75
        # quesval2(cs_rank(ts_corr(vwap, volume, 4)), cs_rank(ts_corr(cs_rank(low), cs_rank(ts_mean(volume, 50)), 12)), 1, 0)
        a75_th = cs_rank(ts_corr(vwap, V, 4))
        a75_x = cs_rank(ts_corr(cs_rank(L), cs_rank(ts_mean(V, 50)), 12))
        features["alpha075"] = quesval2(a75_th, a75_x, 1, 0)

        # Alpha 76: IndNeutralize, Skip

        # Alpha 77
        # ts_less(cs_rank(ts_decay_linear((((high + low) / 2 + high) - (vwap + high)), 20)), cs_rank(ts_decay_linear(ts_corr((high + low) / 2, ts_mean(volume, 40), 3), 6)))
        a77_lhs = cs_rank(ts_decay_linear((H+L)/2 - vwap, 20))
        a77_rhs = cs_rank(ts_decay_linear(ts_corr((H+L)/2, ts_mean(V, 40), 3), 6))
        features["alpha077"] = ts_less(a77_lhs, a77_rhs)

        # Alpha 78
        # pow2(cs_rank(ts_corr(ts_sum((low * 0.352233) + (vwap * (1 - 0.352233)), 20), ts_sum(ts_mean(volume, 40), 20), 7)), cs_rank(ts_corr(cs_rank(vwap), cs_rank(volume), 6)))
        w78 = 0.352233
        a78_term1 = ts_sum(L * w78 + vwap * (1-w78), 20)
        a78_term2 = ts_sum(ts_mean(V, 40), 20)
        a78_base = cs_rank(ts_corr(a78_term1, a78_term2, 7))
        a78_exp = cs_rank(ts_corr(cs_rank(vwap), cs_rank(V), 6))
        features["alpha078"] = pow2(a78_base, a78_exp)

        # Alpha 79, 80: IndNeutralize, Skip

        # Alpha 81
        # quesval2(cs_rank(log(ts_product(cs_rank(pow1(cs_rank(ts_corr(vwap, ts_sum(ts_mean(volume, 10), 50), 8)), 4)), 15))), cs_rank(ts_corr(cs_rank(vwap), cs_rank(volume), 5)), 1, 0) * -1
        a81_corr = ts_corr(vwap, ts_sum(ts_mean(V, 10), 50), 8)
        a81_pow = pow1(cs_rank(a81_corr), 4)
        a81_prod = ts_product(cs_rank(a81_pow), 15)
        a81_th = cs_rank(torch.log(a81_prod + 1e-8))
        a81_x = cs_rank(ts_corr(cs_rank(vwap), cs_rank(V), 5))
        features["alpha081"] = quesval2(a81_th, a81_x, 1, 0) * -1

        # Alpha 82: IndNeutralize, Skip

        # Alpha 83
        # (cs_rank(ts_delay((high - low) / (ts_sum(close, 5) / 5), 2)) * cs_rank(cs_rank(volume))) / (((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close))
        term_83 = (H - L) / (ts_mean(C, 5) + 1e-8)
        a83_num = cs_rank(ts_delay(term_83, 2)) * cs_rank(cs_rank(V))
        a83_den = term_83 / (vwap - C + 1e-8)
        features["alpha083"] = a83_num / (a83_den + 1e-8)

        # Alpha 84
        # pow2(ts_rank(vwap - ts_max(vwap, 15), 21), ts_delta(close, 5))
        features["alpha084"] = pow2(ts_rank(vwap - ts_max(vwap, 15), 21), ts_delta(C, 5))

        # Alpha 85
        # pow2(cs_rank(ts_corr(high * 0.876703 + close * 0.123297, ts_mean(volume, 30), 10)), cs_rank(ts_corr(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7)))
        w85 = 0.876703
        a85_base = cs_rank(ts_corr(H * w85 + C * (1-w85), ts_mean(V, 30), 10))
        a85_exp = cs_rank(ts_corr(ts_rank((H+L)/2, 4), ts_rank(V, 10), 7))
        features["alpha085"] = pow2(a85_base, a85_exp)

        # Alpha 86
        # quesval2(ts_rank(ts_corr(close, ts_sum(ts_mean(volume, 20), 15), 6), 20), cs_rank((open + close) - (vwap + open)), 1, 0) * -1
        a86_th = ts_rank(ts_corr(C, ts_sum(ts_mean(V, 20), 15), 6), 20)
        a86_x = cs_rank(C - vwap)
        features["alpha086"] = quesval2(a86_th, a86_x, 1, 0) * -1

        # Alpha 87: IndNeutralize, Skip

        # Alpha 88
        # ts_less(cs_rank(ts_decay_linear((cs_rank(open) + cs_rank(low)) - (cs_rank(high) + cs_rank(close)), 8)), ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 8), ts_rank(ts_mean(volume, 60), 21), 8), 7), 3))
        a88_lhs = cs_rank(ts_decay_linear(cs_rank(O) + cs_rank(L) - cs_rank(H) - cs_rank(C), 8))
        a88_rhs = ts_rank(ts_decay_linear(ts_corr(ts_rank(C, 8), ts_rank(ts_mean(V, 60), 21), 8), 7), 3)
        features["alpha088"] = ts_less(a88_lhs, a88_rhs)

        # Alpha 89, 90, 91: IndNeutralize, Skip

        # Alpha 92
        # ts_less(ts_rank(ts_decay_linear(quesval2(((high + low) / 2 + close), (low + open), 1, 0), 15), 19), ts_rank(ts_decay_linear(ts_corr(cs_rank(low), cs_rank(ts_mean(volume, 30)), 8), 7), 7))
        a92_th = (H+L)/2 + C
        a92_x = L + O
        a92_lhs = ts_rank(ts_decay_linear(quesval2(a92_th, a92_x, 1, 0), 15), 19)
        a92_rhs = ts_rank(ts_decay_linear(ts_corr(cs_rank(L), cs_rank(ts_mean(V, 30)), 8), 7), 7)
        features["alpha092"] = ts_less(a92_lhs, a92_rhs)

        # Alpha 93: IndNeutralize, Skip

        # Alpha 94
        # pow2(cs_rank(vwap - ts_min(vwap, 12)), ts_rank(ts_corr(ts_rank(vwap, 20), ts_rank(ts_mean(volume, 60), 4), 18), 3)) * -1
        a94_base = cs_rank(vwap - ts_min(vwap, 12))
        a94_exp = ts_rank(ts_corr(ts_rank(vwap, 20), ts_rank(ts_mean(V, 60), 4), 18), 3)
        features["alpha094"] = pow2(a94_base, a94_exp) * -1

        # Alpha 95
        # quesval2(cs_rank(open - ts_min(open, 12)), ts_rank(pow1(cs_rank(ts_corr(ts_sum((high + low) / 2, 19), ts_sum(ts_mean(volume, 40), 19), 13)), 5), 12), 1, 0)
        a95_th = cs_rank(O - ts_min(O, 12))
        a95_corr = ts_corr(ts_sum((H+L)/2, 19), ts_sum(ts_mean(V, 40), 19), 13)
        a95_x = ts_rank(pow1(cs_rank(a95_corr), 5), 12)
        features["alpha095"] = quesval2(a95_th, a95_x, 1, 0)

        # Alpha 96
        # ts_greater(ts_rank(ts_decay_linear(ts_corr(cs_rank(vwap), cs_rank(volume), 4), 4), 8), ts_rank(ts_decay_linear(ts_argmax(ts_corr(ts_rank(close, 7), ts_rank(ts_mean(volume, 60), 4), 4), 13), 14), 13)) * -1
        a96_lhs = ts_rank(ts_decay_linear(ts_corr(cs_rank(vwap), cs_rank(V), 4), 4), 8)
        a96_corr2 = ts_corr(ts_rank(C, 7), ts_rank(ts_mean(V, 60), 4), 4)
        a96_rhs = ts_rank(ts_decay_linear(ts_argmax(a96_corr2, 13), 14), 13)
        features["alpha096"] = ts_greater(a96_lhs, a96_rhs) * -1

        # Alpha 97: IndNeutralize, Skip

        # Alpha 98
        # cs_rank(ts_decay_linear(ts_corr(vwap, ts_sum(ts_mean(volume, 5), 26), 5), 7)) - cs_rank(ts_decay_linear(ts_rank(ts_argmin(ts_corr(cs_rank(open), cs_rank(ts_mean(volume, 15)), 21), 9), 7), 8))
        a98_lhs = cs_rank(ts_decay_linear(ts_corr(vwap, ts_sum(ts_mean(V, 5), 26), 5), 7))
        a98_corr2 = ts_corr(cs_rank(O), cs_rank(ts_mean(V, 15)), 21)
        a98_rhs = cs_rank(ts_decay_linear(ts_rank(ts_argmin(a98_corr2, 9), 7), 8))
        features["alpha098"] = a98_lhs - a98_rhs

        # Alpha 99
        # quesval2(cs_rank(ts_corr(ts_sum((high + low) / 2, 20), ts_sum(ts_mean(volume, 60), 20), 9)), cs_rank(ts_corr(low, volume, 6)), 1, 0) * -1
        a99_th = cs_rank(ts_corr(ts_sum((H+L)/2, 20), ts_sum(ts_mean(V, 60), 20), 9))
        a99_x = cs_rank(ts_corr(L, V, 6))
        features["alpha099"] = quesval2(a99_th, a99_x, 1, 0) * -1

        # Alpha 100: IndNeutralize, Skip

        # Alpha 101
        # ((close - open) / ((high - low) + 0.001))
        features["alpha101"] = (C - O) / (H - L + 1e-3)


        # Volatility
        # ret_1 = C / ts_delay(C, 1) - 1
        # features["volatility_20d"] = ts_std(ret_1, 20)
        # features["volatility_60d"] = ts_std(ret_1, 60)
        
        # Inverse Volatility (Longer term - 60d)
        # Low beta/volatility stocks tend to outperform in bear/stable markets.
        # features["inv_vol_60"] = 1.0 / (features["volatility_60d"] + 1e-4)
        
        # features["atr_ratio_14"] = ta_atr(H, L, C, 14) / C
        # features["daily_range"] = H / L - 1


        # Label
        # ts_delay(close, -5) / close - 1 (Forward returns for next 5 days, consistent with v3)
        # Note: v3 used -5. Alpha101 dataset used -3/-1 - 1.
        # User asked to "migrate alpha101". The dataset has self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")
        # which means Close(t+3)/Close(t+1) - 1. (Return from T+1 to T+3).
        # v3 uses simple 5 day forward return: Close(t+5)/Close(t) - 1.
        # I will stick to a standard forward return for now or use the one from v3 as default label.
        features["label"] = ts_delay(C, -5) / C - 1

        return features
