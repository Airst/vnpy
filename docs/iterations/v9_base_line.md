# V9 基线文档

> 创建日期：2026-04-11
> 代码基础：V8 完整因子 + 3 项验证有效的改动
> 用途：V9 版本所有后续迭代的对照基线

---

## 一、代码构成

V9 基线 = V8 完整因子集（~100 因子） + 以下 3 项经过验证的改动：

| 改动 | 来源 | 验证结果 |
|:---|:---|:---|
| Phase 1: Beta-neutral 标签 | v9_reform_plan.md | 非牛市年化 5% -> 21% |
| Phase 3: 简化 dragon_score | v9_reform_plan.md | 总收益+54%, Sharpe+0.16 |
| Phase 4: turnover_x_bull 交互因子 | v9_reform_plan.md | IC 0.317, 牛市增强 |

### 与 V8 的具体差异

**1. 标签（Phase 1）**

V8:
```python
raw_ret_5 = ts_delay(C, -5) / C - 1
features["label"] = cs_rank(raw_ret_5)
```

V9:
```python
raw_ret_5 = ts_delay(C, -5) / C - 1
mkt_ret_5 = torch.nanmean(raw_ret_5, dim=0, keepdim=True)
excess_ret_5 = raw_ret_5 - features["beta_20d"] * mkt_ret_5
features["label"] = cs_rank(excess_ret_5)
```

**2. Dragon Score（Phase 3）**

V8 第 563-698 行包含三态 regime（Bull/Bear/Chaos）混合逻辑、7 个硬编码惩罚项。

V9 替换为简化版（V6 原始逻辑）：
```python
combined_mom = cs_rank(features["mom_20d"]) * 0.7 + cs_rank(features["mom_60d"]) * 0.3
rank_turnover = cs_rank(features["turnover_mean_20d"])
features["dragon_score"] = combined_mom + rank_turnover * torch.tanh(features["mom_20d"] * 5.0)
```

bear_trap_score 保留为原子因子（不再作为 dragon_score 的惩罚项）。

**3. 新增因子（Phase 4）**

```python
features["turnover_x_bull"] = features["rel_turnover_20d"] * bull_prob
```

---

## 二、因子清单（99 个）

### 动量/反转 (12)
rev_5d, mom_5d, mom_20d, mom_60d, mom_120d, ma_bias_120, price_zscore_20d,
ret_overnight, ret_intraday, bias_5, bias_10, bias_20, bias_60

### 行业 (10)
ind_mom_60d, ind_mom_20d, ind_mom_5d, ind_rel_mom_60d, ind_rel_mom_20d,
ind_pe, ind_rel_pe, ind_rel_turnover_20d, ind_rel_vol_20d, ind_rel_bias_20

### 波动率 (14)
volatility_20d, volatility_60d, volatility_120d, atr_ratio_14,
max_ret_20d, min_ret_20d, daily_range, downside_vol_20d, vol_skew_20,
inv_vol_60, std_5, std_10, std_20, vol_ratio_5_20

### 市场风格 (2)
beta_20d, resid_vol_20d

### 趋势 (5)
trend_rsquare_20, trend_slope_20, trend_sharpe_20, trend_slope_5, slope_div_5_20

### 技术指标 (10)
bollinger_position, boll_width_20, rsi_14, psy_12, drawdown_20d, rebound_20d,
kdj_kd_diff, kdj_kd_velocity, tech_cci_14, alpha040

### 量/换手 (12)
volume_ratio, vol_cv_20, vol_stability_20, turnover_cv_20d, illiquidity_20d,
vwap_dev_20, turnover_mean_5d, turnover_mean_20d, turnover_std_20d,
fund_turnover_growth, price_vol_corr_20, turnover_ratio_5_20

### 基本面 (7)
ep_ratio, val_pb, val_ps, val_dv, size_ln_cap, pe_zscore_60d, pe_rank_change_20d

### 支撑/压力 (6)
dist_support_20, dist_support_60, dist_pressure_20, dist_pressure_60,
rr_ratio_20, rr_ratio_60

### 统计 (5)
qtld_60, klen, min_10, min_20, min_30

### 概念板块 (7)
con_mom_5d, con_mom_20d, con_mom_20d_max, con_turnover_20d,
rel_con_mom_20d, rel_con_mom_max_20d, con_divergence_20d

### 组合/交互 (9)
rel_turnover_20d, mom_x_mkt, tech_reversal, bear_reversal,
dragon_score, inv_vol_20, vol_penalty,
head_lift_signal, camel_hump_score,
resonance_signal, meta_bull_prob, inter_res_bull, inter_camel_bear,
zt_count_20d, bear_trap_score, vol_range_20d, turnover_x_bull

---

## 三、首次训练回测结果

> 训练日期：2026-04-11
> 回测区间：2022-01-04 ~ 2026-04-10

### 整体指标

| 指标 | 值 |
|:---|:---|
| 总收益 | 116.83% |
| 年化 | 27.17% |
| Sharpe | 0.73 |
| MaxDD | -29.24% |
| 最长回撤天数 | 221 |
| 收益回撤比 | 2.64 |
| 盈利交易日 | 553 / 1032 (53.6%) |
| 因子数 | 99 |

### 90 天区间收益

| 区间 | 收益 | 备注 |
|:---|:---|:---|
| 2022-01 ~ 2022-04 | -16.38% | 开局回撤 |
| 2022-04 ~ 2022-07 | +1.97% | |
| 2022-07 ~ 2022-10 | +1.89% | |
| 2022-10 ~ 2023-01 | +18.14% | |
| 2023-01 ~ 2023-04 | +4.53% | |
| 2023-04 ~ 2023-07 | **+20.59%** | 非牛市强势 |
| 2023-07 ~ 2023-10 | +4.97% | |
| 2023-10 ~ 2024-01 | +8.05% | |
| 2024-01 ~ 2024-04 | -5.66% | |
| 2024-04 ~ 2024-07 | -14.97% | 国九条冲击 |
| 2024-07 ~ 2024-10 | +21.08% | 牛市启动 |
| 2024-10 ~ 2025-01 | +11.80% | |
| 2025-01 ~ 2025-04 | -3.62% | |
| 2025-04 ~ 2025-07 | +35.57% | 牛市高峰 |
| 2025-07 ~ 2025-10 | +6.20% | |
| 2025-10 ~ 2026-01 | +7.32% | |
| 2026-01 ~ 2026-04 | -7.51% | 地缘+关税冲击 |

### Top 5 因子 IC

| 因子 | Overall IC | IC_IR |
|:---|:---|:---|
| rel_turnover_20d | 0.321 | 1.98 |
| turnover_mean_20d | 0.319 | 1.96 |
| turnover_x_bull | 0.317 | 1.94 |
| turnover_std_20d | 0.291 | 1.96 |
| turnover_mean_5d | 0.289 | 1.87 |

---

## 四、训练方差说明

MLP 模型对随机种子和训练窗口边界敏感。以下是同一代码（或接近版本）的多次训练结果记录：

| 日期 | 总收益 | 年化 | Sharpe | MaxDD | 备注 |
|:---|:---|:---|:---|:---|:---|
| 2026-04-10 23:59 | 136.53% | 31.75% | 0.86 | -29.58% | 69因子版(删减后) |
| 2026-04-11 09:38 | 18.52% | 4.31% | 0.15 | -62.43% | 69因子版(删减后) |
| 2026-04-11 12:34 | 116.83% | 27.17% | 0.73 | -29.24% | **V9基线(99因子)** |

**结论**：单次回测结论需谨慎，后续迭代应关注多次训练的中位数表现。

---

## 五、已验证失败的方向（不再重复尝试）

| 方向 | 实验 | 结论 |
|:---|:---|:---|
| 质量因子 | ROE/毛利率/ROA 等 9 因子 | 季频数据与日频窗口不匹配，净负面 |
| 资金流因子 | 大/中/小/超大单 10 因子 | 全部 IC < 0.03，纯噪声 |
| Regime 原子因子 | mkt_trend/mkt_vol_regime/mkt_breadth_20d/mkt_dispersion_20d | IC ~0.017，作为独立特征是噪声 |
| value_x_bear 交互 | val_dv * bear_signal | val_dv 在所有时段均为负 IC，底层因子无效 |
| K线微结构 | klow_2_20d (下影线占比) | IC -0.061，反向，总收益 -117% |
| 量价同步性 | cord_20 (ret-vol 相关) | IC 0.061，总收益 -109% |
| 上涨占比 | sump_20 (正向波动比) | IC 0.011，总收益 -186% |
| 板块相对活跃度 | rel_concept_turnover | IC 0.294 但总收益 -161% |
