# V9 改造方案：从 Beta 策略到 Alpha 策略

> 创建日期：2026-04-06
> 基线：V8 (v8_factor_calculator.py)
> 目标：降低对牛市的依赖，在非牛市中产生正收益

---

## 一、V8 诊断：三个结构性问题

### 问题 1：标签奖励 Beta，不是 Alpha

`v8_factor_calculator.py:701-707`：

```python
raw_ret_5 = ts_delay(C, -5) / C - 1
features["label"] = cs_rank(raw_ret_5)
```

`cs_rank(raw_ret_5)` 看似是截面中性的（排名不受市场整体涨跌影响）。但它并没有去除 Beta 效应。

| 股票 | Beta | 未来5日市场+10% | 原始收益 | cs_rank | 超额收益(ret-beta*mkt) | 超额rank |
|:---|:---|:---|:---|:---|:---|:---|
| A | 2.0 | +10% | +22% | 第1 | +2% | 第2 |
| B | 0.5 | +10% | +8% | 第3 | **+3%** | **第1** |
| C | 1.5 | +10% | +15% | 第2 | 0% | 第3 |

在牛市中，`cs_rank(raw_ret_5)` 持续把高 Beta 股票排到最高。模型学到的规律是"高 Beta + 高动量 -> 高排名"。这个规律在牛市正确，但在震荡/熊市中完全反转。

**股票 B 有真正的 Alpha（+3%），但在原始标签中只排第三。模型永远不会学到去选它。**

### 问题 2：10 个财务因子已加载但完全未使用

`data_columns_info.txt` 中显示，`padded_raw` 张量包含以下财务数据列：

| 列索引 | 字段 | 含义 | V8 是否使用 |
|:---|:---|:---|:---|
| 30 | gross_margin | 毛利率 | 未使用 |
| 31 | netprofit_margin | 净利率 | 未使用 |
| 32 | roe | 净资产收益率 | 未使用 |
| 33 | roa | 总资产收益率 | 未使用 |
| 34 | roic | 投入资本回报率 | 未使用 |
| 35 | netprofit_yoy | 净利润同比增速 | 未使用 |
| 36 | tr_yoy | 营收同比增速 | 未使用 |
| 37 | current_ratio | 流动比率 | 未使用 |
| 38 | quick_ratio | 速动比率 | 未使用 |
| 39 | assets_turn | 资产周转率 | 未使用 |

这些是经典的"质量因子"（Quality Factors）。学术研究和实践中，质量因子（高 ROE、高毛利率、盈利稳定性）是与动量因子相关性最低的 alpha 来源之一，在非趋势市场中表现尤其稳健。

系统加载了这些数据（`data_loader.py` 通过 `join_asof` 正确加载了财务指标），但 V8 的 `build_features` 完全没有从中提取因子。

### 问题 3：dragon_score 硬编码了市场观点

`v8_factor_calculator.py:563-698`，dragon_score 的构建包含：

- 三态 regime 概率：`prob_bull`, `prob_bear`, `prob_chaos`（基于 sigmoid 硬编码阈值）
- 策略混合权重：`score_bull * prob_bull + score_bear * prob_bear + score_chaos * prob_chaos`
- 动态换手率权重：`turnover_weight = 0.5 + prob_chaos * 0.3`
- 7 个硬编码惩罚/奖励项：overheat, ceiling, bear_trap, pullback_trap, ST, dead_flat, inflection_bonus

这直接违反了 AGENTS.md 中的原则："不要使用复杂的参数来适应某种特定的市场风格，避免过拟合"。

这些硬编码的 regime 逻辑替代了 MLP 应该自己学习的东西。参数（如 `* 15.0`, `* 100.0`, `0.015` 等）基于历史回测拟合，在样本外可能失效。

---

## 二、V8 回测数据支撑

### 收益时间分布（V8 最新一次回测）

```
Pre-bull (2022-01 ~ 2024-08):  +14.31%  / 2.7年  = 年化 5.39%
Bull     (2024-09 ~ 2025-02):  +74.89%  / 6个月
Post-bull(2025-03 ~ 2026-04):  +33.87%  / 13个月 (惯性消退中)
2026 Q1:                       -12.02%  / 3个月  (已转负)
```

| 阶段 | 收益 | 本质 |
|:---|:---|:---|
| 2022 H1: +23% | 好 | 但2022上半年A股也有结构性反弹 |
| 2022 H2: +10% | 尚可 | |
| 2023 全年: +12% | 平庸 | 年化约12%，仅略好于余额宝 |
| 2024 H1: -20.6% | 灾难 | 模型选出的股票大幅跑输 |
| 2024 Jul-Aug: -5.2% | 继续亏 | |
| 2024 Sep-Dec: +60% | 牛市红利 | |
| 2025: +58% | 牛市惯性 | |
| 2026 Q1: -12% | 打回原形 | |

**结论：去掉牛市区间，模型年化约 5-12%，扣除交易成本后接近零 alpha。**

### 训练不稳定性

```
AGENTS.md 记录:  379%, Sharpe 1.48, 最大回撤 -26.96%
最新回测:        167%, Sharpe 0.95, 最大回撤 -35.28%
```

同一 V8 代码，两次训练差距超过 2 倍。模型对随机种子/训练窗口边界极度敏感。

---

## 三、V9 失败复盘

| 失败版本 | 做了什么 | 为什么失败 |
|:---|:---|:---|
| V9 (12因子) | 一次加入 12 个新因子 | 11/12 因子 IC<0.06，是噪声，稀释模型容量 |
| V9.5~V9.8 | label 工程 + 资金流因子 + 策略层 | 多个变量同时改动，无法隔离原因 |
| V9 策略层 | 基于 model output 做 regime 自适应 | 循环论证，参数过于激进 |

**铁律：每次只改一个变量，验证后再叠加下一个。**

---

## 四、改造方案：三阶段

### 阶段 1：标签从 "原始收益排名" 改为 "超额收益排名"（最高优先级）

**改什么**：修改 label 构造，从 `cs_rank(raw_ret_5)` 改为 `cs_rank(raw_ret_5 - beta * market_ret_5)`

**为什么有效**：
- 去除 Beta 对标签的污染，模型被迫学习 Alpha 而非 Beta
- 在牛市中：模型选出真正跑赢大盘的股票，而非高 Beta 搭便车的股票
- 在非牛市中：高 Beta 股票的超额收益为负，模型自然会避开它们
- 不改任何因子，不改任何策略参数，最小化变量

**具体实现**：

```python
# 当前 V8 标签 (v8_factor_calculator.py:700-707)
raw_ret_5 = ts_delay(C, -5) / C - 1
features["label"] = cs_rank(raw_ret_5)

# 改为：Beta-neutral 超额收益排名
raw_ret_5 = ts_delay(C, -5) / C - 1
mkt_ret_5 = torch.nanmean(raw_ret_5, dim=0, keepdim=True)  # 市场平均5日收益
excess_ret_5 = raw_ret_5 - features["beta_20d"] * mkt_ret_5  # 去除 Beta 成分
features["label"] = cs_rank(excess_ret_5)
```

注意：`beta_20d` 已经在 V8 第 110 行计算好了，`mkt_ret_5` 可直接通过截面均值算出。改动量极小。

**预期效果**：
- 牛市收益下降（不再靠 Beta 搭便车）
- 非牛市收益显著改善（模型不再在震荡市买入高 Beta 票）
- 整体 Sharpe 更稳定，但总收益可能低于 V8 在幸运种子下的结果

**验证**：
- 全区间回测 (2022-01 ~ 2026-04)
- 重点对比 Pre-bull 区间 (2022-01 ~ 2024-08) 的年化收益
- 通过标准：非牛市年化 > 10%（当前约 5%）

**实验结果 (2026-04-06)**：
- 总收益 192.09%, 年化 44.84%, Sharpe 1.0556, MaxDD -25.15%
- Pre-bull 年化: 5.39% -> **21.10%** (4倍改善)
- Bull 贡献: 44.7% -> **29.3%** (降低 15%)
- **阶段 1 通过**

---

### 阶段 2：添加质量因子（已失败，已回退）

**前提**：阶段 1 验证通过后再做。

**改什么**：从 `padded_raw` 中提取 `col_map` 已有的 10 个财务字段，计算质量因子。

**具体因子**（全部是学术界验证过的经典因子）：

```python
ROE = padded_raw[:, :, col_map['roe']]
ROA = padded_raw[:, :, col_map['roa']]
GM  = padded_raw[:, :, col_map['gross_margin']]
NPM = padded_raw[:, :, col_map['netprofit_margin']]
NP_YOY = padded_raw[:, :, col_map['netprofit_yoy']]
TR_YOY = padded_raw[:, :, col_map['tr_yoy']]
CR  = padded_raw[:, :, col_map['current_ratio']]
AT  = padded_raw[:, :, col_map['assets_turn']]

# 盈利能力 (Profitability)
features["quality_roe"] = ROE
features["quality_roa"] = ROA
features["quality_gm"]  = GM
features["quality_npm"] = NPM

# 成长性 (Growth)
features["growth_np_yoy"] = NP_YOY
features["growth_tr_yoy"] = TR_YOY

# 盈利稳定性 (Earnings Stability)
features["earnings_stability"] = 1.0 / (ts_std(ROE, 60) + 1e-4)

# 财务健康 (Financial Health)
features["fin_health_cr"] = CR

# 经营效率 (Operational Efficiency)
features["efficiency_at"] = AT
```

**为什么这些因子能帮到非牛市**：
- 质量因子（ROE、毛利率）在 A 股的研究中长期有效，IC 通常在 0.03-0.08 之间
- 关键特点：与动量因子的相关性极低（<0.1），提供真正独立的信息源
- 在震荡/熊市中，资金倾向流入高质量公司（避险逻辑），质量因子表现最好的时候恰好是动量因子失效的时候

**节奏控制**：先计算每个因子的 IC，只保留 |IC| > 0.03 的因子。遵循 V9 的教训——不添加噪声因子。

**实验结果 (2026-04-06)**：
- 9 因子版本：总收益 159.90%, Sharpe 0.91, MaxDD -21.79%, Pre-bull Ann 11.87%
  - 非牛市防御大幅改善 (2024H1: -20.6% -> +1.35%)，但整体 Sharpe 下降
- IC 分析结果：9 因子中仅 4 个 |IC| >= 0.03:
  - quality_gm (IC -0.091), quality_npm (IC -0.054), growth_np_yoy (IC 0.036), growth_tr_yoy (IC 0.030)
  - 5 个被裁剪：quality_roe/fin_health_cr/efficiency_at (不在 col_map)，quality_roa (IC 0.008)，earnings_stability (IC 0.013)
- 修剪后 4 因子版本：总收益 153.61%, Sharpe 0.90, MaxDD -32.51%, Pre-bull Ann **-5.14%** — 比 V8 还差
- **结论：质量因子对 MLP 是净负面的。**
  - 可能原因：财务数据季频更新，与日频模型重训窗口 (90天) 不匹配，在大部分时间为常数，增加噪声
- **阶段 2 未通过，代码已回退。V9 保持 Phase 1 状态。**

---

### 阶段 3：精简 dragon_score（移除硬编码 regime 逻辑）

**前提**：阶段 1+2 验证通过后再做。这是风险最高的改动。

**改什么**：
- **保留** 所有原子因子（mom_20d, turnover_mean_20d, rsi_14, camel_hump_score 等）
- **移除** dragon_score 中的三态 regime 混合逻辑（prob_bull/bear/chaos 权重分配）和 7 个硬编码惩罚项
- **保留** dragon_score 作为简化版本：`dragon_score = combined_mom + rank_turnover * tanh(mom_20d * 5.0)`（V6 原始版本，不含 regime 逻辑）

**为什么**：
- MLP 有 80+ 个原子因子作为输入，有足够信息自己学习 regime 切换
- 硬编码的 regime 逻辑相当于给 MLP "喂答案"，但这个答案是基于历史拟合的，样本外不一定正确
- 让 MLP 自己从 beta_20d, volatility_20d, meta_bull_prob 等原子因子中学习 regime 交互

**验证**：
- 回测对比
- 通过标准：总收益可接受的下降换取更低的最大回撤和更稳的 Sharpe

**实验结果 (2026-04-06)**：
- 总收益 245.91%, 年化 57.41%, Sharpe 1.22, MaxDD -24.59%, 收益回撤比 2.91
- 与 Phase 1 基线对比：总收益 +54%, 年化 +12.6%, Sharpe +0.16, MaxDD 改善
- 90天区间表现：
  - 非牛市多数季度为正，2024-04~07 亏损 -12.19%
  - 牛市期间收益依然强劲 (2024Q4: +29.92%, 2025Q2: +39.15%)
  - 最近弱势期 2026Q1: -8.29%
- **阶段 3 通过。移除硬编码 regime 逻辑后 MLP 自主学习 regime 交互，全面优于 Phase 1。**

---

## 五、执行节奏

```
阶段 1（标签改造）
  |-- 改动行数：~5 行
  |-- 风险：低（不改因子，不改策略）
  |-- 验证：全区间回测，重点对比 2022-2024H1 非牛市区间
  |-- 通过标准：非牛市年化 > 10%（当前约 5%）

阶段 2（质量因子）
  |-- 改动行数：~20 行
  |-- 风险：中（新增因子可能是噪声）
  |-- 验证：先跑 IC 分析，>0.03 才加入；然后回测
  |-- 通过标准：Sharpe 不降（最好上升），非牛市收益进一步改善

阶段 3（精简 regime 逻辑）
  |-- 改动行数：~60 行删除
  |-- 风险：高（dragon_score 可能是当前牛市收益的核心来源）
  |-- 验证：回测对比
  |-- 通过标准：总收益可接受的下降换取更低的最大回撤和更稳的 Sharpe
```

---

## 六、V8 完整因子清单（参考）

### 动量/反转 (Momentum/Reversal) — 12 个
rev_5d, mom_5d, mom_20d, mom_60d, mom_120d, ma_bias_120, price_zscore_20d,
ret_overnight, ret_intraday, bias_5, bias_10, bias_20, bias_60

### 行业因子 (Industry) — 9 个
ind_mom_60d, ind_mom_20d, ind_mom_5d, ind_rel_mom_60d, ind_rel_mom_20d,
ind_pe, ind_rel_pe, ind_rel_turnover_20d, ind_rel_vol_20d, ind_rel_bias_20

### 波动率 (Volatility) — 14 个
volatility_20d, volatility_60d, volatility_120d, atr_ratio_14,
max_ret_20d, min_ret_20d, daily_range, downside_vol_20d, vol_skew_20,
inv_vol_60, std_5, std_10, std_20, vol_ratio_5_20

### 市场风格 (Market Style) — 2 个
beta_20d, resid_vol_20d

### 趋势 (Trend) — 5 个
trend_rsquare_20, trend_slope_20, trend_sharpe_20, trend_slope_5, slope_div_5_20

### 技术指标 (Technical) — 10 个
bollinger_position, boll_width_20, rsi_14, psy_12, drawdown_20d, rebound_20d,
kdj_kd_diff, kdj_kd_velocity, tech_cci_14, alpha040

### 量/换手 (Volume/Turnover) — 12 个
volume_ratio, vol_cv_20, vol_stability_20, turnover_cv_20d, illiquidity_20d,
vwap_dev_20, turnover_mean_5d, turnover_mean_20d, turnover_std_20d,
fund_turnover_growth, price_vol_corr_20, rel_turnover_20d, turnover_ratio_5_20

### 基本面 (Fundamental) — 7 个
ep_ratio, val_pb, val_ps, val_dv, size_ln_cap, pe_zscore_60d, pe_rank_change_20d

### 支撑/压力 (Support/Resistance) — 6 个
dist_support_20, dist_support_60, dist_pressure_20, dist_pressure_60,
rr_ratio_20, rr_ratio_60

### 统计 (Statistical) — 5 个
qtld_60, klen, min_10, min_20, min_30

### 概念板块 (Concept) — 7 个
con_mom_5d, con_mom_20d, con_mom_20d_max, con_turnover_20d,
rel_con_mom_20d, rel_con_mom_max_20d, con_divergence_20d

### 组合/交互 (Composite) — 14 个
mom_x_mkt, tech_reversal, bear_reversal, head_lift_signal, camel_hump_score,
resonance_signal, inter_res_bull, inter_camel_bear, meta_bull_prob,
dragon_score, vol_range_20d, bear_trap_score, zt_count_20d, inv_vol_20, vol_penalty

---

## 七、V9 Phase 4 失败复盘：资金流因子

**实验日期**：2026-04-09

**改什么**：从 `moneyflow_manager` 加载大/中/小/超大单资金流数据，计算 10 个资金流因子：

| 因子 | 含义 | Overall IC |
|:---|:---|:---|
| mf_net_lg_ratio | 大单净流入占比 | -0.018 |
| mf_net_elg_ratio | 超大单净流入占比 | -0.014 |
| mf_smart_ratio | 主力买卖力量比 | -0.015 |
| mf_retail_pressure | 散户净买入(反向) | 0.020 |
| mf_concentration | 大单成交占比 | -0.003 |
| mf_net_lg_20d | 20日主力流入均值 | (衍生) |
| mf_net_lg_5d | 5日主力流入均值 | -0.025 |
| mf_momentum | 资金流加速度 | -0.006 |
| mf_divergence | 量价背离 | -0.021 |
| mf_persistence | 连续流入天数占比 | -0.032 |

**回测结果**：

| 指标 | Phase 1+3 基线 | +资金流因子 | 变化 |
|:---|:---|:---|:---|
| 总收益 | 249.58% | 100.81% | **-148.77%** |
| 年化 | 58.27% | 23.47% | -34.80% |
| Sharpe | 1.23 | 0.66 | -0.57 |
| MaxDD | -22.59% | -27.52% | 恶化 |

**结论**：全部 10 个资金流因子 IC 均低于 0.03 门槛，属于纯噪声。一次加入 10 个噪声因子导致模型严重退化。

**失败原因**：
1. A 股资金流数据噪声大，大小单划分标准各平台不统一
2. 资金流是同步指标而非领先指标——反映当日交易行为，不预测未来收益
3. 一次加入过多低 IC 因子，放大噪声效应，挤占有效因子权重

**代码已回退**：仅回退 `v9_factor_calculator.py` 中的因子计算，`data_loader.py` 中的数据加载保留（不影响模型）。

---

## 八、最大回撤分析与改进方向

### 8.1 两段主要回撤

| | 最大回撤 | 次大回撤 |
|:---|:---|:---|
| **区间** | 2026-02-26 ~ 2026-04-07 | 2024-01 ~ 2024-07 |
| **峰值** | 3,440,656 | 1,498,835 |
| **谷值** | 2,681,478 | 1,173,233 |
| **幅度** | -22.06% (绝对 -759K) | -21.72% (绝对 -326K) |
| **持续** | ~28 个交易日，尚未恢复 | ~120+ 个交易日 |

### 8.2 回撤期间的重大事件

**2026-02-26 ~ 2026-04-07（最大回撤）：**

| 日期 | 事件 | 市场影响 |
|:---|:---|:---|
| 3月3日 | 中东危机升温，能源股飙升 | 科技股遭"抽血式大跌"，A股全线普跌 |
| 3月17日 | 中东紧张 + 美联储降息预期推迟 | 超4500只个股下跌 |
| 3月23日 | "黑色星期一"，亚太全线崩盘 | A股超5100只下跌，单日蒸发4万亿 |
| 3月底~4月2日 | 特朗普签署关税公告，对华加征关税升级 | 贸易战恐慌叠加前期获利盘回吐 |

**2024-04 ~ 2024-07（次大回撤）：**

| 日期 | 事件 | 市场影响 |
|:---|:---|:---|
| 4月12日 | 新"国九条"发布，严格退市制度 | 微盘股指数两日暴跌近20% |
| 4-6月 | 量化微盘策略连环踩踏 | DMA平仓、小盘股流动性枯竭 |
| 持续 | 垃圾股/ST集中打压 | 高换手率小盘股成重灾区 |
| 持续 | 经济数据疲软，消费/地产低迷 | 市场阴跌磨底 |

### 8.3 模型失效根因分析

两段回撤有共同特征：**系统性外生冲击导致截面 alpha 消失**。

1. **齐跌行情中截面区分度崩溃**：地缘/关税冲击下所有股票同向下跌，"谁比谁好"的区分度急剧下降。模型选出的"相对好"的股票，绝对收益仍为负。

2. **风格急转的惯性错配**：模型训练窗口（90天）学到"最近牛市中换手率高=好"。市场突然转熊时，高换手率股票反而跌最凶（流动性好=被抛售最快）。

3. **现有 regime 信号无效**：`meta_bull_prob`、`inter_res_bull`、`inter_camel_bear` 的 IC 极低（0.006 ~ 0.037），MLP 未能有效学会 regime switching。

### 8.4 核心矛盾：为什么加入价值/资金流因子会让牛市变差

> MLP 是一个无条件的截面打分器——它给每个因子一个相对固定的权重，无法根据市场状态动态调整。

- 加入价值因子 -> 模型在训练集中发现"熊市中低PE跑赢" -> 给 EP_ratio 分配正权重
- 但同一权重在牛市中选出低PE的银行/地产，错过高换手的成长股
- 结果：熊市小赚，牛市大亏，净效果为负

**这不是因子本身无效，而是因子的有效性是 regime-dependent 的**。需要让模型知道"什么时候用什么因子"。

---

## 九、Phase 4（新）：Regime-Conditioned 交互因子

### 9.1 核心思路

不让 MLP 自己"发现" regime（高维空间中二阶交互太难自动学习），而是**显式构建"因子 x 市场状态"的交互项**，降低 MLP 的学习难度。

等价于将隐式模型：
```
y = f(turnover, value, volatility, ...)
```
升级为显式条件模型：
```
y = f(turnover*bull, turnover*bear, value*bull, value*bear, vol*highvol, ...)
```

MLP 理论上能自己学交互，但 100 个因子的高维空间中很难从稀疏信号中发现二阶关系。显式交互项 = feature engineering 降低学习门槛。

### 9.2 实施步骤（严格单变量迭代）

#### Step 1：增强 Regime 原子因子

当前 `meta_bull_prob` 仅基于 `sigmoid(mean(ret_20d) * 15)` ，维度单一、信号弱。增加以下市场状态因子：

```python
# 市场趋势强度（连续值 [-1, 1]）
mkt_ret_20d = torch.nanmean(ret_20d_all_stocks, dim=0, keepdim=True)
features["mkt_trend"] = torch.tanh(mkt_ret_20d * 50.0)

# 市场波动率状态变化（当前波动 vs 长期波动，>0.5 = 波动放大）
mkt_vol_60d = torch.nanmean(features["volatility_60d"], dim=0, keepdim=True)
features["mkt_vol_regime"] = torch.sigmoid((mkt_vol_20d - mkt_vol_60d) * 100.0)

# 市场宽度（上涨家数占比的 20 日均值）
breadth = (ret_1 > 0).float().mean(dim=0, keepdim=True)
features["mkt_breadth_20d"] = ts_mean(breadth, 20)

# 截面离散度（个股收益的截面标准差 -> alpha 环境质量）
cross_dispersion = ret_1.std(dim=0, keepdim=True)
features["mkt_dispersion_20d"] = ts_mean(cross_dispersion, 20)
```

**这些因子不直接预测个股收益（IC 可能低），但它们的价值在于作为交互项的"调制器"。**

**验证**：
- 计算每个 regime 因子的 IC
- 即使 IC < 0.03 也可保留——因为它们的核心用途不是独立预测，而是调制其他因子
- 但关注加入后模型是否退化：对比基线 Sharpe 和总收益

#### Step 2：核心因子的 Regime 交互项

仅对 IC 排名 Top-5 的因子做交互（避免因子爆炸）：

```python
# 提取 regime 信号
bull_signal = features["mkt_trend"].clamp(min=0)       # [0, 1] 牛市强度
bear_signal = (-features["mkt_trend"]).clamp(min=0)     # [0, 1] 熊市强度
high_vol = features["mkt_vol_regime"]                    # [0, 1] 高波动程度

# 换手率：牛市更有效（动量追涨逻辑）
features["turnover_x_bull"] = features["rel_turnover_20d"] * bull_signal

# 换手率：高波动时风险放大（可能选反）
features["turnover_x_highvol"] = features["rel_turnover_20d"] * high_vol

# 价值因子：熊市/低波动时更有效（避险逻辑）
features["value_x_bear"] = features["val_dv"] * bear_signal

# 波动率因子：高波动 regime 下区分度更强
features["vol_x_regime"] = features["volatility_20d"] * high_vol

# 截面离散度：低离散度时所有因子都不可靠，用于"降低下注"
features["turnover_x_dispersion"] = features["rel_turnover_20d"] * features["mkt_dispersion_20d"]
```

**验证**：
- 每次只加 1 个交互因子，对比基线
- 通过标准：Sharpe 不降，且回撤期亏损收窄
- 不通过立即回退

#### Step 3：验证价值因子条件化后的效果

之前直接加 `val_dv` 导致牛市变差。现在只加条件化版本 `value_x_bear`，不加独立的 `val_dv`。

**验证**：
- 对比 Step 2 基线
- 重点关注：牛市区间是否不再被拖累（因为 `bear_signal` 在牛市接近 0，交互项自动关闭）
- 通过标准：牛市收益不降 + 熊市收益改善

#### Step 4：防御模式因子

为模型提供"当前环境不适合选股"的信号：

```python
# 系统性风险占比（市场波动 / 个股平均波动）
avg_stock_vol = torch.nanmean(features["volatility_20d"], dim=0, keepdim=True)
features["systematic_risk_ratio"] = mkt_vol_20d / (avg_stock_vol + 1e-8)
```

当 `systematic_risk_ratio` 高时（齐涨齐跌），个股 alpha 不存在，模型应学会给所有股票打相近分数（不做极端下注）。

**验证**：
- 对比前序基线
- 重点关注回撤期的亏损幅度是否收窄

### 9.3 迭代计划

| Step | 改动 | 新增因子数 | 验证标准 | 风险 |
|:---|:---|:---|:---|:---|
| 1 | Regime 原子因子 | +4 | 加入后模型不退化 | 低 |
| 2 | turnover_x_bull (单个交互) | +1 | Sharpe >= 基线 | 低 |
| 3 | value_x_bear (条件化价值) | +1 | 牛市不降 + 熊市改善 | 中 |
| 4 | 其余交互因子 | +3 | 逐个验证 | 中 |
| 5 | systematic_risk_ratio | +1 | 回撤收窄 | 低 |

每一步严格遵循：**加一个 -> 训练 -> 对比 -> 通过则保留，不通过立即回退**。

### 9.4 设计原则

1. **不新增低 IC 的独立因子** — 将已有因子做"条件化"处理，而非堆砌新维度
2. **交互项数量克制** — 只对 Top-5 因子做，总计不超过 ~10 个交互因子
3. **Regime 信号是调制器，不是预测器** — 通过乘法与原子因子结合，自身不直接参与选股
4. **极端尾部由风控层兜底** — 地缘冲击/关税黑天鹅不在因子信息边界内，由 `risk_controller` 回撤熔断处理，不在模型层硬编码应对

### 9.5 与历史失败实验的区别

| 过去的做法 | 为什么失败 | 本次的做法 |
|:---|:---|:---|
| 一次加 10+ 独立因子 | 低 IC 因子是噪声，稀释有效因子权重 | 只加交互因子，不增加独立维度 |
| 直接加 value/资金流因子 | MLP 给固定权重，牛市被拖累 | 价值因子乘以 bear_signal，牛市时自动关闭 |
| 在 dragon_score 中硬编码 regime | 参数过拟合历史，样本外失效 | 交互项无硬编码参数，由 MLP 自己学权重 |
| 多变量同时改动 | 无法隔离原因 | 严格单变量迭代 |

---

## 十、历史实验汇总

| 阶段 | 日期 | 改动 | 结果 | 状态 |
|:---|:---|:---|:---|:---|
| Phase 1 | 2026-04-06 | Beta-neutral label | 非牛市年化 5.39% -> 21.10% | **通过** |
| Phase 2 | 2026-04-06 | 质量因子(ROE/毛利率等) | Sharpe 下降，季频数据不匹配 | **失败，已回退** |
| Phase 3 | 2026-04-06 | 精简 dragon_score | 总收益 +54%，Sharpe +0.16 | **通过** |
| Phase 4 (资金流) | 2026-04-09 | 10 个资金流因子 | 总收益 -148%，全部 IC<0.03 | **失败，已回退** |
| Phase 4 (交互因子) | 2026-04-10 | Regime 原子因子 + turnover_x_bull | 见下方 Phase 4 实验记录 | **部分通过** |
| Phase 5 Step 1 (klow_2_20d) | 2026-04-10 | K 线微结构: 下影线占比 20d | 总收益 -117%, Sharpe -0.30, IC -0.061 | **失败，已回退** |
| Phase 5 Step 2 (cord_20) | 2026-04-10 | 量价同步性: ret-vol 相关 20d | 总收益 -109%, Sharpe -0.28, IC 0.061 | **失败，已回退** |
| Phase 5 Step 3 (sump_20) | 2026-04-10 | 上涨占比: 正向波动比 20d | 总收益 -186%, Sharpe -0.55, IC 0.011 | **失败，已回退** |
| Phase 5 Step 5 (rel_concept_turnover) | 2026-04-10 | 板块相对活跃度 | 总收益 -161%, Sharpe -0.43, IC 0.294 | **失败，已回退** |
| Phase 6 Exp 1 (纯 IC-Loss) | 2026-04-14 | MSE→IC-Loss + 日期结构化采样 | Sharpe 0.70, 非牛市年化 19.8%, 牛市腰斩 | **失败，已回退** |
| Phase 6 Exp 2 (混合 IC+MSE) | 2026-04-14 | loss = -IC + MSE | Sharpe 0.75, 非牛市年化 5.7%, 两头不讨好 | **失败，已回退** |
| Phase 6 Exp 3 (IC+清理因子) | 2026-04-15 | IC-Loss + 移除 P5P6 因子 | Sharpe 0.71, 非牛市年化 4.0% | **失败，已回退** |
| Phase 6 Exp 4 (时间衰减采样) | 2026-04-15 | 指数衰减采样 decay=0.995 | Sharpe 0.24, MaxDD -57.7%, 全面崩塌 | **失败，已回退** |

### 当前基线（Phase 1 + Phase 3 + turnover_x_bull）

```
总收益: 297.59%  年化: 69.21%  Sharpe: 1.29  MaxDD: -34.28%  收益回撤比: 5.86
问题: 牛市极强，非牛市年化仅 ~5%（Pre-bull 2022-01~2024-07 累计 +11.7%）
```

### Phase 4 交互因子实验记录（2026-04-10）

| Step | 改动 | 结果 | 状态 |
|:---|:---|:---|:---|
| Step 1 (4 regime 原子因子) | mkt_trend/mkt_vol_regime/mkt_breadth_20d/mkt_dispersion_20d | 总收益 116%, Sharpe 0.75, MaxDD -45.77%，严重退化 | **失败** |
| Step 1+2 (仅 turnover_x_bull) | + turnover_x_bull | 总收益 211%, Sharpe 1.05, 收益回撤比 5.16，回撤恢复 65 天 | 基线范围内 |
| Step 3 (value_x_bear) | + val_dv * bear_signal | 总收益 120%, Sharpe 0.76, 牛市被拖累 2025H2 32%→3% | **失败，已回退** |
| 最终保留 | 移除 3 个 regime 原子因子，仅保留 turnover_x_bull | 总收益 298%, Sharpe 1.29, 但非牛市年化 ~5% | **当前状态** |

**教训**：
1. Regime 原子因子（mkt_trend 等）IC 极低（~0.017），作为独立特征对 MLP 是纯噪声，导致严重退化
2. `turnover_x_bull`（IC 0.134）是有效因子，但它是纯进攻因子，只增强牛市信号，对非牛市贡献为零
3. `value_x_bear` 失败的根因：`val_dv` 在所有时段 IC 均为负（-0.105~-0.152），不是"熊市有效牛市无效"，而是全面无效
4. 交互因子方法本身成立（turnover_x_bull 验证），但需要找到**非牛市中有正 IC 的底层因子**

---

## 十一、Phase 5：非牛市 Alpha 挖掘

### 11.1 问题定义

当前模型"牛市极强 + 非牛市接近零 alpha"。**需要的不是防御（降低非牛市亏损），而是找到非牛市中独立的选股信号。**

### 11.2 非牛市 IC 特征分析

从 IC 表提取非牛市 4 个时段（211013-220804, 220805-230602, 230605-240329, 240401-250123）IC 相对整体更强的因子：

| 因子 | 非牛市均值 IC | 整体 IC | 非牛市增幅 | 含义 |
|:---|:---|:---|:---|:---|
| con_turnover_20d | **0.152** | 0.137 | +11% | 概念板块热度 |
| size_ln_cap | **-0.189** | -0.147 | +29% | 小盘溢价 |
| turnover_cv_20d | **0.096** | 0.072 | +33% | 换手率波动性 |
| inter_camel_bear | **0.052** | 0.037 | +41% | 驼峰 * 熊市概率 |

**关键发现**：非牛市中 IC 增强的不是价值/防御类因子，而是"板块轮动"和"交易微结构"类因子。非牛市 alpha 来源：
1. 资金在板块间轮动（con_turnover_20d 非牛市 IC 从 0.10 升至 0.18）
2. 交易行为异常信号（换手率波动性、量价微结构）

### 11.3 当前因子体系的盲区

V9 有 ~100 个因子，但缺少以下维度：
- **K 线微结构**：日内价格行为（实体/影线比例、收盘偏移），在震荡市中日内承接/抛压比趋势动量更有预测力
- **量价背离的时序特征**：收益率变化与成交量变化的相关性（V9 仅有价格-成交量相关性）
- **上涨/下跌不对称性**：累积的上涨占比、连续方向性，捕捉"小幅持续上涨"的机构建仓信号

Alpha158 中已有这些因子的 GPU 实现（`158_factor_calculator.py`），但 V9 未引入。

### 11.4 三个候选方向

#### 方向 A：K 线微结构因子

**来源**：Alpha158 candlestick 特征，V9 完全未使用。

```python
# 下影线占比（买盘承接力）— 20 日均值平滑
klow_2 = (min(O, C) - L) / (H - L + 1e-8)
features["klow_2_20d"] = ts_mean(klow_2, 20)

# 实体方向性（日内多空力量）— 20 日均值平滑
kmid_2 = (C - O) / (H - L + 1e-8)
features["kmid_2_20d"] = ts_mean(kmid_2, 20)

# 收盘偏移（尾盘买卖力量）— 20 日均值平滑
ksft_2 = (2 * C - H - L) / (H - L + 1e-8)
features["ksft_2_20d"] = ts_mean(ksft_2, 20)
```

**为什么可能在非牛市有效**：
- 震荡市中趋势信号（动量、MA）失效，但日内价格行为仍有信息
- 下影线占比高 = 每天有买盘承接，即使收盘没涨也说明有资金在吸筹
- 与现有动量/换手率因子正交：这些因子描述的是"怎么涨的"，而非"涨了多少"
- 不依赖市场趋势方向，在任何 regime 下都有定义

#### 方向 B：量价结构变化因子

**来源**：Alpha158 `cord` / `sump` 系列，V9 未使用。

```python
# 量价同步性（收益率变化 vs 成交量变化的相关性）
ret_1 = C / ts_delay(C, 1) - 1
vol_change = torch.log(V / (ts_delay(V, 1) + 1e-8) + 1e-8)
features["cord_20"] = ts_corr(ret_1, vol_change, 20)

# 上涨占比（上涨日幅度 / 总波动幅度）
delta_c = C - ts_delay(C, 1)
pos_delta = torch.clamp(delta_c, min=0)
abs_delta = torch.abs(delta_c)
features["sump_20"] = ts_sum(pos_delta, 20) / (ts_sum(abs_delta, 20) + 1e-8)
```

**为什么可能在非牛市有效**：
- `cord_20` 高 = 放量涨/缩量跌（健康的量价结构）；低或负 = 放量跌/缩量涨（量价背离）
- `sump_20` 高 = 虽然不一定涨很多，但上涨天的幅度占总波动的比例大（"涨多跌少"）
- 这些因子与 `price_vol_corr_20`（IC 0.077）不同：后者是价格水平与成交量的相关性，前者是变化率之间的相关性
- 在非牛市中，"放量下跌 → 缩量企稳 → 放量上涨"是经典底部形态，cord/sump 可以捕捉

#### 方向 C：板块内相对活跃度

**来源**：`con_turnover_20d` 在非牛市 IC 增强明显（从 0.10 升至 0.18）。

```python
# 个股换手率相对概念板块的活跃度
features["rel_concept_turnover"] = features["turnover_mean_20d"] / (con_turnover_20 + 1e-8)
```

**为什么可能在非牛市有效**：
- 非牛市中资金在板块间轮动，板块整体热度（con_turnover_20d）已是有效信号
- "板块热 + 个股更热"= 板块内领涨股，在轮动环境下可能持续
- 与已有的 `rel_turnover_20d`（相对全市场换手率）不同：这里是相对所属板块的活跃度

### 11.5 迭代计划

| Step | 方向 | 新增因子 | 验证标准 | 优先级 |
|:---|:---|:---|:---|:---|
| 1 | A: K 线微结构 | klow_2_20d (1 个) | IC 分析 + 回测非牛市改善 | 高 |
| 2 | B: 量价结构 | cord_20 (1 个) | IC 分析 + 回测非牛市改善 | 高 |
| 3 | B: 上涨占比 | sump_20 (1 个) | IC 分析 + 回测非牛市改善 | 高 |
| 4 | A: 更多微结构 | kmid_2_20d, ksft_2_20d | 逐个验证 | 中 |
| 5 | C: 板块相对活跃 | rel_concept_turnover | IC + 回测 | 中 |

每一步严格遵循：**加一个 → 训练 → 对比 → 通过则保留，不通过立即回退。**

### 11.6 Phase 5 实验记录

#### Step 1: klow_2_20d（下影线占比 20 日均值）— 失败

**日期**：2026-04-10

**因子 IC**：Overall -0.061（通过 |IC| >= 0.03 门槛），全时段一致为负

| 时段 | IC |
|:---|:---|
| 190422-200218 | -0.108 |
| 211013-220804 | -0.052 |
| 220805-230602 | -0.071 |
| 230605-240329 | -0.098 |
| 240401-250123 | -0.024 |
| Overall | -0.061 |

**回测结果**：

| 指标 | 基线 (Phase 1+3+turnover_x_bull) | + klow_2_20d | 变化 |
|:---|:---|:---|:---|
| 总收益 | 297.59% | 180.83% | **-116.76%** |
| 年化 | 69.21% | 42.05% | -27.16% |
| Sharpe | 1.29 | 0.99 | **-0.30** |
| MaxDD | -34.28% | -29.42% | 改善 |
| 收益回撤比 | 5.86 | 3.64 | -2.22 |

**结论**：虽然 IC 通过门槛，但加入后模型严重退化。IC 方向为负说明下影线占比高的股票未来表现更差——可能反映的是"频繁出现下影线 = 有抛压被短暂承接，但不代表真正买盘力量"。**代码已回退。**

**教训**：
1. IC 通过门槛（|IC| >= 0.03）是必要条件但非充分条件，加入后仍需回测验证
2. 下影线因子的负 IC 表明 A 股中"承接力"信号可能是噪声或反向指标
3. K 线微结构因子方向 A 的前提假设（"日内价格行为在非牛市中有预测力"）可能不成立

#### Step 2: cord_20（量价同步性 20d）— 失败

**日期**：2026-04-10

**因子 IC**：Overall 0.061（通过门槛），全时段正向，非牛市 IC: 0.047~0.094

**回测结果**：

| 指标 | 基线 | + cord_20 | 变化 |
|:---|:---|:---|:---|
| 总收益 | 297.59% | 188.63% | **-108.96%** |
| Sharpe | 1.29 | 1.01 | **-0.28** |
| MaxDD | -34.28% | -28.79% | 改善 |

**结论**：IC 通过门槛且方向合理（量价同步性高 → 未来收益高），但加入后模型整体退化。非牛市 90 天区间有部分改善（2024-04~07 转正），但不足以弥补整体 Sharpe 下降。**代码已回退。**

#### Step 3: sump_20（上涨占比 20d）— 失败

**日期**：2026-04-10

**因子 IC**：Overall **0.011**（未达 0.03 门槛），纯噪声。非牛市时段出现负值（-0.023, -0.032）。

**回测结果**：

| 指标 | 基线 | + sump_20 | 变化 |
|:---|:---|:---|:---|
| 总收益 | 297.59% | 111.14% | **-186.45%** |
| Sharpe | 1.29 | 0.74 | **-0.55** |
| MaxDD | -34.28% | -26.45% | 改善 |

**结论**：IC 远低于门槛，加入后模型严重退化。sump_20 与已有的 psy_12（上涨日占比）高度共线，且 psy_12 本身 IC 也仅 0.010，说明"涨跌方向占比"这类因子在截面选股中预测力极弱。**代码已回退。**

### 11.8 Phase 5 方向 A+B 总结

**三个因子全部失败**。方向 A（K 线微结构）和方向 B（量价结构变化）的前提假设未能得到验证：

| 因子 | IC | 回测变化 | 失败原因 |
|:---|:---|:---|:---|
| klow_2_20d | -0.061 | 总收益 -117% | 反向指标，下影线 = 抛压信号 |
| cord_20 | 0.061 | 总收益 -109% | IC 通过但 MLP 无法有效利用 |
| sump_20 | 0.011 | 总收益 -186% | IC 未达门槛，与 psy_12 共线 |

**核心教训**：
1. **Alpha158 因子不等于 A 股有效因子**：这些因子在学术环境（美股/横截面回归）中有效，但在 A 股 MLP 截面选股框架下，新增单个弱 IC 因子反而稀释有效因子权重
2. **MLP 对噪声因子极度敏感**：即使 IC 通过 0.03 门槛（klow_2, cord_20），加入后模型仍可能大幅退化，说明 MLP 容量有限，新因子占用的权重可能从有效因子处"偷"来
3. **非牛市 alpha 可能不在"新因子"中**：现有 ~100 个因子中已包含换手率、波动率、概念板块等非牛市有效维度，问题可能不是因子不够，而是模型结构/训练方式的限制

**下一步方向建议**：
- 方向 C（板块相对活跃度 rel_concept_turnover）仍可尝试，因为 con_turnover_20d 非牛市 IC 增强最明显
- 或转向模型层面改进（如训练窗口调整、损失函数设计），而非继续加因子

#### Step 5: rel_concept_turnover（板块相对活跃度）— 失败

**日期**：2026-04-10

**因子 IC**：Overall **0.294**（全因子排名第 3，ICIR 2.06）。非牛市 IC: 0.262~0.357，极强。

**回测结果**：

| 指标 | 基线 | + rel_concept_turnover | 变化 |
|:---|:---|:---|:---|
| 总收益 | 297.59% | 136.53% | **-161.06%** |
| Sharpe | 1.29 | 0.86 | **-0.43** |
| MaxDD | -34.28% | -29.58% | 改善 |

**结论**：IC 极强（0.294），但与 rel_turnover_20d (0.321) 和 turnover_mean_20d (0.319) 高度共线——三者本质都是换手率的不同归一化。共线因子不提供边际信息，反而因重复维度占用 MLP 权重，挤压其他有效因子。**代码已回退。**

**关键发现**：高 IC 不等于有效——如果新因子与已有因子共线，即使 IC 达 0.294 也会导致模型退化。未来加因子必须先检查与现有因子的相关性。

### 11.9 Phase 5 全部总结

**四个因子全部失败**。Phase 5 的三个方向均未能为模型带来提升：

| Step | 因子 | IC | 与现有因子关系 | 回测 Sharpe 变化 |
|:---|:---|:---|:---|:---|
| 1 (方向A) | klow_2_20d | -0.061 | 正交但反向 | -0.30 |
| 2 (方向B) | cord_20 | 0.061 | 弱相关 | -0.28 |
| 3 (方向B) | sump_20 | 0.011 | 与 psy_12 共线 | -0.55 |
| 5 (方向C) | rel_concept_turnover | 0.294 | 与换手率因子强共线 | -0.43 |

**根本结论**：
1. 当前 ~100 因子体系下，**新增单个因子无法提升模型**——无论 IC 高低
2. 正交弱因子（klow_2, cord_20）稀释权重；共线强因子（rel_concept_turnover）重复信息
3. 非牛市 alpha 的瓶颈不在因子维度，而在**模型容量和学习方式**
4. 下一步应转向：模型结构改进（更大网络/注意力机制）、训练方式改进（损失函数/样本权重/训练窗口）、或因子体系精简（移除低 IC 噪声因子减轻模型负担）

### 11.7 与历史失败实验的区别

| 过去的做法 | 为什么失败 | 本次的做法 |
|:---|:---|:---|
| 加质量因子（ROE/毛利率） | 季频数据与日频模型不匹配 | K 线/量价因子是日频原生数据 |
| 加资金流因子（10 个） | 全部 IC<0.03，纯噪声 | 从 Alpha158 已验证的学术因子中选取 |
| 做 regime 交互（value_x_bear） | 底层因子（val_dv）本身全面无效 | 先确认底层因子在非牛市有正 IC |
| 加 regime 原子因子（mkt_trend 等） | 截面维度相同的市场级信号对 MLP 是噪声 | 不加市场级信号，加个股级微结构因子 |
| 一次加多个因子 | 无法隔离原因 | 严格单变量迭代 |

---

### 11.10 Phase 5 补充实验：噪声因子批量剪枝

**假设**：模型中 ~30 个 |IC| < 0.03 的因子是噪声，占用 MLP 容量（64/32/16 hidden layers），导致新增任何因子都无法提升表现。移除噪声因子可释放模型容量。

**操作**：从 features dict 中移除约 30 个 |IC| < 0.03 因子，保留依赖中间变量（mom_20d, rsi_14 等转为局部变量）。因子数从 ~100 降至 71。

**移除因子清单**：
ma_bias_120, price_zscore_20d, ret_overnight, ret_intraday, bias_60, trend_rsquare_20, trend_slope_20, vol_skew_20, bollinger_position, rsi_14, psy_12, kdj_kd_diff, kdj_kd_velocity, tech_cci_14, illiquidity_20d, pe_zscore_60d, pe_rank_change_20d, slope_div_5_20, mom_x_mkt, tech_reversal, bear_reversal, dragon_score, con_mom_5d, con_mom_20d_max, rel_con_mom_20d, rel_con_mom_max_20d, resonance_signal, meta_bull_prob, inter_res_bull, head_lift_signal, bear_trap_score, mom_20d

**回测结果（含风控 -15%）**：

| 指标 | 基线 | 批量剪枝 | 变化 |
|:---|:---|:---|:---|
| 总收益 | 297.59% | **18.52%** | **-279%** |
| 年化收益 | ~58% | 4.31% | -54% |
| Sharpe | 1.29 | **0.15** | **-1.14** |
| MaxDD | -34.28% | **-62.43%** | -28% |

**结论：批量剪枝彻底失败。**

**失败原因分析**：
1. **IC ≠ 模型贡献**：MLP 是非线性模型，单因子 IC 测量的是线性相关性。低 IC 因子可通过 hidden layer 的非线性组合产生高贡献（如 rsi_14 + bias_10 + dist_support_20 → camel_hump_score 的交互效应）
2. **特征空间结构破坏**：30 个因子一次移除，等同于改变了输入空间的维度和分布结构，所有已学习的权重矩阵全部失效
3. **IC 门槛不适用于剪枝**：|IC| >= 0.03 是**新增因子**的门槛，不能反向用于判断已有因子是否该移除。因子加入模型后，其价值通过非线性交互体现，不再由线性 IC 衡量

**教训**：
- 因子剪枝不能用 IC 一刀切，应使用 permutation importance（在验证集上逐个置换因子测量模型性能变化）
- 即使要剪枝，也应逐步进行（每次最多移除 3-5 个），而非批量操作
- **已回退代码，当前模型恢复到 Phase 4 基线状态**

---

## Phase 5: Turnover-Neutral Label（换手率中性化标签）

> 日期：2026-04-13
> 基线：Phase 4 (Beta-Neutral + turnover_x_bull)
> 目标：去除 label 中的换手率风险溢价，释放被压制因子的 alpha 空间

### 5.1 问题诊断

通过三层因子分析工具（factor_evaluator / model_attribution / factor_ablation）对 V9 模型的全面分析发现：

**核心问题：模型 77% 的预测能力来自换手率组（9 个因子），其他 89 个因子几乎是摆设**

进一步分析 label 发现：
- Label 与 turnover_mean_20d 的截面 R² = **12.3%** — label 中超过 1/8 的信息就是换手率
- 对 label 做 turnover 正交化后发现：
  - **价值因子被压制**：ep_ratio IC 从 -0.118 提升到 -0.079（提升 33%）
  - **动量因子虚假信号**：mom_60d IC 从 +0.017 翻转为 -0.021（原来是假的）
  - **反转因子被掩盖**：bear_reversal IC 从 +0.012 提升到 +0.032（提升 167%）
  - **换手率因子 1/3 IC 是"自相关"**：turnover IC 从 +0.308 降至 +0.202

**根因**：label 中 baked-in 的换手率信息通过三条路径污染模型：
1. A 股 5 日收益本身与换手率正相关（散户投机驱动）
2. `low_liq_penalty` 显式惩罚低换手率股票
3. MLP 发现 turnover 是预测 label 的"捷径"，不再学习其他因子

### 5.2 理论依据

1. **Miller (1977) 异质信念假说**：高换手率 = 投资者分歧大 = 卖空约束下价格高估 = 低未来收益。A 股卖空约束极强，该机制特别显著
2. **Datar (1998) 流动性溢价**：低换手率股票要求流动性补偿，提供更高期望收益。这是**风险因子**，不是 alpha
3. **Barra CNE6 风险模型**：将流动性（STOM/STOQ/STOA）列为风格风险因子，与 beta、size 并列
4. **Gu, Kelly, Xiu (2020)**：确认 stock-level liquidity 是 ML 模型中最重要的三类预测因子之一，但优秀模型能在多维度均衡获取 alpha

**核心逻辑**：V9 Phase 1 已从 label 中去除 beta（市场风险因子），理论一致的做法是**同样去除 turnover（流动性风险因子）**

### 5.3 方案设计

模拟测试了 4 种方案（见 `core/tools/label_reform_simulation.py`）：

| 方案 | Turnover R² | Turnover IC | bear_reversal IC | illiquidity IC |
|:---|:---:|:---:|:---:|:---:|
| V9 基线 | 12.3% | +0.308 | +0.012 | +0.061 |
| 全量 turnover 中性 | **4.9%** | +0.202 | **+0.032** | +0.074 |
| turnover+size 中性 | 3.8% | +0.174 | +0.024 | **-0.061**(翻转) |
| 半中性(50%) | 8.2% | +0.255 | +0.023 | +0.068 |

**选择方案 1（全量 turnover 中性）**：
- Turnover R² 降 60%，但 IC 仍保留 +0.202（真实预测力完整保留）
- 被压制因子释放最充分（bear_reversal +167%, tech_reversal +238%）
- 不会过度矫正（方案 2 导致 illiquidity 翻转）

### 5.4 代码改动

**文件 1: `core/alpha/factor_calculator.py`**
- 新增 `cs_neutralize(y, x)` 函数：截面回归中性化，返回残差

**文件 2: `core/alpha/v9_factor_calculator.py`**
- Label 构造改为：
  ```python
  excess_ret_5 = raw_ret_5 - beta * mkt_ret_5       # Step 1: Beta neutral (Phase 1)
  turnover_rank = cs_rank(turnover_mean_20d)
  alpha_ret_5 = cs_neutralize(excess_ret_5, turnover_rank)  # Step 2: Turnover neutral (Phase 5)
  label = cs_rank(alpha_ret_5)
  ```
- 移除 `low_liq_penalty`（不再硬编码换手率偏见）

### 5.5 验证标准

| 指标 | 通过标准 |
|:---|:---|
| Sharpe | >= V9 基线 (1.29) |
| 非牛市年化 | > V9 基线 (~5%) — **核心改善目标** |
| 因子组贡献分散度 | Turnover 组贡献占比 < 60%（V9 为 77%）|
| 牛市收益 | 允许适度下降，但不低于 V9 的 70% |

### 5.6 回测结果

Phase 5 Turnover-Neutral Label 未单独训练验证（直接与 Phase 6 一起测试，见 Phase 6）。

---

## Phase 6: 模型训练层改造实验

> 日期：2026-04-14 ~ 2026-04-15
> 基线：Phase 4 (Beta-Neutral + turnover_x_bull)，MSE 损失 + 均匀采样
> 目标：通过改变模型学习方式（而非因子/标签）改善非牛市表现

### 6.1 背景

Phase 5 的因子实验（4 个新因子全部失败）和噪声因子剪枝（批量移除 30 个导致崩溃）表明：因子维度已充分探索，瓶颈在模型学习方式。核心问题：MLP 77% 预测能力来自换手率组，非牛市年化仅 ~14%。

### 6.2 实验记录

#### Exp 1: 纯 IC-Loss（截面排序损失）

**改动**：MSE 替换为 per-day Pearson IC loss + 日期结构化批次采样（每批采样 10 个完整交易日）

| 指标 | MSE 基线 | 纯 IC-Loss | 变化 |
|:---|:---|:---|:---|
| 总收益 | 259.0% | 117.6% | -141.4% |
| Sharpe | 1.10 | 0.70 | -0.40 |
| Pre-2024-11 年化 | 14.2% | **19.8%** | **+5.6%** |
| MaxDD | -36.9% | -30.4% | +6.5% |
| Q2 2024 | -17.6% | -11.2% | +6.4% |
| 牛市核心(Nov24-Mar25) | +66.6% | +17.0% | -49.6% |

**结论**：IC-Loss 是唯一成功改善非牛市的方法（19.8% vs 14.2%），但牛市收益腰斩导致整体 Sharpe 大降。IC-Loss 让模型不再过度依赖换手率/波动率信号，在非牛市中有效，但牛市中这些信号恰恰是 alpha 来源。

#### Exp 2: 混合损失（IC + MSE）

**改动**：loss = -mean_IC + MSE，早停从 40 降至 20

| 指标 | MSE 基线 | 混合 IC+MSE |
|:---|:---|:---|
| Sharpe | 1.10 | 0.75 |
| Pre-2024-11 年化 | 14.2% | 5.7% |
| Q1 2024 | -8.9% | -25.2% |

**结论**：两个损失目标梯度方向冲突（IC 优化截面排序 vs MSE 优化绝对值），模型两头都没学好。比纯 IC 和纯 MSE 都差。

#### Exp 3: IC-Loss + 清理噪声因子

**改动**：纯 IC-Loss + 回退因子集到 Phase 1+3+4（移除 Phase 5+6 的资金流/质量因子）

| 指标 | IC+P5P6 | IC+Clean(P1+3+4) |
|:---|:---|:---|
| Sharpe | 0.70 | 0.71 |
| Pre-2024-11 年化 | **19.8%** | 4.0% |

**结论**：Phase 5+6 的低 IC 因子在 IC-Loss 框架下有正贡献。不同损失函数下同一因子集的有效性完全不同。

#### Exp 4: 时间衰减采样（Temporal Decay Sampling）

**改动**：训练采样从均匀分布改为指数衰减（decay=0.995），近期样本被采样概率约为 500 天前的 12 倍

| 指标 | MSE 基线 | 衰减采样 |
|:---|:---|:---|
| Sharpe | ~0.68 | **0.24** |
| MaxDD | ~-29% | **-57.7%** |
| Q2 2024 | ~-16% | **-33.8%** |

**结论**：catastrophic failure。decay=0.995 过于激进，模型严重过拟合近期 regime。A 股 regime 切换剧烈，时间偏向采样反而有害。均匀采样的 500 天窗口本身是有效的隐式正则化。

### 6.3 核心教训

1. **损失函数改造是双刃剑**：IC-Loss 唯一改善非牛市，但必然损害牛市。两个目标不可调和
2. **混合损失不可行**：梯度冲突导致两败俱伤，不是比例调参问题
3. **采样策略改造高风险**：时间衰减破坏了均匀采样的隐式正则化效果
4. **因子有效性与损失函数耦合**：IC-Loss 下低 IC 因子有价值，MSE 下是噪声
5. **均匀采样 500 天窗口不可轻易改变**：它提供了 regime 多样性，是模型泛化的基础

### 6.4 代码状态

**全部实验已回退。当前代码 = Phase 1+3+4 基线（MSE + 均匀采样）。**
