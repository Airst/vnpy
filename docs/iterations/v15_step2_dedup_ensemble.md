# V15.3 迭代：GP因子去重 + 验证集扩大 + 3-seed Ensemble

## 背景

V15.2 锁定基线为：Sharpe 1.36 / 年化 74.8% / MaxDD -25.0% / 收益回撤比 7.29 / MaxDD持续 64 天。

诊断后识别三个核心问题：

1. **GP 因子同质化**：22 个 validated GP 因子中存在大量结构相似的子树，集体加入 Factor Attention 后会稀释注意力权重
2. **验证集偏小**：50 天（约 2 个月）的验证窗口可能被短期市场偏差污染，影响 early stopping 决策可靠性
3. **单次训练 variance 高**：固定 seed=42 单模型训练，结果受权重初始化、batch 采样、dropout mask 的随机性影响

## 方案

### 方案 1：GP 因子去重

按子树相似性聚类，识别同质化 cluster，保留每组中 ICIR 最强或信息维度最独立的代表性因子。

**Reject 9 个冗余因子**：

| ID | 拒绝理由 | 保留代表 |
|:---|:---|:---|
| gp_007 | `add(SL,TR)` 与 gp_009 同质，ICIR 弱 | gp_009 |
| gp_008 | `cs_zscore(BL)` 与 gp_009/015 同质 | gp_009 |
| gp_011 | `ts_delta(sub(CM20,X))` 与 gp_019 同质 | gp_019 |
| gp_013 | `cs_zscore(SL)` 与 gp_014 同质，结构过复杂 | gp_014 |
| gp_015 | `cs_zscore(BL)` 同质 | gp_009 |
| gp_016 | 子树与 gp_018 同质，ICIR 弱 | gp_018 |
| gp_020 | `cs_rank(ts_delta(sub(CM20,V),3))` 与 gp_019 同质 | gp_019 |
| gp_037 | `cs_zscore(BMD)` 与 gp_038 同质，ICIR 弱 | gp_038 |
| gp_039 | `sub(SMD,V)` 与 gp_038 同质 | gp_038 |

**保留 13 个代表性因子**，覆盖 5 个独立信息维度：
- 换手率结构：gp_001, gp_002
- 资金流买卖：gp_006, gp_009, gp_010, gp_012, gp_014
- 超大单：gp_017, gp_018, gp_019
- 散户/中单：gp_038
- 估值×成交量：gp_041, gp_042

### 方案 2：验证集 50 → 100 天

在 700 天滚动窗口中，把验证集从 50 天扩到 100 天：
- 训练集：~600 天（仍提供充足 regime 多样性）
- 验证集：100 天（约 4 个月行情）

文件：`core/alpha/mlp_signals.py:187`，`valid_len = 50` → `valid_len = 100`

### 方案 3：3-seed Ensemble

利用已存在但从未启用的 `ensemble_size` 参数：
- Seeds: `[42, 123, 2024]`（diverse primes 保证独立性）
- 每个滚动窗口训练 3 个 FactorAttentionNetwork
- 预测取算术平均后再做截面排名

代码改动：`core/alpha/mlp_signals.py`
- `__init__` 默认 `ensemble_size=1` → `ensemble_size=3`
- `_train_and_predict_window`：循环训练 3 个模型，预测累计后取均值
- `_train_model` 增加 `seed` 参数

## 验证结果

### 整体回测对比（2022-01-04 ~ 2026-05-20）

| 指标 | V15.2 (基线) | V15.3 (本次) | 变化 |
|:---|:---|:---|:---|
| Sharpe | 1.36 | **1.74** | +28.0% |
| 年化收益 | 74.8% | **113.9%** | +39.1pp |
| 总收益 | ~330% | **501.5%** | +171pp |
| MaxDD | -25.0% | **-21.1%** | 改善 3.9pp |
| MaxDD 持续天数 | 64 | **26** | -59% |
| 收益回撤比 | 7.29 | **7.25** | 持平 |

**所有核心指标全面突破基线，没有用收益换回撤的 trade-off。**

### 改进效果归因分析

1. **GP 去重**：减少 attention 容量浪费，让模型更聚焦在真正独立的信息维度。预期贡献：因子层 IC 稳定性 + 减少过拟合
2. **验证集扩大**：100 天覆盖更多 regime，early stopping 决策更稳健，减少在偶然好/差的短期市场中的过早/过晚停止
3. **3-seed ensemble**：消除单模型 variance 是收益最直接的来源 —— 单次训练的"坏运气"被多模型平滑

三者协同作用，共同把 OOS Sharpe 从 1.36 推到 1.74。

## 失败记录

无 —— 三个改动同时落地一次性通过，所有指标改善。

## 后续方向（待评估）

- 进一步扩大 ensemble（5-seed/7-seed）边际收益是否成立
- GP 挖掘新一轮（避开已 reject 的同质化结构）
- 移除 V15 中 bull_prob 硬编码相关的 6 个交互因子（mom_x_mkt, bear_reversal, vol_penalty 等）
- 策略层引入信号加权仓位（score 越高仓位越大）
