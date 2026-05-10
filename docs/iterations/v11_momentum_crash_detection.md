# V11 动量崩溃检测 (Momentum Crash Detection) - 失败记录

> 创建日期：2026-05-05
> 基线：V10 (Factor Self-Attention + cord_20, ~110 factors, Sharpe 1.76-1.84)
> 目标：修复603598.SSE等股票的错误高评分问题
> 状态：**失败** - Sharpe 0.89 (-49%), MaxDD -43.11% (+109%)

---

## 一、问题诊断

### 1.1 核心矛盾

股票603598.SSE在V10模型下持续获得高分信号（mean=1.02, max=1.73），但其价格表现矛盾：
- 120d动量: +24.33%（长期趋势向上）
- 60d动量: -19.57%（中期趋势向下）

这是典型的"动量崩溃"特征：长期赢家开始反转。

### 1.2 理论依据

**Daniel & Moskowitz (2016) - Momentum Crashes**
- 动量策略在美股呈现不对称性：牛市表现好，熊市崩溃
- 崩溃条件：过去赢家（高mom_120d）在市场反弹时急剧反转
- 检测：mom_120d > 0 但 mom_60d < mom_120d * threshold

---

## 二、方案实施

### 2.1 新增因子（6个）

```python
# 1. mom_crash_120_60: 长期动量崩溃检测
mom_120_pos = features["mom_120d"].clamp(min=0)
mom_decay_ratio = features["mom_60d"] / (mom_120_pos + 1e-8)
features["mom_crash_120_60"] = (mom_decay_ratio - 0.5).clamp(min=-1.0, max=1.0)

# 2. mom_crash_60_20: 中期动量崩溃检测
mom_60_pos = features["mom_60d"].clamp(min=0)
mom_decay_ratio_60_20 = features["mom_20d"] / (mom_60_pos + 1e-8)
features["mom_crash_60_20"] = (mom_decay_ratio_60_20 - 0.3).clamp(min=-1.0, max=1.0)

# 3. mom_recovery_signal: 动量恢复信号
mom_60_turning = features["mom_60d"] - ts_delay(features["mom_60d"], 20)
features["mom_recovery_signal"] = (
    (features["mom_120d"] > 0).float() *
    (mom_60_turning > 0).float() *
    (features["mom_60d"] < 0).float() * 1.0
)

# 4. mom_crash_interaction: 动量崩溃×长期动量交互
mom_120_rank = cs_rank(features["mom_120d"])
features["mom_crash_interaction"] = features["mom_crash_120_60"] * mom_120_rank

# 5. mom_stability_120: 动量稳定性（120日动量标准差）
mom_120_series = C / ts_delay(C, 120) - 1
features["mom_stability_120"] = -1.0 * ts_std(mom_120_series, 120)

# 6. mom_consistency_score: 动量一致性得分
mom_signs = (
    torch.sign(features["mom_20d"]) + 
    torch.sign(features["mom_60d"]) + 
    torch.sign(features["mom_120d"])
)
features["mom_consistency_score"] = torch.abs(mom_signs) / 3.0
```

---

## 三、实验结果

### 3.1 因子IC分析

| 因子 | 整体IC | 2022 | 2023 | 2024 | 2025 | 最近 |
|:---|:---|:---|:---|:---|:---|:---|
| mom_crash_120_60 | -0.019 (-0.17) | 0.049 | -0.009 | -0.004 | -0.030 | -0.019 |
| mom_crash_60_20 | -0.049 (-0.45) | -0.009 | -0.051 | -0.025 | -0.039 | -0.049 |
| mom_recovery_signal | -0.001 (-0.03) | 0.014 | 0.003 | 0.001 | -0.002 | -0.001 |
| mom_crash_interaction | +0.001 (+0.01) | 0.059 | -0.030 | 0.024 | -0.000 | +0.001 |
| mom_stability_120 | **-0.194 (-1.70)** | - | - | - | - | **-0.194** |
| mom_consistency_score | -0.011 (-0.14) | -0.002 | -0.036 | -0.001 | -0.018 | -0.011 |

**关键发现**：
- 5/6因子IC为负
- `mom_stability_120` IC=-0.194严重为负，表明A股中动量波动大的股票反而表现更好（与理论相反）

### 3.2 回测对比

| 指标 | V10基线 | V11动量崩溃 | 变化 |
|:---|:---|:---|:---|
| 总收益 | 533~606% | 185.03% | -66~70% |
| 年化 | 124~140% | 42.45% | -66~70% |
| **Sharpe** | **1.76~1.84** | **0.89** | **-49~51%** |
| MaxDD | -20.64% | -43.11% | +109% |
| 收益回撤比 | 6.04~8.72 | 2.47 | -59~72% |

### 3.3 90天收益分析（V11）

- 2024-04~07（国九条）: -23.09%（严重亏损）
- 2025-04~07: +40.69%
- 2026-01~04: -11.84%

非牛市表现并未改善。

---

## 四、失败原因分析

### 4.1 核心问题

1. **A股动量特征与美股不同**
   - Daniel & Moskowitz (2016) 基于美股数据
   - A股动量效应本身较弱，甚至常表现为反转
   - `mom_stability_120` IC=-0.194证明A股中"大起大落"的股票反而跑赢"稳定上涨"的股票

2. **比率因子不稳定性**
   - `mom_decay_ratio = mom_60d / mom_120d` 在分母接近零时极不稳定
   - 即使做了clamp，噪声仍然很大

3. **因子与框架不匹配**
   - 动量崩溃是**组合管理**层面的概念（Daniel & Moskowitz用于调整动量策略仓位）
   - 本系统是**截面选股**MLP，因子通过横截面排名后输入
   - 截面排名已经消除了部分绝对水平信息

### 4.2 更深层认知

603598的问题不是"偶尔极端高分"，而是"持续中等偏高分数"：
- V10下mean signal=1.02, max=1.73，从未超过2.0
- 信号公式 `final_signal = ((rank/count - 0.5) * 3.46).clip(-3, 3)`
- 该股票在横截面排名中始终处于60-70分位

**结论**：问题不在因子层面，可能在：
- 信号处理层（横截面排名/标准化方式）
- 模型架构（Attention对某些模式的学习）
- 需要因子正交化或信号层面的变换

---

## 五、V11尝试历史

| 版本 | 方案 | Sharpe | 年化 | MaxDD | 结论 |
|:---|:---|:---|:---|:---|:---|
| V11-1 | 单天差分动量加速度 | 1.13 | - | - | 失败，金融时间序列差分=白噪声 |
| V11-2 | 动量交互+质量因子 | 1.23 | - | - | 失败，10/12因子负IC |
| V11-3 | 动量崩溃检测 | 0.89 | 42.45% | -43.11% | 失败，A股动量特征不同 |
| V11-4 | Gate Network条件化因子权重 | 0.84 | - | - | 失败，Gate增加噪声，正则化不足以补偿 |

---

## 六、回退决定

V11四次尝试全部失败，代码已回退到V10基线。Gate Network代码已从模型中移除。

### V11-4 Gate Network 失败分析

**方案**：在FactorAttentionNetwork中引入Gate Network，基于市场状态因子（meta_bull_prob, mom_x_mkt等5个）动态调整每个因子token的权重。

**理论依据**：MLP是无条件截面打分器，无法根据市场状态动态调整因子权重。Gate Network通过Sigmoid门控实现条件化因子加权。

**失败原因**：
1. Gate Network增加了额外参数（5→32→110 Sigmoid），在500天滚动训练窗口中容易过拟合
2. Gate输出的Sigmoid在初始阶段接近0.5，对所有因子等权缩放，训练早期提供的信息量不足
3. L1 gate_reg=0.01过小，无法有效约束Gate的噪声学习
4. 市场状态因子（bull_prob等）本身已作为输入特征参与Attention，Gate Network对其二次使用造成信息冗余
5. Attention机制已经具备动态加权能力（通过注意力权重），Gate Network的功能与之重叠

**关键认知**：Factor Self-Attention本身已经是"条件化因子加权"的更优解——它通过因子间相互关注实现动态权重分配，无需额外的Gate模块。

**未来可能的方向**（未执行）：
1. 信号层面变换：分位数标准化、因子正交化
2. 完全不同的思路（非动量类因子探索）
3. 标签改造（V9 Phase 1证明标签改造ROI最高）
