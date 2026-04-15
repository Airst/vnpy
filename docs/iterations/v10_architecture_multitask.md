# V10 改造方案：模型架构 + 多任务学习

> 创建日期：2026-04-15
> 基线：V9 Phase 1+3+4 (Beta-neutral label + simplified dragon_score + turnover_x_bull)
> 目标：通过模型结构改造和多任务学习，打破换手率主导，改善非牛市表现

---

## 一、基线状态

V9 当前基线（Phase 1+3+4），AGENTS.md 记录值：

| 指标 | 值 |
|:---|:---|
| 总收益 | 297.59% |
| 年化 | 69.21% |
| Sharpe | 1.29 |
| MaxDD | -34.28% |
| 收益回撤比 | 5.86 |
| 因子数 | ~99 |
| 核心问题 | 77% 预测力来自换手率组，非牛市年化 ~5% |

---

## 二、改造路径（三步递进）

### Step 1：扩大网络 + 强化正则（本轮）

**改什么**：
- hidden_sizes: (64, 32, 16) → (256, 128, 64)
- weight_decay: 0.001 → 0.005
- 每层 hidden layer 后加 Dropout(0.10)

**为什么**：
- 99 个因子压缩到 64 维丢失二阶交互信息
- 训练数据量 ~110 万样本（500 天 x 2200 股），支撑 4 万参数绰绰有余
- 更大的第一层（99→256）保留更多因子交互空间
- 更强的 Dropout + weight_decay 防止过拟合

**验证标准**：Sharpe 不降，关注非牛市表现变化

### Step 2：多任务多周期学习（下一轮）

共享特征提取层 + 多周期独立预测头（1d/5d/10d/20d）

### Step 3：Gate Network 条件化因子权重（再下一轮）

动态调整因子权重的门控网络

---

## 三、Step 1 实验记录

### 3.1 代码改动

**文件 1: `core/alpha/mlp_signals.py`**
- `hidden_sizes`: (64, 32, 16) → (256, 128, 64)
- `weight_decay`: 0.001 → 0.005

**文件 2: `vnpy/alpha/model/models/mlp_model.py`**
- `MlpNetwork.__init__`: 每个 hidden layer 的 activation 后加 `Dropout(0.10)`
- 输入层 Dropout: 0.05 → 0.10
- 输出层 Dropout: 0.05 → 0.10

### 3.2 回测结果

（训练后填写）

### 3.3 结论

（训练后填写）
