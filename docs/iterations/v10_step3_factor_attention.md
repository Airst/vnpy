# V10 Step 3: Factor Self-Attention (因子自注意力交互层)

> 创建日期：2026-04-16
> 基线：V10 Step 1 (MLP 256/128/64, Dropout 0.10, weight_decay 0.002)
> 目标：通过因子自注意力机制让模型自动学习因子间交互关系，替代手工构建交互因子

---

## 一、问题诊断

### 1.1 当前瓶颈

V9 Phase 4 验证了"因子交互"方法有效（`turnover_x_bull` IC=0.134），但存在根本限制：

| 问题 | 说明 |
|:---|:---|
| 手工枚举效率低 | 99 个因子的二阶交互有 ~4900 种组合，人工只能试个位数 |
| 依赖先验知识 | 必须预先知道"换手率在牛市更有效"才能构建 `turnover_x_bull` |
| 底层因子约束 | `value_x_bear` 失败因为 `val_dv` 本身全面无效，交互无法挽救无效因子 |
| 模型层已穷尽 | 损失函数改造连续 3 次失败（IC-Loss、混合损失、多任务损失），该方向暂停 |

### 1.2 MLP 的结构性缺陷

当前 MLP 第一层 `Linear(99, 256)` 将所有因子一次性压缩到 256 维。这意味着：

- 每个因子的信号被立即混合，无法先做"成对交互"
- 高维空间中二阶交互信号被淹没在一阶线性组合中
- MLP 理论上能学交互，但实际训练中很难从 4900 种组合中发现稀疏的有效交互

### 1.3 核心假设

> **在因子送入 MLP 之前，增加一个 Self-Attention 层，让每个因子能"看见"其他因子并动态调整自身信号强度。这等价于模型自动学习所有因子间的交互关系。**

例如：Attention 可以学到"当市场波动率因子的值很高时，降低换手率因子的权重"——这正是手工构建 `turnover_x_highvol` 要实现的效果，但无需人工指定。

---

## 二、技术方案：FT-Transformer 架构

### 2.1 架构来源

FT-Transformer（Feature Tokenizer + Transformer）由 Gorishniy et al. 2021 提出（"Revisiting Deep Learning Models for Tabular Data"），在多个 tabular data benchmark 上优于纯 MLP 和 GBDT。

核心创新：将每个数值特征独立映射为 token embedding，然后用 Transformer 的 Self-Attention 建模特征间交互。

### 2.2 架构设计

```
输入: [f1, f2, ..., f99]  (99 个标量因子)
        |
        v
  Factor Tokenizer: 每个 fi -> Linear(1, d_token) + bias
        |
        v
  [e1, e2, ..., e99]  (99 个 d_token 维向量)
        |
  Prepend [CLS] token
        |
        v
  [CLS, e1, e2, ..., e99]  (100 个 d_token 维向量)
        |
        v
  Transformer Block x N_layers:
    ├── LayerNorm
    ├── Multi-Head Self-Attention (n_heads)
    ├── Residual Connection
    ├── LayerNorm
    ├── FFN: Linear(d_token, d_ffn) -> GELU -> Dropout -> Linear(d_ffn, d_token) -> Dropout
    └── Residual Connection
        |
        v
  Extract [CLS] token -> [d_token]
        |
        v
  Prediction Head:
    Linear(d_token, head_dim) -> BN -> LeakyReLU -> Dropout
    Linear(head_dim, 1)
        |
        v
  Output: 标量预测值
```

### 2.3 Self-Attention 的交互机制

对于第 i 个因子 token `ei`，Self-Attention 计算：

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

其中 Q = ei * Wq,  K = [e1..e99] * Wk,  V = [e1..e99] * Wv
```

这意味着：
- 每个因子通过 Q/K 内积计算与其他所有因子的"相关性权重"
- 权重是 **数据驱动的、动态的**：同一个换手率因子，在不同股票、不同时间点上，会关注不同的其他因子
- 输出 = 其他因子的加权组合，等价于"交互增强后的因子表示"

**与手工交互因子的对比**：

| 手工交互因子 | Self-Attention |
|:---|:---|
| `turnover * bull_signal` (固定组合) | turnover 根据当前所有因子值动态决定关注谁 |
| 每次只能测 1 个交互 | 一次建模所有因子的交互 |
| 需要先验知识选择组合 | 数据驱动，自动发现有效交互 |
| 交互权重固定为 1.0 | 交互权重由 Attention 学习 |

### 2.4 [CLS] Token 的作用

[CLS] 是一个可学习的特殊 token，不对应任何实际因子。经过 Self-Attention 后，[CLS] 聚合了所有因子的全局交互信息，作为最终预测的"摘要向量"。

这比 mean pooling 更好：mean pooling 对所有因子等权平均，而 [CLS] 通过 Attention 学习"关注哪些因子的交互对预测最重要"。

---

## 三、两步递进实施

### Step 3a：轻量级 FactorAttention（最小可验证方案）

**目标**：用最小的参数增量验证"因子交互层"假设是否成立。

**超参数**：

| 参数 | 值 | 理由 |
|:---|:---|:---|
| d_token | 32 | 每个因子映射到 32 维，足够表达交互关系，参数量可控 |
| n_heads | 4 | 4 个注意力头，每头 8 维，捕捉不同类型的因子关系 |
| n_layers | 1 | 仅 1 层 Attention，最小化过拟合风险 |
| d_ffn | 64 | FFN 扩展比 2x（32→64→32） |
| attn_dropout | 0.15 | Attention 权重 dropout |
| ffn_dropout | 0.15 | FFN 中间层 dropout |
| head_hidden | 64 | 预测头隐藏层维度 |

**参数量估算**：

| 组件 | 参数量 |
|:---|:---|
| Factor Tokenizer: 99 × (1×32 + 32) | ~6,300 |
| CLS Token | 32 |
| Attention QKV + Out: 4 × 32 × 32 | ~4,100 |
| FFN: 32×64 + 64×32 | ~4,100 |
| LayerNorm × 2 | ~130 |
| Head: 32×64 + 64×1 | ~2,100 |
| **总计** | **~16,800** |

对比当前 MLP：~66,000 参数。Step 3a 参数量反而更小。
训练数据：~110 万样本（500 天 × 2200 股），参数/样本比 = 1:65，远低于过拟合阈值。

**预测头设计**：

```python
# 轻量预测头（替代原 256/128/64 MLP）
Head:
  Linear(32, 64) -> BatchNorm1d(64) -> LeakyReLU -> Dropout(0.10)
  Linear(64, 1)
```

预测头保持简单，让 Attention 层承担因子交互的核心工作，预测头只做最终打分。

**正则化策略**：

| 正则化 | 位置 | 值 |
|:---|:---|:---|
| Attention Dropout | Attention 权重 | 0.15 |
| FFN Dropout | FFN 中间层 | 0.15 |
| Head Dropout | 预测头隐藏层后 | 0.10 |
| Weight Decay | Adam 优化器 | 0.002（与 Step 1 一致） |

**验证标准**：

| 指标 | 标准 |
|:---|:---|
| Sharpe | >= 1.10（当前基线 1.24，允许 Attention 学习曲线造成的轻微下降） |
| 总收益 | >= 250% |
| MaxDD | 不恶化超过 5% |
| 非牛市表现 | 关注 2022-01~2024-07 区间，是否优于基线 |

### Step 3b：完整 FT-Transformer（仅 Step 3a 通过后执行）

**改动**：在 Step 3a 基础上增加模型容量：

| 参数 | Step 3a | Step 3b |
|:---|:---|:---|
| d_token | 32 | 64 |
| n_heads | 4 | 8 |
| n_layers | 1 | 2 |
| d_ffn | 64 | 128 |
| head_hidden | 64 | 128→64 (两层) |
| 预估参数量 | ~16,800 | ~75,000 |

Step 3b 的参数量与当前 MLP（~66,000）相当，但结构不同：参数集中在因子交互层而非简单全连接。

---

## 四、代码改动方案

### 4.1 改动文件清单

| 文件 | 改动 | 说明 |
|:---|:---|:---|
| `vnpy/alpha/model/models/mlp_model.py` | 新增 `FactorAttentionNetwork` 类 | 核心架构实现 |
| `vnpy/alpha/model/models/mlp_model.py` | `MlpModel.__init__` 支持 `model_type` 参数 | 在 MlpNetwork 和 FactorAttentionNetwork 间切换 |
| `core/alpha/mlp_signals.py` | `model_settings` 增加 `model_type` | 传递模型类型选择 |

### 4.2 FactorAttentionNetwork 伪代码

```python
class FactorAttentionNetwork(nn.Module):
    """
    Factor Self-Attention Network
    
    将每个因子映射为 token embedding，通过 Self-Attention 建模因子间交互，
    CLS token 聚合全局信息后送入预测头。
    """
    def __init__(
        self,
        input_size: int,         # 因子数量 (99)
        d_token: int = 32,       # 每个因子的 embedding 维度
        n_heads: int = 4,        # 注意力头数
        n_layers: int = 1,       # Transformer 层数
        d_ffn: int = 64,         # FFN 中间层维度
        head_hidden: int = 64,   # 预测头隐藏层维度
        attn_dropout: float = 0.15,
        ffn_dropout: float = 0.15,
        head_dropout: float = 0.10,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_token = d_token
        
        # === Factor Tokenizer ===
        # 每个因子独立的 Linear(1, d_token) 映射
        # 实现为 weight (input_size, d_token) + bias (input_size, d_token)
        self.token_weight = nn.Parameter(torch.empty(input_size, d_token))
        self.token_bias = nn.Parameter(torch.empty(input_size, d_token))
        
        # === CLS Token ===
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        
        # === Transformer Blocks ===
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(d_token),
                'attn': nn.MultiheadAttention(
                    embed_dim=d_token,
                    num_heads=n_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                ),
                'norm2': nn.LayerNorm(d_token),
                'ffn': nn.Sequential(
                    nn.Linear(d_token, d_ffn),
                    nn.GELU(),
                    nn.Dropout(ffn_dropout),
                    nn.Linear(d_ffn, d_token),
                    nn.Dropout(ffn_dropout),
                ),
            }))
        
        # === Prediction Head ===
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )
        
        # 初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Token embedding: 缩放初始化
        d = self.d_token
        nn.init.uniform_(self.token_weight, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d))
        nn.init.uniform_(self.token_bias, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d))
        nn.init.uniform_(self.cls_token, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d))
        
        # Linear 层 Kaiming 初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, input_size) — 99 个因子的标量值
        output: (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # === Factor Tokenizer ===
        # x: (B, 99) -> (B, 99, 1) -> 乘以 weight 加 bias -> (B, 99, d_token)
        tokens = x.unsqueeze(-1) * self.token_weight.unsqueeze(0) + self.token_bias.unsqueeze(0)
        
        # === Prepend CLS ===
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_token)
        tokens = torch.cat([cls, tokens], dim=1)          # (B, 100, d_token)
        
        # === Transformer Blocks ===
        for block in self.blocks:
            # Pre-norm Self-Attention
            normed = block['norm1'](tokens)
            attn_out, _ = block['attn'](normed, normed, normed)
            tokens = tokens + attn_out  # Residual
            
            # Pre-norm FFN
            normed = block['norm2'](tokens)
            ffn_out = block['ffn'](normed)
            tokens = tokens + ffn_out   # Residual
        
        # === Extract CLS ===
        cls_output = tokens[:, 0, :]    # (B, d_token)
        
        # === Prediction Head ===
        output = self.head(cls_output)  # (B, 1)
        
        return output
```

### 4.3 MlpModel 改动

```python
class MlpModel(AlphaModel):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int] = (256,),
        model_type: str = "mlp",       # 新增: "mlp" 或 "factor_attention"
        # Factor Attention 超参数
        d_token: int = 32,
        n_heads: int = 4,
        n_attn_layers: int = 1,
        ...
    ):
        ...
        if model_type == "factor_attention":
            self.model = FactorAttentionNetwork(
                input_size=input_size,
                d_token=d_token,
                n_heads=n_heads,
                n_layers=n_attn_layers,
                ...
            )
        else:
            self.model = MlpNetwork(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
            )
```

### 4.4 mlp_signals.py 改动

```python
# Step 3a 配置
self.model_settings = {
    "model_type": "factor_attention",
    "d_token": 32,
    "n_heads": 4,
    "n_attn_layers": 1,
    # 训练超参数保持不变
    "n_epochs": 1000,
    "batch_size": 2048,
    "lr": 0.001,
    "early_stop_rounds": 40,
    "weight_decay": 0.002,
    "optimizer": "adam",
}
```

---

## 五、风险分析

### 5.1 与历史失败实验的区别

| 历史失败 | 失败原因 | 本方案的区别 |
|:---|:---|:---|
| IC-Loss / 混合损失 / 多任务损失 | 损失函数梯度冲突 | **不改损失函数**，仍用 MSE |
| 新增因子（Phase 5, 4个全失败） | 弱/共线因子稀释权重 | **不新增因子**，改变因子组合方式 |
| 批量因子剪枝 | 破坏输入空间结构 | **不删因子**，增加交互能力 |
| 多任务学习 | 辅助任务梯度干扰 | 单任务，无辅助头 |
| 扩大网络（Step 1） | 更多参数但同样结构 | **结构性改变**：从全连接到 Attention |

### 5.2 可能的风险

| 风险 | 概率 | 应对 |
|:---|:---|:---|
| 训练不收敛 | 低 | Attention 机制成熟，lr scheduler + early stopping 保底 |
| 过拟合 | 中 | 参数量（~17K）远小于当前 MLP（~66K），多层 dropout 防护 |
| 训练速度变慢 | 确定 | Attention 计算量 O(100^2 × 32) ≈ 320K FLOPs/sample，可接受 |
| Attention 退化为均匀分布 | 低 | 如发生说明因子交互无增量信息，属于有效的零结果 |
| 非牛市仍无改善 | 中 | 核心瓶颈可能不在因子交互而在因子信号本身，Attention 无法创造不存在的信号 |

### 5.3 预期效果分析

**乐观情景**：Attention 有效学习到 regime-dependent 的因子交互权重，等价于自动发现多个类似 `turnover_x_bull` 的条件化组合。非牛市表现改善，Sharpe 提升。

**中性情景**：Attention 未能发现比 MLP 更好的交互模式，但参数量更小、正则化更强，Sharpe 持平或略降。这说明当前因子体系的交互空间有限。

**悲观情景**：Attention 机制增加的复杂度超过收益，训练不稳定或过拟合，Sharpe 明显下降。需回退。

---

## 六、实验记录

### 6.1 Step 3a Round 1 (d_token=32, 1 层, 4 heads) — 未通过

| 指标 | Step 1 MLP 基线 | Step 3a R1 | 变化 |
|:---|:---|:---|:---|
| 总收益 | 322.66% | 244.60% | -78% |
| Sharpe | 1.24 | 1.09 | -0.15 |
| MaxDD | -36.38% | -36.81% | 持平 |
| 最长回撤天数 | 24 | 127 | 恶化 |
| 收益回撤比 | 5.61 | 3.38 | -2.23 |

非牛市时段（2022Q1、2023Q2）显著改善，但牛市收益严重缩水（2024Q4: +23%→+3%）。
参数量 ~17K 远小于 MLP ~66K，模型容量不足以同时兼顾牛市和非牛市。

### 6.2 Step 3a Round 2 (d_token=64, 1 层, 4 heads) — 通过

**最优配置，当前采用。**

| 指标 | Step 1 MLP 基线 | Step 3a R2 | 变化 |
|:---|:---|:---|:---|
| 总收益 | 322.66% | **374.96%** | **+52%** |
| 年化 | 74.82% | ~87% | +12% |
| Sharpe | 1.24 | **1.50** | **+0.26** |
| MaxDD | -36.38% | **-24.55%** | **改善 12%** |
| 最长回撤天数 | 24 | 21 | 改善 |
| 收益回撤比 | 5.61 | **8.22** | **+46%** |

**90 天区间非牛市对比**：

| 时段 | MLP 基线 | FactorAttention | 说明 |
|:---|:---|:---|:---|
| 2022-01~04 | -9.34% | +11.08% | 大幅改善 |
| 2023-04~07 | -15.72% | +9.10% | 显著改善 |
| 2023-07~10 | -9.22% | -1.71% | 改善 |
| 2024-04~07 (国九条) | -16.80% | -13.08% | 改善 |
| 2026-01~04 (关税冲击) | -4.21% | +0.58% | 转正 |
| 2024-07~10 (牛市) | 58.98% | 31.36% | 下降但仍强 |
| 2025-04~07 (牛市) | 31.53% | 26.16% | 轻微下降 |

### 6.3 Step 3b (d_token=64, 2 层, 8 heads) — 未通过

| 指标 | Step 3a R2 (1层) | Step 3b (2层) | 变化 |
|:---|:---|:---|:---|
| 总收益 | 374.96% | 314.92% | -60% |
| Sharpe | 1.50 | 1.31 | -0.19 |
| MaxDD | -24.55% | -33.13% | 恶化 |
| 最长回撤天数 | 21 | 79 | 恶化 |
| 收益回撤比 | 8.22 | 5.66 | -2.56 |

增加 Attention 深度（2 层）导致过拟合，性能全面不如 1 层版本。

### 6.4 结论

最优架构 = **1 层 Self-Attention, d_token=64, 4 heads, FFN=128, head=128**。

关键发现：
1. **Factor Self-Attention 结构性突破**：在不改因子、不改标签、不改损失函数的前提下，仅通过模型结构改变实现 Sharpe 1.24→1.50 的提升
2. **模型容量敏感**：d_token=32 容量不足（Sharpe 1.09），d_token=64 最优（Sharpe 1.50），2 层过拟合（Sharpe 1.31）
3. **非牛市改善机制**：Attention 自动学习到因子间 regime-dependent 交互，等价于模型自动发现了多个类似 turnover_x_bull 的条件化组合
4. **与 IC-Loss 的本质区别**：IC-Loss 改善非牛市但必然损害牛市（损失函数层面的 trade-off）。Factor Attention 同时改善两者，因为它提升的是因子利用效率而非改变优化目标

---

## 七、经验沉淀

1. **模型结构改变 > 损失函数改造**：相同因子+相同标签+相同 MSE 损失，仅换 MLP→FactorAttention 就能打破牛市/非牛市 trade-off
2. **1 层 Attention 在日频截面选股中最优**：110 万训练样本足以支撑 ~50K 参数的 1 层 Attention，但 2 层过拟合
3. **d_token 是核心超参数**：32→64 带来质变（Sharpe +0.41），影响远大于 n_heads 和 n_layers
4. **FT-Transformer 在金融 tabular data 上的有效性得到验证**：因子间动态交互是 MLP 无法隐式学习的关键信息

---

## 八、与后续方向的关系

Factor Attention 成功后的潜在探索方向：

1. **Attention 权重可解释分析**：提取 Attention map 看"换手率最关注哪些因子"，为因子工程提供数据驱动方向
2. **在 Attention 框架下重新测试弱因子**：已验证，见 Step 4
3. **加入 regime token**：将市场状态作为特殊 token 参与 Attention，进一步增强条件化能力

---

## 九、V10 Step 4: Attention 框架下弱因子重测

> 基线：V10 Step 3a (Factor Attention d=64, 1层, cord_20 未加入前 Sharpe 1.50)

### 动机

Factor Attention 可能让之前在 MLP 下失败的弱因子重新发挥作用（Attention 通过动态交互提取弱信号，MLP 给因子固定权重无法做到）。

### Step 4a: cord_20 (量价同步性)

- **因子定义**：20 日收益率变化与成交量变化的相关系数
- **IC**: 0.061 (Overall)
- **之前 MLP 表现**：Sharpe 1.29→1.01，彻底失败

**Attention 框架下结果**：

| 指标 | Step 3a 基线 | + cord_20 (Run 1) | + cord_20 (Run 2) |
|:---|:---|:---|:---|
| Sharpe | 1.50 | 1.84 | 1.76 |
| 总收益 | 375% | 606% | 533% |
| MaxDD | -24.55% | -20.64% | -20.64% |
| 收益回撤比 | 8.22 | 6.04 | 8.72 |

**结论：通过。** cord_20 在 Attention 框架下全面提升，两次训练均显著优于基线。验证了 Attention 可以利用 MLP 无法利用的弱信号。

### Step 4b: klow_2_20d (下影线占比)

- **因子定义**：(min(O,C) - L) / (H - L) 的 20 日均值，衡量买盘承接力
- **IC**: -0.061
- **之前 MLP 表现**：Sharpe 降低 -0.30

**Attention 框架下结果（在 cord_20 基础上叠加）**：

| 指标 | cord_20 基线 | + klow_2_20d |
|:---|:---|:---|
| Sharpe | 1.84 | 1.24 |
| 总收益 | 606% | 262% |
| MaxDD | -20.64% | -29.71% |
| 收益回撤比 | 6.04 | 3.31 |

**结论：失败，已回退。** klow_2_20d 即使在 Attention 框架下仍然有害。可能原因：
1. 下影线在 A 股涨跌停机制下信息含义模糊（涨停时 H=L=O=C，下影线为 0）
2. 该因子与现有波动率因子高度相关，引入冗余噪声
3. 不是所有 IC 通过门槛的弱因子都能被 Attention 利用

### Step 4 经验总结

1. **Attention 并非万能**：cord_20 成功但 klow_2_20d 失败，说明 Attention 提升弱因子利用率有条件——因子本身需要有独立信息量
2. **量价同步性是有效弱信号**：cord_20 (IC=0.061) 虽然 IC 低，但包含了价量关系的独立信息维度，Attention 通过交互放大了这个信号
3. **K 线形态因子在日频截面模型中可能不适用**：klow_2_20d 的信息已被波动率/振幅因子覆盖
