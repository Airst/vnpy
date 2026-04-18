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

## 六、实施节奏

```
Step 3a（轻量级 FactorAttention）
  |-- 新增代码行数：~100 行（FactorAttentionNetwork 类）
  |-- 改动文件：3 个
  |-- 风险：中（纯结构改变，不碰因子/标签/损失函数）
  |-- 验证：全区间回测，对比 V10 Step 1 基线
  |-- 通过标准：Sharpe >= 1.10, 总收益 >= 250%

Step 3b（完整 FT-Transformer，仅 Step 3a 通过后执行）
  |-- 在 Step 3a 基础上调整超参数
  |-- d_token: 32→64, n_layers: 1→2, n_heads: 4→8
  |-- 验证标准同上，期望在 Step 3a 基础上进一步提升
```

---

## 七、与后续方向的关系

Factor Attention 如果成功，会产生额外价值：

1. **Attention 权重可解释**：提取 Attention map 可以看到"换手率最关注哪些因子"，为后续因子工程提供方向
2. **为后续因子扩展降低风险**：新增因子不再直接挤占 MLP 权重，而是通过 Attention 机制动态融合。之前失败的弱因子（如 cord_20）可能在 Attention 框架下有正贡献
3. **渐进式升级路径**：Step 3a → Step 3b → 更深层 Transformer → 加入 regime token 等
