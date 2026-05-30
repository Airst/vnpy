# vnpy 量化交易系统

---

## 一、项目基础信息

### 1.1 项目说明

构建基于 vnpy 的 A 股量化分析系统，使用 Factor Attention（FT-Transformer）模型进行日线级别因子选股，持续获取 alpha 收益。

- 量化的内核是寻找市场波动的规律，从波动中获利
- 挖掘因子让模型发现市场波动规律、适应市场风格；不用复杂参数适应特定风格，避免过拟合
- 截面选股模型回答"选哪些股票"，不回答"市场涨还是跌"——系统性风险由风控层处理
- 结合 A 股量化主流研究成果进行分析

### 1.2 环境与命令

- 操作系统：WSL Ubuntu
- Python 路径：`/home/airst/Workspace/.venv/bin/python`
- 训练命令：`python training.py -v{版本号} -t --index {指数代码}`（**必须传 --index**，否则全市场股票会 GPU OOM。常用：`000852.SH,399303.SZ` 或单独 `399303.SZ`）
- GP 挖掘命令：`python gp_mining.py -v{版本号}`（独立脚本，支持 `--pop`/`--gen`/`--max-factors` 等超参数）
- GP 注册表管理：`python gp_mining.py -v{版本号} --status/--accept/--reject/--test/--note`
- 训练日志：`log/run_v{版本号}.log`
- 回测结果：`core/alpha_db/backtest/` 下 JSON 文件

### 1.3 信息索引

AI 在工作时应按需读取以下位置获取信息，而非依赖本文件中的静态记录：

| 需要了解的信息 | 去哪里看 |
|:---|:---|
| 当前版本状态、因子数量、关键指标 | 当前主力版本的 factor_calculator.py 文件头部 docstring |
| 模型架构、超参数配置 | `core/alpha/mlp_signals.py` 文件头部 docstring |
| 模型实现细节 | `vnpy/alpha/model/models/mlp_model.py` 文件头部 docstring |
| 风控设计与参数 | `core/risk_controller.py` 文件头部 docstring |
| 策略逻辑 | `core/strategies/multifactor_strategy.py` 文件头部 docstring |
| 历史迭代的详细实验数据 | `docs/iterations/` 下对应版本文档 |
| 量化研究知识沉淀 | `docs/knowledge/` 下各知识条目 |
| 各版本的因子演进和失败记录 | 对应版本 `v*_factor_calculator.py` 文件头部 docstring |
| GP 因子注册表（生命周期状态） | `core/alpha/gp_factors.json` |
| GP 挖掘模块设计与算子体系 | `core/alpha/gp_factor_miner.py` 文件头部 docstring |

### 1.4 代码目录

```
core/
├── alpha/                        # 因子计算与信号生成（核心）
│   ├── factor_calculator.py      # 基类：GPU 张量准备、横截面/时间序列辅助函数
│   ├── v*_factor_calculator.py   # 各版本因子计算器（v8/v9/v10/v11/v101/v158）
│   ├── data_loader.py            # 数据加载：OHLCV、日频基础、财务、概念、资金流、筹码
│   ├── gp_factor_miner.py        # GP 遗传编程因子挖掘模块
│   ├── gp_factors.json           # GP 因子注册表（生命周期管理）
│   ├── engine.py                 # AlphaEngine 编排器
│   ├── mlp_signals.py            # 滚动训练与信号生成
│   └── concept_embedding.py      # 概念板块特征
├── strategies/
│   └── multifactor_strategy.py   # 多因子组合策略
├── risk_controller.py            # 组合级风险控制
├── selector/
│   └── selector.py               # FundamentalSelector 选股宇宙过滤
├── core_service.py               # 后端服务
├── trade_service.py              # 实盘交易
├── main_controller.py            # FastAPI REST API
└── alpha_db/                     # 数据存储（Parquet + JSON）

data_manager/ts_downloader/       # Tushare 数据下载管理
vnpy/alpha/                       # vnpy 框架 Alpha 模块（模型、回测、策略模板）
core/tools/                       # 分析工具脚本
docs/iterations/                  # 迭代实验文档
docs/knowledge/                   # 量化知识库
```

入口脚本（根目录）：
- `main.py`: FastAPI/Uvicorn Web UI 服务器
- `run.py`: vnpy 桌面 GUI
- `training.py`: 主训练脚本（动态版本发现 → 数据下载 → 因子计算 → 训练 → 回测）
- `gp_mining.py`: GP 因子挖掘独立脚本（因子发现 → 滚动IC验证 → 注册表管理）

---

## 二、迭代工作流程

每一轮迭代严格遵循以下步骤闭环。**每轮迭代计划中仅保留一个方案阶段**，多种思路拆分为小版本分别迭代。

### Step 1：回测数据分析

- 运行当前版本训练和回测，获取最新回测结果
- 分析整体指标：总收益、年化、Sharpe、MaxDD、收益回撤比
- **重点分析非牛市表现**：提取 90 天区间收益，单独计算非牛市时段的年化收益
- 识别当前版本的核心问题和瓶颈

### Step 2：解决方案分析

- 基于 Step 1 识别的问题，结合"量化研究准则"中的经验教训，分析可能的改进方向
- **必须先阅读 `docs/knowledge/` 下相关知识条目**，确认方案不违反已有结论
- **必须阅读当前版本 factor_calculator 文件头部的失败记录**，避免重复尝试
- 关键问题的解决方案需要从网络上搜索量化交易研究相关知识进行分析
- 每个方案必须明确：改什么、为什么有效、预期效果、验证标准

### Step 3：制定迭代计划

- 明确改动范围（改标签 / 加因子 / 改模型 / 改风控等）
- 记录迭代计划到 `docs/iterations/` 对应的版本文档，多种思路拆分成多个小版本迭代

### Step 4：执行因子计算与训练

- 按计划修改代码
- 训练命令：`python training.py -v{版本号} -t`，训练耗时较长
- 使用 `ps` 命令查看进程执行情况。杜绝在进程仍在运行是直接读取日志输出

### Step 5：回测数据分析（验证）

- 对比迭代前后的回测指标（整体 + 非牛市分时段）
- 判定迭代结果：
  - **通过**：Sharpe 不降 + 目标指标改善 → 保留改动，更新基线
  - **失败**：Sharpe 下降或目标指标未改善 → 询问用户是否回退代码，等待下一步指令
- 如需计算，必须通过创建分析计算脚本，不要猜测

**整体指标对比**：

| 指标 | 含义 | 判断标准 |
|:---|:---|:---|
| 总收益 / 年化 | 绝对盈利能力 | 不低于基线 |
| Sharpe | 风险调整收益 | **核心指标**，不降为通过底线 |
| MaxDD | 最大回撤 | 越小越好，但不以牺牲 Sharpe 为代价 |
| 收益回撤比 | 综合效率 | 辅助判断 |

**非牛市表现（重点）**：

- 提取 90 天区间收益分解，划分牛市/非牛市时段
- 迭代目标：非牛市年化显著改善，同时牛市收益不大幅下降
- 关注亏损集中的时段，分析是否属于系统性外生事件

**关键认知**：

- 系统性外生冲击（地缘政治、关税黑天鹅）超出截面因子信息边界，此类回撤只能由风控层兜底
- Factor Attention 通过注意力权重实现动态因子加权，但仍是截面打分器——无法预测市场整体方向
- 同一代码多次训练结果可能差异较大（随机种子/窗口边界敏感），单次回测结论需谨慎

### Step 6：记录与沉淀

- 将本轮迭代的方案、数据、结论写入 `docs/iterations/` 版本文档
- 如产生新的经验教训，更新本文件"量化研究准则"部分
- **更新相关代码文件的模块级 docstring**（版本演进、设计决策、失败记录）
- 新发现的量化知识沉淀到 `docs/knowledge/` 中
- 代码提交

**迭代失败时**：记录失败结论到代码文件 docstring 的"失败记录"中，询问用户是否回退，等待用户指令。不自行发起新一轮迭代。

---

## 三、代码自描述规范

每个核心代码文件的头部必须包含模块级 docstring，格式如下：

```python
"""
<文件标题>

== 版本演进 ==
V8 → V9: <变更摘要>
V9 → V10: <变更摘要>
...

== 当前状态 ==
<当前版本、关键配置、因子数、指标等>

== 设计决策 ==
- <为什么选择这个方案>
- <关键参数的选择理由>

== 失败记录 ==
- <失败实验1>: <原因简述>
- <失败实验2>: <原因简述>
"""
```

**规则**：
1. 每次迭代修改代码文件时，必须同步更新该文件的模块级 docstring
2. 失败记录必须保留，避免后续重复尝试已证明无效的方向
3. docstring 应简洁，每条记录控制在 1~2 行；详细实验数据放 `docs/iterations/`
4. 需要维护 docstring 的核心文件：`v*_factor_calculator.py`、`mlp_signals.py`、`mlp_model.py`、`engine.py`、`risk_controller.py`、`multifactor_strategy.py`

---

## 四、量化研究准则

以下准则从历次迭代实验中提炼，是迭代工作中必须遵守的规则。随着新实验的开展持续沉淀更新。

### 4.1 因子评估准则

1. **因子理论支撑**：新增的因子必须要有量化研究理论支撑
2. **结构性提升**：补全当前因子体系的结构性缺失（例如当前体系以换手率、动量为主导）
3. **禁止唯IC论**：IC绝对值不能作为评估因子是否有效的唯一标准
4. **因子与框架匹配性检查**：引入学术因子前，必须验证数据粒度、更新频率、调仓周期、选股宇宙是否匹配本系统（日频截面选股）。详见 `docs/knowledge/factor_data_granularity.md`

### 4.2 核心方法论

1. **失败反思**：因子引入后效果不佳时，不要首先想到回退，而是分析原因，是否应该改造后重试
2. **不硬编码市场观点**：让模型从原子因子中自主学习，不用复杂参数适应特定市场风格
3. **标签设计优先于因子工程**：Phase 1 改标签一步将非牛市年化从 5% 提到 21%
4. **交互因子方法成立但受底层因子约束**：底层因子本身必须有效
5. **学术因子不等于 A 股有效因子**：在美股/回归模型中有效，不代表在 A 股日频截面选股框架下有效
6. **模型层与风控层职责分离**：模型层解决长期因子有效性，风控层解决短期极端事件，不混淆
7. **参考知识库**：每次迭代决策前，必须先阅读 `docs/knowledge/` 下的相关知识条目
8. **损失函数改造是双刃剑**：IC-Loss 改善非牛市但损害牛市，混合损失梯度冲突效果最差。详见 `docs/knowledge/ic_loss_experiment.md`
9. **训练采样策略不可轻易改变**：均匀采样窗口提供 regime 多样性，是隐式正则化
10. **因子有效性与损失函数深度耦合**：同一因子集在不同损失函数下表现截然不同
11. **多任务学习在截面选股框架下失败**：损失函数层面改造已连续 3 次失败，该方向暂停。详见 `docs/iterations/v10_architecture_multitask.md`
12. **Factor Self-Attention 有效**：模型结构改变 > 损失函数改造。详见 `docs/iterations/v10_step3_factor_attention.md`
13. **Attention 框架下弱因子重测有条件成功**：需提供独立信息维度，信息冗余的因子仍然失败
14. **Gate Network 在 Factor Attention 框架下失败**：Attention 已实现动态加权，Gate 冗余且过拟合
15. **动量崩溃检测在 A 股截面选股中无效**：动量维度因子工程已触及天花板。详见 `docs/iterations/v11_momentum_crash_detection.md`
16. **GP 因子需要去重**：结构同质化的 GP 因子集体加入会稀释 attention 权重，应按子树相似性聚类后保留代表性因子。V15.3 将 22 个 validated 因子缩减到 13 个后 Sharpe 从 1.36 提升到 1.74。详见 `docs/iterations/v15_step2_dedup_ensemble.md`
17. **验证集长度影响 early stopping 可靠性**：50 天验证集易被短期市场偏差误导，100 天（覆盖约 4 个月行情）可提供更稳定的 stop signal
18. **Multi-seed ensemble 显著降低 OOS variance**：3-seed (42/123/2024) 训练 + 预测均值聚合，能有效消除单次训练的随机性（权重初始化、batch 采样、dropout mask），是低成本高收益的稳健化手段
19. **季频数据在日频截面框架下信号过滤效率低**：股东人数等季频数据公告滞后 1~3 个月，被价量数据先行反映；强行引入会带来同质化重复信号（如 holder_change_qoq 与 avg_holding_size_change 数学等价、avg_holding_size_log 是 size 因子变种），稀释 attention 权重导致 Sharpe 1.74→1.09。详见 `docs/iterations/v16_holder_number_failed.md`
20. **GP 因子信号空间在当前算子体系下已趋于饱和**：经过 6 轮挖掘（50 个候选，13 个 validated），新一轮发现的候选因子大量集中于 `cs_zscore(BMD)`/`ts_cov(log(PB),neg(X))` 等已有信号维度的变体，无法提供真正增量信息。GP 算子扩展（加入新终端如财务数据）或更换搜索空间是下一步方向
21. **执行层复杂仓位调节是 Sharpe 杀手**：在 5 持仓 + 日频 + T+1 + 整手 100 + 排名归一化信号的框架下，"等权入场 + score 阈值清仓"已是接近最优的简单规则；V18 系列四类机制（SignalScale 入场分档 / loss_cut 浮亏减仓 / signal_fade 信号衰减分档 / pyramid 金字塔加仓）全部 Sharpe 下降。详见 `docs/iterations/v18_dynamic_position.md`
22. **A 股牛市轮动反对集中持仓**：基于信号强度的入场分档（Top1=1.5x）在 A 股板块快速轮动的牛市中损害收益捕获（V18.1: 24Q3 牛市少赚 43%）。学术上的 IC × signal_strength 加权（Markowitz 改良）在 A 股日频截面框架下不成立
23. **减半仓在 A 股日频框架下产生双重摩擦**："Cut losses short" 经典法则在 T+1 + 整手 100 约束下：减半后反弹错失收益 OR 减半后继续跌触发硬止损（两次卖在不同价位），交易笔数飙升 13%~28%，手续费 + 滑点累计成本超过 alpha 改善
24. **pyramid 加仓与排名退出存在结构性互斥**：基于持仓盈亏的加仓机制必须先解决与"基于信号排名退出"的优先级冲突；不改 sell_threshold 则 pyramid 0 次触发，改了则副作用大于加仓收益（V18.3b 实证）。该方向暂停

---

## 五、工作方法

1. **坚守量化研究准则**：每次迭代决策前对照第四章准则检查，避免重复踩已知的坑
2. **持续沉淀经验教训**：每轮迭代结束后，将新发现的规律更新到第四章
3. **代码自描述优先**：迭代信息写入代码文件 docstring，而非本文件
4. **严格遵循迭代流程**：按第二章的 6 步闭环执行，不跳步、不合并多个方案到一轮迭代
5. **迭代失败时止步等待**：记录结论后询问用户是否回退，等待用户指令，不自行发起下一轮
6. **知识库驱动决策**：迭代前阅读 `docs/knowledge/` 和代码文件 docstring 中的失败记录
