# vnpy 量化交易系统

---

## 一、项目基础信息

### 1.1 项目说明

构建基于 vnpy 的 A 股量化分析系统，使用 MLP 模型进行日线级别因子选股，持续获取 alpha 收益。

- 量化的内核是寻找市场波动的规律，从波动中获利
- 挖掘因子让 MLP 发现市场波动规律、适应市场风格；不用复杂参数适应特定风格，避免过拟合
- 截面选股模型回答"选哪些股票"，不回答"市场涨还是跌"——系统性风险由风控层处理
- 结合 A 股量化主流研究成果进行分析

### 1.2 环境与命令

- 操作系统：WSL Ubuntu
- Python 路径：`/home/airst/Workspace/.venv/bin/python`

### 1.3 当前版本状态

**V9（Phase 1 + Phase 3 + Phase 4 turnover_x_bull）**

| 指标 | 值 |
|:---|:---|
| 总收益 | 297.59% |
| 年化 | 69.21% |
| Sharpe | 1.29 |
| MaxDD | -34.28% |
| 收益回撤比 | 5.86 |
| 因子数 | ~100 个 |
| 待解决问题 | 非牛市年化仅 ~5%，alpha 集中在牛市 |

改造路径：V8（beta 策略）→ Phase 1（beta-neutral label）→ Phase 3（去掉 dragon_score 硬编码 regime）→ Phase 4（仅保留 turnover_x_bull 交互因子）

### 1.4 文档结构

```
AGENTS.md                           # 本文件：项目知识、迭代流程、量化准则、工作方法
docs/iterations/
├── v9_base_line.md
└── v9_reform_plan.md               # V9 完整改造记录：Phase 1~5 全部实验过程和数据
docs/knowledge/                     # 量化交易知识库：每轮对话沉淀的研究结论和经验
├── factor_data_granularity.md      # 因子有效性与数据粒度/模型框架的匹配性
└── turnover_label_design.md        # 换手率因子与标签设计
```

| 文件 | 内容 | 用途 |
|:---|:---|:---|
| `docs/iterations/v9_base_line.md` | V9 基线说明 |  |
| `docs/iterations/v9_reform_plan.md` | V9 各 Phase 的实验方案、IC 分析、回测数据、结论 | 追溯具体实验细节 |
| `docs/knowledge/*.md` | 量化交易知识条目：因子研究、标签设计、模型机制等 | 迭代决策时参考，避免重复踩坑 |

新增版本迭代时，在 `docs/iterations/` 下创建对应文件（如 `v10_xxx.md`），并将提炼的经验教训更新到本文件"量化研究准则"中。每轮对话产生的量化知识沉淀到 `docs/knowledge/` 中。

### 1.5 代码目录

```
core/
├── alpha/                      # 因子计算与信号生成（核心）
│   ├── factor_calculator.py    # 基类：GPU 张量准备、横截面/时间序列辅助函数
│   ├── v8_factor_calculator.py # V8 基线，~80+ 因子
│   ├── v9_factor_calculator.py # V9 当前版本
│   ├── 101_factor_calculator.py # WorldQuant Alpha101 GPU 实现
│   ├── 158_factor_calculator.py # Alpha158 因子实现
│   ├── data_loader.py          # 数据加载：OHLCV、日频基础、财务指标、概念、资金流
│   ├── engine.py               # AlphaEngine 编排器
│   ├── mlp_signals.py          # MLP 滚动训练（90 天重训，隐藏层 64/32/16）
│   ├── concept_embedding.py    # 概念板块特征
│   └── data_columns_info.txt   # 88 列数据张量的列名索引映射
├── strategies/
│   └── multifactor_strategy.py # 多因子组合策略（信号驱动，止损，追踪止损，风控集成）
├── risk_controller.py          # 组合级风险控制（回撤熔断，波动率缩仓，动态 max_holdings）
├── selector/
│   └── selector.py             # FundamentalSelector（EP>0, 换手率>1%, ln_cap>=11.5）
├── core_service.py             # 后端服务：策略加载、回测编排
├── trade_service.py            # 实盘交易：ToraStock 网关
├── main_controller.py          # FastAPI REST API
└── alpha_db/                   # 数据存储（Parquet + JSON）
    ├── daily/                  # ~2214 只 A 股日频数据
    ├── model/                  # MLP 模型 pickle
    ├── signal/                 # 每日信号 parquet
    └── backtest/               # 回测结果 JSON
```

入口脚本（根目录）：
- `main.py`: FastAPI/Uvicorn Web UI 服务器，监听 8000 端口
- `run.py`: vnpy 桌面 GUI，启动 Qt + ToraStock + PortfolioStrategy
- `training.py`: 主训练脚本，编排：数据下载 → 因子计算 → MLP 训练 → 信号生成 → 回测

数据管理（data_manager/ts_downloader/）：
- `daily_basic_manager.py`: PE/PB/PS/换手率/市值
- `stock_info_manager.py`: 股票列表
- `fina_indicator_manager.py`: 财务指标（ROE/ROA/利润率/增长率）
- `concept_manager.py`: 概念板块分类及日频数据
- `moneyflow_manager.py`: 资金流（小/中/大/超大单）

vnpy 框架 Alpha 模块（vnpy/alpha/）：
- `lab.py`: AlphaLab 数据管理
- `dataset/template.py`: AlphaDataset 特征工程流水线
- `model/models/mlp_model.py`: MlpModel（PyTorch MLP，early stopping，特征重要性）
- `strategy/backtesting.py`: 回测引擎（Plotly 可视化，滑点/佣金建模）
- `strategy/template.py`: AlphaStrategy 抽象策略类

分析工具模块（core/tools/）：
- `ab_test_risk_control.py`: 风险管理模块ABtest分析工具
- `factor_toolkit_common.py`: 因子分析工具包公共模块（数据加载、模型加载、因子分组定义）
- `factor_evaluator.py`: 因子独立评估器（分组多空收益、IC/IC_IR、增量IC、因子相关性、IC衰减、因子换手率）
- `model_attribution.py`: 模型归因分析器（Permutation Importance、因子组贡献、分 regime 重要性、预测分布）
- `factor_ablation.py`: 因子消融框架（Leave-One-Out、因子组消融、分 regime 组消融、冗余因子检测）
- `v8_v9_compare.py`: V8 vs V9 回测全时段对比分析
- `v8_v9_nonbull_compare.py`: V8 vs V9 非牛市深度对比分析

---

## 二、迭代工作流程

每一轮迭代严格遵循以下步骤闭环。**每轮迭代计划中仅保留一个方案阶段**，多种思路拆分为小版本分别迭代。

### Step 1：回测数据分析

- 运行当前版本训练和回测，获取最新回测结果
- 分析整体指标：总收益、年化、Sharpe、MaxDD、收益回撤比
- **重点分析非牛市表现**：提取 90 天区间收益，单独计算非牛市时段（如 2022-01~2024-07）的年化收益
- 识别当前版本的核心问题和瓶颈

### Step 2：解决方案分析

- 基于 Step 1 识别的问题，结合"量化研究准则"中的经验教训，分析可能的改进方向
- 关键问题的解决方案需要从网络上搜索量化交易研究相关知识进行分析
- 每个方案必须明确：改什么、为什么有效、预期效果、验证标准

### Step 3：制定迭代计划

- 明确改动范围（改标签 / 加因子 / 改模型 / 改风控等）
- 记录迭代计划到 `docs/iterations/` 对应的版本文档，多种思路拆分成多个小版本迭代

### Step 4：执行因子计算与训练

- 按计划修改代码
- 训练命令：`python training.py -v9 -t`（`-t` 为必需参数，`-v` 指定因子版本），训练耗时较长
- 使用BashOutput观测输出结果

### Step 5：回测数据分析（验证）

- 训练日志：脚本自动写入 `trainingV9.txt`（项目根目录，文件后缀为因子版本号）
- 回测结果：`core/alpha_db/backtest/` 下 JSON 文件，按日期时间戳命名
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
- 当前非牛市参考区间：2022-01 ~ 2024-07（分析区间内累计收益：非牛市年华收益率）
- 迭代目标：非牛市年化显著改善，同时牛市收益不大幅下降
- 关注亏损集中的时段（如 2024-04~07 国九条冲击、2026-02~04 地缘+关税冲击），分析是否属于系统性外生事件

**关键认知**：

- 系统性外生冲击（地缘政治、关税黑天鹅）超出截面因子信息边界，此类回撤只能由风控层兜底，不应在模型层硬编码应对
- MLP 是无条件截面打分器，给因子固定权重，无法根据市场状态动态调整——这是"加价值因子牛市变差"的根因
- 同一代码多次训练结果可能差异较大（随机种子/窗口边界敏感），单次回测结论需谨慎

### Step 6：记录与沉淀

- 将本轮迭代的方案、数据、结论写入 `docs/iterations/` 版本文档
- 如产生新的经验教训，更新本文件"量化研究准则"部分
- 分析过程脚本更新到AGENTS.md中，代码目录的分析工具模块
- 代码提交，将最新迭代改造内容以及回测结果作为提交记录

**迭代失败时**：记录失败结论后，询问用户是否回退代码，并等待用户的下一步指令。不自行发起新一轮迭代。

---

## 三、量化研究准则

以下准则从历次迭代实验中提炼，是迭代工作中必须遵守的规则。随着新实验的开展持续沉淀更新。

### 3.1 因子评估准则

1. **因子理论支撑**：新增的因子必须要有量化研究理论支撑
2. **结构性提升**：补全当前因子体系的结构性缺失（例如当前体系以换手率、动量为主导）
3. **禁止唯IC论**：IC绝对值不能作为评估因子是否有效的唯一标准
4. **因子与框架匹配性检查**：引入学术因子前，必须验证数据粒度、更新频率、调仓周期、选股宇宙是否匹配本系统（日频MLP截面选股）。详见 `docs/knowledge/factor_data_granularity.md`


### 3.3 核心方法论

1. **失败反思**：因子引入后，回测分析效果不佳或者其反作用时，不要首先想到会退，而是更进一步分析原因，是否应该对因子进行改造后重试，寻找充分的理论依据
2. **不硬编码市场观点**：让 MLP 从原子因子中自主学习，不用复杂参数适应特定市场风格
3. **标签设计优先于因子工程**：Phase 1 改标签一步将非牛市年化从 5% 提到 21%；Phase 5 试了 4 个新因子全部失败
4. **交互因子方法成立但受底层因子约束**：turnover_x_bull 验证了"因子 x regime"思路，但底层因子本身必须有效（val_dv 全面无效导致 value_x_bear 失败）
5. **学术因子不等于 A 股有效因子**：Alpha158 因子在美股/回归模型中有效，不代表在 A 股 MLP 截面选股框架下有效
6. **模型层与风控层职责分离**：模型层解决长期因子有效性，风控层解决短期极端事件，不混淆
7. **参考知识库**：每次迭代决策前，必须先阅读 `docs/knowledge/` 下的相关知识条目，确认当前方案不违反已有结论。新发现的知识在迭代结束后沉淀到知识库中

---

## 四、工作方法

1. **坚守量化研究准则**：每次迭代决策前对照第三章准则检查，避免重复踩已知的坑
2. **持续沉淀经验教训**：每轮迭代结束后，将新发现的规律或教训更新到第三章"量化研究准则"中
3. **必要时构建新的分析脚本**：如需新的分析能力（如 permutation importance、因子相关性矩阵等），可构建独立脚本，并更新第一章"文档结构"
4. **严格遵循迭代流程**：按第二章的 6 步闭环执行，不跳步、不合并多个方案到一轮迭代
5. **迭代失败时止步等待**：记录结论后询问用户是否回退，等待用户指令，不自行发起下一轮
6. **知识库驱动决策**：迭代 Step 2（解决方案分析）阶段，必须先阅读 `docs/knowledge/` 下与当前问题相关的知识条目；迭代完成（无论成功或失败）后，将本轮新增的量化知识沉淀为新条目或更新已有条目
