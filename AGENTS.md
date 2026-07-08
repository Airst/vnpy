# vnpy 量化交易系统

> 本文件是项目入口与信息索引。详细知识已沉淀到 `docs/` 下各专门文档，按需读取，不在此处重复记录。

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
- LLM 辅助挖掘：`python gp_mining_llm.py`（LLM 假设生成 + 表达式翻译，见 `core/alpha/llm_factor_miner.py`）
- GP 注册表管理：`python gp_mining.py -v{版本号} --status/--accept/--reject/--test/--note`
- 训练日志：`log/run_v{版本号}.log`
- 回测结果：`core/alpha_db/backtest/` 下 JSON 文件

### 1.3 信息索引

AI 在工作时应按需读取以下位置获取信息，而非依赖本文件中的静态记录：

| 需要了解的信息 | 去哪里看 |
|:---|:---|
| 迭代闭环工作流程（6 步 SOP） | `docs/loop/process.md` |
| 当前迭代目标与验收标准 | `docs/loop/goals.md` |
| 开放问题清单 | `docs/loop/problems.md` |
| 验证记录流水 | `docs/loop/verification_log.md` |
| 设计方案 | `docs/loop/design/` 下各文件 |
| Loop engineering 理念与文档空间说明 | `docs/loop/README.md` |
| 量化研究准则（必须遵守的硬约束） | `docs/knowledge/research_principles.md` |
| 代码自描述 docstring 规范 | `docs/knowledge/code_docstring_spec.md` |
| 工作方法 | `docs/knowledge/work_methods.md` |
| 量化研究知识条目 | `docs/knowledge/` 下各文件 |
| 当前版本状态、因子数量、关键指标 | 当前主力版本的 factor_calculator.py 文件头部 docstring |
| 模型架构、超参数配置 | `core/alpha/mlp_signals.py` 文件头部 docstring |
| 模型实现细节 | `vnpy/alpha/model/models/mlp_model.py` 文件头部 docstring |
| 风控设计与参数 | `core/risk_controller.py` 文件头部 docstring |
| 策略逻辑 | `core/strategies/multifactor_strategy.py` 文件头部 docstring |
| 历史迭代的详细实验数据 | `docs/iterations/` 下对应版本文档 |
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
│   ├── llm_factor_miner.py       # LLM 辅助因子挖掘模块（假设生成 + 表达式翻译）
│   ├── expression_translator.py  # 自然语言/结构 → GP 表达式翻译
│   ├── hypothesis_generator.py   # LLM 因子假设生成
│   ├── knowledge_base.py         # 因子挖掘知识库
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
docs/loop/                        # 迭代闭环运行状态（目标/问题/验证/设计）
docs/iterations/                  # 迭代实验文档（单版本归档）
docs/knowledge/                   # 量化知识库（准则/规范/方法论）
```

入口脚本（根目录）：
- `main.py`: FastAPI/Uvicorn Web UI 服务器
- `run.py`: vnpy 桌面 GUI
- `training.py`: 主训练脚本（动态版本发现 → 数据下载 → 因子计算 → 训练 → 回测）
- `gp_mining.py`: GP 因子挖掘独立脚本（因子发现 → 滚动IC验证 → 注册表管理）

---

## 二、文档结构迁移说明

本文件原为单体文档，现已按 loop engineering 理念拆分到 `docs/` 下各专门空间，便于持续迭代更新：

| 原章节 | 迁移去向 | 说明 |
|:---|:---|:---|
| 一、项目基础信息 | 保留本文件 §一 | 入口信息，不变 |
| 二、迭代工作流程 | `docs/loop/process.md` | 6 步闭环 SOP，归入 loop 空间 |
| 三、代码自描述规范 | `docs/knowledge/code_docstring_spec.md` | 长期规范，归入知识库 |
| 四、量化研究准则 | `docs/knowledge/research_principles.md` | 26 条硬约束，持续追加 |
| 五、工作方法 | `docs/knowledge/work_methods.md` | 长期方法论 |

> 历史文档（`docs/iterations/*.md`、代码 docstring）中出现的"AGENTS.md 第四章准则""AGENTS.md §4.x"等引用，对应内容现位于 `docs/knowledge/research_principles.md`，章节编号保持一致（4.1 / 4.2）。

新增的 loop engineering 文档空间（`docs/loop/`）用于记录跨版本闭环运行状态，与 `docs/iterations/`（单版本归档）和 `docs/knowledge/`（长期知识）分工互补，详见 `docs/loop/README.md`。
