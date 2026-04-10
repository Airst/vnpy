# vnpy 量化交易系统

## 环境与命令

- 操作系统：WSL Ubuntu
- Python 路径：`/home/airst/Workspace/.venv/bin/python`
- 训练命令：`python training.py -v9 -t`（`-t` 为必需参数，`-v` 指定因子版本），训练耗时较长
- 训练日志：脚本自动写入 `trainingV9.txt`（项目根目录， 文件后缀V9是因子版本号）
- 回测结果：`core/alpha_db/backtest/` 下 JSON 文件，按日期时间戳命名

## 项目目的

构建基于 vnpy 的 A 股量化分析系统，使用 MLP 模型进行日线级别因子选股，持续获取 alpha 收益。

## 当前版本：V9（Phase 1 + Phase 3）

V9 从 V8 重新出发，通过两阶段改造将模型从 beta 策略转为 alpha 策略：

### Phase 1：Beta-Neutral Label
- 将标签从 `cs_rank(raw_ret_5)` 改为 `cs_rank(raw_ret_5 - beta_20d * mkt_ret_5)`
- 去除市场 beta 对标签的污染，迫使模型学习真正的 alpha
- 效果：非牛市年化从 5.39% 提升至 21.10%

### Phase 3：简化 Dragon Score
- 移除 dragon_score 中 ~130 行硬编码 regime 逻辑（三态概率混合、7 个惩罚项）
- 保留简化版 `dragon_score = combined_mom + rank_turnover * tanh(mom_20d * 5.0)`
- 所有原子因子（meta_bull_prob, bear_trap_score 等）保留为 MLP 输入，由模型自主学习 regime 交互
- 效果：全面优于 Phase 1，总收益 +54%，Sharpe +0.16

### Phase 2（已失败回退）
- 尝试添加质量因子（ROE/毛利率/增长率等），IC 分析后仅 4 个过阈值
- 财务数据季频更新与日频模型 90 天重训窗口不匹配，净负面
- 教训：季频数据不适合日频 MLP 模型

### 当前回测表现（含风控模块，-15% 起调）
```
总收益: 249.58%  年化: 58.27%  Sharpe: 1.23  MaxDD: -22.59%  收益回撤比: 2.48
```

### V8 基线（对照）
```
总收益: 167%  年化: ~40%  Sharpe: 0.95  MaxDD: -35.28%
核心问题: 过度依赖 beta，非牛市年化仅 5.39%，2024H1 亏损 -20.6%
```

## 工作方法论

### 因子迭代铁律
1. **每次只改一个变量**，验证后再叠加下一个
2. **IC 门槛**：新因子 |IC| >= 0.03 才保留，低于此值视为噪声
3. **不硬编码市场观点**：不用复杂参数适应特定市场风格，让 MLP 从原子因子中自主学习
4. **记录实验结果**：每次迭代将结果写入 `V9_REFORM_PLAN.md`，避免长周期迭代遗忘

### 因子评估流程
1. 计算因子 IC（训练日志中自动输出 Rolling Window IC 表）
2. 筛选 |IC| >= 0.03 的因子
3. 单独或分组加入训练，对比回测指标
4. 未通过的因子立即回退，不累积噪声

### 回测分析要点
- 不只看总收益，必须分时段对比（非牛市 / 牛市 / 回调期）
- 关注 Sharpe、MaxDD、收益回撤比的综合表现
- 90 天区间收益分解可快速定位亏损时段

### 风险控制方法论
- **模型层**（alpha model）解决长期有效性：因子质量、标签设计、防过拟合
- **组合管理层**（portfolio risk）解决短期极端事件：回撤熔断、波动率缩仓
- 两者职责分离：地缘政治等外生冲击不在价格因子信息边界内，不应在模型层硬编码应对逻辑

## 文件目录

```
core/
├── alpha/                      # 因子计算与信号生成（核心）
│   ├── factor_calculator.py    # 基类：GPU 张量准备、横截面/时间序列辅助函数
│   ├── v8_factor_calculator.py # V8 基线，~80+ 因子
│   ├── v9_factor_calculator.py # V9 当前版本（Phase 1 + Phase 3）
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

### 入口脚本（根目录）
- `main.py`: FastAPI/Uvicorn Web UI 服务器，监听 8000 端口
- `run.py`: vnpy 桌面 GUI，启动 Qt + ToraStock + PortfolioStrategy
- `training.py`: 主训练脚本，编排：数据下载 → 因子计算 → MLP 训练 → 信号生成 → 回测

### 数据管理（data_manager/ts_downloader/）
- `daily_basic_manager.py`: PE/PB/PS/换手率/市值
- `stock_info_manager.py`: 股票列表
- `fina_indicator_manager.py`: 财务指标（ROE/ROA/利润率/增长率）
- `concept_manager.py`: 概念板块分类及日频数据
- `moneyflow_manager.py`: 资金流（小/中/大/超大单）

### vnpy 框架 Alpha 模块（vnpy/alpha/）
- `lab.py`: AlphaLab 数据管理（K 线、数据集、模型、信号的存取）
- `dataset/template.py`: AlphaDataset 特征工程流水线
- `model/models/mlp_model.py`: MlpModel（PyTorch MLP，early stopping，特征重要性）
- `strategy/backtesting.py`: 回测引擎（Plotly 可视化，滑点/佣金建模）
- `strategy/template.py`: AlphaStrategy 抽象策略类

### 关键文档
- `V9_REFORM_PLAN.md`: V9 三阶段改造方案及实验结果记录

## 风险控制模块（risk_controller.py）

组合级风险控制，独立于模型层，在策略执行阶段动态调整 max_holdings。

### 风险信号
1. **组合回撤**（主信号）：从滚动净值峰值计算 drawdown，阶梯式减仓
   - < -15%: max-1, < -20%: max-2, < -25%: max-3, < -30%: max-4, < -35%: 全现金
   - 教训：-5% 起调过于敏感，导致正常波动中频繁触发，反复「卖低→错过反弹」，全面恶化表现
2. **波动率飙升**（辅助）：20 日组合收益率 std > 60 日 std × 2 时，额外 -1

### 执行逻辑
- **减仓**：即时生效，按信号分数升序强制卖出超额持仓
- **恢复**：每次 +1，间隔至少 3 个交易日，由正常信号自然买入填充
- **零仓死锁修复**：max_holdings=0 时净值冻结，drawdown 永远无法恢复；cooldown 后重置 peak_equity 打破死锁

### A/B 测试验证（同一信号文件）
```
阈值 -15% 起调: 总收益 249.58%, 年化 58.27%, Sharpe 1.23, MaxDD -22.59%, 收益回撤比 2.48
无风控基线:      总收益 232.91%, 年化 54.37%, Sharpe 1.18, MaxDD -24.59%, 收益回撤比 2.26
```

## 历史经验总结

### 失败实验记录
| 版本 | 改动 | 结果 | 教训 |
|------|------|------|------|
| V9 12 因子 | 一次加 12 个新因子 | 11/12 IC<0.06，退化 | 低 IC 因子是噪声 |
| V9.5~V9.8 | label/资金流/牛熊不对称 | 全部失败 | 多变量同时改动无法隔离原因 |
| V9 策略层 | 基于 score 做 regime 自适应 | 379%→86% | 不理解数据分布就调参必败 |
| V9 Phase 2 | 质量因子（ROE/毛利率等） | Sharpe 下降 | 季频数据不匹配日频模型 |
| V9 风控 -5% | 回撤 -5% 起调减仓 | 全面恶化（MaxDD -32%） | 截面选股正常波动中频繁触发，卖低错过反弹 |

### 策略执行优化（已生效）
- 硬止损 5%，追踪止损 15%
- 冷却期 3 天（止损后不立即买回）
- 信号持续性校验（persistence_days=3）
- Anti-chasing：近 5 日涨幅 >12% 不买入
- high_price 追踪使用 bar.high_price

## 量化思路

- 量化的内核是寻找市场波动的规律，从波动中获利
- 挖掘因子让 MLP 发现市场波动规律、适应市场风格；不用复杂参数适应特定风格，避免过拟合
- 结合 A 股量化主流研究成果进行分析
- 截面选股模型回答"选哪些股票"，不回答"市场涨还是跌"——系统性风险由风控层处理
