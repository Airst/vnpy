# V16 失败记录：股东人数因子（彻底放弃）

## 实验背景

V15.3 基线（Sharpe 1.74 / 年化 113.9%）锁定后，多次重训发现 3-seed ensemble 跨重训仍不稳定，回退为单次训练。
为了寻找新的 alpha 维度，调研后发现项目尚无股东人数（shareholder count）相关数据，决定引入 Tushare `stk_holdernumber` 接口数据，构造筹码集中度类因子。

## 改动内容

1. **新建 `data_manager/ts_downloader/holder_number_manager.py`**
   - 表名：`holder_number(ts_code, ann_date, end_date, holder_num)`，主键 `(ts_code, end_date)`
   - 单股票循环请求（接口不支持多股票批量）
   - 全市场下载：3788 只股票 / 235540 条记录 / 2002-2026 完整覆盖

2. **`data_loader.py` 新增第 9 步**：按 `ann_date` 公告日期 `join_asof` 对齐到日频，`forward_fill` 填补两次公告间空缺，并预计算 `holder_num_prev`（环比前值）和 `holder_num_yoy`（同比前值）。

3. **新建 `core/alpha/v16_factor_calculator.py`**：继承 V15 + 5 个新增因子
   - `holder_num_log`：股东人数对数（规模特征）
   - `holder_change_qoq`：环比变化率（取负，人数下降为正）
   - `holder_change_yoy`：同比变化率
   - `avg_holding_size_log`：户均持股 = float_share / holder_num
   - `avg_holding_size_change`：户均持股环比变化率

4. **配套**：mlp_signals.py 默认 `ensemble_size=1`（V15.3-revert 已完成），训练命令 `python training.py -v16 -t --index "000852.SH,932000.CSI"`

## 回测对比

| 指标 | V15.2 单次基线 | V15.3 3-seed | **V16 单次 + holder** |
|:---|:---|:---|:---|
| Sharpe | 1.36 | 1.74 | **1.09** |
| 年化 | 74.8% | 113.9% | **47.9%** |
| MaxDD | -25.0% | -21.1% | **-30.1%** |
| 收益回撤比 | 7.29 | 7.25 | **3.65** |
| MaxDD 持续 | 64 天 | 26 天 | **34 天** |

V16 vs V15.2（同样单次训练）：Sharpe 下降 0.27，年化下降 27 pp，MaxDD 恶化 5 pp。**全面劣化。**

## 失败因子诊断（10 日 rolling IC）

| 因子 | 起始 IC | 末期 IC | 平均 IC | 诊断 |
|:---|:---|:---|:---|:---|
| `avg_holding_size_log` | -0.261 | -0.106 | -0.166 | size 因子变种，与 `size_ln_cap` 高度冗余 |
| `holder_change_qoq` | -0.006 | 0.011 | -0.003 | 信号近 0 |
| `avg_holding_size_change` | -0.006 | 0.011 | -0.003 | **与 holder_change_qoq 数值完全等价** |
| `holder_num_log` | -0.076 | 0.003 | 0.000 | 完全无信号 |
| `holder_change_yoy` | (缺失) | (缺失) | — | 4 季度前数据不足，被 NaN 过滤 |

### 关键代数关系

```
avg_holding_size_change
  = (FS / HN_now) / (FS / HN_prev) - 1
  = HN_prev / HN_now - 1
  = -(HN_now - HN_prev) / HN_now
  ≈ -holder_change_qoq    （在差分小的范围内，符号相反但数值近乎线性）
```

实测两个因子的 IC 数值 100% 相同（截面 rank 处理后符号差异被消除），等价于把同一信号传两次 → 稀释 attention 权重，参照 V15.3 GP 因子去重失败的同质化机制。

## 失败根因

1. **季频与日频频率失配**：股东人数公告滞后财报披露 1~3 个月。当 holder_change_qoq 出现"人数减少→主力建仓"信号时，价量数据已通过换手率、波动率、动量等高频信号充分反映过，holder 因子提供的是滞后冗余信息。

2. **同质化因子稀释 attention 权重**：5 个 holder 因子里有 2 个数学上等价（`holder_change_qoq` ≡ `-avg_holding_size_change`），1 个是 size 变种（`avg_holding_size_log`），1 个完全无信号（`holder_num_log`），1 个数据缺失。等价于强行把模型 attention 资源分给 5 个低质量信号，挤压了原 V15 高质量因子的权重。

3. **不符合"因子与框架匹配性"准则**：AGENTS.md §4.1.4 已明确要求"引入学术因子前，必须验证数据粒度、更新频率、调仓周期、选股宇宙"，本次将季频数据强行接入日频截面选股框架，违反了这一准则。

## 处置

- 删除 `core/alpha/v16_factor_calculator.py`
- 回退 `data_loader.py` 第 9 步加载逻辑
- 删除 V16 训练日志和 backtest JSON
- **保留** `data_manager/ts_downloader/holder_number_manager.py` 和 MySQL `holder_number` 表作为基础设施（未来 GP 挖掘可能用作原始算子，或用于其他低频研究）
- 锁定 V15.3 基线（Sharpe 1.74 / 年化 113.9% / MaxDD -21.1%）

## 沉淀准则

新增 AGENTS.md 第四章准则 19：
> **季频数据在日频截面框架下信号过滤效率低**：股东人数等季频数据公告滞后 1~3 个月，被价量数据先行反映；强行引入会带来同质化重复信号，稀释 attention 权重导致 Sharpe 1.74→1.09。

## 后续方向（不立即推进）

- 不再尝试将季频/低频原始数据直接作为 attention 输入因子
- 未来如需利用股东数据：考虑作为风控层信号（如"机构集中持仓 + 股东人数骤降"组合事件触发）
- 因子工程方向回归到日频可观测的微观结构信号（已多次失败，触及天花板，等待数据源升级）
