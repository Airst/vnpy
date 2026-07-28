# Auto-Research Loop — 交接文档

> 供下一个 Agent 接手。本文自包含：背景、已建文件、工作方法、验证状态、**未完成事项**、关键设计约束。
> 计划原件：`/home/airst/.claude/plans/cuddly-jingling-mist.md`。Steering 契约：`docs/loop/auto_research.md`。

---

## 1. 背景与目标

**问题**：因子选完后无法快速验证——一次完整训练+回测 ≈ 1 小时（MLP 滚动窗 35 个窗口）。因子组合/裁剪/挖掘无法高效验证。用户要 Agent 自主跑 research loop，但"试错成本高"（过拟 + 单 seed 噪声 + 算力浪费）。

**根因（已从 `log/run_v15.log` 实测确认）**：1 小时是 **MLP 训练循环**，不是因子计算（因子 calc 仅 ~10 秒）。`--max-windows N` 本就是有效杠杆（35 窗→2 窗 ≈ 1hr→~10min）。问题是 10 分钟结果不可信（单 seed + in-sample），所以人退回到 1hr 全量。

**解法**：让快速路径可信——多 seed + 配对方差门槛 + OOS held-out + 实验账本 + guardrails，agent 在便宜层（Tier-0/1）自主循环，只有存活候选才烧 1hr Tier-3（人在环）。

**对标**：karpathy/autoresearch（单文件可改 + 固定便宜预算 + 标量指标 + keep/discard + git revert + 人改 steering doc）+ 本仓库已有的 `gp_factor_miner`（propose→evaluate→keep/reject→record + `gp_factors.json` 原子写注册表）。

---

## 2. 已建文件地图

### 新建（4）
| 文件 | 职责 |
|---|---|
| `core/alpha/research_runner.py` | **核心编排器**。`load_session`（数据+因子只算一次）/`_run_single_seed`（单 seed 训练+独立回测）/`compute_baseline`/`tier0_factor_gate`/`tier1_quick_validate`/`variance_keep_or_revert`（配对 seed 门槛）/`tier3_full_validate`（人在环）/`record_experiment`（账本原子写）|
| `core/alpha/research_loop.py` | **agent 入口**。`load_and_baseline`/`run_one_iteration`/`run_tier3`/`_git_commit`/`main()` CLI |
| `core/alpha/experiments.json` | 实验账本（仿 `gp_factors.json` v2，原子 `os.replace`，append-only）。初始空 |
| `docs/loop/auto_research.md` | steering 契约（`program.md` 等价物）：循环契约 + 硬约束 + 分层 Tier 表 + 配对门槛公式 |

### 修改（4）
| 文件 | 改动 |
|---|---|
| `core/alpha/mlp_signals.py` | `__init__` 加 `seed` 参数；`generate_signals` 用 `self.seed`；`_train_and_predict_window` 中 `ensemble_size==1` 用 `[self.seed]`，`>1` 仍用 `ENSEMBLE_SEEDS`（保留 ensemble 平均）|
| `core/alpha/engine.py` | `analyze_factor_performance` 返回 `(factors_df, factor_metrics)`；factor_metrics 含 `ic/icir/t_stat/n_periods/direction_ratio/turnover/per_window`；加 polars 原生 turnover（两阶段、内存安全）|
| `training.py` | 抽出 `resolve_version_config()` + `run_training(args,version,CalcClass,description)`；`__main__` 成薄 shim；**CLI 字节不变**；更新 `analyze_factor_performance` 调用解包 tuple |
| `AGENTS.md` | §1.3 索引表加 3 行（auto_research 契约/账本/编排器）|

### 校准产物
- `core/alpha_db/baseline_calibration.json` — w=8 baseline 标定（median 1.7536, spread 1.2207, seed_scores, paired_margin 0.05）

---

## 3. 工作方法（怎么用）

### CLI（单实验，最常用）
```bash
# 基线锚点（3 seed × 8 窗 × lgb，in-sample，~14min；数据缓存后 load ~70s）
/home/airst/Workspace/.venv/bin/python -m core.alpha.research_loop --index 399303.SZ --baseline-only --max-windows 8

# 验证一个因子裁剪（≤3 个，#27 硬约束）：tier0→tier1→配对门槛→记账→(keep则commit)
/home/airst/Workspace/.venv/bin/python -m core.alpha.research_loop --index 399303.SZ --remove pool_size_x_regime --max-windows 8 --commit-on-keep
```

### 程序化（agent / 人 import）
```python
from core.alpha.research_loop import load_and_baseline, run_one_iteration, run_tier3, FactorChange
s = load_and_baseline(index="399303.SZ", version="v15", max_windows=8, backend="lgb", margin=0.05)
run_one_iteration(s, FactorChange("remove", factors=["pool_size_x_regime"], desc="drop low-IC"), commit_on_keep=True)
# 存活候选 → 人工签字 →
run_tier3(s, change, exp_id, human_approved=True)   # 3 seed × 35 窗 × attention + OOS, ~3hr
```

### 运行环境关键点（踩过坑）
- **venv**：`/home/airst/Workspace/.venv/bin/python`（不在仓库内，别用相对 `.venv/bin/python`，会 exit 127）
- **PYTHONPATH**：跑仓库外脚本（如 `/tmp/x.py`）必须 `PYTHONPATH=/home/airst/Workspace/vnpy`，否则 `No module named 'core'`（exit 1）。仓库内 `python -m` 不需要。
- **`--index` 强制**（AGENTS.md:23）：全市场 GPU OOM。`load_session` 对 `index=None` 直接 raise。
- **后台长跑**：`run_in_background: true` + `Monitor`（`tail -f log | grep --line-buffered ...`），别用前台 sleep（被 block）。
- **scratch 信号**：每次 `_run_single_seed` 写 `ar_{ver}_{suffix}_s{seed}.parquet`；session 结束应清理（`lab.remove_signal`）。基线跑后已清理。

---

## 4. 已验证状态

| 项 | 状态 |
|---|---|
| `training.py --help` CLI 不变 | ✅ |
| seed 参数化（ens=1→[seed], ens=3→[42,123,2024]） | ✅ |
| `resolve_version_config` 发现全部版本 | ✅ |
| `load_session` 真实数据（~145s，135 因子结构化指标含 turnover） | ✅ |
| `_run_single_seed` 端到端（1 seed×1 窗×lgb，65s，得分=3.16，974 daily_data 点） | ✅ |
| 账本 I/O（原子写、append、reload） | ✅ |
| #27 批量移除护栏（4→拒, ≤3→过） | ✅ |
| add 因子门（IC/ICIR/方向不达标→拒） | ✅ |
| **配对 seed 门槛**（单元测试：真改善 +0.10/seed→KEEP，旧 median 门槛会漏；噪声→REVERT；null→REVERT） | ✅ |
| **w=8 baseline 标定**（median 1.754, spread 1.221） | ✅ |

---

## 5. ⚠️ 未完成事项（按优先级）

### P0 — 必做：真实改动端到端验证（配对门槛在真改动上的行为）
**从未用配对门槛跑过一次真实 `run_one_iteration`**（只单元测试过）。必须拿一个真实因子改动验证整条链路：
```bash
/home/airst/Workspace/.venv/bin/python -m core.alpha.research_loop --index 399303.SZ --remove <某低IC因子> --max-windows 8 --commit-on-keep
```
预期：tier0 过 → tier1 3 seed×8 窗×lgb ~14min → 配对门槛判 keep/revert → 记账（看 `core/alpha/experiments.json` 是否有 `paired_deltas/n_positive` 字段）。
**注意**：若 keep 且 `--commit-on-keep`，会真 `git commit`——先确认工作树干净或去掉该 flag。

### P1 — 可选：null-change 精标 margin
`paired_margin=0.05` 是保守默认。要精确标定，跑一次 **null-change**（候选==baseline，同 seed→delta 应≈0），残余 delta spread 即不可约配对噪声，据此设 `margin = max(0.02, 1.5×null_delta_std)`。~6min（3 seed×8 窗×lgb，复用缓存数据）。
**不做的后果**：margin 可能偏松/偏严。但符号检验（≥2/3）是鲁棒核心，0.05 已安全可用——不跑也能用。

### P2 — 未实现：`propose_change` 的 LLM 集成
`research_loop.py` 目前**没有** `propose_change`——agent 手动构造 `FactorChange`。计划是用 `knowledge_base.build_knowledge_base`（knowledge_base.py:277-294，已镜像 27 原则）+ `hypothesis_generator._build_prompt`（232-268）做 steering，让 agent/LLM 自主提因子假设。骨架在 `knowledge_base.py`/`hypothesis_generator.py` 已有（GP/LLM mining 在用），需接到 `research_loop` 的 `propose_change`。

### P2 — 未实现：`add` 因子改动的完整 wiring
Tier-0 的 `add` 门（`tier0_factor_gate` 检 `candidate_metrics`）已写，但 `research_loop` CLI 只支持 `--remove`。`add` 需要：
- 改动 `gp_factors.json` 把因子 status 设 `testing` → `load_session(gp_status_filter=["validated","testing"])` 重算因子（~10s）→ 构造 `FactorChange(change_type="add", candidate_factor_df=<新factor_df>, candidate_metrics=<新因子IC/ICIR>)`。
- `_build_experiment_factor_df` 的 `add` 分支已支持 `candidate_factor_df`，但**没有** CLI 入口和"算新因子 metrics"的辅助函数。需补。

### P3 — 从未跑过：Tier-3 全量终验
`tier3_full_validate`（3 seed×35 窗×attention，~3hr，含 OOS 切片 `_rdd_from_daily`）**只验证了接线逻辑，从未实跑**。需在 P0 存活的候选上、人工签字后跑一次。OOS 切片重算 return_drawdown_ratio 的 `_rdd_from_daily` 也只在合成数据测过。

### P3 — 数据缓存（明确 out of scope，但高价值）
`engine.load_data()` 每次 ~2.5min，未缓存。`lab.save_dataset/load_dataset`（lab.py:389-415）存在但从未被调用，`alpha_db/dataset/` 空。缓存 joined raw DataFrame 可把 session 启动 2.5min→~10s，但有 staleness 风险（新数据下载后缓存过期）。当前 in-process 复用已在 session 内消除重复 load；跨 session 缓存是独立增强。

### P3 — 真训练留出（out of scope）
当前 OOS 是**评估留出**（backtest `end_date=oos_start`，训练仍用全历史，但每窗只用过去数据，已是 walk-forward）。真训练留出需改 `mlp_signals` 的 `train_period` 在 OOS 前截止——大改、收益边际。`auto_research.md` 已显式记录此设计。

---

## 6. 关键设计约束（接手必读，别踩）

1. **多 seed = `ensemble_size=1` × 3，不是 `ensemble_size=3`**。`ensemble_size=3` 会把 3 个 seed 的预测**平均成 1 个信号**（mlp_signals.py:407-409 `np.mean(all_preds)`），只产出 1 个 backtest 分、**无 spread**。配对门槛需要 3 个独立分，所以必须 `ensemble_size=1` 跑 3 次、seed-suffixed `signal_name` 隔离（`ar_{ver}_{suffix}_s{seed}`）。

2. **配对 seed 门槛，不是 median-vs-median**。标定实测 `return_drawdown_ratio` 是比值型，between-seed spread（1.22）≈ median（1.75），独立 median 比较无信噪比。baseline 与候选跑**同一组 seed [42,123,2024]**且 `_run_single_seed` 确定，"seed 倒霉"噪声共享相消。门槛：
   ```
   delta_s = score_cand(seed_s) - score_base(seed_s)
   keep iff median(delta_s) > 0.05  AND  ≥2/3 个 delta_s > 0
   ```
   实现见 `research_runner.variance_keep_or_revert`。`DEFAULT_MARGIN=0.05`（配对差值尺度，**不是**旧的 0.15）。

3. **收益回撤比是比值型、噪声大**——这是 #2 的根因。若日后换指标（如 Sharpe/annual_return），需重跑 baseline 标定其 spread；配对门槛公式不变，但 margin 尺度要重标。

4. **`--index` 强制**（AGENTS.md:23，GPU OOM）。`load_session` 对 `index=None` raise。

5. **#27 硬约束**：`tier0_factor_gate` 对 `remove/prune` 且 `len(factors)>3` 直接拒。**严禁**自动套用 `factor_screening.removal_set`（V15.9: 移除 41 个 → Sharpe 1.50→1.19）。

6. **#18 ≥3 seed**：单 seed 结果永远不能用于 keep 判决。

7. **#20/#26 饱和**：若 3 次连续 factor-space 改动在 Tier-1 被 revert，steering 指示 agent 转向数据/模型层（从 `experiments.json` verdict 历史可查；agent 推理，非硬锁）。

8. **OOS = 评估留出**：Tier-1 回测 `end_date=oos_start`（最近 6 月不评分）；Tier-3 跑全段并从 `daily_data` 切 `[oos_start, latest]` 重算 OOS 收益回撤比。

9. **Tier-3 人在环**：`tier3_full_validate(human_approved=True)` 不签字不跑、不动生产 signal DB。

10. **seed 匹配**：`variance_keep_or_revert` 用 `seeds`+`seed_scores` 建 {seed:score} 映射按 seed 值配对（不依赖顺序）。baseline 和 tier1 必须用**同一组 seeds**（`run_one_iteration` 已让 tier1 用 `session.baseline_scores["seeds"]`）。

---

## 7. 校准数据快照（w=8, lgb, in-sample）

```
seeds:         [42, 123, 2024]
seed_scores:   {42: 1.7536, 123: 0.7353, 2024: 1.9561}
median:        1.7536   (return_drawdown_ratio)
spread:        1.2207   (between-seed, ≈ median → 比值型噪声大)
sharpe_median: 0.7733
paired_margin: 0.05
```
注：max_windows=2 时 median=0.933/spread=0.718；**spread 随窗口数增长**（比值放大），所以不是"拉长回测降噪声"，而是"配对消噪声"。

---

## 8. 接手第一步建议

1. 读 `docs/loop/auto_research.md`（steering 契约）+ 本文件 §6（约束）。
2. 跑 P0：`--remove` 一个低 IC 因子（从 `core/alpha_db/baseline_calibration.json` 或 factor_screening 找），`--max-windows 8`，先**不带** `--commit-on-keep`，看 `experiments.json` 的 `paired_deltas/n_positive` 是否符合预期。
3. 若 P0 通过，按需做 P1（null-change 标 margin）或 P2（propose_change LLM 集成 / add 因子 wiring）。
4. 任何 Tier-3 终验前，确认 `git status` 干净并人工签字。
