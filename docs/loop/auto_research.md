# Auto-Research Loop Steering Contract

> **角色**：这是 auto-research 的 `program.md` 等价物——agent 在启动循环前必读的"研究组织代码"。
> 与 `process.md`（人在环 SOP）互补：本文件规定 **agent 自主循环** 的契约与硬约束。

## 为什么需要这个

一次完整训练+回测验证 ≈ 1 小时（MLP 滚动窗 35 个窗口）。因子计算本身只要 ~10 秒，**不是瓶颈**。
`--max-windows 2` 能把单轮压到 ~10 分钟，但单 seed + in-sample 的结果不可信，所以人会退回到 1 小时全量重训。

Auto-research 的解法：让快速路径**可信**（多 seed + 方差门槛 + OOS held-out），
agent 在便宜层（Tier-0/Tier-1）自主循环，只有存活候选才烧 1 小时全量（Tier-3，人在环）。

**核心架构**（实现见 `core/alpha/research_runner.py`）：

```
load_session   (数据+因子只算一次，跨 seed 复用)
  → compute_baseline   3 seed × 2 窗 × lgb, in-sample  → 锚点 median+spread
  → [propose change
       → tier0_factor_gate          秒级，无训练，#27 硬约束 + 因子 IC/ICIR/方向门
       → tier1_quick_validate        3 seed × 2 窗 × lgb, in-sample(OOS held out) → median+spread
       → variance_keep_or_revert     keep iff delta > max(spreads)+margin  (原则#18)
       → record_experiment           experiments.json 原子追加 + (keep则git commit)
    ] × N
  → tier3_full_validate   (人工签字) 3 seed × 35 窗 × attention, 全段+OOS → oos_score
```

## 标量目标

keep/discard 判据 = **收益回撤比 `return_drawdown_ratio`**（回测直接返回，兼顾收益与风险，对 MaxDD 敏感）。
多目标合成成这一个标量，agent 才能自动判 keep/revert。

## 分层验证（Tier）

| Tier | 后端 | 窗口 | OOS | 墙钟 | 谁决策 |
|------|------|------|-----|------|--------|
| 0 因子门 | — | — | — | 秒级 | agent 自主（#27 硬约束） |
| 1 快验 | lgb | 2 | in-sample only（OOS held out） | ~15 min | agent 自主 + 方差门槛 |
| 3 终验 | attention | 35(全量) | 含 OOS 切片 | ~3 hr | **人在环**（human_approved=True） |

**Tier-1 lgb 是必要非充分**：lgb 与 attention 归纳偏置不同（原则#27 "attention 能用 corr=1.0 因子的边际信息" 是 attention 特有）。
Tier-1 的职责是**廉价杀噪**，不保证 attention 也提升。Tier-3 attention 是 ground truth。

## 硬约束（guardrails，全部来自 research_principles.md，此处引用不重复）

- **#18**：单 seed 不可信，**至少 3 seed**。keep 判据必须看 median + spread，禁止用单次结果 keep。
- **#27**：禁止批量移除高相关因子（V15.9：移除 41 个 → Sharpe 1.50→1.19）。**每次最多移除 2-3 个**，
  且必须经过 Tier-1/3 验证。**严禁**自动套用 `factor_screening.removal_set`。
- **#20/#26**：GP/因子工程空间已饱和（6 轮后新候选多为已有变体）。若 **3 次连续** factor-space 改动在
  Tier-1 被 revert，agent 应**转向数据层/模型层**改动而非继续堆因子（从 `experiments.json` 的 verdict
  历史可查；agent 自行推理，非硬锁死）。
- **#16**：新增因子前先去重（corr>0.7 与现有因子则不加入）。
- **`--index` 强制**（AGENTS.md）：全市场训练 GPU OOM，`load_session` 对 `index=None` 直接拒绝。

## 方差门槛公式（核心去风险，配对 seed）

**校准发现**：`return_drawdown_ratio` 是总收益/最大回撤的比值，between-seed spread（~0.7–1.2）≈ median，
独立 median-vs-median 比较无信噪比。但 baseline 与候选跑**同一组 seed**且 `_run_single_seed` 确定，
"seed 倒霉"的噪声在两侧**共享、相消**。改用配对差值：

```
delta_s = score_cand(seed_s) - score_base(seed_s)     # 同 seed 配对
keep iff  median(delta_s) > margin  AND  ≥2/3 个 delta_s > 0   (符号检验)
```

- 符号检验（≥2/3 同向）是鲁棒核心；`margin`（默认 0.05，配对差值尺度）防"微小但一致"的噪声。
- 真正 null 的改动：同 seed → 同分 → delta=0；残余 delta spread 来自改动对训练的 seed-相关扰动。
  需要时可跑一次 null-change 标定 margin。
- 实现见 `research_runner.variance_keep_or_revert`。校准数据存 `core/alpha_db/baseline_calibration.json`。

## OOS = 评估留出（evaluation-holdout）

Tier-1 的回测 `end_date = oos_start`（最近 N 个月不参与评分）；
Tier-3 跑全段并从 `daily_data` 切 [oos_start, latest] 重算 OOS 收益回撤比 = `oos_score`。

**这是评估留出，不是训练留出**——滚动窗仍用全部历史训练（每窗只用过去数据，已是 walk-forward）。
等价于 autoresearch 的 `val_bpb`（held-out 评估）。

## 实验账本与沉淀

- **`core/alpha/experiments.json`**：机器可读，append-only，原子写（仿 `gp_factors.json`）。
  字段：exp_id / change / seed_scores / median / spread / baseline / verdict / oos_score / commit_hash。
  每条 keep → 一个 git commit → 零成本 `git revert`。
- **`docs/loop/verification_log.md`**：人读散文，time-reverse。Tier-3 `oos_passed` 时由
  `record_experiment` 蒸馏一行（基线/本次/OOS/判定/结论/关联=exp_id+commit）。
  账本喂日志，日志不喂账本，不重复。
- verdict 生命周期：`pending → keep/revert(Tier-1后) → oos_passed/oos_failed(Tier-3后)`，
  `oos_failed` 则强制 `git revert`。

## Agent 自主度边界

- **自主**：Tier-0、Tier-1、方差门槛、账本记录、git commit-on-keep。
- **人在环**：Tier-3（`tier3_full_validate(human_approved=True)`）——不签字不跑全量、不动生产 signal DB。
- **永远不自主**：批量因子移除（#27）、单 seed keep（#18）、跳过 OOS 终验进生产。

## 入口

- 程序化（agent / 人 import）：
  ```python
  from core.alpha.research_loop import load_and_baseline, run_one_iteration, run_tier3, FactorChange
  session = load_and_baseline(index="399303.SZ", version="v15")
  run_one_iteration(session, FactorChange("remove", factors=["pool_size_x_regime"], desc="..."), commit_on_keep=True)
  # 存活候选 → 人工签字 → run_tier3(session, change, exp_id, human_approved=True)
  ```
- CLI 单实验：`.venv/bin/python -m core.alpha.research_loop --index 399303.SZ --remove pool_size_x_regime`
- 基线 only：`.venv/bin/python -m core.alpha.research_loop --index 399303.SZ --baseline-only`

## 相关文件

- `core/alpha/research_runner.py` — 编排器（load_session / tier0 / tier1 / variance / tier3 / record）
- `core/alpha/research_loop.py` — agent 入口（run_one_iteration / run_tier3 / main CLI）
- `core/alpha/experiments.json` — 实验账本
- `docs/loop/process.md` — 6 步人在环 SOP（本文件的姊妹：人工迭代纪律）
- `docs/knowledge/research_principles.md` — 27 条硬约束（guardrails 的唯一真源）
- `core/alpha/knowledge_base.py::build_criteria_list` — 27 原则已镜像进 LLM prompt（steering 复用此机制）
