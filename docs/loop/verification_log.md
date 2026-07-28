# 验证记录流水

> 按时间倒序记录每轮验证（Step 5）的输入/输出/判定。最新在最上。单次回测结论不可信（随机种子敏感），看趋势。

## 记录格式

```
## YYYY-MM-DD vX.{子版本} — {一句话主题}
- **基线**：Sharpe / 年化 / MaxDD
- **本次**：Sharpe / 年化 / MaxDD
- **非牛市对比**：基线 vs 本次
- **判定**：通过 / 失败
- **结论**：保留 / 回退 + 一句话原因
- **去向**：沉淀到 knowledge/... 或 problems/...
- **关联**：design/xxx.md, iterations/vX_xxx.md, commit hash
```

---

<!-- 新记录追加在此分隔线下方 -->

## 2026-07-28 持有期控制（min_hold/max_hold）— 双区间验证通过，采纳 min=3/max=7 为生产默认
- **依据**（信号评测多维矩阵，Top-1~10 × 1/3/5/7/10d，1051 交易日）：alpha 在持有第 1~7 天持续兑现（前 5 天 0.107%/天、第 6~7 天 0.098%/天），第 8~10 天钝化到 0.042%/天；Top-1~10 准确率持平（无"越集中越准"）
- **策略改造**（multifactor_strategy.py 新参数）：min_hold_days=持有不足 N 天屏蔽信号卖出（止损/风控不受限）；max_hold_days=持满后不在当日 Top-K 强制换仓（TIME_EXIT）、仍在 Top-K 重置计时
- **全区间**（2022-01~2026-07，8 配置）：基线 Sharpe 1.155/RDD 3.013/210.8%；**min=3 max=7 → Sharpe 1.462/RDD 3.465/333.3%**；min_hold=3 单独 → 1.396/3.783/308.9%；min=5 max=10 → 1.264/3.878/265.9%
- **子区间**（2025-01~2026-07，含 26Q2 逆风）：基线 1.005/35.0%；**min=3 max=7 → Sharpe 1.688/65.7%**；min_hold=3 → 1.514/58.5% → 两区间排序一致，非单区间过拟合
- **反例**：mh=3+min5+max7 崩溃（Sharpe 0.715/MaxDD -45.1%）——矩阵里 Top-3 超额略优不足以补偿集中度风险，印证 #28（N=5 最优）；单独 max_hold=7 无效（dSharpe -0.008），换仓收益主要来自 min_hold 屏蔽过早信号卖出
- **判定**：通过，采纳 min_hold=3/max_hold=7 为生产默认（设 0 可回退旧行为）
- **关联**：core/tools/signal_evaluator.py（多维矩阵）, scripts/holding_period_sweep.py, log/holding_period_sweep.json, log/holding_recent_check.log

## 2026-07-28 标签风格中性化 R5+R5x（双池 Tier-1 + 扩种子消歧）— 5 seeds 后信号崩塌，标签方向五轮终局
- **假设**（用户指示继续标签方向后的第五轮）：vol_scaled 转正靠的是削掉标签风格暴露，除法是钝器；外科手术版 = 组内排名中性化（不奖励押风格，只保风格内相对强弱，风格暴露交风控层）
- **变体**：size_neutral（大/小市值组内 rank）/ vol_neutral（高/低波组内 rank，cs_rank 分组避开 CUDA nanmedian 无确定性实现）/ style_neutral（2×2 四组）；冒烟断言全过（组内百分位相等、组均值相等、NaN 传播）
- **结果**（双池 3 seeds × 3 窗，26-04~07 total_return delta vs 5d）：
  - size_neutral：-11.60 / +0.28 / +1.33pp → 噪声级
  - **vol_neutral：-10.10 / +22.51 / +21.62pp** → 两个 WIN 幅度为全战役 Tier-1 最强，但 seed-42 为负
  - style_neutral：-4.23 / +8.17 / +6.11pp → 方向一致但弱于 vol_neutral
- **关键疑点**：基线 seed-42 = -6.86% 是离群值（另两 seed -27.91/-19.10%），三变体被同一个"运气好的基线抽样"压死，与 R2/R4 的噪声级 WEAK-WIN 性质不同
- **处置与终判**：R5x 扩种子消歧（5d vs vol_neutral 各补 seeds 7/777 至 5 个）—— delta = -10.10 / -1.62 / -0.84 / +21.62 / +22.51pp，**2/5 为正、median -0.84pp → 5d HOLD**。两个 +22pp 是双峰分布里的运气抽样而非被 seed-42 压死的稳健信号；若只看 3 seeds 会被 median +21.62pp 严重误导
- **教训沉淀**：① Tier-1 判定遇到"大幅 WIN 但 seed 间双峰"时，扩种子比直接升级 Tier-3 便宜一个量级且能避免重蹈 R1 覆辙；② 五轮（11 变体、3 次 Tier-3、1 次扩种子）后结论加固：26Q2 逆风不存在标签层免费修复，风格/波动暴露的处置属风控层职责（仓位/波动率目标），标签方向关闭
- **生产状态**：未触碰——label_mode 默认值仍为 "5d"，实验全程走 research_runner scratch 信号（ar_v15_lr5*），无残留进程
- **旁证**：途中修复 torch.nanmedian 在 use_deterministic_algorithms(True) 下无 CUDA 实现 → 改 cs_rank(argsort) 分组；脚本增加 jsonl 断点续跑
- **关联**：scripts/label_rebound_paired_r5.py, scripts/label_rebound_paired_r5x.py, log/label_rebound_paired_r5.jsonl

## 2026-07-28 vol_blend Tier-3 REJECT — 标签改造方向四轮终局：关闭
- **结果**（同 seed=42 配对全量，双池，两侧重跑）：
  - 5d 基线：Total 349.4% / Sharpe 1.537 / RDD 4.55 / MaxDD -18.4% / 子区间 +0.17%（注：随新交易日数据，基线子区间已自然回正）
  - vol_blend：Total 171.5% / Sharpe 0.945 / RDD 1.88 / MaxDD -29.0% / 子区间 -14.73% → **REJECT**（三项全受损）
- **重要发现：seed-42 代理假设的适用边界**——R3 大幅 delta（+10.9pp）在 Tier-3 复现（+21.9pp），R4 小幅 delta（+2.9pp）完全反转（-14.9pp）：Tier-1 仅 3 窗，小幅优势在噪声内，不足以支撑升级。后续升级门槛应加：seed-42 delta 须 > +5pp
- **四轮终局总结**（8 变体，3 次 Tier-3 全量）：
  - R1 地平线融合（单池 WIN）→ Tier-3 REJECT；R2 双池地平线/路径 → 无稳健优势
  - R3 vol_scaled：唯一真实效应（子区间首次转正 +21.9pp）但 Sharpe 腰斩；R4 两个半剂量中间点均失败（剂量-响应非单调，继续网格搜索同轴属数据窥探）
  - **结论**：标签层改造无法免费修复 26Q2 逆风——vol_scaled 的改善本质是风险偏好取舍（用全时段 alpha 换逆风保护），这类取舍应由风控层（仓位/波动率目标）实现，而非污染选股标签；且最新数据下基线子区间已回正，修复紧迫性下降
- **去向**：生产保持 5d 标签；label_mode 基础设施（8 变体）与 3 个配对脚本 + 参数化 Tier-3 脚本保留备查；方向关闭
- **关联**：log/label_rebound_tier3_r4.log, log/label_rebound_tier3.jsonl

## 2026-07-28 半剂量波动率缩放 R4（双池 Tier-1）— vol_sqrt 淘汰，vol_blend 弱 WIN 升级 Tier-3
- **背景**：R3 vol_scaled 满缩放剂量过猛（子区间转正但 Sharpe 腰斩），R4 试两种中间点：vol_sqrt（√vol 幂插值 p=0.5）、vol_blend（0.5*rank(原始超额)+0.5*rank(满缩放)）。冒烟断言确认两者排序介于 raw 与 full 之间
- **结果**（3 seeds × 3 窗，双池，delta vs 5d；基线 -14.67/-15.39/-13.66%）：
  - vol_sqrt：delta -4.96/+5.19/-8.89pp，median -4.96，seed-42 负 → HOLD 淘汰（幂插值不改变相邻排序关系的程度不够，或介于两个局部最优之间）
  - vol_blend：delta +2.94/+2.73/-18.35pp，median +2.73，2/3 正，seed-42 正 → 形式 WIN，按预登记协议升级 Tier-3；**但证据弱于 R3**（seed-2024 大幅负，预期不高）
- **Tier-3 规则修订已落地**：scripts/label_rebound_tier3.py 判定新增 Sharpe delta > -0.10 门槛（堵 R3 暴露的 RDD 口径漏洞）
- **去向**：vol_blend Tier-3 进行中，结论见后续条目
- **关联**：scripts/label_rebound_paired_r4.py, log/label_rebound_paired_r4.jsonl

## 2026-07-28 vol_scaled Tier-3 — 形式 CANDIDATE 但实质取舍过猛：子区间首次转正，Sharpe 腰斩
- **结果**（同 seed=42 配对全量 35 窗，双池，两侧重跑）：
  - 5d 基线：Total 299.1% / Sharpe 1.361 / RDD 2.39 / MaxDD -24.9% / 子区间 -14.99%
  - vol_scaled：Total 121.4% / Sharpe 0.706 / RDD 2.45 / MaxDD -35.9% / **子区间 +6.90%（三轮以来首次在逆风区间转正，+21.89pp）**
- **判定**：预登记规则（RDD 不受损 + 子区间改善）形式 CANDIDATE，但 RDD 口径掩盖了全时段实质受损：Sharpe 1.361→0.706、Total -177.7pp、MaxDD -11pp。**不采纳为生产标签**；规则漏洞记录在案：后续 Tier-3 判定须同时要求 Sharpe 不显著受损
- **解读**：波动率缩放信号真实存在（逆风区间转正不是噪声能解释的幅度），但满缩放（p=1）把 2022-2025 反转 alpha 稀释过狠——方向对、剂量过猛
- **去向**：R4 试 √vol 缩放（p=0.5 插值），寻找保留反转 alpha 与逆风保护的中间点
- **关联**：scripts/label_rebound_tier3.py, log/label_rebound_tier3.jsonl, log/label_rebound_tier3_r3.log

## 2026-07-27 波动率缩放标签族 R3（双池 Tier-1）— vol_scaled WIN 且过 seed-42 门槛，升级 Tier-3
- **背景**：R1/R2 地平线方向关闭后换标签族：不改"看多远"（仍 5 日，无额外 NaN 损失），改"奖励什么风险形态"。两变体：vol_scaled（5 日超额收益 / 过去 20 日波动率）、fwd_sharpe（5 日超额收益 / 未来 5 日实现波动率）。合成数据冒烟断言全过（同收益下低波>高波、平稳>锯齿、NaN 尾部 5 日）
- **新判定门槛**（R2 教训沉淀）：除 median>0 且 ≥2/3 为正外，seed-42 delta 须为正才升级（双池 seed-42 已验证为 Tier-3 子区间的可靠代理）
- **结果**（3 seeds × 3 窗，双池，26-04～07 total_return，delta vs 5d；基线重跑因数据更新至新交易日）：
  - 5d 基线：-17.55 / -20.29 / -15.10%
  - **vol_scaled：delta +10.91/+3.92/-0.83pp，median +3.92，2/3 正，seed-42 强正 → WIN，升级 Tier-3**
  - fwd_sharpe：delta +3.90/-3.36/-0.14pp，median -0.14 → 5d HOLD，淘汰
- **去向**：vol_scaled 已提交 Tier-3 同 seed 配对全量（scripts/label_rebound_tier3.py 已参数化，两侧重跑），结论见后续条目
- **关联**：scripts/label_rebound_paired_r3.py, log/label_rebound_paired_r3.jsonl

## 2026-07-27 反弹标签 R2（生产双池 Tier-1）— 地平线/路径方向无稳健优势，方向关闭
- **背景**：用户指令继续改良标签。吸收 R1 教训，Tier-1 直接在生产双池 000852.SH,399303.SZ 上跑，新增 3 变体：rebound_confirm（R1 双 WIN 未进 Tier-3）、rebound_2h（只融合 5/10 日 0.6/0.4，砍掉 20d）、rebound_path（未来 10 日每日累计收益均值，冗余 cs_rank 已省）。合成数据冒烟断言全过（早涨>晚涨、NaN 尾部 10 日、值域）
- **结果**（3 seeds × 3 窗，26-04～07 total_return，delta vs 5d）：
  - 5d 基线：-7.66 / -27.19 / -15.33%（seed 散布 19.5pp；seed 42 与 Tier-3 子区间 -7.75% 几乎重合，验证了双池 Tier-1 的代表性）
  - rebound_confirm：delta -3.96/+15.08/+0.82pp，median +0.82 → 形式 WIN 但噪声级
  - rebound_2h：delta -10.77/+4.80/+9.00pp，median +4.80 → 形式 WIN 但符号混杂
  - rebound_path：delta -17.75/+13.22/-1.81pp，median -1.81 → 5d HOLD
- **判定：不升级 Tier-3，方向关闭**。三点依据：
  1. Tier-3 用 seed 42，而双池 Tier-1 的 seed-42 子区间≈Tier-3 子区间（已被 -7.66 vs -7.75 验证）；两个"WIN"变体的 seed-42 delta 均为负（-3.96/-10.77），升级几乎必然 REJECT
  2. R1 的 rebound_avg 单池 Tier-1 是 3/3 WIN median +10.68pp，仍 Tier-3 REJECT；R2 两个变体的证据强度远低于当时
  3. 基线 seed 散布 19.5pp 远大于任何变体 median delta：26Q2 子区间表现由 seed 噪声主导，标签地平线调整不是该区间亏损的解释变量
- **沉淀结论**：两轮（R1 单池 + Tier-3，R2 双池）共 4 个地平线/路径变体，无一在生产宇宙上稳健优于 5d；26Q2 错配不可由标签地平线单独修复。若再试标签方向，应换标签族（如波动率缩放/风险调整收益）而非继续调地平线权重
- **去向**：label_mode 基础设施保留（含 4 变体分支）；生产保持 5d
- **关联**：scripts/label_rebound_paired_r2.py, log/label_rebound_paired_r2.jsonl

## 2026-07-26 反弹趋势确认标签 Tier-3 终验 — REJECT，Tier-1 优势在全量双池下反转
- **前置**：Tier-1 定向配对（26-04~07，3 seeds × 3 窗，399303.SZ 单池）rebound_avg 3/3 WIN，median delta +10.68pp
- **本次**：同 seed=42 配对全量（35 窗，生产双池 000852.SH,399303.SZ，attention，回测 2022→最新）：
  - 5d 基线：Total 224.7% / Sharpe 1.174 / RDD 3.22 / MaxDD -29.6% / **26-04~07 子区间 -7.75%**
  - rebound_avg：Total 284.0% / Sharpe 1.205 / RDD 2.01 / MaxDD -27.2% / **26-04~07 子区间 -19.25%**
- **判定**：REJECT（预登记规则：目标区间未复现即拒）。目标区间 delta **-11.50pp，与 Tier-1 方向完全反转**；RDD 3.22→2.01 也显著受损
- **结论**：回退，生产保持 5d 标签（label_mode 默认值未动，无需代码回退）。两点发现：
  1. 担心的"长地平线稀释 2022-2025 反转 alpha"未发生（全时段 Total/Sharpe/MaxDD 反而小幅改善），但换来的不是目标区间改善
  2. Tier-1 优势不可迁移的可能原因：单池(399303)→双池宇宙差异 + 单 seed 全量本身的抽样方差（seed 42 在 Tier-1 恰是 delta 最小的 +4.42pp）。若重启该方向，应先在双池上重跑 Tier-1 确认优势存在再升级
- **去向**：标签实验基础设施保留（label_mode 参数 + 两个配对脚本）；方向暂停
- **关联**：scripts/label_rebound_tier3.py, log/label_rebound_tier3.jsonl

## 2026-07-26 反弹趋势确认标签（5/10/20 日融合）— 26-04~07 定向配对，两变体均 3/3 WIN
- **背景**：用户指令——用未来 5~10~20 日走势构造"反弹趋势确认"标签，快速验证 26 年 4~7 月选股收益。前次单地平线实验（10d vs 5d 全 in-sample，07-19）结论 5d HOLD；本次改多地平线融合 + 聚焦近期区间
- **设计**：v15 calculator 新增 label_mode 参数（因子完全相同仅标签不同）：5d=生产基线；rebound_avg=5/10/20 日 beta-neutral 超额收益各自 cs_rank 后加权融合 (0.40/0.35/0.25)；rebound_confirm=三地平线 rank 取 min（最差地平线决定标签，惩罚冲高回落）。各 3 seeds × 3 窗 × attention，回测 2026-04-01→最新 (N=5)，主指标 total_return
- **基线 (5d)**：total_return -9.47% / -25.23% / -18.92%（median -18.92%，spread 15.8pp），印证"超跌反弹猎手"在 26Q2 动量 regime 的系统性错配
- **rebound_avg**：-5.05% / -6.95% / -8.24%（median -6.95%，spread 仅 3.2pp），配对 delta median **+10.68pp，3/3 为正** → WIN
- **rebound_confirm**：-6.71% / -13.32% / -4.38%（median -6.71%，spread 8.9pp），配对 delta median **+11.91pp，3/3 为正** → WIN；MaxDD 也系统性收窄（-19~-34% → -17~-22%）
- **判定**：通过（两变体均 3/3 seeds 配对为正，幅度远超噪声）。rebound_avg 方差最小最稳，rebound_confirm median 略高但散布大
- **结论**：多地平线趋势确认标签显著缓解 5 日反转标签在动量 regime 的错配，但区间绝对收益仍为负；采纳前需 Tier-3 全量训练（35 窗，2022-2026）验证全时段不受损（风险：反转 alpha 在 2022-2025 是主收益来源，长地平线标签可能稀释其强度）。首选 rebound_avg（稳定性优先）
- **注意**：20 日标签在训练/测试窗口边界的前视重叠比 5 日多 15 天（框架无 embargo，配对双方处理一致），长地平线一侧略偏乐观，结论留安全边际
- **去向**：待 Tier-3 全量验证后再决定是否沉淀准则
- **关联**：scripts/label_rebound_paired.py, log/label_rebound_paired.jsonl, core/alpha/v15_factor_calculator.py (label_mode)

## 2026-07-23 SSL embedding 生产集成上线 — 全量 Sharpe 1.41/RDD 3.48，超出无 SSL 历史分布上沿
- **基线**（无 SSL 生产配置 6 次全量分布）：Sharpe 0.81~1.29 / RDD 1.70~3.27；切换前生产信号 Sharpe 0.911 / 年化 34.4% / MaxDD -27.8% / RDD 1.70
- **本次**（143+10 SSL 维度，生产默认配置）：**Sharpe 1.411 / 年化 67.9% / MaxDD -21.0% / RDD 3.48**，总收益 311%
- **逐年**：2022 +52.9% / 2023 +15.7% / 2024 +27.8% / 2025 +61.2% / **2026 YTD +12.9%**（此前弱市年为负，SSL 特征在逆风 regime 同样有效）
- **集成实现**：core/alpha/ssl_embedding.py（join 处理与 Tier-3 验证逐字一致）接入 AlphaEngine.calculate_factors（training.py 与 research_runner 共用，研究基线自动含 SSL）；日级增量 scripts/minute_ssl_update.py（分钟下载合并写回+冻结编码器推理，已端到端验证：1181 只增量下载 + 5904 条新 embedding，训练日覆盖率 100%）
- **判定**：通过。单次全量仍有抽样方差，但 Sharpe/RDD 双双超出无 SSL 历史分布最大值，与 Tier-3 配对证据（3/3 全正 median +1.44）方向一致
- **运营约束**：盘后生成信号前必须先跑 minute_ssl_update.py，否则当日 embedding 缺失稀释末日信号（attach 时有覆盖率警告）；编码器永久冻结禁止重训
- **关联**：core/alpha/ssl_embedding.py, core/alpha/engine.py, scripts/minute_ssl_update.py, core/alpha_db/backtest/ashare_mlp_signal_v15_20220101_20260723_20260723_205136.json

## 2026-07-24 SSL embedding Tier-3 终验 — 3/3 轮全正，REPRODUCIBLE（首个通过全部门禁的新特征方向）
- **基线**：143 因子，同会话配对全量 35 窗，seed=42，score {2.341, 2.152, 2.149}
- **本次**：143+10 SSL embedding 维度，score {3.777, 3.941, 2.916}
- **配对 delta**：{+1.436, +1.790, +0.768}，median +1.436，**3/3 为正** → REPRODUCIBLE
- **对照组**（同尺子）：手工微观因子同测试 {+0.377, -0.330, -0.436} 不可复现——SSL 表征与手工聚合存在本质差异，幅度（最小 +0.77）远超残余非确定性噪声带
- **判定**：通过。Tier-0（10/16 过门）→ Tier-1 KEEP（median +0.075）→ Tier-3 REPRODUCIBLE，历史首个走完全链路的新特征
- **结论**：分钟数据的价值兼容于日频监督范式，但必须经无监督预训练压缩（而非手工聚合）。待生产集成：需接入 training 管线（embedding join）+ 日级分钟数据更新 + 盘后编码器推理，用户确认后实施
- **关联**：scripts/minute_ssl_tier3.py, log/minute_ssl_tier3.jsonl（含逐轮 detail）

## 2026-07-23 分钟数据自监督预训练 embedding（路径 B 最小实验）— Tier-0 强通过，Tier-1 边缘 KEEP，待 Tier-3
- **背景**：参数量级讨论结论——监督任务吃不下大模型，但无监督预训练可用 56M 行分钟数据喂大编码器；用户批准花一注验证
- **方法**：Day2Vec masked bottleneck autoencoder（840K 参数，生产模型 42 倍），每日 48×5min bar（5 特征/bar，全日内信息）→ 16 维瓶颈 z。防泄漏：只用 2019-07~2021-12（回测期前）预训练 68 万序列后冻结；recon MSE 0.233→0.201
- **Tier-0**（2022+ 评估）：**10/16 维过门**，最强 emb_15 IC -0.090/ICIR -0.73/dir 0.79、emb_9 IC -0.082，超过多数手工因子；与 143 生产因子 max_corr 0.15~0.67（非换皮）
- **Tier-1**（3 seeds × 8 窗配对，10 维 cs_zscore 接入）：delta {42: +0.776, 123: +0.075, 2024: -0.036}，median +0.075 > 0.05 且 2/3 为正 → **KEEP（边缘）**
- **判定**：形式过门，但证据质量一般：一个大正外点 + 一个刚过线 + 一个小负。微观因子教训（Tier-0/单 seed 全量好看→配对复现翻车）要求必须过 Tier-3 同会话配对全量（≥3 轮）才能谈集成
- **与微观因子的关键差异**：手工微观因子 Tier-1 两轮均 REVERT，SSL embedding 首轮即 KEEP——无监督压缩学到的表征与手工聚合不等价
- **关联**：scripts/minute_ssl_pretrain.py, scripts/minute_ssl_tier1.py, log/minute_ssl_tier1.jsonl, core/alpha_db/minute_ssl_emb.parquet, core/alpha_db/model/minute_ssl_encoder.pt

## 2026-07-22 探底企稳确认过滤诊断 — 方向为负，就地关闭
- **背景**：用户观察模型喜欢抄底，问单针/双针/MACD底背离等探底信号能否改善入场时机
- **方法**：生产回测 610 笔 round-trip，买入日 T-1 可见信息判定三条件（缩量 vol_dry、低点抬高 higher_low、CLV偏强），按企稳评分分组看胜率/均值
- **验证输出**：
  - 全时段：未企稳(≤1) 53.5%/+1.17% vs 已企稳(≥2) 50.6%/+0.78% —— **方向相反**
  - 2026 前：未企稳 58.6%/+1.75% vs 已企稳 52.0%/+0.92%，差距更大；放量比缩量好（+1.44% vs +0.72%）
  - 2026 年(n=38)：所有企稳状态组均亏 -3~-6%，企稳过滤对逆风期无任何保护
- **判定**：企稳确认过滤被直接证伪（会滤掉最赚钱的"未企稳"交易），方向关闭
- **结论**：V15 的反转 alpha 本质是向恐慌市场提供流动性的补偿——钱就藏在"别人不敢买的时刻"，等企稳确认后补偿已被吃掉。这也解释动量 regime 伤它的机制：市场不恐慌时无人付流动性溢价。2026 亏损与入场时机无关，执行层修不了 regime
- **关联**：scripts/bottom_confirm_diag.py

## 2026-07-21 SWA+微观因子 seed=42 可复现性测试 — seed 运气，不可复现（终局）
- **背景**：微观因子单 seed 全量训练（Sharpe 1.23/RDD 4.50 vs 基线 1.16/3.27）似有改善，需确认是否可复现还是 seed 运气
- **本次**：3 轮同会话配对（每轮 load_session 一次，基线因子计算共享）→ 基线(143) 与 候选(143+13micro) 同 seed=42 全量 35 窗，看 delta 一致性
- **验证输出**：
  - 轮1: 基线 2.638 / 候选 3.015，delta +0.377（Sharpe 1.13→1.32）
  - 轮2: 基线 1.981 / 候选 1.650，delta -0.330（Sharpe 0.81→0.99）
  - 轮3: 基线 2.554 / 候选 2.118，delta -0.436（Sharpe 1.29→1.25）
  - **median delta -0.330，1/3 为正 → 不可复现**
- **判定**：seed 运气。微观因子在 seed 42 的"改善"不可复现，非真效应
- **结论**：微观结构因子探索终局——Tier-0 反复证明真实正交 alpha（7/14、21/31 过门），但：Tier-1 两轮 REVERT（v1 6 因子 -0.068、v2 13 因子 -0.50），可复现性测试证 seed 42 改善是运气。微观 alpha 与现有因子重叠，模型无法稳定利用（#20 饱和同构）。**生产不集成微观因子**。分钟数据（7 年 ~56M 行）与 31 因子库（micro_factors_v2.parquet）留存备用
- **关联**：scripts/micro_repro_test.py, log/micro_repro.jsonl, problems.md

## 2026-07-21 微观因子 v2 扩挖（31 因子/21 过门）— Tier-1 再 REVERT，证模型饱和非因子数量问题
- **背景**：用户观察"微观因子 Q2 表现好"（已证：候选在坏 regime Q2/6/7月减亏增益，是 regime 对冲，但单 seed），问能否挖更多微观因子
- **扩挖**：v1 14 因子 → v2 31 因子（新增订单流/大单/趋势结构/价差流动性/时段/量价），Tier-0 过门 21 个（cs_spread -0.075、vol_weighted_vol -0.073、vol_autocorr -0.053、obv_slope、intraday_mdd 等）
- **去重**：13 个正交代表因子（剔除 intraday_range（与现有 klen 完全相关）、n_reversals（与 trend_persist 完全反向）等 8 个冗余）
- **关键 bug（二次教训）**：候选训练 loss nan 反复——根因是 `joined.drop(keep).join(micro_clean)` 只删了 keep 却留下全部 31 个原始微观列（含 vwap_dev 99.5 亿离群值）。修：`cand = base.join(micro_clean)`（micro_clean 只含清洗后的 keep 因子）+ 截面 z-score + clip ±5
- **Tier-1（基线 [3.19, 7.30, 5.99] vs 候选 [3.99, 4.55, 5.49]）**：配对 deltas {42:+0.79, 123:-2.75, 2024:-0.50}，median -0.50，1/3 为正 → **REVERT**
- **判定**：失败。与 v1（6 因子）同模式——Tier-0 强 alpha、Tier-1 不稳定 REVERT
- **结论**：**能挖到更多微观因子（31 个、21 过门），但挖更多不改变结论——瓶颈不是因子数量，是模型对新增 alpha 的稳定提取能力**。微观 alpha 与现有因子重叠，模型无法稳定利用（seed 42 受益是 seed-fitting 非稳健效应，不可用于生产）。与 #20 信号饱和同构。模型层改善的方向在聚合（SWA 已证）与组合构建（N=5 已证），不在新增因子。v2 因子库（micro_factors_v2.parquet）留存备用
- **关联**：scripts/compute_micro_factors_v2.py, scripts/micro_v2_cand_only.py, log/micro_v2_tier1.log

## 2026-07-20 P0 日内微观结构因子 — Tier-0 强 alpha 但 Tier-1 REVERT（附 label 泄漏 bug 教训）
- **背景**：用户采纳 P0 方向（日内微观结构，准则 #26 数据层突破）。下载分钟线（tushare 5min，1179 只 × 4 年，~56M 行），聚合 14 个日频微观结构因子
- **Tier-0（IC 门）**：**7/14 过门**，机制清晰且与现有 143 因子正交（最大相关仅 0.77 < 0.90 冗余线）——kyle_lambda（非流动性溢价 IC +0.095）、realized_vol_5m/intraday_range（低波动异象 -0.075）、ushape_vol/vol_of_vol（量能结构）、first60_vol_r（开盘反转）、tail30_ret（尾盘反转）
- **严重 bug（过程教训）**：首次 Tier-1 候选 Sharpe 5.74/RDD 26139——**label 泄漏**。join 把 micro 列追加到 label 之后，模型 `df.columns[2:-1]` 把 label 当特征（完美预测）。修复：join 后重排让 label 回最后一列。**教训：任何向 factor_df 追加因子的操作，必须校验 label 仍在最后一列**
- **Tier-1（修复后，基线 [3.87, 4.13, 4.93] vs 候选 [3.80, 2.50, 7.10]）**：配对 deltas {42:-0.068, 123:-1.62, 2024:+2.18}，median -0.068，1/3 为正 → **REVERT**。且候选 seed 散布 4.6 远大于基线 1.06——**增加 seed 方差**
- **判定**：失败。微观因子有真实正交 alpha（Tier-0），但加入模型不可靠提升且增不稳（Tier-1）
- **结论**：P0 方向证伪于 Tier-1。alpha 与现有因子（amihud/volatility）的**超额部分重叠**——虽然因子本身不高度相关，但可提取的 alpha 已被模型覆盖（与 #20 信号饱和同构）。且微观因子（非流动性/低波动）是防御性异常，本就不对治用户的动量 regime 核心问题。数据基础设施保留（分钟线 ~56M 行 + 因子计算/验证脚本 scripts/download_minute*.py, compute_micro_factors.py, micro_factors_tier*.py）
- **关联**：core/alpha_db/micro_factors.parquet, log/micro_tier1.log, problems.md

## 2026-07-18 TimesFM 零样本验证 — 无可用 alpha（全期 IC +0.011，回测 -36.9%）
- **背景**：用户提议引入 TimesFM（Google 时序基础模型）+ 股票数据微调 + 融合回测框架验证 alpha。按"零样本先试信号→有信号再微调"路径
- **本次**：TimesFM 2.5 200M（torch）零样本，双池宇宙每只股 512 日收盘预测 5 日 → 隐含收益排名。20 日 IC 初测 +0.067（疑似有信号），生成 2026 YTD 完整信号（151523 行）回测验证
- **验证输出**：
  - 全期 IC（2026 YTD 5 日前向）：**+0.011（≈噪声）**，正占比 53%；逐月：1 月 +0.049、4 月 -0.083、7 月 +0.201（100% 正日——趋势崩盘月假象）
  - 交易回测 N=5：**总收益 -36.9%，Sharpe -2.95**；月度 1 月 +9.1% 后连续 6 个月亏损
  - 组合（生产 SWA + TimesFM rank 平均）：**-8.1% vs 生产单独 +14.0%**——拖累而非互补
  - top-5 核查（7/6 均值 -6.6%，7/8 -4.1%）：无信号方向 bug——宽截面排名微弱正 IC 但极端 top-N 是动量强势股，反转市里崩最狠
- **判定**：失败。TimesFM 零样本无可用 alpha（全期 IC 噪声级，回测灾难）
- **结论**：**单变量点预测范式与截面选股不匹配**——TimesFM 只做每股自身价格路径外推（动量延续），看不到截面关系（当前模型 alpha 的核心）；只在极端趋势市（7 月崩盘）偶尔有效，反转主导的 A 股小盘里 top-N 动量必崩。**微调预期价值低**：要么学成生产模型已有的反转（重复），要么保持动量（仍失败）。若要基础模型，正确姿势是把 TimesFM 预测作为一个**因子**喂给生产模型（让 attention 决定何时信任），而非独立信号；但零样本 IC 噪声级，作为因子增量也有限。该方向记录为已探索-不推荐
- **关联**：scripts/timesfm_zeroshot_ic.py, scripts/timesfm_signal_gen.py, scripts/timesfm_combo_backtest.py, log/timesfm_zeroshot.log

## 2026-07-18 TimesFM 表征微调探针 — 微调前提不成立（表征无股票信号，IC -0.015）
- **背景**：目标含"微调训练"。零样本已证无 alpha后，验证微调前提——TimesFM 的表征是否含股票收益信息（若含，全量微调才有意义）
- **本次**：头微调（冻结 TimesFM 主体，forward() 提取 output_embeddings 1280 维，2025 年训练 MLP 头预测 5 日收益，2026 H1 测试 IC）
- **验证输出**：探针 IC **-0.015**（正占比 43%）；MLP 训练 loss 0.0370→0.0371（**平坦，表征里无可学习收益信息**）；7 月逐日 IC -0.072/-0.143/-0.165（与所有方法一样反预测）
- **判定**：失败。TimesFM 表征不含股票 alpha，全量微调失去意义
- **结论**：**TimesFM 探索终局**——零样本（IC +0.011，回测 -36.9%）与头微调（IC -0.015）双重证伪。单变量时序基础模型的表征对截面股票选股无信息价值，不迁移。通用时序基础模型 ≠ 截面选股工具。生产模型（手工因子+FT-Transformer 截面排名）范式更强
- **关联**：scripts/timesfm_finetune_probe.py, log/timesfm_probe.log

## 2026-07-18 标签地平线配对验证（10d vs 5d）— 失败，风格非标签周期问题
- **背景**：用户诊断模型超跌反弹风格不适应 2026 动量 regime，疑与标签/训练窗口有关。10 日标签（V14 前代最优 Sharpe 1.42）更慢更不接飞刀
- **本次**：两 session（因子全同仅标签 horizon 不同）各 3 seeds × 8 窗 × attention 配对
- **验证输出**：5d [2.55, 6.86, 5.60]，10d [3.59, 4.90, 4.39]；配对 deltas {42:+1.05, 123:-1.96, 2024:-1.20}，median -1.20，1/3 为正
- **判定**：失败（median -1.20 < 0.05，仅 1/3 为正）
- **结论**：**风格不是标签周期能改的**——5 日反转风格深嵌于整个因子体系（turnover/momentum/流动性全是 5-20 日短周期信号），只改标签 horizon 而因子不动，模型仍是反转。要真改风格需因子+标签一起向动量重构，等于另起一个模型。至此风格问题的机制性修复全部证伪：池切换/池扩容/北向/IC门控（创可贴）/标签周期
- **关联**：scripts/label_horizon_paired.py, log/label_horizon_paired.jsonl, problems.md alpha 反转条目

## 2026-07-18 alpha 效能门控（IC gate）原型测试 — 方向有希望但不干净，非根治
- **背景**：用户诊断模型风格（超跌反弹）不适应 2026 动量 regime。信号 IC 实证：4 月 +0.037 → 5 月 -0.054 → 6 月 -0.062 → 7 月 -0.239（连续 3 月为负恶化，反预测）。门控思路：信号近期 IC 为负时钳制持仓
- **本次**：compute_ic_gate.py 生成滞后 5 日的滚动 10 日 IC 门控（IC≥0.01→5 仓，-0.03~0.01→3 仓，<-0.03→1 仓），策略 ic_gate_enabled 钩子，同一 SWA 信号回测
- **验证输出**（SWA 信号 N=5）：门控 Sharpe 1.24/总收益 322.7% vs 无门控 1.16/228.7%（收益提升）；但 RDD 3.07 vs 3.27、MaxDD -37.1% vs -31.9%（回撤更深）；2026 问题段中性（6 月 -8.6% 未躲开，7 月 -4.4%）
- **判定**：不明确——收益/Sharpe 升但回撤/RDD 差，2026 段无针对性改善
- **结论**：IC 门控是创可贴非根治——它在 IC 正时更激进赚收益，但不解决模型在动量 regime 押错方向的本质。根治在标签地平线（方向 ①）与 regime 均衡采样（方向 ②）。门控代码保留（compute_ic_gate.py + 策略 ic_gate_enabled，默认关闭）
- **关联**：scripts/compute_ic_gate.py, scripts/signal_ic_analysis.py, problems.md alpha 反转条目

## 2026-07-18 股票池方向两实验均失败 — 2026 H1 回撤是 regime 方差，不可池切换
- **背景**：用户报告 26 年 4 月起持续回撤，建议从沪深300 入手。诊断：2026-06 策略 -8.3% 而沪深300 +1.8%（alpha 衰减非 beta）；主导因子 turnover_mean_10d 滚动 IC 从 ~0.35-0.40 衰减到 0.174（-50%）
- **实验一（HS300 单池）**：同因子集同配置仅换池，全量 35 窗。N=5 Sharpe 0.08/RDD 0.14/总收益 8.0%/DD持续 607 天；N=10 Sharpe 0.10/DD持续 759 天。**因子体系在大盘池无 alpha**（小盘微观结构 alpha 不可迁移：大盘机构主导、截面离散度低、仅 264 只可排名）
- **实验二（三池并集 000300+000852+399303，1443 只）**：N=5 RDD 2.54/Sharpe 0.86（不如双池 3.27/1.16），2026-06 **-12.3% 比双池 -8.3% 更差**；且 6 月买入中新增大盘名仅 10%——**模型不轮动**：个股级信号（换手率/流动性）是小盘特异的，大盘名排不进 top5，大盘敞口徒增稀释
- **判定**：两实验均失败，池方向（换池/扩池）证伪
- **结论**：2026 H1 回撤是小盘 alpha 的 regime 性失宠，非模型失效（当前 ~8% DD 远小于历史 MaxDD -32%；2024 年 Sharpe 0.78 也曾是弱年）。正确姿态：持仓观望 + 风控层应对，不追 regime 改模型（V15.5/V15.5b 已证此陷阱）。若真要大盘敞口，需独立的大盘因子体系（基本面+北向，V17 已证北向 alpha 仅大盘有效，hk_hold_manager 基础设施可复用）——是数周的研究轨道，非快修
- **关联**：scripts/hs300_experiment.py, scripts/tripool_experiment.py, log/hs300_experiment.log, log/tripool_experiment.log, problems.md

## 2026-07-18 N 选择修正 — SWA 信号下 N=5 优于 N=10（#28 修正，生产回退 N=5）
- **背景**：用户实测 SWA 新信号发现 N=5 在 2024 牛市更好、N=10 仅 2024 前回撤周期更短。#28（N=10 最优）建立在 pre-SWA 信号上，SWA 信号的 N 敏感性需重估
- **本次**：① SWA 生产信号纯净 A/B（同一信号只改 N，双池全时段）；② 跨 3 个 SWA seed 信号复核（399303.SZ，8 窗）
- **验证输出**：
  - 生产 SWA 信号：N=5 RDD 3.27/Sharpe 1.16/总收益 228.7%/DD持续 39 天 vs N=10 RDD 2.39/Sharpe 1.08/183.9%/55 天；2024 牛市 +31.9% vs +17.2%（924 段 +44.6% vs +37.7%），2026 +8.6% vs -7.2%；仅 MaxDD N=10 更浅（-24.6% vs -31.9%）
  - 跨 seed（399303.SZ）：N=5 胜 2/3（s123 6.41 vs 4.85，s2024 5.83 vs 5.05，s42 3.88 vs 4.02），median delta +0.78
- **判定**：通过（同信号 A/B + 跨 seed 方向一致 + 机制自洽）
- **结论**：生产回退 N=5（training.py/trade_service/策略/风控默认值）。机制：广度与 SWA 是信号噪声的替代解药——SWA 稳定信号后，集中持仓在牛市/高确信名上的收益捕获重新占优；附带 N=5 风控阶梯可 -35% 全退、DD 持续更短。代价：MaxDD 更深
- **去向**：research_principles.md #28 修正（N 最优值随信号噪声水平变化；若信号稳定性退化需重估广度）
- **关联**：exp_053, log/breadth_sweep_20260717_085855.json

## 2026-07-18 exp_053 — 检查点平均化三候选：SWA 决定性 KEEP，设为默认
- **背景**：34/35 窗口训满 1000 epoch（早停耐心 800 形同虚设），验证损失 ~epoch 500 后进噪声高原，best checkpoint 是噪声选择的点（同 seed 重跑信号 rank 相关仅 0.989）
- **本次**：三候选配对验证（attention × 8 窗 × 3 seeds × 399303.SZ，in-sample，vl=100 基线）
- **验证输出**（基线 best: [3.52, 4.69, 3.89]，median 3.89）：
  - ① topk_pred（top-3 检查点分别预测+rank平均）：[3.56, 4.64, 5.47]，deltas {+0.04, -0.05, +1.58}，median +0.04 → REVERT（边界，同训练的检查点相关性太高，预测平均无增量）
  - ② swa（top-3 检查点权重平均，greedy model soup）：[3.88, 6.41, 5.83]，deltas {+0.36, +1.72, +1.94}，**median +1.71，3/3 为正 → KEEP**（median 3.89→5.83，+50%）
  - ③ ema（全轨迹权重 EMA，decay=0.999）：[3.16, 0.85, 1.62]，deltas {-0.36, -3.85, -2.27}，median -2.27 → REVERT（记忆过长混入高原前劣质权重）
- **判定**：② 通过（远超 0.05 门槛）；①③ 失败
- **结论**：SWA 设为生产默认（mlp_signals model_settings checkpoint_mode="swa"，exp_053 记账）。这是 2026-07-17 诊断中"噪声高原检查点选择"机制的首次直接修复——权重平均落到更平更稳的点。生产信号以 swa 重训
- **生产重训结果（2026-07-18 04:50，vl=100 + swa + N=10）**：Sharpe 0.70→**1.08**，RDD 1.88→**2.39**，MaxDD -29.1%→**-24.7%**，总收益 99.6%→183.9%。逐年全部改善（2022 +15.6%→+33.5%，2023 +17.1%→+29.7%，2024 +14.7%→+17.2%，2025 +44.4%→+50.7%，2026 -11.0%→-7.2%）。与验证方向一致
- **关联**：scripts/checkpoint_modes_validate.py, log/checkpoint_modes_validate.jsonl, vnpy/alpha/model/models/mlp_model.py docstring

## 2026-07-18 valid_len 池依赖实锤 — 双池 3-seed 分布对比（维持 vl=100 默认）
- **背景**：exp_052 在 399303.SZ 单池判定 vl=100 胜；但当日生产全量训练（双池）vl=100 新信号 N=10 Sharpe 仅 0.70，低于 vl=50 分布范围，疑似 verdict 池依赖
- **本次**：vl=100 双池 3-seed 全量稳定性测试（stability_test.py --max-windows 0，与 vl=50 的 2026-07-16 稳定性测试同构）
- **验证输出**（双池，N=5）：
  - vl=100：RDD [2.19, 1.50, 3.03]，median 2.19，min 1.50；Sharpe [0.93, 0.68, 0.88]
  - vl=50：RDD [2.55, 1.04, 3.14]，median 2.55，min 1.04；Sharpe [1.03, 0.55, 1.10]
  - 配对 delta（100-50）：{42: -0.36, 123: +0.46, 2024: -0.11}，median -0.11，1/3 为正
- **对比 399303.SZ Tier-3（exp_052）**：vl=100 median 3.15 vs vl=50 median 2.34（vl=100 胜 0.81）
- **判定**：**valid_len 效应池依赖且两处都在噪声范围内**——399303 单池 vl=100 胜，双池 vl=50 微胜（median -0.11 非决定性）；vl=100 在双池的最低值更高（1.50 vs 1.04，地板更厚）
- **结论**：维持 vl=100 默认（依据：唯一受控配对测试 exp_052 支持 100 + 准则 #17 + 双池地板更厚），不因非受控的弱信号反复横跳。生产信号（vl=100，N=5 RDD 1.70）是分布内的一次抽取（高于 min 1.50），不为追更好抽取而重训（seed-shopping）
- **教训**：**Tier-3 验证池必须与生产池一致**——exp_052 用 399303.SZ 单池得出的 verdict 对双池生产的适用性有限；后续 Tier-3 默认用双池
- **关联**：log/stability_vl100.log, log/breadth_sweep_20260717_085855.json, exp_052

## 2026-07-17 exp_052 — exp_050 (valid_len=50) 配对 Tier-3 终验：推翻，回退 100
- **背景**：exp_050 在 Tier-1（lgb/8窗/in-sample）keep 了 valid_len 100→50（delta +0.43），但违反准则 #17 且从未经 Tier-3 确认
- **本次**：配对 Tier-3，3 seeds × 35 窗 × attention，全时段含 OOS，pool=399303.SZ，同 session 两配置
- **验证输出**：
  - valid_len=100：seed RDD [5.42, 2.37, 3.15]，median 3.15；OOS RDD [0.05, -0.26, -0.08]，median -0.08
  - valid_len=50：seed RDD [2.22, 2.34, 3.49]，median 2.34；OOS RDD [-0.69, -0.49, +1.03]，median -0.49
  - 配对 delta（50-100）：{42: -3.20, 123: -0.02, 2024: +0.34}，median -0.023，1/3 为正
- **判定**：失败（median delta < 0 且仅 1/3 为正；OOS median 也支持 100）
- **结论**：回退。mlp_signals.py valid_len 默认恢复 100（exp_052 记账）。准则 #17 再次被证实：100 天验证集的 early stopping 可靠性在全量 attention 框架下成立；exp_050 的 Tier-1 改善是 lgb+短窗+in-sample 假象
- **流程教训（重要）**：Tier-1 keep 必须经 Tier-3 确认才能进生产——本次 exp_050 未经终验即上线，生产信号在错误 valid_len 下训练了一天。后续 keep 流程：Tier-1 keep → 人工签字 → Tier-3 → 通过才改生产默认
- **关联**：scripts/valid_len_tier3_paired.py, log/valid_len_tier3.jsonl, core/alpha/experiments.json exp_052

## 2026-07-17 持仓广度扫描 — N=5→10 是 seed 稳健性最强杠杆（准则 #28）
- **基线**：生产信号 N=5：Sharpe 1.49 / RDD 3.45 / MaxDD -28.3%（单 seed 抽取）
- **本次**：同一信号仅改 max_holdings 的纯净 A/B，4 信号（生产 + stability 3 seeds 全量版）× N∈{5,10,15,20}
- **验证输出**（跨 4 信号聚合）：
  - N=5：RDD mean 2.55 / min 1.04 / 散布 2.41；Sharpe min 0.55；最坏 MaxDD -29.8%
  - N=10：RDD mean 2.80 / min 2.51 / 散布 0.62；Sharpe min 0.98；最坏 MaxDD -26.1%；成本 322K→279K
  - N=15：RDD mean 2.77 / min 2.19；N=20：RDD mean 2.68 / min 2.48 / 散布 0.36，但 DD 持续 88 天
  - 生产信号逐年：N=5 2024 年 +21.6% → N=10 +38.1%；2026 年 +8.7% → -1.8%（弱市年份互有胜负，聚合指标 N=10 全胜）
- **判定**：通过（同信号 A/B，无 seed 噪声混入；4 信号方向一致）
- **结论**：保留为准则 #28。N=10 为最优平衡点：期望收益不降（mean RDD 2.55→2.80），最坏情形提升 2.4 倍，成本反降。建议生产 max_holdings 5→10（待用户确认后改实盘配置）
- **补充实验（风控阶梯）**：N=10 下阶梯每级 -1 最多减到 5 仓（不会清仓）。测试每级 -2（reduction_per_level=2，恢复 -35% 清仓能力）：生产信号 RDD 2.96→3.01（+0.05），s123 RDD 3.13→2.99（-0.14），MaxDD 不变。方向不一致、幅度微小 → 保持每级 -1 默认，阶梯无需改动。代码已参数化（risk_controller.py reduction_per_level，策略 setting risk_reduction_per_level，默认 1）
- **补充分析（2026 弱 regime 逐年分解）**：2026 年收益 mean：N=5 -4.7% → N=10 -5.5% → N=15/20 -8.4%/-8.5%；但 2026 最坏信号：N=5 -19.3% → N=10 -10.0%。生产信号 2026 +8.7%（N=5）→ -1.8%（N=10）——幸运 seed 的 2026 正收益部分是集中度运气；s123 -19.3%→-1.3%。结论：弱 regime 下广度同样是用微小均值换最坏情形保护，且 N=10 在 2026 依然优于 N=15/20，甜点结论跨 regime 成立
- **去向**：knowledge/research_principles.md #28；problems.md 新增"同 seed 重跑发散"
- **关联**：scripts/breadth_sweep.py, log/breadth_sweep_20260717_085855.json

## 2026-07-17 exp_051 — Vintage ensemble（时序模型平均）Tier-1 REVERT
- **基线**：3 seeds × 8 窗 × lgb，vintage=0：seed_scores [2.39, 4.30, 4.11]，spread 1.91
- **本次**：vintage_ensemble=2（当前窗口模型 + 过去 2 窗口模型逐日截面 rank 平均）：[3.04, 3.81, 4.15]，spread 1.12
- **配对 delta**：{42: +0.64, 123: -0.49, 2024: +0.04}，median +0.044 ≤ margin 0.05，2/3 为正
- **判定**：失败（差 0.006 未过门槛）
- **结论**：回退。但注意：seed spread 压缩 41%（1.91→1.12）——该机制对"降方差"有效，只是配对门槛量的是中位数提升。seed 稳健性问题已由 #28（广度）以更强效应解决，vintage 方向暂停。代码保留（mlp_signals.py vintage_ensemble 参数，默认 0=关闭）
- **关联**：core/alpha/experiments.json exp_051, scripts/vintage_ensemble_validate.py

## 2026-07-08 News v1 — 股票资讯采集与展示（端到端验收）
- **基线**：无（新功能，首次构建）
- **本次**：定时任务 + LLM 联网采集 + 板块/个股映射 + FastAPI 接口 + 前端资讯页面
- **验证输入**：
  - 采集器：core/llm/news_collector.py（gp llm 连接：OpenAI SDK + 百炼 + DASHSCOPE_API_KEY；模型 qwen3.7-max + enable_search；glm-5.2 不支持 search 已记录）
  - 接口：/api/news、/api/news/sectors、/api/news/dates、/api/news/history、/api/news/status、POST /api/news/collect
  - 前端：core/web_ui/src/components/NewsDashboard.jsx + App.jsx /news 路由
  - 定时：main_controller news_scheduler（09:00 / 15:30，已注册并启动）
- **验证输出**：
  - 手动 POST /api/news/collect → 后台线程 → LLM web search → 映射 → 落盘。status 轮询：running→false，last_count=8，message="采集完成: ok"
  - 落盘：core/alpha_db/news/2026-07-08.json，14 条（6+8 合并去重）
  - info_date 校验：14/14 ≤ 今日；11/14 为昨日(2026-07-07)，时效达标
  - 板块映射：12/14 命中 dc_concept（含 concept_pct_change）；14/14 含 mapped_stocks（真实 vt_symbol）
  - 情绪分布：利好10 / 利空2 / 中性2（覆盖负面，非只报喜）
  - 时效：high 10 / medium 4
  - 前端：headless chrome 渲染 /news 页面，DOM 含"立即采集/资讯条数/利好/利空/代表性个股/板块当日涨跌幅"及真实板块名；截图确认卡片渲染（板块 Tag、情绪色标、高时效徽标、标题、摘要、影响分析、轮动含义、关联板块、代表性个股 vt_symbol）
- **判定**：通过
- **结论**：保留。功能满足"可服务于一个 A 股投资交易者"的验收：交易者打开页面可看到当日影响板块/轮动/情绪的资讯、板块当日涨跌幅、领涨股与代表性标的，并据此判断板块强弱与轮动方向。
- **去向**：problems.md 三项（glm 不支持 search / 日期漂移 / 板块口径）均标记解决或兜底达标；不沉淀回测知识（非量化因子迭代）。
- **关联**：design/news_v1_collect.md, design/news_v1_frontend.md, goals.md
- **备注**：运行环境——同机 8000 端口被 sibling 工程 /home/airst/Workspace/vnpy 的 start_vnpy_rs.sh 看门狗占用并自重启；验证用 8001 端口跑 vnpy2。用户若要在主端口 8000 使用，需停掉该看门狗与 sibling 服务后用 main.py 启动 vnpy2。

（待填）
