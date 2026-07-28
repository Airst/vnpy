"""
SSL embedding 接入模块（分钟数据自监督预训练日频特征）

== 当前状态 ==
验证链路: Tier-0 10/16 维过门（最强 IC -0.090/ICIR -0.73）
        → Tier-1 KEEP（3 seeds median +0.075）
        → Tier-3 同会话配对全量 35 窗 ×3 轮 REPRODUCIBLE（delta {+1.44, +1.79, +0.77} 全正）
        —— 历史首个通过全部门禁的新特征方向（2026-07-24, verification_log.md）
编码器: Day2Vec masked bottleneck autoencoder（840K 参数），
        仅用 2019-07~2021-12 分钟数据预训练后永久冻结（防泄漏），
        每日 48×5min bar → 16 维瓶颈 z，取其中 10 个过门维度
数据流: scripts/minute_ssl_pretrain.py 产出 → minute_ssl_emb.parquet
        日级增量: scripts/minute_ssl_update.py（盘后分钟下载 + 冻结编码器推理）

== 设计决策 ==
- 接入点: AlphaEngine.calculate_factors 因子计算之后 join（training.py 与
  research_runner.load_session 共用此路径，研究基线自动包含 SSL 维度）
- 处理与 Tier-3 验证脚本逐字一致: left join → fill 0 → 截面 z-score → clip ±5
- 优雅降级: parquet 缺失时警告并原样返回（不阻断训练）
- 覆盖率监控: 最新一日覆盖率 <50% 时打警告（分钟数据未更新会让当日
  embedding 全 0，稀释末日信号）
"""
from pathlib import Path

import polars as pl

SSL_EMB_PATH = "core/alpha_db/minute_ssl_emb.parquet"

# Tier-0 过门的 10 个维度（log/minute_ssl_tier1.jsonl, 2026-07-23）
SSL_DIMS = ["ssl_emb_1", "ssl_emb_2", "ssl_emb_6", "ssl_emb_7", "ssl_emb_8",
            "ssl_emb_9", "ssl_emb_10", "ssl_emb_12", "ssl_emb_14", "ssl_emb_15"]


def attach_ssl_embeddings(factor_df: pl.DataFrame) -> pl.DataFrame:
    """把 SSL embedding 过门维度 join 进因子表（处理方式与 Tier-3 验证一致）。

    label 列（若存在）保持在最后一列；parquet 缺失时原样返回。
    """
    if not Path(SSL_EMB_PATH).exists():
        print(f"[SSLEmb] 警告: {SSL_EMB_PATH} 不存在，跳过 SSL embedding 接入")
        return factor_df

    emb = pl.read_parquet(SSL_EMB_PATH, columns=["datetime", "vt_symbol"] + SSL_DIMS)
    emb = emb.with_columns(pl.col("datetime").cast(factor_df.schema["datetime"]))

    out = factor_df.join(emb, on=["datetime", "vt_symbol"], how="left")
    out = out.with_columns([
        ((pl.when(pl.col(k).is_infinite()).then(0.0).otherwise(pl.col(k)).fill_nan(0.0).fill_null(0.0)
          - pl.col(k).fill_nan(0.0).fill_null(0.0).mean().over("datetime"))
         / (pl.col(k).fill_nan(0.0).fill_null(0.0).std().over("datetime") + 1e-8))
        .clip(-5.0, 5.0).alias(k)
        for k in SSL_DIMS
    ])

    # label 保持最后一列（模型训练约定）
    if "label" in out.columns:
        cols = [c for c in out.columns if c != "label"] + ["label"]
        out = out.select(cols)

    # 末日覆盖率监控（分钟数据滞后 → 当日 embedding 缺失会稀释信号）
    last_dt = factor_df["datetime"].max()
    last_cover = emb.filter(pl.col("datetime") == last_dt).height
    last_total = factor_df.filter(pl.col("datetime") == last_dt).height
    ratio = last_cover / last_total if last_total else 0.0
    tag = "" if ratio >= 0.5 else " ⚠️ 覆盖率过低，请先运行 scripts/minute_ssl_update.py"
    print(f"[SSLEmb] 已接入 {len(SSL_DIMS)} 维, 最新日 {str(last_dt)[:10]} 覆盖率 {ratio:.0%}{tag}")
    return out
