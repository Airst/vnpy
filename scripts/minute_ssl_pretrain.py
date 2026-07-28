"""
分钟数据自监督预训练（day2vec）→ 日频 embedding 生成

路径 B 最小实验（用户批准 2026-07-22）:
- 假设: 56M 行分钟数据可通过无监督预训练支撑更大参数量的表征学习，
  产出的日频 embedding 可能携带手工微观因子未覆盖的模式
- 防泄漏: 编码器只用 2019-07-01 ~ 2021-12-31 预训练（回测期 2022+ 之前），
  之后冻结；特征全部日内计算，无跨日信息
- 架构: masked bottleneck autoencoder (~1M 参数, 比生产模型大 50 倍)
  编码: 每根 5min bar 5 特征 → d=128 → 4 层 Transformer → 可见 bar 池化 → z(16)
  解码: (z + 位置嵌入) → MLP → 重构被 mask bar 的 5 特征 (MSE)
  z 被迫成为"一天的摘要" = 日频 embedding
- 输出: core/alpha_db/minute_ssl_emb.parquet (datetime, vt_symbol, emb_0..15)
        core/alpha_db/model/minute_ssl_encoder.pt

用法: /home/airst/Workspace/.venv/bin/python scripts/minute_ssl_pretrain.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")
import glob
import numpy as np
import polars as pl
import torch
import torch.nn as nn

SEED = 42
SEQ = 48          # 每日 48 根 5min bar
NF = 5            # 每 bar 特征数
D = 128
Z = 16            # 瓶颈 = 日频 embedding 维数
PRETRAIN_END = "2021-12-31"
EPOCHS = 8
BS = 1024
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIN_FILES = sorted(glob.glob("core/alpha_db/minute/*.parquet"))
EMB_OUT = "core/alpha_db/minute_ssl_emb.parquet"
ENC_OUT = "core/alpha_db/model/minute_ssl_encoder.pt"

torch.manual_seed(SEED)
np.random.seed(SEED)


def day_features(g: np.ndarray) -> np.ndarray | None:
    """一天的 bar 矩阵 (n,5)=[close,open,high,low,vol] → (SEQ,NF) 特征, 全日内信息"""
    c, o, h, l, v = g[:, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4]
    n = len(c)
    if n < 40 or c[0] <= 0 or np.any(~np.isfinite(g)):
        return None
    day_vol = v.sum()
    if day_vol <= 0:
        return None
    prev_c = np.concatenate([[o[0]], c[:-1]])
    ret = np.where(prev_c > 0, c / prev_c - 1, 0.0) * 100          # bar 收益 (%)
    vol_share = v / day_vol * SEQ                                   # 归一量占比 (均值~1)
    rng = np.where(c > 0, (h - l) / c, 0.0) * 100                   # bar 振幅 (%)
    hl = h - l
    clv = np.where(hl > 0, (c - l) / np.where(hl > 0, hl, 1.0), 0.5)  # 收盘位置
    body = np.where(c > 0, (c - o) / c, 0.0) * 100                  # 实体 (%)
    f = np.stack([ret, vol_share, rng, clv, body], axis=1).astype(np.float32)
    f = np.clip(f, -10, 10)
    if n >= SEQ:
        return f[:SEQ]
    out = np.zeros((SEQ, NF), dtype=np.float32)
    out[:n] = f
    return out


def load_stock_days(fp: str):
    """返回 [(trade_date_str, (SEQ,NF))], 按日期升序"""
    df = pl.read_parquet(fp, columns=["trade_time", "close", "open", "high", "low", "vol", "trade_date"])
    df = df.sort("trade_time")
    out = []
    for td, g in df.group_by("trade_date", maintain_order=True):
        td_val = td[0] if isinstance(td, (list, tuple)) else td
        arr = g.select(["close", "open", "high", "low", "vol"]).to_numpy()
        f = day_features(arr)
        if f is not None:
            out.append((str(td_val), f))
    return out


class Day2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(NF, D)
        self.pos = nn.Parameter(torch.randn(SEQ, D) * 0.02)
        layer = nn.TransformerEncoderLayer(D, 8, dim_feedforward=512, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=4)
        self.to_z = nn.Linear(D, Z)
        self.dec = nn.Sequential(nn.Linear(Z + D, 256), nn.GELU(), nn.Linear(256, NF))

    def encode(self, x):  # x: (B,SEQ,NF) → z: (B,Z)
        h = self.enc(self.inp(x) + self.pos)
        return self.to_z(h.mean(dim=1))

    def forward(self, x, mask):  # mask: (B,SEQ) bool, True=被遮
        x_vis = x.masked_fill(mask.unsqueeze(-1), 0.0)
        h = self.enc(self.inp(x_vis) + self.pos)
        vis = (~mask).unsqueeze(-1).float()
        z = self.to_z((h * vis).sum(1) / vis.sum(1).clamp(min=1.0))
        pos = self.pos.unsqueeze(0).expand(x.size(0), -1, -1)
        rec = self.dec(torch.cat([z.unsqueeze(1).expand(-1, SEQ, -1), pos], dim=-1))
        return rec, z


def main():
    # ---- Phase A: 收集预训练数据 (<= 2021-12-31) ----
    print(f"[A] 加载 {len(MIN_FILES)} 只股票的分钟数据 (预训练截止 {PRETRAIN_END}) ...")
    cutoff = PRETRAIN_END.replace("-", "")
    train_feats = []
    all_days = {}  # vt_symbol -> [(td, feat)]
    for k, fp in enumerate(MIN_FILES):
        vt = os.path.basename(fp)[:-8]
        days = load_stock_days(fp)
        all_days[vt] = days
        train_feats.extend(f for td, f in days if td <= cutoff)
        if (k + 1) % 200 == 0:
            print(f"  [{k+1}/{len(MIN_FILES)}] 预训练样本 {len(train_feats):,}")
    X = np.stack(train_feats)
    del train_feats
    print(f"[A] 预训练序列: {X.shape} ({X.nbytes/1e9:.1f} GB)")

    # ---- Phase A: 训练 ----
    model = Day2Vec().to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[A] Day2Vec 参数量: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(X)
    steps_per_epoch = n // BS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
    model.train()
    for ep in range(EPOCHS):
        perm = np.random.permutation(n)
        tot, cnt = 0.0, 0
        for s in range(steps_per_epoch):
            idx = perm[s * BS:(s + 1) * BS]
            xb = torch.from_numpy(X[idx]).to(DEV)
            mask = torch.rand(xb.size(0), SEQ, device=DEV) < 0.4
            rec, _ = model(xb, mask)
            loss = ((rec - xb) ** 2)[mask].mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            tot += loss.item(); cnt += 1
        print(f"[A] epoch {ep+1}/{EPOCHS} recon_mse={tot/cnt:.4f}")
    del X
    os.makedirs(os.path.dirname(ENC_OUT), exist_ok=True)
    torch.save(model.state_dict(), ENC_OUT)
    print(f"[A] 编码器已存: {ENC_OUT}")

    # ---- Phase B: 冻结, 生成全时段 embedding ----
    print("[B] 生成全时段日频 embedding ...")
    model.eval()
    rows_dt, rows_vt, rows_emb = [], [], []
    with torch.no_grad():
        for k, (vt, days) in enumerate(all_days.items()):
            if not days:
                continue
            feats = torch.from_numpy(np.stack([f for _, f in days]))
            embs = []
            for s in range(0, len(feats), 4096):
                embs.append(model.encode(feats[s:s+4096].to(DEV)).cpu().numpy())
            embs = np.concatenate(embs)
            rows_dt.extend(td for td, _ in days)
            rows_vt.extend([vt] * len(days))
            rows_emb.append(embs)
            if (k + 1) % 200 == 0:
                print(f"  [{k+1}/{len(all_days)}]")
    E = np.concatenate(rows_emb)
    df = pl.DataFrame({"trade_date": rows_dt, "vt_symbol": rows_vt,
                       **{f"ssl_emb_{i}": E[:, i] for i in range(Z)}})
    df = df.with_columns(pl.col("trade_date").str.strptime(pl.Datetime("us"), format="%Y%m%d").alias("datetime")).drop("trade_date")
    df = df.select(["datetime", "vt_symbol"] + [f"ssl_emb_{i}" for i in range(Z)])
    df.write_parquet(EMB_OUT)
    print(f"[B] embedding 已存: {EMB_OUT} shape={df.shape}, 日期 {df['datetime'].min()} -> {df['datetime'].max()}")


if __name__ == "__main__":
    main()
