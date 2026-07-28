"""
TimesFM 表征微调探针（head fine-tuning on frozen embeddings）

验证核心问题: TimesFM 的时序表征是否含股票收益信息（微调能否成立的前提）。
方法:
- 冻结 TimesFM 主体，forward() 提取 output_embeddings（每股 512 日收益序列 → 1280 维）
- 在 2025 年训练一个小型 MLP 头: embedding → 未来 5 日收益（MSE）
- 在 2026 H1 测试: 全池排名 IC，对照零样本(+0.011)与生产模型
- 若探针 IC > 0.03: 表征含信号, 值得全量微调; 若 ≈0: 范式确认无 alpha

用法:
  /home/airst/Workspace/.venv/bin/python scripts/timesfm_finetune_probe.py
"""
import sys, os
sys.path.insert(0, "/home/airst/Workspace/vnpy")
os.chdir("/home/airst/Workspace/vnpy")

import numpy as np
import polars as pl
from datetime import datetime
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from vnpy.trader.database import get_database
from vnpy.trader.constant import Exchange, Interval

CONTEXT = 512
PATCH = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_embeddings(model, series_dict, dates):
    """对每 (stock, date)，取 512 日收益序列的 TimesFM output_embeddings（mean-pool over patches）"""
    feats, tgts, keys = [], [], []
    for td in dates:
        batch_syms, batch_inputs = [], []
        for s, (dts, ret) in series_dict.items():
            if td not in dts:
                continue
            i = list(dts).index(td)
            if i + 5 >= len(ret) or i + 1 < CONTEXT:
                continue
            batch_syms.append(s)
            batch_inputs.append(ret[i + 1 - CONTEXT: i + 1])
            tgts.append(ret[i + 1: i + 6].sum())  # 未来 5 日收益
        if not batch_syms:
            continue
        X = np.stack(batch_inputs)  # (B, 512)
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        xt = torch.tensor(X, dtype=torch.float32, device=DEVICE).reshape(len(X), -1, PATCH)
        mask = torch.zeros(xt.shape, dtype=torch.bool, device=DEVICE)
        with torch.no_grad():
            (_, out_emb, _, _), _ = model(xt, mask)
        emb = out_emb.mean(dim=1).cpu().numpy()  # mean-pool over patches → (B, 1280)
        feats.append(emb)
        keys.extend([(td, s) for s in batch_syms])
    return np.concatenate(feats, axis=0), np.array(tgts), keys


def main():
    sig = pl.read_parquet("core/alpha_db/signal/ashare_mlp_signal_v15.parquet")
    syms = sorted(sig["vt_symbol"].unique().to_list())
    db = get_database()
    series = {}
    for s in syms:
        code, ex = s.split(".")
        bars = db.load_bar_data(code, Exchange(ex), Interval.DAILY, datetime(2023, 1, 1), datetime(2026, 7, 17))
        if bars and len(bars) > CONTEXT + 10:
            close = np.array([b.close_price for b in bars])
            dts = np.array([b.datetime.strftime("%Y-%m-%d") for b in bars])
            series[s] = (dts, close[1:] / close[:-1] - 1)  # 日收益序列（对应 dts[1:]）
    for s in series:
        series[s] = (series[s][0][1:], series[s][1])
    all_dates = sorted({d for dts, _ in series.values() for d in dts})
    train_dates = [d for d in all_dates if "2025-01-01" <= d <= "2025-12-31"][::6]  # 每 6 日采样
    test_dates = [d for d in all_dates if "2026-01-01" <= d <= "2026-07-10"][::3]
    print(f"universe {len(series)}, train dates {len(train_dates)}, test dates {len(test_dates)}")

    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m = model.model
    print("TimesFM loaded, extracting embeddings ...")

    Xtr, ytr, _ = extract_embeddings(m, series, train_dates)
    print(f"train embeddings {Xtr.shape}")
    Xte, yte, te_keys = extract_embeddings(m, series, test_dates)
    print(f"test embeddings {Xte.shape}")

    # 训练 MLP 头
    torch.manual_seed(42)
    head = nn.Sequential(
        nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1),
    ).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    n = len(xt)
    for epoch in range(200):
        head.train()
        perm = torch.randperm(n, device=DEVICE)
        tot = 0.0
        for i in range(0, n, 4096):
            idx = perm[i: i + 4096]
            opt.zero_grad()
            loss = lossf(head(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}: loss {tot:.4f}")

    # 测试 IC
    head.eval()
    with torch.no_grad():
        preds = head(torch.tensor(Xte, dtype=torch.float32, device=DEVICE)).squeeze(1).cpu().numpy()
    by_date = {}
    for (td, s), p, t in zip(te_keys, preds, yte):
        by_date.setdefault(td, []).append((p, t))
    ics = []
    for td, pairs in by_date.items():
        if len(pairs) > 100:
            ic, _ = spearmanr([p for p, _ in pairs], [t for _, t in pairs])
            if not np.isnan(ic):
                ics.append((td, ic))
    v = np.array([ic for _, ic in ics])
    print(f"\n=== TimesFM 表征微调探针（2026 H1 {len(ics)} 日）: mean IC={v.mean():+.4f}, 正占比 {(v > 0).mean():.0%} ===")
    print("判定参考: IC>0.03 表征含信号值得全量微调; ≈0 范式无 alpha")
    for td, ic in ics[-8:]:
        print(f"  {td}: {ic:+.3f}")


if __name__ == "__main__":
    main()
