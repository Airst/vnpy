"""
训练轮次(Run)产物管理

== 设计 ==
- Run: 一次全量训练(-t)产生一个 run, run_id = {version}_{YYYYMMDD_HHMMSS}
- 存储布局: core/alpha_db/runs/{run_id}/
  - manifest.json          版本/signal_name/配置/因子IC摘要/窗口清单/回测引用/信号覆盖范围
  - models/{ps_str}.pkl    每个滚动窗口的模型 (ps_str = 预测窗口起始日 YYYY-MM-DD)
  - signal.parquet         该 run 的完整信号
- 全局因子库: core/alpha_db/factors/{version}.parquet (每版本仅一份, 不随 run 快照)
  因子总是按最新代码重算, per-run 快照无消费方且重复占用 ~1GB/次;
  改为按因子名增量合并: 新因子加列/已删因子去列, 重叠日期以新值为准,
  store 中新 df 未覆盖的历史日期行保留 (见 save_factors)
- Active run: runs/active.json 记录生产 run;
  set_active / sync_signal_to_production 时把 run 的 signal.parquet
  复制到现有 signal/{signal_name}.parquet, 生产策略/实盘读取路径不变

== 增量补全 ==
training.py 无 -t 时默认补全 active run; --run {run_id} 指定任意 run.
从该 run 信号的最后日期起逐窗口补全: 窗口模型已存在则纯推理,
跨过窗口边界则训练新窗口模型并存入该 run (见 MLPSignals 的 run 模式).
"""
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Set

import polars as pl

MANIFEST_NAME = "manifest.json"
ACTIVE_NAME = "active.json"
SIGNAL_NAME = "signal.parquet"
MODELS_DIR = "models"
FACTOR_KEY_COLS = ["datetime", "vt_symbol"]


class RunManager:
    """训练轮次产物管理器 (runs/{run_id}/ 目录的唯一读写入口)"""

    def __init__(self, lab_path: str = "core/alpha_db"):
        self.lab_path = Path(lab_path)
        self.runs_path = self.lab_path.joinpath("runs")
        self.signal_path = self.lab_path.joinpath("signal")
        self.factors_path = self.lab_path.joinpath("factors")
        self.runs_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 路径
    # ------------------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        return self.runs_path.joinpath(run_id)

    def _manifest_path(self, run_id: str) -> Path:
        return self.run_dir(run_id).joinpath(MANIFEST_NAME)

    def _models_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id).joinpath(MODELS_DIR)

    def run_exists(self, run_id: str) -> bool:
        return self._manifest_path(run_id).exists()

    # ------------------------------------------------------------------
    # Run 生命周期
    # ------------------------------------------------------------------
    def create_run(self, version: str, signal_name: str, config: Optional[dict] = None) -> str:
        """创建新 run, 返回 run_id"""
        now = datetime.now()
        run_id = f"{version}_{now.strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._models_dir(run_id).mkdir(exist_ok=True)

        manifest = {
            "run_id": run_id,
            "version": version,
            "signal_name": signal_name,
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config or {},
            "factors": {},          # {factor: {ic, icir}}
            "windows": [],          # [{ps_str, pe_str, saved_at}]
            "signal_range": {},     # {start, end, rows}
            "backtests": [],        # 回测结果 JSON 文件名引用
        }
        self._write_manifest(run_id, manifest)
        print(f"[RunManager] Created run: {run_id}")
        return run_id

    def delete_run(self, run_id: str) -> bool:
        """删除 run 目录 (禁止删除 active run)"""
        if self.get_active() == run_id:
            raise ValueError(f"Cannot delete active run: {run_id}")
        run_dir = self.run_dir(run_id)
        if not run_dir.exists():
            return False
        shutil.rmtree(run_dir)
        print(f"[RunManager] Deleted run: {run_id}")
        return True

    def list_runs(self) -> List[dict]:
        """列出所有 run 的 manifest 摘要 (新→旧)"""
        runs = []
        for d in sorted(self.runs_path.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            manifest = self.load_manifest(d.name)
            if manifest is None:
                continue
            runs.append(self._summarize(manifest))
        return runs

    def _summarize(self, manifest: dict) -> dict:
        run_id = manifest.get("run_id", "")
        models_dir = self._models_dir(run_id)
        n_models = len(list(models_dir.glob("*.pkl"))) if models_dir.exists() else 0
        signal_file = self.run_dir(run_id).joinpath(SIGNAL_NAME)
        return {
            "run_id": run_id,
            "version": manifest.get("version"),
            "signal_name": manifest.get("signal_name"),
            "created_at": manifest.get("created_at"),
            "config": manifest.get("config", {}),
            "n_factors": len(manifest.get("factors", {})),
            "n_models": n_models,
            "signal_range": manifest.get("signal_range", {}),
            "backtests": manifest.get("backtests", []),
            "has_signal": signal_file.exists(),
            "is_active": self.get_active() == run_id,
        }

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    def load_manifest(self, run_id: str) -> Optional[dict]:
        path = self._manifest_path(run_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[RunManager] Failed to read manifest of {run_id}: {e}")
            return None

    def update_manifest(self, run_id: str, updates: dict) -> None:
        manifest = self.load_manifest(run_id)
        if manifest is None:
            raise FileNotFoundError(f"Run {run_id} manifest not found")
        manifest.update(updates)
        self._write_manifest(run_id, manifest)

    def _write_manifest(self, run_id: str, manifest: dict) -> None:
        with open(self._manifest_path(run_id), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def add_backtest(self, run_id: str, filename: str) -> None:
        """把回测结果文件名登记到 manifest (供回测轮转清理豁免)"""
        manifest = self.load_manifest(run_id)
        if manifest is None:
            return
        backtests = manifest.setdefault("backtests", [])
        if filename not in backtests:
            backtests.append(filename)
            self._write_manifest(run_id, manifest)

    def list_referenced_backtests(self) -> Set[str]:
        """所有 run 引用的回测文件名集合 (回测 keep-4 轮转清理时豁免)"""
        referenced: Set[str] = set()
        if not self.runs_path.exists():
            return referenced
        for d in self.runs_path.iterdir():
            if not d.is_dir():
                continue
            manifest = self.load_manifest(d.name)
            if manifest:
                referenced.update(manifest.get("backtests", []))
        return referenced

    # ------------------------------------------------------------------
    # 模型 (每个滚动窗口一个)
    # ------------------------------------------------------------------
    def save_model(self, run_id: str, ps_str: str, pe_str: str, model) -> None:
        models_dir = self._models_dir(run_id)
        models_dir.mkdir(parents=True, exist_ok=True)
        with open(models_dir.joinpath(f"{ps_str}.pkl"), "wb") as f:
            pickle.dump(model, f)
        # 登记窗口清单 (按 ps_str 去重更新)
        manifest = self.load_manifest(run_id)
        if manifest is not None:
            windows = manifest.setdefault("windows", [])
            entry = {"ps_str": ps_str, "pe_str": pe_str,
                     "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            for i, w in enumerate(windows):
                if w.get("ps_str") == ps_str:
                    windows[i] = entry
                    break
            else:
                windows.append(entry)
            windows.sort(key=lambda w: w.get("ps_str", ""))
            self._write_manifest(run_id, manifest)

    def load_model(self, run_id: str, ps_str: str):
        path = self._models_dir(run_id).joinpath(f"{ps_str}.pkl")
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def has_model(self, run_id: str, ps_str: str) -> bool:
        return self._models_dir(run_id).joinpath(f"{ps_str}.pkl").exists()

    def list_window_models(self, run_id: str) -> List[str]:
        models_dir = self._models_dir(run_id)
        if not models_dir.exists():
            return []
        return sorted(f.stem for f in models_dir.glob("*.pkl"))

    # ------------------------------------------------------------------
    # 信号
    # ------------------------------------------------------------------
    def save_signal(self, run_id: str, signal_df: pl.DataFrame) -> None:
        """保存/合并 run 信号: 已有信号时只追加最后日期之后的新行"""
        if signal_df is None or signal_df.is_empty():
            print(f"[RunManager] Empty signal for {run_id}, skip saving.")
            return
        path = self.run_dir(run_id).joinpath(SIGNAL_NAME)
        existing = pl.read_parquet(path) if path.exists() else None
        if existing is not None and not existing.is_empty():
            last_dt = existing["datetime"].max()
            new_rows = signal_df.filter(pl.col("datetime") > last_dt)
            if new_rows.is_empty():
                print(f"[RunManager] No new signal rows after {last_dt} for {run_id}.")
                signal_df = existing
            else:
                print(f"[RunManager] Appending {len(new_rows)} signal rows (after {last_dt}) to {run_id}.")
                signal_df = pl.concat([existing, new_rows])
        signal_df = signal_df.sort(["datetime", "vt_symbol"])
        signal_df.write_parquet(path)
        self.update_manifest(run_id, {"signal_range": {
            "start": str(signal_df["datetime"].min())[:10],
            "end": str(signal_df["datetime"].max())[:10],
            "rows": len(signal_df),
        }})

    def load_signal(self, run_id: str) -> Optional[pl.DataFrame]:
        path = self.run_dir(run_id).joinpath(SIGNAL_NAME)
        if not path.exists():
            return None
        return pl.read_parquet(path)

    # ------------------------------------------------------------------
    # 全局因子库 (每版本仅一份, 按因子名增量合并, 不随 run 快照)
    # ------------------------------------------------------------------
    def factor_store_path(self, version: str) -> Path:
        return self.factors_path.joinpath(f"{version}.parquet")

    def save_factors(self, version: str, factors_df: pl.DataFrame) -> None:
        """增量合并因子到全局因子库 factors/{version}.parquet

        合并规则 (因子总是按最新代码重算, 新值权威):
        - 列级: 以新 df 的因子集合为准 —— 新增因子加列, 已从代码移除的因子去列
        - 行级: 与新 df 日期重叠的行整体用新值覆盖;
          store 中新 df 未覆盖的历史日期行保留 (新增因子在这些行为 null)
        - float64 → float32 降体积, zstd 压缩
        """
        if factors_df is None or factors_df.is_empty():
            print("[FactorStore] Empty factors df, skip saving.")
            return
        self.factors_path.mkdir(parents=True, exist_ok=True)
        path = self.factor_store_path(version)
        casted = factors_df.with_columns([
            pl.col(c).cast(pl.Float32)
            for c, dtype in factors_df.schema.items() if dtype == pl.Float64
        ])

        if path.exists():
            existing = pl.read_parquet(path)
            old_cols = [c for c in existing.columns if c not in FACTOR_KEY_COLS]
            new_cols = [c for c in casted.columns if c not in FACTOR_KEY_COLS]
            added = sorted(set(new_cols) - set(old_cols))
            dropped = sorted(set(old_cols) - set(new_cols))
            # 重叠日期以新值为准: 仅保留 store 中新 df 未覆盖的日期行 (并去掉已删因子列)
            kept = existing.filter(
                ~pl.col("datetime").is_in(casted["datetime"].unique())
            ).select([c for c in existing.columns if c in casted.columns])
            merged = casted if kept.is_empty() else pl.concat([kept, casted], how="diagonal")
            if added:
                print(f"[FactorStore] 新增因子列 {len(added)}: {', '.join(added[:10])}{' ...' if len(added) > 10 else ''}")
            if dropped:
                print(f"[FactorStore] 移除因子列 {len(dropped)}: {', '.join(dropped[:10])}{' ...' if len(dropped) > 10 else ''}")
            print(f"[FactorStore] 保留历史行 {len(kept)} + 重算行 {len(casted)}")
        else:
            merged = casted

        merged = merged.sort(FACTOR_KEY_COLS)
        merged.write_parquet(path, compression="zstd")
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"[FactorStore] Factor store updated: {path} "
              f"({len(merged)} rows x {len(merged.columns) - len(FACTOR_KEY_COLS)} factors, {size_mb:.1f} MB)")

    def load_factors(self, version: str) -> Optional[pl.DataFrame]:
        path = self.factor_store_path(version)
        if not path.exists():
            return None
        return pl.read_parquet(path)

    def factor_store_info(self, version: str) -> dict:
        """因子库元信息 (仅读 schema, 不加载数据)"""
        path = self.factor_store_path(version)
        if not path.exists():
            return {"exists": False}
        try:
            n_cols = len(pl.scan_parquet(path).collect_schema().names())
        except Exception:
            n_cols = 0
        return {
            "exists": True,
            "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
            "n_columns": max(n_cols - len(FACTOR_KEY_COLS), 0),
        }

    # ------------------------------------------------------------------
    # Active run
    # ------------------------------------------------------------------
    def get_active(self) -> Optional[str]:
        path = self.runs_path.joinpath(ACTIVE_NAME)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("run_id")
        except Exception:
            return None

    def set_active(self, run_id: str) -> None:
        """设为生产 run, 并把该 run 的信号同步到 signal/{signal_name}.parquet"""
        if not self.run_exists(run_id):
            raise FileNotFoundError(f"Run {run_id} not found")
        with open(self.runs_path.joinpath(ACTIVE_NAME), "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id,
                "activated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }, f, ensure_ascii=False, indent=2)
        print(f"[RunManager] Active run set to: {run_id}")
        self.sync_signal_to_production(run_id)

    def sync_signal_to_production(self, run_id: str) -> None:
        """把 run 的 signal.parquet 复制到生产信号路径 (策略/实盘读取路径不变)"""
        manifest = self.load_manifest(run_id)
        if manifest is None:
            raise FileNotFoundError(f"Run {run_id} manifest not found")
        src = self.run_dir(run_id).joinpath(SIGNAL_NAME)
        if not src.exists():
            print(f"[RunManager] Run {run_id} has no signal yet, skip production sync.")
            return
        self.signal_path.mkdir(parents=True, exist_ok=True)
        dst = self.signal_path.joinpath(f"{manifest['signal_name']}.parquet")
        shutil.copy2(src, dst)
        print(f"[RunManager] Synced run signal to production: {dst}")

    # ------------------------------------------------------------------
    # 详情 (供 API)
    # ------------------------------------------------------------------
    def get_run_detail(self, run_id: str) -> Optional[dict]:
        manifest = self.load_manifest(run_id)
        if manifest is None:
            return None
        detail = dict(manifest)
        detail["window_models"] = self.list_window_models(run_id)
        detail["is_active"] = self.get_active() == run_id
        signal_file = self.run_dir(run_id).joinpath(SIGNAL_NAME)
        detail["has_signal"] = signal_file.exists()
        # 全局因子库状态 (按版本共享, 非 run 级产物)
        detail["factor_store"] = self.factor_store_info(manifest.get("version", ""))
        return detail
