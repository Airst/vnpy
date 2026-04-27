"""
LLM Rating Task Runner.

Standalone module for running LLM evaluations on signal Top-K stocks.
Decoupled from model training - reads signals from AlphaLab parquet files,
selects Top-K stocks by score, and runs LLM rating evaluation.

Supports:
  - Reading signals by version (e.g., v8, v9)
  - Force mode (-f): re-evaluate even if stock has a valid (non-expired) rating
  - Per-stock file persistence with history appending
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from core.llm.openclaw_client import OpenClawClient
from core.llm.risk_screener import StockRating, StockRatingScreener, save_ratings
from vnpy.alpha.lab import AlphaLab


# Default paths
DEFAULT_SIGNAL_DIR = "core/alpha_db/signal"
DEFAULT_RATING_DIR = "core/alpha_db/llm_tasks"


class LLMRatingTask:
    """
    Orchestrates LLM rating evaluation on Top-K signal stocks.

    Workflow:
        1. Load signal parquet by version name
        2. Get latest date's Top-K stocks by total_score
        3. Check existing ratings for validity (skip if still valid, unless force mode)
        4. Run LLM rating on valid candidates
        5. Save results to per-stock JSON files with history

    Parameters
    ----------
    signal_dir : str
        Directory containing signal parquet files.
    rating_dir : str
        Directory for per-stock rating JSON files.
    lab_path : str, optional
        AlphaLab path (unused, signals read directly from parquet).
    """

    def __init__(
        self,
        signal_dir: str = DEFAULT_SIGNAL_DIR,
        rating_dir: str = DEFAULT_RATING_DIR,
    ):
        self.signal_dir = Path(signal_dir)
        self.rating_dir = Path(rating_dir)
        self.rating_dir.mkdir(parents=True, exist_ok=True)
        self.screener = StockRatingScreener()

    def get_signal_path(self, version: str) -> Optional[Path]:
        """Get signal parquet file path for a given version."""
        # Normalize version: "v9" -> "ashare_mlp_signal_v9"
        if not version.startswith("v"):
            version = "v" + version
        signal_name = f"ashare_mlp_signal_{version}"
        signal_path = self.signal_dir / f"{signal_name}.parquet"
        if not signal_path.exists():
            return None
        return signal_path

    def load_top_k(
        self, version: str, top_k: int = 20
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Load signal parquet and return (latest_date, top_k_candidates).

        Parameters
        ----------
        version : str
            Signal version (e.g., "v8", "v9").
        top_k : int
            Number of top stocks to select.

        Returns
        -------
        tuple of (latest_date_str, candidates)
            candidates: list of {"vt_symbol": str, "score": float}
        """
        signal_path = self.get_signal_path(version)
        if signal_path is None:
            raise FileNotFoundError(
                f"Signal file not found for version '{version}'. "
                f"Expected: {self.signal_dir / f'ashare_mlp_signal_{version}.parquet'}"
            )

        df = pl.read_parquet(signal_path)

        if df is None or df.is_empty():
            raise ValueError(f"Signal data is empty for version '{version}'")

        # Get latest date
        latest_dt = df["datetime"].max()
        latest_df = df.filter(pl.col("datetime") == latest_dt)

        # Score column: try total_score, then final_signal, then score
        score_col = None
        for col_name in ["total_score", "final_signal", "score"]:
            if col_name in latest_df.columns:
                score_col = col_name
                break

        if score_col is None:
            raise ValueError(
                f"No score column found in signal data. Columns: {latest_df.columns}"
            )

        # Sort by score descending, take top_k
        top_k_df = latest_df.sort(score_col, descending=True).head(top_k)

        candidates = []
        for row in top_k_df.iter_rows(named=True):
            candidates.append({
                "vt_symbol": row["vt_symbol"],
                "score": float(row[score_col]),
            })

        # Format date
        if hasattr(latest_dt, "strftime"):
            latest_dt_str = latest_dt.strftime("%Y-%m-%d")
        else:
            latest_dt_str = str(latest_dt).split(" ")[0]

        return latest_dt_str, candidates

    def is_rating_valid(self, vt_symbol: str, check_date: str) -> bool:
        """
        Check if a stock already has a valid (non-expired, non-error) rating.

        A rating is valid if:
          - The stock file exists
          - The latest entry has no error
          - The latest entry's date + expiry_days >= check_date

        Parameters
        ----------
        vt_symbol : str
            Stock symbol (e.g., "000001.SZ").
        check_date : str
            Reference date in YYYY-MM-DD format.

        Returns
        -------
        bool
            True if valid rating exists, False otherwise.
        """
        stock_file = self.rating_dir / f"{vt_symbol}.json"
        if not stock_file.exists():
            return False

        try:
            with open(stock_file, "r", encoding="utf-8") as f:
                history = json.load(f)

            if not isinstance(history, list) or not history:
                return False

            latest = history[-1]

            # Check for error
            if latest.get("error"):
                return False

            # Check expiry
            rating_date_str = latest.get("date", "")
            expiry_days = latest.get("expiry_days", 60)

            if not rating_date_str:
                return False

            try:
                rating_date = datetime.strptime(rating_date_str, "%Y-%m-%d")
                check_date_dt = datetime.strptime(check_date, "%Y-%m-%d")
                valid_until = rating_date + timedelta(days=expiry_days)
                return check_date_dt <= valid_until
            except ValueError:
                return False

        except (json.JSONDecodeError, Exception):
            return False

    def filter_candidates(
        self,
        candidates: List[Dict[str, Any]],
        check_date: str,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Filter out candidates that already have valid ratings.

        Parameters
        ----------
        candidates : list of {"vt_symbol": str, "score": float}
        check_date : str
            Reference date in YYYY-MM-DD format.
        force : bool
            If True, skip validity check and evaluate all candidates.

        Returns
        -------
        list of candidates that need evaluation
        """
        if force:
            return candidates

        valid_count = 0
        to_evaluate = []

        for c in candidates:
            vt_symbol = c["vt_symbol"]
            if self.is_rating_valid(vt_symbol, check_date):
                valid_count += 1
            else:
                to_evaluate.append(c)

        if valid_count > 0:
            print(f"[LLMRatingTask] Skipped {valid_count} stocks with valid ratings")

        return to_evaluate

    def run(
        self,
        version: str,
        top_k: int = 20,
        force: bool = False,
        batch_size: int = 4,
        max_workers: int = 4,
    ) -> List[StockRating]:
        """
        Execute the full LLM rating pipeline.

        Parameters
        ----------
        version : str
            Signal version (e.g., "v8", "v9").
        top_k : int
            Number of top stocks to select.
        force : bool
            If True, re-evaluate all stocks even if they have valid ratings.
        batch_size : int
            Number of stocks per LLM batch.
        max_workers : int
            Number of concurrent LLM threads per batch.

        Returns
        -------
        list of StockRating
        """
        # Step 1: Load Top-K candidates
        print(f"\n=== LLM Rating Task: version={version}, top_k={top_k} ===")
        check_date, candidates = self.load_top_k(version, top_k)
        print(f"=== Latest signal date: {check_date} ===")
        print(f"=== Total candidates: {len(candidates)} ===")

        if not candidates:
            print("[LLMRatingTask] No candidates found.")
            return []

        # Step 2: Filter by validity
        to_evaluate = self.filter_candidates(candidates, check_date, force)

        if not to_evaluate:
            print("[LLMRatingTask] All stocks have valid ratings. Nothing to do.")
            return []

        print(f"=== Stocks to evaluate: {len(to_evaluate)} ===")
        print(f"=== Batch size={batch_size}, Max workers={max_workers} ===")

        # Step 3: Run LLM rating
        ratings = self.screener.rate_many(
            to_evaluate, check_date, batch_size=batch_size, max_workers=max_workers
        )

        # Step 4: Save to per-stock files
        save_ratings(ratings, str(self.rating_dir))

        # Summary
        good_count = sum(1 for r in ratings if r.is_good())
        bad_count = sum(1 for r in ratings if r.is_bad())
        neutral_count = sum(1 for r in ratings if r.is_neutral())
        error_count = sum(1 for r in ratings if r.error)
        print(f"\n=== Summary: Good={good_count}, Bad={bad_count}, Neutral={neutral_count}, Errors={error_count} ===")

        return ratings
