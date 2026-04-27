"""
Stock rating screener.

Takes top-N signal candidates, sends each to OpenClaw for LLM analysis,
and returns structured stock ratings (Good/Bad/Neutral) with predictions.

Supports both sequential and parallel (batched) rating.
"""

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.llm.openclaw_client import OpenClawClient, _extract_json
from core.llm.prompts import build_stock_rating_messages


VALID_RATINGS = {"good", "bad", "neutral"}


@dataclass
class StockRating:
    """Structured result from LLM stock rating prediction for one stock."""

    vt_symbol: str
    rating: str  # "Good", "Bad", "Neutral"
    reason: str
    confidence: float
    analysis_dimensions: Dict[str, str]  # technical, fundamental, event, sentiment
    key_factors: List[Dict[str, str]]  # [{type: positive/negative, dimension, content}]
    target_direction: str  # "up", "down", "flat"
    stop_loss_price: Optional[float]
    expiry_days: int
    raw_response: Optional[str] = None
    error: Optional[str] = None

    def is_good(self) -> bool:
        return self.rating == "good"

    def is_bad(self) -> bool:
        return self.rating == "bad"

    def is_neutral(self) -> bool:
        return self.rating == "neutral"


class StockRatingScreener:
    """
    Orchestrates LLM-based stock rating prediction of candidates.

    Workflow:
        1. Accept list of (vt_symbol, score) candidates
        2. Call OpenClaw per stock (LLM uses tushare/web-search tools internally)
        3. Parse and validate JSON response
        4. Return list of StockRating

    On LLM errors or validation failures, the result for that stock is
    marked as error with rating='neutral' (fail-safe default).
    """

    def __init__(self, client: Optional[OpenClawClient] = None):
        self.client = client or OpenClawClient()

    def rate_one(self, vt_symbol: str, score: float, check_date: str) -> StockRating:
        """Rate a single stock and return structured result."""
        system, user = build_stock_rating_messages(vt_symbol, score, check_date)

        try:
            raw = self.client.chat(
                system=system,
                user=user,
                response_format_json=True,
                temperature=0.2,
                max_tokens=2048,
            )
            return self._parse_response(vt_symbol, raw)
        except Exception as e:
            return self._error_result(vt_symbol, str(e))

    def rate_many(
        self,
        candidates: List[Dict[str, Any]],
        check_date: Optional[str] = None,
        batch_size: int = 4,
        max_workers: int = 4,
    ) -> List[StockRating]:
        """
        Rate multiple stocks with parallel batch execution.

        Parameters
        ----------
        candidates : list of {"vt_symbol": str, "score": float}
        check_date : str, optional
            Reference date in YYYY-MM-DD format (default: today)
        batch_size : int
            Number of stocks per batch (default: 4)
        max_workers : int
            Number of concurrent threads within each batch (default: 4)

        Returns
        -------
        list of StockRating
        """
        if check_date is None:
            check_date = datetime.now().strftime("%Y-%m-%d")

        if not candidates:
            return []

        total = len(candidates)
        num_batches = math.ceil(total / batch_size)
        print(f"[StockRatingScreener] Rating {total} stocks in {num_batches} batches (batch_size={batch_size}, workers={max_workers})")

        all_results: List[StockRating] = [None] * total

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch = candidates[start:end]

            print(f"[StockRatingScreener] Batch {batch_idx + 1}/{num_batches}: processing {len(batch)} stocks...")

            # Parallel processing within batch
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {}
                for i, c in enumerate(batch):
                    idx = start + i
                    vt_symbol = c["vt_symbol"]
                    score = float(c.get("score", 0.0))
                    print(f"[StockRatingScreener]   [{idx + 1}/{total}] Submitting {vt_symbol} (score={score:.4f})...")
                    future = executor.submit(self.rate_one, vt_symbol, score, check_date)
                    future_to_idx[future] = idx

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        all_results[idx] = result
                        print(f"[StockRatingScreener]   [{idx + 1}/{total}] → {result.rating} (confidence={result.confidence:.2f}): {result.reason[:60]}")
                    except Exception as exc:
                        c = batch[min(idx % batch_size, len(batch) - 1)]
                        vt_symbol = c["vt_symbol"]
                        print(f"[StockRatingScreener]   [{idx + 1}/{total}] {vt_symbol} failed: {exc}")
                        all_results[idx] = self._error_result(vt_symbol, str(exc))

        # Filter out None (should not happen, but safety)
        final_results = [r for r in all_results if r is not None]

        # Summary
        good = sum(1 for r in final_results if r.is_good())
        bad = sum(1 for r in final_results if r.is_bad())
        neutral = sum(1 for r in final_results if r.is_neutral())
        errors = sum(1 for r in final_results if r.error)
        print(f"[StockRatingScreener] Done: Good={good}, Bad={bad}, Neutral={neutral}, Errors={errors}")

        return final_results

    def _parse_response(self, vt_symbol: str, raw: str) -> StockRating:
        """Parse and validate LLM JSON response."""
        extracted = _extract_json(raw)
        try:
            data = json.loads(extracted)
        except json.JSONDecodeError as e:
            return self._error_result(vt_symbol, f"Invalid JSON: {e}", raw=raw)

        try:
            return self._build_result_from_dict(vt_symbol, data, raw=raw)
        except Exception as e:
            return self._error_result(vt_symbol, f"Validation error: {e}", raw=raw)

    def _build_result_from_dict(
        self,
        vt_symbol: str,
        data: Dict[str, Any],
        raw: Optional[str] = None,
    ) -> StockRating:
        """Validate and build a StockRating from a dict."""
        rating = str(data.get("rating", "neutral")).lower()
        if rating not in VALID_RATINGS:
            rating = "neutral"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        # Downgrade low-confidence ratings
        original_rating = rating
        if confidence < 0.6:
            if rating == "good":
                rating = "neutral"
            elif rating == "bad":
                rating = "neutral"

        target_direction = str(data.get("target_direction", "flat")).lower()
        if target_direction not in ("up", "down", "flat"):
            target_direction = "flat"

        stop_loss_price = data.get("stop_loss_price")
        if stop_loss_price is not None:
            try:
                stop_loss_price = float(stop_loss_price)
            except (ValueError, TypeError):
                stop_loss_price = None

        expiry_days = int(data.get("expiry_days", 60))
        expiry_days = max(1, min(180, expiry_days))

        analysis_dimensions = data.get("analysis_dimensions", {}) or {}
        key_factors = data.get("key_factors", []) or []

        return StockRating(
            vt_symbol=vt_symbol,
            rating=rating,
            reason=str(data.get("reason", ""))[:500],
            confidence=confidence,
            analysis_dimensions=analysis_dimensions,
            key_factors=key_factors,
            target_direction=target_direction,
            stop_loss_price=stop_loss_price,
            expiry_days=expiry_days,
            raw_response=raw,
        )

    def _error_result(
        self, vt_symbol: str, error_msg: str, raw: Optional[str] = None
    ) -> StockRating:
        """Fail-safe: on any error, return neutral / no-op result."""
        return StockRating(
            vt_symbol=vt_symbol,
            rating="neutral",
            reason=f"[ERROR] {error_msg}",
            confidence=0.0,
            analysis_dimensions={},
            key_factors=[],
            target_direction="flat",
            stop_loss_price=None,
            expiry_days=60,
            raw_response=raw,
            error=error_msg,
        )


def save_ratings(results: List[StockRating], output_dir: str) -> None:
    """Persist stock ratings to per-stock JSON files, appending to existing history.
    
    Each stock gets its own file: {vt_symbol}.json
    The file contains a list of rating entries (historical evaluations).
    
    Parameters
    ----------
    output_dir : str
        Directory path where per-stock JSON files will be saved.
    """
    ratings_dir = Path(output_dir)
    ratings_dir.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    for rating in results:
        stock_file = ratings_dir / f"{rating.vt_symbol}.json"
        
        # Load existing history or start new
        history = []
        if stock_file.exists():
            try:
                with open(stock_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                if not isinstance(history, list):
                    history = []
            except (json.JSONDecodeError, Exception):
                history = []
        
        # Append new rating
        history.append({
            "date": today,
            "vt_symbol": rating.vt_symbol,
            "rating": rating.rating,
            "reason": rating.reason,
            "confidence": rating.confidence,
            "analysis_dimensions": rating.analysis_dimensions,
            "key_factors": rating.key_factors,
            "target_direction": rating.target_direction,
            "stop_loss_price": rating.stop_loss_price,
            "expiry_days": rating.expiry_days,
            "error": rating.error,
        })
        
        # Save updated history
        with open(stock_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    # Summary
    good = sum(1 for r in results if r.is_good())
    bad = sum(1 for r in results if r.is_bad())
    neutral = sum(1 for r in results if r.is_neutral())
    errors = sum(1 for r in results if r.error)
    print(f"[save_ratings] Saved {len(results)} stocks: Good={good}, Bad={bad}, Neutral={neutral}, Errors={errors}")


# Legacy aliases for backward compatibility
ScreeningResult = StockRating
RiskScreener = StockRatingScreener
