"""
Stock entry timing screener.

Takes top-N signal candidates, sends each to OpenClaw for LLM analysis,
and returns structured entry timing assessments (buy_now/wait/avoid).

Supports both sequential and parallel (batched) evaluation.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.llm.openclaw_client import OpenClawClient, _extract_json
from core.llm.prompts import build_stock_rating_messages


VALID_ACTIONS = {"buy_now", "wait", "avoid"}
# Legacy compatibility
VALID_RATINGS = VALID_ACTIONS


@dataclass
class StockRating:
    """Structured result from LLM entry timing evaluation for one stock."""

    vt_symbol: str
    action: str  # "buy_now", "wait", "avoid"
    risk_level: str  # "low", "medium", "high"
    reason: str
    confidence: float
    analysis_dimensions: Dict[str, str]  # risk_event, earnings_quality, entry_timing, sentiment
    key_factors: List[Dict[str, Any]]  # [{type, dimension, content, info_date, timeliness}]
    entry_timing: Dict[str, Any]  # {recommendation, wait_reason, wait_days, upcoming_events}
    risk_events: List[Dict[str, Any]]  # [{event, date, severity, source, priced_in}]
    stop_loss_price: Optional[float]
    expiry_days: int
    raw_response: Optional[str] = None
    error: Optional[str] = None

    # Backward-compatible property: old code using .rating still works
    @property
    def rating(self) -> str:
        """Map action to legacy rating: buy_now->good, avoid->bad, wait->neutral."""
        return {"buy_now": "good", "avoid": "bad", "wait": "neutral"}.get(
            self.action, "neutral"
        )

    def is_good(self) -> bool:
        return self.action == "buy_now"

    def is_bad(self) -> bool:
        return self.action == "avoid"

    def is_neutral(self) -> bool:
        return self.action == "wait"


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
            return self._parse_response(vt_symbol, raw, system, user)
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
                        print(f"[StockRatingScreener]   [{idx + 1}/{total}] → {result.action} (confidence={result.confidence:.2f}): {result.reason[:60]}")
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
        print(f"[StockRatingScreener] Done: BuyNow={good}, Avoid={bad}, Wait={neutral}, Errors={errors}")

        return final_results

    def _parse_response(self, vt_symbol: str, raw: str, system: str = "", user: str = "") -> StockRating:
        """Parse and validate LLM JSON response. Retries once on JSON errors using original session."""
        extracted = _extract_json(raw)
        try:
            data = json.loads(extracted)
        except json.JSONDecodeError as e:
            # Retry: continue original conversation asking LLM to fix its JSON
            data = self._retry_json_fix(raw, system, user)
            if data is None:
                return self._error_result(vt_symbol, f"Invalid JSON: {e}", raw=raw)

        try:
            return self._build_result_from_dict(vt_symbol, data, raw=raw)
        except Exception as e:
            return self._error_result(vt_symbol, f"Validation error: {e}", raw=raw)

    def _retry_json_fix(self, malformed_response: str, system: str, user: str) -> Optional[Dict[str, Any]]:
        """Continue original session to ask LLM to fix its malformed JSON response."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": malformed_response},
            {"role": "user", "content": "你的回复JSON格式有误，无法解析。请重新输出正确的JSON，只返回JSON，不要有其他内容。"},
        ]
        try:
            retry_raw = self.client.chat_messages(
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
            )
            extracted = _extract_json(retry_raw)
            return json.loads(extracted)
        except Exception:
            return None

    def _build_result_from_dict(
        self,
        vt_symbol: str,
        data: Dict[str, Any],
        raw: Optional[str] = None,
    ) -> StockRating:
        """Validate and build a StockRating from a dict."""
        action = str(data.get("action", "wait")).lower()
        if action not in VALID_ACTIONS:
            action = "wait"

        risk_level = str(data.get("risk_level", "medium")).lower()
        if risk_level not in ("low", "medium", "high"):
            risk_level = "medium"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        # Downgrade low-confidence actions to wait
        if confidence < 0.6 and action != "wait":
            action = "wait"

        stop_loss_price = data.get("stop_loss_price")
        if stop_loss_price is not None:
            try:
                stop_loss_price = float(stop_loss_price)
            except (ValueError, TypeError):
                stop_loss_price = None

        expiry_days = int(data.get("expiry_days", 30))
        expiry_days = max(1, min(180, expiry_days))

        analysis_dimensions = data.get("analysis_dimensions", {}) or {}
        key_factors = data.get("key_factors", []) or []
        entry_timing = data.get("entry_timing", {}) or {}
        risk_events = data.get("risk_events", []) or []

        return StockRating(
            vt_symbol=vt_symbol,
            action=action,
            risk_level=risk_level,
            reason=str(data.get("reason", ""))[:500],
            confidence=confidence,
            analysis_dimensions=analysis_dimensions,
            key_factors=key_factors,
            entry_timing=entry_timing,
            risk_events=risk_events,
            stop_loss_price=stop_loss_price,
            expiry_days=expiry_days,
            raw_response=raw,
        )

    def _error_result(
        self, vt_symbol: str, error_msg: str, raw: Optional[str] = None
    ) -> StockRating:
        """Fail-safe: on any error, return wait / no-op result."""
        return StockRating(
            vt_symbol=vt_symbol,
            action="wait",
            risk_level="medium",
            reason=f"[ERROR] {error_msg}",
            confidence=0.0,
            analysis_dimensions={},
            key_factors=[],
            entry_timing={"recommendation": "wait", "wait_reason": "评估失败，建议人工复核", "wait_days": 0, "upcoming_events": []},
            risk_events=[],
            stop_loss_price=None,
            expiry_days=30,
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
            "action": rating.action,
            "rating": rating.rating,  # backward-compat mapped field
            "risk_level": rating.risk_level,
            "reason": rating.reason,
            "confidence": rating.confidence,
            "analysis_dimensions": rating.analysis_dimensions,
            "key_factors": rating.key_factors,
            "entry_timing": rating.entry_timing,
            "risk_events": rating.risk_events,
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
    print(f"[save_ratings] Saved {len(results)} stocks: BuyNow={good}, Avoid={bad}, Wait={neutral}, Errors={errors}")


# Legacy aliases for backward compatibility
ScreeningResult = StockRating
RiskScreener = StockRatingScreener
