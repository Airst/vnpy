# Lazy imports to avoid hard dependency on openai for non-LLM use cases


def __getattr__(name):
    if name == "OpenClawClient":
        from core.llm.openclaw_client import OpenClawClient
        return OpenClawClient
    if name == "parse_json_response":
        from core.llm.openclaw_client import parse_json_response
        return parse_json_response
    if name in ("StockRatingScreener", "StockRating", "save_ratings"):
        from core.llm.risk_screener import (
            StockRatingScreener,
            StockRating,
            save_ratings,
        )
        if name == "StockRatingScreener":
            return StockRatingScreener
        if name == "StockRating":
            return StockRating
        if name == "save_ratings":
            return save_ratings
    if name == "LLMRatingTask":
        from core.llm.rating_task import LLMRatingTask
        return LLMRatingTask
    # Legacy aliases
    if name in ("RiskScreener", "ScreeningResult"):
        from core.llm.risk_screener import RiskScreener, ScreeningResult
        if name == "RiskScreener":
            return RiskScreener
        if name == "ScreeningResult":
            return ScreeningResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OpenClawClient",
    "parse_json_response",
    "StockRatingScreener",
    "StockRating",
    "save_ratings",
    "LLMRatingTask",
    "RiskScreener",
    "ScreeningResult",
]
