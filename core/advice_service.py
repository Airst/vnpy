"""
投资建议读取服务（read side）。

从 core/alpha_db/advice/{date}.json 读取每日建议（潜力股/风险板块/持仓动向），
提供列表与最新查询。写 side（LLM 生成）见 core/llm/advisor.py。
持仓股票池 CRUD 直接透传 advisor 模块（文件锁在写侧统一管理）。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from vnpy.alpha.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADVICE_DIR = PROJECT_ROOT / "core" / "alpha_db" / "advice"


def list_advice_dates() -> List[str]:
    """已生成建议的日期列表，降序（排除 watchlist.json）。"""
    if not ADVICE_DIR.exists():
        return []
    return sorted(
        (p.stem for p in ADVICE_DIR.glob("*.json") if p.stem[:2].isdigit()),
        reverse=True,
    )


def get_advice(date: Optional[str] = None) -> Dict[str, Any]:
    """读取某日建议，date 为空取最新。无数据返回 {status: empty}。"""
    if not date:
        dates = list_advice_dates()
        date = dates[0] if dates else None
    if not date:
        return {"status": "empty", "date": None}
    fp = ADVICE_DIR / f"{date}.json"
    if not fp.exists():
        return {"status": "empty", "date": date}
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[advice_service] load {fp} failed: {e}")
        return {"status": "error", "date": date, "message": str(e)}
