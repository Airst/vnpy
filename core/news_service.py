"""
资讯读取服务（read side）。

从 core/alpha_db/news/{date}.json 读取已采集的板块资讯，
提供列表/过滤/板块聚合。写 side 见 core/llm/news_collector.py。
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from vnpy.alpha.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEWS_DIR = PROJECT_ROOT / "core" / "alpha_db" / "news"


def _load_day(date_str: str) -> List[Dict[str, Any]]:
    fp = NEWS_DIR / f"{date_str}.json"
    if not fp.exists():
        return []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        if isinstance(data, list):
            return data
    except Exception as e:
        logger.error(f"[news_service] load {fp} failed: {e}")
    return []


def _load_meta(date_str: str) -> Dict[str, Any]:
    fp = NEWS_DIR / f"{date_str}.json"
    if not fp.exists():
        return {}
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k != "items"}
    except Exception:
        pass
    return {}


def list_dates() -> List[str]:
    """已采集的日期列表，降序。"""
    if not NEWS_DIR.exists():
        return []
    dates = sorted(
        (p.stem for p in NEWS_DIR.glob("*.json") if p.stem),
        reverse=True,
    )
    return dates


def _latest_date() -> Optional[str]:
    dates = list_dates()
    return dates[0] if dates else None


def get_news(
    date: Optional[str] = None,
    sector: Optional[str] = None,
    sentiment: Optional[str] = None,
    impact_type: Optional[str] = None,
    direction: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """读取资讯列表，支持过滤。date 为空则取最新采集日。

    direction 过滤投资分析方向：bullish/bearish/neutral。
    """
    if not date:
        date = _latest_date()
    if not date:
        return {"date": None, "items": [], "count": 0, "meta": {}}

    items = _load_day(date)
    meta = _load_meta(date)

    if sector:
        items = [it for it in items if sector in (it.get("sector", "") + " " + " ".join(it.get("related_sectors", [])))]
    if sentiment:
        items = [it for it in items if it.get("sentiment") == sentiment]
    if impact_type:
        items = [it for it in items if it.get("impact_type") == impact_type]
    if direction:
        items = [it for it in items if it.get("direction") == direction]

    # 排序：timeliness high 优先，info_date 新优先
    order = {"high": 0, "medium": 1, "low": 2}
    items = sorted(items, key=lambda x: (order.get(x.get("timeliness"), 1), x.get("info_date", "")), reverse=False)
    items = sorted(items, key=lambda x: order.get(x.get("timeliness"), 1))

    total = len(items)
    items = items[:limit]

    return {"date": date, "items": items, "count": total, "meta": meta}


def get_sectors(date: Optional[str] = None) -> Dict[str, Any]:
    """板块聚合：distinct sector + 计数 + 平均概念涨跌幅。"""
    if not date:
        date = _latest_date()
    if not date:
        return {"date": None, "sectors": []}

    items = _load_day(date)
    agg: Dict[str, Dict[str, Any]] = {}
    for it in items:
        sec = it.get("sector", "")
        if not sec:
            continue
        a = agg.setdefault(sec, {"sector": sec, "count": 0, "concept_pct_change": None, "sentiment": None})
        a["count"] += 1
        if a["concept_pct_change"] is None and it.get("concept_pct_change") is not None:
            a["concept_pct_change"] = it["concept_pct_change"]
        if a["sentiment"] is None:
            a["sentiment"] = it.get("sentiment")

    sectors = sorted(agg.values(), key=lambda x: x.get("concept_pct_change") if x.get("concept_pct_change") is not None else -999, reverse=True)
    return {"date": date, "sectors": sectors}


def get_history(days: int = 14) -> Dict[str, Any]:
    """最近 N 天的采集概况（日期 + 条数）。"""
    dates = list_dates()[:days]
    out = []
    for d in dates:
        items = _load_day(d)
        out.append({"date": d, "count": len(items)})
    return {"days": out}
