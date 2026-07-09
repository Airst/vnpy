"""
LLM 驱动的 A 股板块资讯采集 + 投资影响分析器

连接方式参考 llm 股票评估代码（core/llm/risk_screener.py + prompts.py）：
通过 OpenClawClient 接入本地 OpenClaw 编排器（OpenAI 兼容端点）。
OpenClaw 内置 web-search / tushare 等 MCP 工具，LLM 会自动 function calling
联网获取最新资讯，无需 extra_body.enable_search。

角色定位：A 股专业投资分析师。不做新闻总结，而是以投资者视角分析每条
资讯对市场的影响预期——方向、确信度、投资逻辑、传导链条、催化剂、风险、
预期差。分析范式对齐 STOCK_RATING_SYSTEM 的结构化投资者判断。

采集聚焦：影响板块、影响轮动、影响市场情绪的消息资讯（非逐股覆盖）。
每条资讯后处理映射到 dc_concept（板块当日涨跌幅/热度/领涨股）与
stock_basic（代表性个股 vt_symbol），让交易者能落地到标的。

落盘：core/alpha_db/news/{date}.json（每日一份，列表），参考 llm_tasks 的 JSON 模式。
"""

import json
import os
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.llm.openclaw_client import OpenClawClient, _extract_json
from vnpy.alpha.logger import logger


NEWS_DIR = Path(__file__).resolve().parent.parent.parent / "core" / "alpha_db" / "news"

VALID_IMPACT_TYPES = {"positive", "negative", "neutral"}
VALID_SENTIMENTS = {"positive", "negative", "neutral"}
VALID_DIRECTIONS = {"bullish", "bearish", "neutral"}
VALID_TIMELINESS = {"high", "medium", "low"}
VALID_HORIZONS = {"短期", "中期", "长期"}

# direction → impact_type/sentiment 一致性映射
_DIRECTION_TO_SENTIMENT = {
    "bullish": "positive",
    "bearish": "negative",
    "neutral": "neutral",
}


# =============================================================================
# Prompt（投资分析师角色，参考 STOCK_RATING_SYSTEM 范式）
# =============================================================================
NEWS_SYSTEM = """你是 A 股专业投资分析师，服务对象是 A 股投资交易者。

你的任务：使用联网搜索工具获取**最新时效**的行业资讯，并**以投资者视角**分析每条资讯对市场的影响预期——不是做新闻总结，而是判断「这件事对谁有利、对谁不利、传导路径是什么、市场是否已 price-in、预期差在哪」。

## 分析原则（投资者视角）
- 每条资讯必须给出明确的**方向**（bullish 利好 / bearish 利空 / neutral 中性）与**确信度** conviction（0.0-1.0）。
- 必须给出**投资逻辑** thesis：一句话说清"为什么这条消息会移动价格"，要有经济/产业逻辑支撑，不要复述事实。
- 必须给出**传导链条** transmission_chain：从事件 → 产业链环节 → 受益/受损标的 → 价格反应的路径。
- 必须给出**预期差** expectation_gap：市场当前可能已 price-in 多少、超预期方向在哪；难以判断则如实说明，不要编造。
- 催化剂 catalysts 与风险/证伪点 risks 要具体、可观测（如某项数据发布日、某产能投产节点、某政策落地时点）。
- 只收录会移动板块、驱动轮动、扭转情绪的消息（政策、产业事件、价格异动、资金行为、地缘/宏观冲击等）。宁缺毋滥，没有真实增量信息的不要硬凑。
- 个股仅在"直接受益/受损的代表性标的"层面出现，不做逐股覆盖。

## 输出要求（严格 JSON）
只返回一个 JSON 对象：
```json
{
  "items": [
    {
      "sector": "板块/主题名（如 半导体设备、储能、创新药）",
      "title": "一句话标题",
      "summary": "2-3 句事实摘要，含关键数字",
      "direction": "bullish | bearish | neutral",
      "conviction": 0.0到1.0的浮点数,
      "thesis": "投资逻辑核心一句话",
      "transmission_chain": "事件→产业链→标的→价格反应的传导路径",
      "time_horizon": "短期 | 中期 | 长期",
      "catalysts": ["可观测催化剂1", "催化剂2"],
      "risks": ["证伪点/风险1", "风险2"],
      "expectation_gap": "市场已price-in程度与超预期方向；难以判断则说明",
      "impact": "影响分析：利好/利空/中性 + 逻辑（一句话）",
      "impact_type": "positive | negative | neutral",
      "rotation": "轮动含义：资金从哪切到哪 / 情绪如何切换（一句话，无则填 null）",
      "sentiment": "positive | negative | neutral",
      "timeliness": "high | medium | low",
      "info_date": "YYYY-MM-DD",
      "related_sectors": ["关联板块名1", "关联板块名2"],
      "related_stocks": ["代表性个股名称1", "代表性个股名称2"],
      "source": "信息来源（财联社/澎湃/公司公告/统计局等）"
    }
  ]
}
```

## 约束
- items 数量 5-12 条，按 timeliness=high 优先、info_date 新者优先、conviction 高者优先排序。
- 今天是 {collect_date}。info_date 不得晚于 {collect_date}。
- direction 与 impact_type/sentiment 必须一致（bullish↔positive, bearish↔negative, neutral↔neutral）。
- conviction < 0.4 视为弱信号；信息不足时降低 conviction 而非编造逻辑。
- 全中文。title≤30 字，summary≤120 字，thesis≤80 字，transmission_chain≤200 字，expectation_gap≤150 字，impact≤80 字。
- 只返回 JSON，不要任何额外文字。"""

NEWS_USER = """请使用联网搜索工具采集近期（最近 1-3 个交易日内）影响 A 股板块、轮动与情绪的资讯，并以投资者视角逐条分析影响预期。

重点方向（按当日实际情况取舍，不要强行凑齐）：
1. 政策与监管（财政/货币/产业政策、监管动向）
2. 产业事件（订单/涨价/产能/技术突破/招投标）
3. 资金与情绪（北向/两融/成交额/涨停板结构/高低切换）
4. 宏观与地缘冲击（进出口/利率/汇率/外围事件）

按 system 的 JSON schema 输出 {n_items} 条左右。每条都必须包含完整的投资分析字段（direction/conviction/thesis/transmission_chain/time_horizon/catalysts/risks/expectation_gap），不要留空。"""


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class NewsItem:
    """一条板块资讯（含投资影响分析 + 后处理映射结果）。"""
    sector: str
    title: str
    summary: str
    impact: str
    impact_type: str  # positive/negative/neutral
    rotation: Optional[str]
    sentiment: str
    timeliness: str
    info_date: str
    # —— 投资影响分析字段（LLM 作为投资分析师产出）——
    direction: str = "neutral"            # bullish/bearish/neutral
    conviction: float = 0.5               # 0.0-1.0
    thesis: str = ""                      # 投资逻辑核心
    transmission_chain: str = ""          # 传导链条
    time_horizon: str = "中期"            # 短期/中期/长期
    catalysts: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    expectation_gap: str = ""             # 市场预期差

    related_sectors: List[str] = field(default_factory=list)
    related_stocks: List[str] = field(default_factory=list)
    source: str = ""

    # 后处理映射（非 LLM 原文，由 _map_to_market 填充）
    concept_code: Optional[str] = None        # dc_concept theme_code
    concept_pct_change: Optional[float] = None  # 板块当日涨跌幅 %
    concept_hot: Optional[float] = None       # 热度
    lead_stock: Optional[str] = None          # 领涨股名
    lead_stock_code: Optional[str] = None     # 领涨股 vt_symbol
    mapped_stocks: List[Dict[str, str]] = field(default_factory=list)  # [{vt_symbol, name}]

    collect_date: str = ""
    error: Optional[str] = None


# =============================================================================
# 采集器
# =============================================================================
class NewsCollector:
    """LLM 板块资讯采集 + 投资影响分析器（基于 OpenClaw）。"""

    def __init__(
        self,
        client: Optional[OpenClawClient] = None,
        n_items: int = 8,
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ):
        self.client = client or OpenClawClient()
        self.n_items = n_items
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ---------- 主流程 ----------
    def collect(self, collect_date: Optional[str] = None) -> List[NewsItem]:
        """采集一轮资讯并做投资影响分析，返回 NewsItem 列表（已映射板块/个股）。"""
        if collect_date is None:
            collect_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"[NewsCollector] start collect_date={collect_date} model={self.client.model}")
        raw_items = self._llm_collect(collect_date)
        if not raw_items:
            logger.warning("[NewsCollector] LLM returned no items")
            return []

        items = [self._build_item(d, collect_date) for d in raw_items]
        items = [it for it in items if it is not None]

        # 后处理：映射板块/个股
        try:
            self._map_to_market(items, collect_date)
        except Exception as e:
            logger.error(f"[NewsCollector] map_to_market failed (non-fatal): {e}")

        logger.info(f"[NewsCollector] done: {len(items)} items")
        return items

    # ---------- LLM 调用 ----------
    def _llm_collect(self, collect_date: str) -> List[Dict[str, Any]]:
        system = NEWS_SYSTEM.replace("{collect_date}", collect_date)
        user = NEWS_USER.format(n_items=self.n_items)

        try:
            raw = self.client.chat(
                system=system,
                user=user,
                response_format_json=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error(f"[NewsCollector] OpenClaw call failed: {e}")
            raise

        items = self._parse_items(raw)
        if not items:
            # 一次 JSON 修复重试（参考 risk_screener._retry_json_fix）
            logger.warning("[NewsCollector] JSON 解析为空，尝试修复重试")
            retry_raw = self._retry_json_fix(system, user, raw)
            if retry_raw:
                items = self._parse_items(retry_raw)
        return items

    def _retry_json_fix(self, system: str, user: str, malformed: str) -> str:
        """继续原会话，请 LLM 修复 JSON。"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": malformed},
            {"role": "user", "content": "你的回复JSON格式有误，无法解析。请重新输出正确的JSON，只返回JSON对象，不要有任何额外文字。"},
        ]
        try:
            return self.client.chat_messages(
                messages=messages,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error(f"[NewsCollector] JSON fix retry failed: {e}")
            return ""

    def _parse_items(self, raw: str) -> List[Dict[str, Any]]:
        if not raw:
            return []
        extracted = _extract_json(raw)
        try:
            data = json.loads(extracted)
        except json.JSONDecodeError as e:
            logger.error(f"[NewsCollector] JSON parse failed: {e}\nraw[:500]: {raw[:500]}")
            return []
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        if isinstance(data, list):
            return data
        return []

    def _build_item(self, d: Dict[str, Any], collect_date: str) -> Optional[NewsItem]:
        try:
            direction = str(d.get("direction", "")).lower().strip()
            if direction not in VALID_DIRECTIONS:
                # 由 impact_type/sentiment 反推
                direction = "neutral"

            impact_type = str(d.get("impact_type", "")).lower().strip()
            sentiment = str(d.get("sentiment", "")).lower().strip()

            # direction 与 impact_type/sentiment 一致性校正
            expected = _DIRECTION_TO_SENTIMENT[direction]
            if impact_type not in VALID_IMPACT_TYPES:
                impact_type = expected
            if sentiment not in VALID_SENTIMENTS:
                sentiment = expected

            timeliness = str(d.get("timeliness", "medium")).lower()
            if timeliness not in VALID_TIMELINESS:
                timeliness = "medium"

            time_horizon = str(d.get("time_horizon", "中期")).strip()
            if time_horizon not in VALID_HORIZONS:
                time_horizon = "中期"

            # conviction 数值与下限
            try:
                conviction = float(d.get("conviction", 0.5))
            except (TypeError, ValueError):
                conviction = 0.5
            conviction = max(0.0, min(1.0, conviction))

            info_date = str(d.get("info_date", "") or "")
            if info_date and info_date > collect_date:
                info_date = collect_date

            return NewsItem(
                sector=str(d.get("sector", "")).strip(),
                title=str(d.get("title", "")).strip()[:200],
                summary=str(d.get("summary", "")).strip()[:600],
                impact=str(d.get("impact", "")).strip()[:400],
                impact_type=impact_type,
                rotation=(str(d.get("rotation")).strip()[:400] if d.get("rotation") else None),
                sentiment=sentiment,
                timeliness=timeliness,
                info_date=info_date or collect_date,
                direction=direction,
                conviction=conviction,
                thesis=str(d.get("thesis", "")).strip()[:400],
                transmission_chain=str(d.get("transmission_chain", "")).strip()[:600],
                time_horizon=time_horizon,
                catalysts=[str(x).strip() for x in (d.get("catalysts") or []) if str(x).strip()][:10],
                risks=[str(x).strip() for x in (d.get("risks") or []) if str(x).strip()][:10],
                expectation_gap=str(d.get("expectation_gap", "")).strip()[:500],
                related_sectors=[str(x).strip() for x in (d.get("related_sectors") or []) if str(x).strip()][:10],
                related_stocks=[str(x).strip() for x in (d.get("related_stocks") or []) if str(x).strip()][:10],
                source=str(d.get("source", "")).strip()[:200],
                collect_date=collect_date,
            )
        except Exception as e:
            logger.error(f"[NewsCollector] build item failed: {e}")
            return None

    # ---------- 板块/个股映射 ----------
    def _map_to_market(self, items: List[NewsItem], collect_date: str) -> None:
        """用 dc_concept（板块行情）+ stock_basic（个股）映射，结果挂在 item 上。"""
        concepts = self._load_concepts()        # [{theme_code, name, pct_change, hot, lead_stock, lead_stock_code}]
        stock_basic = self._load_stock_basic()  # {name: vt_symbol, industry: {industry: [(vt_symbol,name)]}}

        for item in items:
            # 1) 板块名 → dc_concept（取 sector 与 related_sectors 中首个命中）
            cand_sectors = [item.sector] + list(item.related_sectors)
            hit = self._match_concept(cand_sectors, concepts)
            if hit:
                item.concept_code = hit.get("theme_code")
                item.concept_pct_change = hit.get("pct_change")
                item.concept_hot = hit.get("hot")
                item.lead_stock = hit.get("lead_stock")
                item.lead_stock_code = _to_vt_symbol(hit.get("lead_stock_code"))

            # 2) 代表性个股：LLM 命名 + 行业模糊匹配
            mapped: Dict[str, Dict[str, str]] = {}  # vt_symbol -> {vt_symbol, name}

            # 2a) LLM 给的个股名 → stock_basic.name
            for nm in item.related_stocks:
                vs = self._match_stock_name(nm, stock_basic["by_name"])
                if vs:
                    mapped.setdefault(vs[0], {"vt_symbol": vs[0], "name": vs[1]})

            # 2b) 领涨股
            if item.lead_stock_code:
                mapped.setdefault(item.lead_stock_code, {
                    "vt_symbol": item.lead_stock_code,
                    "name": item.lead_stock or "",
                })

            # 2c) 板块名 → industry 模糊匹配，补足代表性个股（最多 8 个）
            if len(mapped) < 8:
                industry_stocks = self._match_industry(cand_sectors, stock_basic["by_industry"])
                for vs, nm in industry_stocks:
                    if len(mapped) >= 12:
                        break
                    mapped.setdefault(vs, {"vt_symbol": vs, "name": nm})

            item.mapped_stocks = list(mapped.values())[:15]

    # 概念匹配
    def _match_concept(self, cand_sectors: List[str], concepts: List[Dict]) -> Optional[Dict]:
        for sec in cand_sectors:
            if not sec:
                continue
            # 精确
            for c in concepts:
                if c["name"] == sec:
                    return c
            # 包含
            for c in concepts:
                if sec in c["name"] or c["name"] in sec:
                    return c
            # 关键词拆分
            for c in concepts:
                if any(k and k in c["name"] for k in re.split(r"[、/／\s]+", sec) if len(k) >= 2):
                    return c
        return None

    def _match_stock_name(self, nm: str, by_name: Dict[str, Tuple[str, str]]) -> Optional[Tuple[str, str]]:
        if not nm:
            return None
        if nm in by_name:
            return by_name[nm]
        # 模糊：包含
        for name, (vs, cn) in by_name.items():
            if nm in name or name in nm:
                return (vs, cn)
        return None

    def _match_industry(self, cand_sectors: List[str], by_industry: Dict[str, List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for sec in cand_sectors:
            if not sec:
                continue
            for ind, stocks in by_industry.items():
                if ind == sec or ind in sec or sec in ind:
                    out.extend(stocks)
                    if len(out) >= 30:
                        return out
        return out

    # ---------- 数据加载 ----------
    def _load_concepts(self) -> List[Dict]:
        """从 tushare dc_concept 取最新交易日的概念行情（含名称/涨跌/领涨股）。"""
        try:
            from vnpy.trader.setting import SETTINGS
            import tushare as ts
            pro = ts.pro_api(SETTINGS["datafeed.password"])
            df = pro.dc_concept()
            if df is None or df.empty:
                return []
            # 取最新交易日
            latest = sorted(df["trade_date"].unique())[-1]
            df = df[df["trade_date"] == latest]
            cols = ["theme_code", "name", "pct_change", "hot", "lead_stock", "lead_stock_code"]
            out = []
            for _, r in df.iterrows():
                out.append({
                    "theme_code": r.get("theme_code"),
                    "name": str(r.get("name", "")).strip(),
                    "pct_change": _to_float(r.get("pct_change")),
                    "hot": _to_float(r.get("hot")),
                    "lead_stock": (str(r.get("lead_stock")).strip() if r.get("lead_stock") else None),
                    "lead_stock_code": (str(r.get("lead_stock_code")).strip() if r.get("lead_stock_code") else None),
                })
            return out
        except Exception as e:
            logger.error(f"[NewsCollector] load dc_concept failed: {e}")
            return []

    def _load_stock_basic(self) -> Dict:
        """从 MySQL stock_basic 加载 {name: (vt_symbol, name)} 与 {industry: [(vt_symbol,name)]}。"""
        by_name: Dict[str, Tuple[str, str]] = {}
        by_industry: Dict[str, List[Tuple[str, str]]] = {}
        try:
            from vnpy.trader.setting import SETTINGS
            import pymysql
            cfg = {
                k.replace("database.", ""): SETTINGS[k]
                for k in SETTINGS
                if k.startswith("database.")
                and k not in ("database.timezone", "database.name")
            }
            conn = pymysql.connect(**cfg, charset="utf8mb4")
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT ts_code, name, industry FROM stock_basic "
                        "WHERE list_status='L'"
                    )
                    for ts_code, name, industry in cur.fetchall():
                        if not name:
                            continue
                        vt = _to_vt_symbol(ts_code)
                        by_name[name] = (vt, name)
                        if industry:
                            by_industry.setdefault(industry, []).append((vt, name))
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"[NewsCollector] load stock_basic failed: {e}")
        return {"by_name": by_name, "by_industry": by_industry}


# =============================================================================
# 工具
# =============================================================================
def _to_vt_symbol(ts_code: str) -> Optional[str]:
    """tushare ts_code (000001.SZ / 600519.SH / 872925.BJ) → vnpy vt_symbol (000001.SZSE / 600519.SSE / 872925.BSE)。

    vnpy 的 bar/signal/llm_tasks 数据均按全称交易所后缀落盘（SZSE/SSE/BSE），
    而 tushare dc_concept / stock_basic 给的是短后缀（SZ/SH/BJ）。需要统一到 vnpy 格式，
    否则下游 load_bar_data / llm_ratings 查不到文件。
    """
    if not ts_code or not isinstance(ts_code, str):
        return ts_code
    if ts_code.endswith(".SZ"):
        return ts_code[:-3] + ".SZSE"
    if ts_code.endswith(".SH"):
        return ts_code[:-3] + ".SSE"
    if ts_code.endswith(".BJ"):
        return ts_code[:-3] + ".BSE"
    return ts_code


def _to_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None


# =============================================================================
# 落盘
# =============================================================================
def save_news(items: List[NewsItem], collect_date: str, output_dir: Optional[str] = None) -> str:
    """落盘到 core/alpha_db/news/{date}.json（每日一份，列表）。

    同一日多次采集：按 (sector, title) 去重后追加，保留历史条目。
    """
    out_dir = Path(output_dir) if output_dir else NEWS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"{collect_date}.json"

    existing: List[Dict[str, Any]] = []
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "items" in data:
                    existing = data["items"]
                elif isinstance(data, list):
                    existing = data
        except (json.JSONDecodeError, Exception):
            existing = []

    # 去重键
    seen = {(it.get("sector"), it.get("title")) for it in existing}
    new_rows = []
    for it in items:
        key = (it.sector, it.title)
        if key in seen:
            continue
        seen.add(key)
        new_rows.append(_item_to_dict(it))

    merged = existing + new_rows
    # 排序：timeliness high 优先 → info_date 新→旧（倒序）→ conviction 高→低
    # 三次稳定排序（字符串日期无法在 tuple 内取负），后一次保留前一次的相对顺序。
    order = {"high": 0, "medium": 1, "low": 2}
    merged.sort(key=lambda x: _sort_conviction(x), reverse=True)
    merged.sort(key=lambda x: _sort_date(x), reverse=True)
    merged.sort(key=lambda x: order.get(x.get("timeliness"), 1))

    payload = {
        "collect_date": collect_date,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(merged),
        "items": merged,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"[save_news] {filepath}: +{len(new_rows)} new, {len(merged)} total")
    return str(filepath)


def _sort_date(x: Dict[str, Any]) -> str:
    return x.get("info_date", "") or ""


def _sort_conviction(x: Dict[str, Any]) -> float:
    try:
        return float(x.get("conviction", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _item_to_dict(it: NewsItem) -> Dict[str, Any]:
    d = asdict(it)
    return d


# =============================================================================
# 一键采集（供定时任务/手动触发调用）
# =============================================================================
_collect_lock = threading.Lock()


def run_collection(
    collect_date: Optional[str] = None,
    n_items: int = 8,
) -> Dict[str, Any]:
    """同步采集 + 投资分析 + 映射 + 落盘。返回简报 dict。供后台线程调用。"""
    if not _collect_lock.acquire(blocking=False):
        return {"status": "skipped", "message": "已有采集任务在执行"}
    try:
        collect_date = collect_date or datetime.now().strftime("%Y-%m-%d")
        collector = NewsCollector(n_items=n_items)
        items = collector.collect(collect_date)
        if not items:
            return {"status": "empty", "collect_date": collect_date, "message": "LLM 未返回有效资讯"}
        filepath = save_news(items, collect_date)
        return {
            "status": "ok",
            "collect_date": collect_date,
            "count": len(items),
            "filepath": filepath,
            "items": [_item_to_dict(it) for it in items],
        }
    finally:
        _collect_lock.release()
