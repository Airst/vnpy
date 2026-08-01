"""
LLM 投资建议模块（每日汇总 + 持仓池动向监控）

定位：资讯采集（news_collector）的下游消费者。每天采集的资讯条目多（20+），
交易者没时间逐条读——本模块用 LLM 把当日全部资讯**蒸馏**成一份可直接行动的建议：
1. 最具潜力股票 Top 3-5：跨资讯交叉验证（多条资讯共振的标的优先），给出评估依据；
2. 最具风险 / 退潮板块：利空冲击、情绪退潮、拥挤兑现三类，给出证据；
3. 持仓股票池动向：用户跟单的股票，逐只扫描当日资讯，提示风险与利好。

与 news_collector 的分工：
- news_collector 负责"广度"——自主挖掘 + 逐条投资分析（stock_implications）；
- advisor 负责"深度汇总"——跨条目共振分析、排序、去重，输出当日结论。

LLM 输入是已采集的资讯 JSON（不联网重新搜索，保证与资讯页看到的内容一致、
可溯源），evidence 字段引用资讯标题。持仓扫描做双轨：LLM 判断 + 代码侧
stock_implications/mapped_stocks 兜底匹配（LLM 漏报时仍有信号）。

落盘：
- core/alpha_db/advice/{date}.json     每日建议（digest + watchlist_alerts + holding_analysis）
- core/alpha_db/advice/watchlist.json  持仓股票池（用户跟单提交）

持仓深度跟踪（holding_analysis，v2 新增）：
watchlist_alerts 只覆盖"当日资讯命中"的持仓，无新资讯时显示"无动向"——
但行情与情绪变化本身就是信号（如持仓跟随板块上涨却无个股资讯）。
holding_analysis 对每只持仓独立分析：tushare 行情快照（当日/5日/20日涨幅、
量能比、行业）作为硬数据防幻觉，LLM 联网搜索个股+板块近期消息面/资金面，
输出市场情绪/动量/短期走势研判/驱动/风险。LLM 失败时降级为纯行情快照条目
（llm_ok=False），保证持仓跟踪永远有内容。

自动化：main_controller 在资讯采集成功后自动触发生成（也支持手动）；
持仓分析可通过 /api/watchlist/analyze 单独刷新（不重跑全量蒸馏）。
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.llm.openclaw_client import OpenClawClient, parse_json_response
from core.llm.news_collector import load_stock_basic
from vnpy.alpha.logger import logger

ADVICE_DIR = Path(__file__).resolve().parent.parent.parent / "core" / "alpha_db" / "advice"
WATCHLIST_FILE = ADVICE_DIR / "watchlist.json"
NEWS_DIR = Path(__file__).resolve().parent.parent.parent / "core" / "alpha_db" / "news"

VALID_TONES = {"risk_on", "risk_off", "neutral"}
VALID_RISK_TYPES = {"利空冲击", "情绪退潮", "拥挤兑现"}
VALID_ALERTS = {"risk", "positive"}
# 持仓动向关联强度：direct 直接关联 / chain 产业链 / concept 仅概念口径（弱关联，前端默认折叠）
VALID_RELEVANCE = {"direct", "chain", "concept"}

_watchlist_lock = threading.Lock()
_advice_lock = threading.Lock()


# =============================================================================
# Prompt
# =============================================================================
ADVICE_SYSTEM = """你是 A 股首席投资顾问，负责把当日采集的全部资讯蒸馏成一份**可直接行动的每日投资建议**，服务对象是 A 股交易者。

输入是当日已采集的资讯列表（每条含板块、标题、摘要、方向、确信度、投资逻辑、受影响标的推导）。你的价值不是复述，而是**跨条目综合**：
- 同一标的/板块被多条资讯从不同角度印证（政策+订单+资金），置信度应显著提升；
- 单条资讯孤证、且缺乏可观测催化剂的，降权处理；
- 资讯之间有矛盾的（一条利好一条利空），如实指出冲突而非选边站。

## 输出三部分

### 1. top_stocks（当日最具潜力股票，3-5 只，宁缺毋滥）
- 从资讯的 stock_implications 与你的产业认知中选出**证据最强**的标的。
- 评估依据 rationale 必须落到"哪几条资讯、什么逻辑、为什么是这只而不是同板块其他"。
- evidence 引用资讯标题（原文），供用户溯源。
- 给出 entry_risk：买入这只票当下最大的风险一句话（追高、兑现、证伪点）。

### 2. risk_sectors（最具风险 / 退潮板块，2-4 个）
- risk_type 三选一：利空冲击（有实质利空资讯）/ 情绪退潮（前期热点资金撤退迹象）/ 拥挤兑现（涨幅大+利好落地兑现风险）。
- 依据 rationale + evidence（资讯标题）。持仓相关板块的风险要优先提示。

### 3. watchlist_alerts（持仓股票池动向，仅当提供了持仓池时输出）
- 逐只核对当日资讯：直接提及、所属板块被提及、产业链上下游被提及，都算动向。
- alert 二选一：risk（利空/退潮/兑现风险）/ positive（利好/催化）。
- relevance 关联强度必须如实评级，三选一：
  - direct：资讯直接点名该公司，或其核心主营业务实质受益/受损（如股权关系、订单、产品涨价）；
  - chain：处于资讯影响的产业链上下游，主营业务有实质关联；
  - concept：仅因所属概念板块口径被波及，主营业务无实质关联（如压缩机公司被划入光刻机概念）。
  不确定时宁可降级到 concept，**不要为持仓硬找利好**。
- 当日资讯与该股无任何关联的，不要硬凑，直接不输出该股。
- action_hint 给一句话操作提示（如"关注回调风险，留意 XX 证伪点"），不是指令。

## 输出格式（严格 JSON，只返回 JSON）
```json
{
  "market_summary": {
    "tone": "risk_on | neutral | risk_off",
    "comment": "当日市场基调一句话（主线、情绪、资金特征）"
  },
  "top_stocks": [
    {
      "name": "股票名",
      "direction": "bullish",
      "conviction": 0.0到1.0,
      "sector": "所属板块/主题",
      "rationale": "评估依据：哪些资讯共振+核心逻辑+为何是这只（≤150字）",
      "evidence": ["引用的资讯标题1", "资讯标题2"],
      "entry_risk": "当下买入最大风险一句话",
      "time_horizon": "短期 | 中期 | 长期"
    }
  ],
  "risk_sectors": [
    {
      "sector": "板块名",
      "risk_type": "利空冲击 | 情绪退潮 | 拥挤兑现",
      "rationale": "评估依据（≤120字）",
      "evidence": ["资讯标题1"]
    }
  ],
  "watchlist_alerts": [
    {
      "name": "持仓股票名",
      "alert": "risk | positive",
      "relevance": "direct | chain | concept",
      "rationale": "该股与当日哪条资讯有何关联、影响逻辑（≤100字）",
      "evidence": ["资讯标题1"],
      "action_hint": "一句话操作提示"
    }
  ]
}
```

## 约束
- 全中文；conviction 反映证据强度，孤证不超过 0.6，多条资讯共振才允许 0.7+。
- top_stocks 中的股票名必须是 A 股真实股票名（来自资讯 stock_implications 或明确的产业链标的）。
- evidence 必须严格引用输入资讯的 title 原文，不得虚构。
- 结论仅供研究参考，不构成投资建议，但输出中不需要重复此声明。"""

ADVICE_USER = """今天是 {date}。以下是当日采集的 {news_count} 条资讯（JSON），请蒸馏为每日投资建议。

【当日资讯】
{news_json}

【持仓股票池】（用户跟单持仓，逐只核对当日资讯动向；为空则 watchlist_alerts 输出空数组）
{watchlist}

按 system 的 JSON schema 输出。top_stocks 3-5 只、risk_sectors 2-4 个，评估依据必须引用资讯标题。只返回 JSON。"""


# 持仓个股深度跟踪 prompt：不依赖当日资讯，联网搜索个股/板块近期动向，
# 结合调用方提供的行情快照（硬数据，防止 LLM 编造涨跌幅）做情绪与走势研判。
HOLDING_SYSTEM = """你是 A 股个股跟踪分析师，负责对用户持仓股票做每日体检。

你会收到一只持仓股的行情快照（当日/近5日/近20日涨跌幅、量能变化、所属行业）。
请使用联网搜索工具查该股及所属板块最近 1 周的消息面与资金面（关键词如"股票名 最新"
"行业名 板块 行情"），结合行情快照给出：
- 市场情绪 sentiment：bullish/bearish/neutral（资金与消息面对该股的当前态度）
- 动量 momentum：强势/震荡/走弱（行情快照为主，不要与给定数据矛盾）
- 短期走势研判 view：未来 1-2 周的情景判断，要落到驱动与条件（如"若铝价维持高位则…，若板块回调则关注…支撑"），不是模棱两可的套话
- 驱动 drivers：近期支撑该股的具体因素（板块主线/产品涨价/业绩/资金）
- 风险 risks：具体可观测的证伪点

要求：情绪判断必须有依据（搜到的消息/行情特征），搜不到个股消息就如实基于板块与行情判断，
不要编造。conviction 0-1 反映依据强度。

只返回一个 JSON：
{"sentiment": "bullish|neutral|bearish", "momentum": "强势|震荡|走弱",
 "view": "短期走势研判（≤120字）", "drivers": ["驱动1"], "risks": ["风险1"],
 "conviction": 0.0到1.0}
全中文，不要任何额外文字。"""

HOLDING_USER = """持仓股：{name}（{ts_code}，行业：{industry}）{note}

【行情快照（真实数据，以此为准）】
- 最新交易日：{trade_date}，收盘 {close}，当日 {d1:+.2f}%
- 近 5 日累计 {d5:+.2f}%，近 20 日累计 {d20:+.2f}%
- 量能：近 5 日日均成交额是前 5 日的 {vol_ratio:.2f} 倍

请联网搜索该股与「{industry}」板块近期消息，按 system 格式输出情绪与走势研判 JSON。"""


# =============================================================================
# 持仓股票池（watchlist）
# =============================================================================
def load_watchlist() -> List[Dict[str, Any]]:
    """读取持仓股票池 [{vt_symbol, name, note, added_at}]。"""
    if not WATCHLIST_FILE.exists():
        return []
    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items") if isinstance(data, dict) else data
        return items if isinstance(items, list) else []
    except Exception as e:
        logger.error(f"[advisor] load watchlist failed: {e}")
        return []


def _save_watchlist(items: List[Dict[str, Any]]) -> None:
    ADVICE_DIR.mkdir(parents=True, exist_ok=True)
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump({"items": items, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                  f, ensure_ascii=False, indent=2)


def resolve_stock(query: str) -> Optional[Tuple[str, str]]:
    """把用户输入（代码 600895 / vt_symbol / 名称"张江高科"）解析为 (vt_symbol, name)。

    匹配顺序：名称精确 → 代码前缀 → 名称包含。查 MySQL stock_basic，失败返回 None。
    """
    query = (query or "").strip()
    if not query:
        return None
    by_name = load_stock_basic()["by_name"]
    # 名称精确
    if query in by_name:
        return by_name[query]
    # 代码：取数字前缀匹配 vt_symbol
    digits = "".join(c for c in query if c.isdigit())
    if len(digits) == 6:
        for vt, nm in by_name.values():
            if vt.startswith(digits):
                return (vt, nm)
    # 名称模糊（包含）
    for nm, (vt, cn) in by_name.items():
        if query in nm:
            return (vt, cn)
    return None


def add_watch_stock(query: str, note: str = "") -> Dict[str, Any]:
    """添加跟单股票。query 支持代码/名称，自动解析；重复添加幂等。"""
    resolved = resolve_stock(query)
    if not resolved:
        return {"status": "error", "message": f"无法识别股票: {query}（请输入 6 位代码或准确名称）"}
    vt_symbol, name = resolved
    with _watchlist_lock:
        items = load_watchlist()
        for it in items:
            if it.get("vt_symbol") == vt_symbol:
                return {"status": "exists", "message": f"{name} 已在股票池中", "item": it}
        item = {
            "vt_symbol": vt_symbol,
            "name": name,
            "note": (note or "").strip()[:100],
            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        items.append(item)
        _save_watchlist(items)
    return {"status": "ok", "item": item}


def remove_watch_stock(vt_symbol: str) -> Dict[str, Any]:
    """移除跟单股票。"""
    with _watchlist_lock:
        items = load_watchlist()
        kept = [it for it in items if it.get("vt_symbol") != vt_symbol]
        if len(kept) == len(items):
            return {"status": "not_found", "message": f"{vt_symbol} 不在股票池中"}
        _save_watchlist(kept)
    return {"status": "ok"}


# =============================================================================
# 每日建议生成
# =============================================================================
class InvestmentAdvisor:
    """每日投资建议生成器：资讯蒸馏 + 持仓池动向扫描（基于 OpenClaw）。"""

    def __init__(
        self,
        client: Optional[OpenClawClient] = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ):
        self.client = client or OpenClawClient()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, date: Optional[str] = None) -> Dict[str, Any]:
        """生成当日建议。date 为空取最新资讯日。返回完整 advice dict（未落盘）。"""
        news_date, news_items = _load_news(date)
        if not news_items:
            return {"status": "empty", "message": f"无可用资讯（date={date or '最新'}），请先采集资讯"}

        watchlist = load_watchlist()
        raw = self._llm_digest(news_date, news_items, watchlist)
        advice = self._normalize(raw, news_items, watchlist)

        # 代码侧兜底：持仓池 × 资讯 stock_implications/mapped_stocks 直接匹配，
        # 合并 LLM 漏报的动向（derived=True 标记来源）
        derived = _derive_watchlist_alerts(news_items, watchlist)
        seen_names = {a["name"] for a in advice["watchlist_alerts"]}
        for a in derived:
            if a["name"] not in seen_names:
                advice["watchlist_alerts"].append(a)

        # top_stocks 映射 vt_symbol（供前端点击查看详情）
        by_name = load_stock_basic()["by_name"]
        for s in advice["top_stocks"]:
            hit = by_name.get(s["name"])
            if not hit:
                for nm, (vt, cn) in by_name.items():
                    if s["name"] in nm or nm in s["name"]:
                        hit = (vt, cn)
                        break
            s["vt_symbol"] = hit[0] if hit else None
        for a in advice["watchlist_alerts"]:
            if not a.get("vt_symbol"):
                hit = by_name.get(a["name"])
                a["vt_symbol"] = hit[0] if hit else None

        advice.update({
            "status": "ok",
            "date": news_date,
            "news_count": len(news_items),
            "watchlist_count": len(watchlist),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        # 持仓深度跟踪：行情快照 + LLM 情绪/走势研判（失败非致命，降级为纯行情条目）
        try:
            advice["holding_analysis"] = self.analyze_holdings(watchlist)
        except Exception as e:
            logger.error(f"[advisor] holding analysis failed (non-fatal): {e}")
            advice["holding_analysis"] = []
        return advice

    # ---------- 持仓个股深度跟踪 ----------
    def analyze_holdings(self, watchlist: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """对每只持仓做行情快照 + LLM 情绪/走势分析。

        不依赖当日资讯，解决"持仓跟随板块上涨却因无个股资讯显示无动向"的盲区。
        LLM 逐只分析（联网搜索，串行避免 OpenClaw 并发过载），单只失败降级为
        纯行情快照条目（llm_ok=False，momentum 由 5 日涨幅规则推导）。
        """
        if watchlist is None:
            watchlist = load_watchlist()
        if not watchlist:
            return []

        quotes = _load_quote_snapshots([w["vt_symbol"] for w in watchlist])
        out: List[Dict[str, Any]] = []
        for w in watchlist:
            q = quotes.get(w["vt_symbol"])
            entry: Dict[str, Any] = {
                "vt_symbol": w["vt_symbol"],
                "name": w["name"],
                "note": w.get("note", ""),
                "quote": q,
                "llm_ok": False,
                "sentiment": "neutral",
                "momentum": _rule_momentum(q),
                "view": "",
                "drivers": [],
                "risks": [],
                "conviction": None,
            }
            if q is not None:
                try:
                    llm = self._llm_holding(w, q)
                    if llm:
                        entry.update(llm)
                        entry["llm_ok"] = True
                except Exception as e:
                    logger.error(f"[advisor] holding LLM failed for {w['name']} (non-fatal): {e}")
            if not entry["view"]:
                entry["view"] = "LLM 分析暂不可用，仅展示行情快照" if q else "行情数据暂不可用"
            out.append(entry)
            logger.info(f"[advisor] holding {w['name']}: llm_ok={entry['llm_ok']} "
                        f"sentiment={entry['sentiment']} momentum={entry['momentum']}")
        return out

    def _llm_holding(self, w: Dict[str, Any], q: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """单只持仓的 LLM 情绪/走势分析。agent 错误整次重发，最多 2 轮。"""
        note = f"，备注：{w['note']}" if w.get("note") else ""
        user = HOLDING_USER.format(
            name=w["name"], ts_code=q["ts_code"], industry=q.get("industry") or "未知",
            note=note, trade_date=q["trade_date"], close=q["close"],
            d1=q["d1"], d5=q["d5"], d20=q["d20"], vol_ratio=q["vol_ratio"],
        )
        for attempt in range(2):
            try:
                raw = self.client.chat(
                    system=HOLDING_SYSTEM,
                    user=user,
                    response_format_json=True,
                    temperature=self.temperature,
                    max_tokens=3000,
                )
            except Exception as e:
                logger.warning(f"[advisor] holding chat failed ({attempt + 1}/2): {e}")
                continue
            try:
                data = parse_json_response(raw)
            except ValueError:
                logger.warning(f"[advisor] holding JSON invalid ({attempt + 1}/2)")
                continue
            if not isinstance(data, dict):
                continue
            sentiment = str(data.get("sentiment", "")).lower().strip()
            momentum = str(data.get("momentum", "")).strip()
            try:
                conviction = max(0.0, min(1.0, float(data.get("conviction", 0.5))))
            except (TypeError, ValueError):
                conviction = 0.5
            return {
                "sentiment": sentiment if sentiment in ("bullish", "bearish", "neutral") else "neutral",
                "momentum": momentum if momentum in ("强势", "震荡", "走弱") else _rule_momentum(q),
                "view": str(data.get("view", "")).strip()[:300],
                "drivers": [str(x).strip() for x in (data.get("drivers") or []) if str(x).strip()][:5],
                "risks": [str(x).strip() for x in (data.get("risks") or []) if str(x).strip()][:5],
                "conviction": conviction,
            }
        return None

    # ---------- LLM ----------
    def _llm_digest(
        self,
        date: str,
        news_items: List[Dict[str, Any]],
        watchlist: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        compact = [_compact_news(it) for it in news_items]
        wl_text = (
            json.dumps([{"name": w["name"], "note": w.get("note", "")} for w in watchlist], ensure_ascii=False)
            if watchlist else "（空）"
        )
        user = ADVICE_USER.format(
            date=date,
            news_count=len(compact),
            news_json=json.dumps(compact, ensure_ascii=False),
            watchlist=wl_text,
        )
        # OpenClaw agent 偶发 "Agent couldn't generate a response"（非 JSON 错误文本），
        # 整次重发最多 3 轮；真正的 JSON 畸形再走会话内修复重试。
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                raw = self.client.chat(
                    system=ADVICE_SYSTEM,
                    user=user,
                    response_format_json=True,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                last_err = e
                logger.warning(f"[advisor] chat failed (attempt {attempt + 1}/3): {e}")
                continue
            try:
                data = parse_json_response(raw)
                return data if isinstance(data, dict) else {}
            except ValueError as e:
                last_err = e
                if "couldn't generate a response" in (raw or ""):
                    # agent 层错误，会话内修复无意义，直接整次重发
                    logger.warning(f"[advisor] agent error, full retry (attempt {attempt + 1}/3)")
                    continue
                # JSON 畸形：会话内修复一次
                logger.warning("[advisor] JSON 解析失败，会话内修复重试")
                try:
                    retry = self.client.chat_messages(
                        messages=[
                            {"role": "system", "content": ADVICE_SYSTEM},
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": raw},
                            {"role": "user", "content": "你的回复JSON格式有误。请重新输出正确的JSON，只返回JSON对象。"},
                        ],
                        temperature=0.0,
                        max_tokens=self.max_tokens,
                    )
                    data = parse_json_response(retry)
                    return data if isinstance(data, dict) else {}
                except Exception as e2:
                    last_err = e2
                    continue
        raise RuntimeError(f"LLM digest failed after 3 attempts: {last_err}")

    # ---------- 校验归一 ----------
    def _normalize(
        self,
        raw: Dict[str, Any],
        news_items: List[Dict[str, Any]],
        watchlist: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        titles = {it.get("title", "") for it in news_items}
        wl_names = {w["name"] for w in watchlist}

        ms = raw.get("market_summary") or {}
        tone = str(ms.get("tone", "neutral")).lower().strip()
        market_summary = {
            "tone": tone if tone in VALID_TONES else "neutral",
            "comment": str(ms.get("comment", "")).strip()[:300],
        }

        top_stocks = []
        for s in (raw.get("top_stocks") or [])[:5]:
            if not isinstance(s, dict) or not str(s.get("name", "")).strip():
                continue
            try:
                conviction = max(0.0, min(1.0, float(s.get("conviction", 0.5))))
            except (TypeError, ValueError):
                conviction = 0.5
            top_stocks.append({
                "name": str(s["name"]).strip(),
                "direction": "bullish",
                "conviction": conviction,
                "sector": str(s.get("sector", "")).strip()[:50],
                "rationale": str(s.get("rationale", "")).strip()[:500],
                "evidence": _valid_evidence(s.get("evidence"), titles),
                "entry_risk": str(s.get("entry_risk", "")).strip()[:200],
                "time_horizon": str(s.get("time_horizon", "短期")).strip()[:10],
            })

        risk_sectors = []
        for r in (raw.get("risk_sectors") or [])[:4]:
            if not isinstance(r, dict) or not str(r.get("sector", "")).strip():
                continue
            rt = str(r.get("risk_type", "")).strip()
            risk_sectors.append({
                "sector": str(r["sector"]).strip()[:50],
                "risk_type": rt if rt in VALID_RISK_TYPES else "利空冲击",
                "rationale": str(r.get("rationale", "")).strip()[:400],
                "evidence": _valid_evidence(r.get("evidence"), titles),
            })

        watchlist_alerts = []
        for a in (raw.get("watchlist_alerts") or [])[:20]:
            if not isinstance(a, dict):
                continue
            name = str(a.get("name", "")).strip()
            alert = str(a.get("alert", "")).lower().strip()
            # 只保留真实持仓股的动向（防 LLM 幻觉扩散）
            if not name or name not in wl_names or alert not in VALID_ALERTS:
                continue
            relevance = str(a.get("relevance", "")).lower().strip()
            if relevance not in VALID_RELEVANCE:
                relevance = "chain"
            watchlist_alerts.append({
                "name": name,
                "vt_symbol": next((w["vt_symbol"] for w in watchlist if w["name"] == name), None),
                "alert": alert,
                "relevance": relevance,
                "rationale": str(a.get("rationale", "")).strip()[:300],
                "evidence": _valid_evidence(a.get("evidence"), titles),
                "action_hint": str(a.get("action_hint", "")).strip()[:150],
                "derived": False,
            })

        return {
            "market_summary": market_summary,
            "top_stocks": top_stocks,
            "risk_sectors": risk_sectors,
            "watchlist_alerts": watchlist_alerts,
        }


# =============================================================================
# 行情快照（持仓深度跟踪的硬数据层）
# =============================================================================
def _vt_to_ts_code(vt_symbol: str) -> Optional[str]:
    """vnpy vt_symbol (600895.SSE) → tushare ts_code (600895.SH)。"""
    if not vt_symbol or "." not in vt_symbol:
        return None
    code, ex = vt_symbol.rsplit(".", 1)
    suffix = {"SSE": "SH", "SZSE": "SZ", "BSE": "BJ"}.get(ex)
    return f"{code}.{suffix}" if suffix else None


def _rule_momentum(q: Optional[Dict[str, Any]]) -> str:
    """LLM 不可用时的动量规则降级：近 5 日涨幅 >3% 强势 / <-3% 走弱 / 否则震荡。"""
    if not q or q.get("d5") is None:
        return "震荡"
    if q["d5"] > 3:
        return "强势"
    if q["d5"] < -3:
        return "走弱"
    return "震荡"


def _load_quote_snapshots(vt_symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """tushare 拉取持仓股近 20 交易日行情，计算快照：
    {vt_symbol: {ts_code, industry, trade_date, close, d1, d5, d20, vol_ratio}}。
    单只失败跳过（返回中缺失该 key），不影响其他持仓。"""
    out: Dict[str, Dict[str, Any]] = {}
    try:
        from vnpy.trader.setting import SETTINGS
        import tushare as ts
        from datetime import timedelta
        pro = ts.pro_api(SETTINGS["datafeed.password"])

        # 行业映射（一次拉全量）
        industry_map: Dict[str, str] = {}
        try:
            sb = pro.stock_basic(list_status="L", fields="ts_code,industry")
            industry_map = {r["ts_code"]: r["industry"] for _, r in sb.iterrows()}
        except Exception as e:
            logger.warning(f"[advisor] load industry map failed: {e}")

        start = (datetime.now() - timedelta(days=45)).strftime("%Y%m%d")
        for vt in vt_symbols:
            ts_code = _vt_to_ts_code(vt)
            if not ts_code:
                continue
            try:
                df = pro.daily(ts_code=ts_code, start_date=start)
                if df is None or len(df) < 6:
                    continue
                df = df.sort_values("trade_date")  # 旧→新
                closes = df["close"].tolist()
                amounts = df["amount"].tolist()
                d1 = float(df["pct_chg"].iloc[-1])
                d5 = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else None
                d20 = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else (closes[-1] / closes[0] - 1) * 100
                recent5 = sum(amounts[-5:]) / 5
                prev5 = sum(amounts[-10:-5]) / 5 if len(amounts) >= 10 else recent5
                out[vt] = {
                    "ts_code": ts_code,
                    "industry": industry_map.get(ts_code, ""),
                    "trade_date": str(df["trade_date"].iloc[-1]),
                    "close": float(closes[-1]),
                    "d1": round(d1, 2),
                    "d5": round(d5, 2) if d5 is not None else None,
                    "d20": round(d20, 2),
                    "vol_ratio": round(recent5 / prev5, 2) if prev5 else 1.0,
                }
            except Exception as e:
                logger.warning(f"[advisor] quote snapshot {vt} failed: {e}")
    except Exception as e:
        logger.error(f"[advisor] quote snapshots failed: {e}")
    return out


def _compact_news(it: Dict[str, Any]) -> Dict[str, Any]:
    """压缩资讯条目供 LLM 输入（保留分析要素，去掉映射类冗余字段）。"""
    return {
        "sector": it.get("sector"),
        "title": it.get("title"),
        "summary": it.get("summary"),
        "direction": it.get("direction"),
        "conviction": it.get("conviction"),
        "thesis": it.get("thesis"),
        "rotation": it.get("rotation"),
        "expectation_gap": it.get("expectation_gap"),
        "timeliness": it.get("timeliness"),
        "info_date": it.get("info_date"),
        "catalysts": (it.get("catalysts") or [])[:3],
        "risks": (it.get("risks") or [])[:3],
        "stock_implications": it.get("stock_implications") or [],
        "related_sectors": (it.get("related_sectors") or [])[:5],
    }


def _valid_evidence(raw: Any, titles: set) -> List[str]:
    """evidence 只保留真实存在的资讯标题（防虚构引用），容忍轻微截断差异。"""
    out: List[str] = []
    if not isinstance(raw, list):
        return out
    for e in raw[:5]:
        s = str(e).strip()
        if not s:
            continue
        if s in titles or any(s in t or t in s for t in titles if t):
            out.append(s[:100])
    return out


def _derive_watchlist_alerts(
    news_items: List[Dict[str, Any]],
    watchlist: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """代码侧兜底：持仓股名匹配资讯 stock_implications / mapped_stocks / 正文。

    LLM 漏报时保证仍有动向信号。alert 方向取 stock_implications 的 direction，
    没有明确方向时按资讯整体 direction。
    关联强度：stock_implications 命中（LLM 推导时点名）→ direct；
    仅 mapped_stocks/正文命中（概念成分映射口径）→ concept。
    """
    out: List[Dict[str, Any]] = []
    for w in watchlist:
        name = w["name"]
        for it in news_items:
            hit_dir: Optional[str] = None
            relevance = "concept"
            for imp in (it.get("stock_implications") or []):
                if imp.get("name") == name:
                    hit_dir = imp.get("direction")
                    relevance = "direct"
                    break
            if hit_dir is None:
                in_mapped = any(m.get("name") == name for m in (it.get("mapped_stocks") or []))
                in_text = name in (str(it.get("title", "")) + str(it.get("summary", "")))
                if in_text:
                    hit_dir = it.get("direction")
                    relevance = "direct"
                elif in_mapped:
                    hit_dir = it.get("direction")
                    relevance = "concept"
            if hit_dir in ("bullish", "bearish"):
                out.append({
                    "name": name,
                    "vt_symbol": w.get("vt_symbol"),
                    "alert": "positive" if hit_dir == "bullish" else "risk",
                    "relevance": relevance,
                    "rationale": f"资讯《{it.get('title', '')}》关联该股：{(it.get('thesis') or it.get('impact') or '')[:80]}",
                    "evidence": [str(it.get("title", ""))[:100]],
                    "action_hint": "",
                    "derived": True,
                })
                break  # 每只股取首条命中即可，避免刷屏
    return out


# =============================================================================
# 资讯加载 / 建议落盘
# =============================================================================
def _load_news(date: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """加载指定日（默认最新）资讯。返回 (date, items)。"""
    if not date:
        dates = sorted((p.stem for p in NEWS_DIR.glob("*.json")), reverse=True) if NEWS_DIR.exists() else []
        date = dates[0] if dates else None
    if not date:
        return "", []
    fp = NEWS_DIR / f"{date}.json"
    if not fp.exists():
        return date, []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items") if isinstance(data, dict) else data
        return date, items if isinstance(items, list) else []
    except Exception as e:
        logger.error(f"[advisor] load news {fp} failed: {e}")
        return date, []


def save_advice(advice: Dict[str, Any]) -> str:
    """落盘 core/alpha_db/advice/{date}.json（同日覆盖，保留最新一份）。"""
    ADVICE_DIR.mkdir(parents=True, exist_ok=True)
    fp = ADVICE_DIR / f"{advice['date']}.json"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(advice, f, ensure_ascii=False, indent=2)
    logger.info(f"[advisor] saved {fp}: {len(advice.get('top_stocks', []))} stocks, "
                f"{len(advice.get('risk_sectors', []))} risk sectors, "
                f"{len(advice.get('watchlist_alerts', []))} alerts")
    return str(fp)


def run_advice(date: Optional[str] = None) -> Dict[str, Any]:
    """同步生成 + 落盘。供后台线程调用，重复触发直接跳过。"""
    if not _advice_lock.acquire(blocking=False):
        return {"status": "skipped", "message": "已有生成任务在执行"}
    try:
        advisor = InvestmentAdvisor()
        advice = advisor.generate(date)
        if advice.get("status") != "ok":
            return advice
        filepath = save_advice(advice)
        advice["filepath"] = filepath
        return advice
    finally:
        _advice_lock.release()


def run_holdings_analysis(date: Optional[str] = None) -> Dict[str, Any]:
    """单独刷新持仓深度跟踪（不重跑全量资讯蒸馏），合并写回当日 advice 文件。

    advice 文件不存在时创建骨架（只含 holding_analysis），前端仍可展示持仓跟踪。
    """
    if not _advice_lock.acquire(blocking=False):
        return {"status": "skipped", "message": "已有生成任务在执行"}
    try:
        date = date or datetime.now().strftime("%Y-%m-%d")
        advisor = InvestmentAdvisor()
        analysis = advisor.analyze_holdings()

        ADVICE_DIR.mkdir(parents=True, exist_ok=True)
        fp = ADVICE_DIR / f"{date}.json"
        advice: Dict[str, Any] = {}
        if fp.exists():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    advice = json.load(f)
            except Exception:
                advice = {}
        if not advice:
            # 骨架：当日还没生成过建议（如采集失败），持仓跟踪也能独立落盘
            advice = {"status": "ok", "date": date, "news_count": 0,
                      "market_summary": {"tone": "neutral", "comment": ""},
                      "top_stocks": [], "risk_sectors": [], "watchlist_alerts": []}
        advice["holding_analysis"] = analysis
        advice["holdings_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(advice, f, ensure_ascii=False, indent=2)
        logger.info(f"[advisor] holdings analysis saved: {len(analysis)} stocks -> {fp}")
        return {"status": "ok", "date": date, "count": len(analysis), "holding_analysis": analysis}
    finally:
        _advice_lock.release()
