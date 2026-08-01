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

v2 改进（2026-07-28，针对"光刻机资讯遗漏"问题）：
1. 覆盖完整性机制：采集前先加载 dc_concept 提取当日异动板块（涨幅前15+跌幅前8），
   连同固定重点产业主题台账（KEY_THEMES，光刻机/半导体设备/AI算力等）注入 prompt，
   要求 LLM 逐项核查，不再依赖 LLM 自由发挥。
2. 定向补采轮：第一轮结束后代码侧做覆盖核查（_find_missed_themes），
   对未覆盖的异动板块/重点主题发起第二轮定向补采（_llm_collect_supplement）。
   补采按 5 个主题一批分批调用（避免 OpenClaw agent 单次 tool 循环过载返回
   "Agent couldn't generate a response"），每批解析为空时原样重试一次。
   另将 coverage_check 标 covered 但 items 语料未覆盖的主题一并纳入补采
   （LLM 声称有增量却没输出条目，必须补采落地）。
3. coverage_check 审计：要求 LLM 输出已核查主题清单（covered/no_news），
   落盘到 JSON meta，可审计每日覆盖完整性。
4. 显式启用 OpenClaw skill「a-share-news-radar」（~/.openclaw/skills/），
   skill 内沉淀资讯搜索地址清单 / 主题台账 / 采集 SOP，prompt 中声明启用。
5. dc_concept / stock_basic 全流程只加载一次（collect 前置加载后透传），
   避免补采轮与映射阶段重复请求 tushare/MySQL。

v4 改进（2026-07-29，方法论纠偏——LLM 自主挖掘优先）：
1. 主流程重新定位：LLM 自主挖掘时效资讯 → 筛选市场冲击 → 推导受影响个股与
   涨跌方向（新增 stock_implications 字段）。盘面异动板块从"必须逐一反查"
   降级为可选背景参考——单纯依靠已有板块涨幅会局限 LLM 挖掘能力，陷入
   "涨了找原因"的确认偏误，漏掉尚未被定价的资讯（预期差才是 alpha 来源）。
2. prompt 中显式声明 OpenClaw tushare 技能可用：LLM 挖掘标的时可调用
   tushare 查询板块成分、个股行情验证影响面。
3. 盘面数据（dc_member 成分 × 当日涨幅）保留为后处理验证层，不参与束缚
   LLM 的资讯挖掘视野。

v3 改进（2026-07-29，针对"光刻机条目代表个股不对"反馈）：
1. 代表个股从「LLM 命名 + 行业模糊匹配」改为「盘面验证口径」：
   dc_concept(000xxx) 经 dc_index 桥接东财 BK 代码 → dc_member 成分股 ×
   当日全市场涨跌幅（pro.daily）排序，涨幅 Top 5 置前（如光刻机 7-28
   盘面领涨为波长光电/张江高科/海立股份，而非 LLM 命名的产业链大票）。
   领涨股 lead_stock 同步改为成分股当日涨幅第一名。
2. 板块匹配精度修复：复合板块名（如"半导体设备/光刻机"）拆分关键词后
   从后往前优先精确匹配（中文习惯"大类/细分"，细分在后），避免被
   大类板块（"半导体"）的包含匹配截胡。
3. dc_member 调用注意：必须同时传 ts_code+trade_date，否则 ts_code 过滤
   不生效且返回被 8000 行上限截断（实测）。
"""

import json
import os
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.llm.openclaw_client import OpenClawClient, parse_json_response
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

# 重点产业主题台账：与 OpenClaw skill「a-share-news-radar」第二节对齐。
# 第一轮采集未覆盖的主题会进入定向补采清单，防止"LLM 没想到"导致遗漏（如光刻机）。
# 维护原则：只放长期高关注度、事件驱动型产业主题，20 个左右，随市场主线演进更新。
KEY_THEMES = [
    "光刻机", "光刻胶", "半导体设备", "存储芯片", "先进封装",
    "AI算力", "光模块", "CPO", "人形机器人", "固态电池",
    "创新药", "CXO", "稀土", "锂矿", "券商",
    "军工", "低空经济", "商业航天", "华为产业链", "信创",
]


# =============================================================================
# Prompt（投资分析师角色，参考 STOCK_RATING_SYSTEM 范式）
# =============================================================================
NEWS_SYSTEM = """你是 A 股专业投资分析师，服务对象是 A 股投资交易者。

【显式启用 skill】本任务已启用 OpenClaw skill「a-share-news-radar」（A股资讯雷达）。
若运行环境中存在该 skill，遵循其资讯搜索地址清单与覆盖核查方法，但挖掘判断由你自主完成。

你的任务是一条**自主挖掘链路**，分三步：
1. **抓取**：用联网搜索工具自主抓取最近 1-3 个交易日最具时效的行业资讯——不是等待给定主题，而是主动发现。
2. **筛选**：从投资者视角判断哪些资讯会对 A 股产生真实冲击（资金会因此重新定价）。只收录会移动板块、驱动轮动、扭转情绪的消息；没有真实增量的不要硬凑。
3. **推导**：对每条入选资讯，挖掘它会影响的**具体股票及涨跌方向**——从事件逻辑出发推导受益/受损标的（可挖掘弹性更大的二线标的，不只列人人皆知的龙头），每只标的给出方向与一句话逻辑。

【可用工具】环境中已启用 **tushare 技能**（OpenClaw tushare-pro skill）。推导影响标的时，可调用 tushare 查询 A 股板块成分、个股行情、涨跌停、资金流等数据，验证个股归属与近期表现，让标的推导有据可依。

## 挖掘原则（预期差导向）
- **最有价值的资讯是市场尚未充分定价的**。不要只解释已经有涨幅的板块（那是后视镜），要挖掘"今天发生、市场还没反应过来"的增量信息。
- 漏掉当日重要产业主题（如光刻机、半导体设备、AI算力的重大事件）是最大的失败。
- 每条资讯必须给出明确的**方向**（bullish 利好 / bearish 利空 / neutral 中性）与**确信度** conviction（0.0-1.0）。
- 必须给出**投资逻辑** thesis：一句话说清"为什么这条消息会移动价格"，要有经济/产业逻辑支撑，不要复述事实。
- 必须给出**传导链条** transmission_chain：从事件 → 产业链环节 → 受益/受损标的 → 价格反应的路径。
- 必须给出**预期差** expectation_gap：市场当前可能已 price-in 多少、超预期方向在哪；难以判断则如实说明，不要编造。
- 催化剂 catalysts 与风险/证伪点 risks 要具体、可观测（如某项数据发布日、某产能投产节点、某政策落地时点）。
- 核查过但确认无增量的主题，在 coverage_check 中标注 no_news，**不要为了凑数编造资讯**。

## 输出要求（严格 JSON）
只返回一个 JSON 对象（coverage_check 与 items 同级，必须输出，供调用方审计覆盖完整性）：
```json
{
  "coverage_check": [
    {"theme": "光刻机", "status": "covered | no_news", "note": "一句话说明核查结果"}
  ],
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
      "stock_implications": [
        {"name": "受影响个股名", "direction": "bullish | bearish", "logic": "一句话逻辑（为何受益/受损）"}
      ],
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
- 全中文。title≤30 字，summary≤120 字，thesis≤80 字，transmission_chain≤200 字，expectation_gap≤150 字，impact≤80 字，stock_implications 每条 logic≤50 字。
- stock_implications 是本模块的核心产出：每条资讯 2-6 只受影响个股，必须是从资讯逻辑推导出来的（可用 tushare 验证归属），direction 只允许 bullish/bearish。
- 只返回 JSON，不要任何额外文字。"""

NEWS_USER = """请执行自主挖掘链路：抓取最近 1-3 个交易日最具时效的行业资讯 → 筛选其中会对 A 股产生真实冲击的 → 对每条推导受影响的股票及涨跌方向。

【推荐搜索入口】（via web_search / web_fetch 工具）：
- 快讯电报：财联社 cls.cn/telegraph、华尔街见闻 wallstreetcn.com/live、东方财富 kuaixun.eastmoney.com、同花顺 news.10jqka.com.cn
- 主题定向检索：百度资讯（按时间排序）、必应资讯、搜狗微信，关键词如"光刻机 最新""半导体设备 订单"
- 推导标的时可用 tushare 技能查询板块成分股、个股行情，验证影响面。

【当日盘面背景】（仅供参考，了解市场当前热点结构；不要求逐一覆盖，更不要只为解释已有涨幅而收录——你的首要任务是挖掘尚未被定价的增量资讯）：
{hot_themes}

【覆盖核查台账】（完成自主挖掘后逐项核查，防止遗漏重要产业主题；这不是挖掘范围的限制，只是防遗漏清单）：
{key_themes}

重点方向（按当日实际情况取舍，不要强行凑齐）：
1. 政策与监管（财政/货币/产业政策、监管动向）
2. 产业事件（订单/涨价/产能/技术突破/招投标）
3. 资金与情绪（北向/两融/成交额/涨停板结构/高低切换）
4. 宏观与地缘冲击（进出口/利率/汇率/外围事件）

按 system 的 JSON schema 输出 {n_items} 条左右。每条都必须包含完整的投资分析字段（direction/conviction/thesis/transmission_chain/time_horizon/catalysts/risks/expectation_gap）以及 **stock_implications**（受影响个股+方向+逻辑，本模块核心产出），不要留空。
coverage_check 必须覆盖你核查过的所有主题（covered 或 no_news）。"""

NEWS_USER_SUPPLEMENT = """第一轮资讯采集已结束，但以下【当日异动板块 / 重点产业主题】未被覆盖，存在遗漏风险：

{missed_themes}

请对上述主题**逐个用关键词定向搜索**（如"光刻机 最新""光刻胶 涨价""XX板块 异动原因"），核查最近 1-3 个交易日是否有影响 A 股的增量资讯：
- 有真实增量的，按 system 的 JSON schema 输出 items（有几条收几条，宁缺毋滥）；
- 确认当日无增量的，不要编造，在 coverage_check 中标注 no_news；
- coverage_check 必须覆盖上述全部主题。只返回 JSON。"""


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
    # LLM 自主推导的受影响个股（核心产出）：[{name, direction, logic}]
    stock_implications: List[Dict[str, str]] = field(default_factory=list)

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
        # 最近一次采集 LLM 返回的 coverage_check（覆盖审计），由 _parse_items 记录
        self.last_coverage: List[Dict[str, Any]] = []
        # dc_concept 最新交易日（_load_concepts 设置），供 daily quotes/dc_member 对齐口径
        self._concepts_trade_date: Optional[str] = None
        # dc_member 成分股缓存 {bk_code: [(con_code, name)]}
        self._members_cache: Dict[str, List[Tuple[str, str]]] = {}

    # ---------- 主流程 ----------
    def collect(self, collect_date: Optional[str] = None) -> List[NewsItem]:
        """采集资讯并做投资影响分析，返回 NewsItem 列表（已映射板块/个股）。

        两轮机制：
        1. 广覆盖轮：注入当日异动板块（dc_concept 涨跌幅领先）+ 重点主题台账；
        2. 定向补采轮：代码侧覆盖核查后，对遗漏主题定向搜索补采。
        """
        if collect_date is None:
            collect_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"[NewsCollector] start collect_date={collect_date} model={self.client.model}")

        # 前置加载（全程复用，避免补采轮/映射阶段重复请求 tushare/MySQL）
        concepts = self._load_concepts()
        stock_basic = self._load_stock_basic()
        market_ctx = self._build_market_ctx()
        hot_themes = self._extract_hot_themes(concepts)
        logger.info(f"[NewsCollector] hot themes ({len(hot_themes)}): {'、'.join(hot_themes[:10])}...")

        # 第一轮：广覆盖采集
        self.last_coverage = []
        raw_items = self._llm_collect(collect_date, hot_themes)
        if not raw_items:
            logger.warning("[NewsCollector] LLM returned no items")
            return []

        items = [self._build_item(d, collect_date) for d in raw_items]
        items = [it for it in items if it is not None]
        if not items:
            # 主轮全部解析/构建失败说明 LLM 严重不可用（如 OpenClaw 网关故障），
            # 此时全量补采只会白耗十几分钟（2026-07-30 09:00 档实测），直接放弃
            logger.warning("[NewsCollector] all items failed to build, skip supplement")
            return []

        # 覆盖核查 + 定向补采
        missed = self._find_missed_themes(items, hot_themes)
        if missed:
            logger.info(f"[NewsCollector] missed themes, supplement round: {'、'.join(missed)}")
            extra_raw = self._llm_collect_supplement(collect_date, missed)
            extra_items = [self._build_item(d, collect_date) for d in extra_raw]
            seen = {(it.sector, it.title) for it in items}
            for it in extra_items:
                if it is not None and (it.sector, it.title) not in seen:
                    seen.add((it.sector, it.title))
                    items.append(it)

        # 后处理：映射板块/个股（复用前置加载数据）
        try:
            self._map_to_market(items, collect_date, concepts=concepts, stock_basic=stock_basic, market_ctx=market_ctx)
        except Exception as e:
            logger.error(f"[NewsCollector] map_to_market failed (non-fatal): {e}")

        logger.info(f"[NewsCollector] done: {len(items)} items")
        return items

    # ---------- LLM 调用 ----------
    def _llm_collect(self, collect_date: str, hot_themes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """主轮采集。OpenClaw agent 偶发 "Agent couldn't generate a response"（非 JSON
        错误文本），会话内修复无意义，整次重发最多 3 轮；真正的 JSON 畸形才走
        会话内修复重试（与 advisor._llm_digest 同策略，2026-07-30 09:00 档实测教训）。"""
        system = NEWS_SYSTEM.replace("{collect_date}", collect_date)
        user = NEWS_USER.format(
            n_items=self.n_items,
            hot_themes=("、".join(hot_themes) if hot_themes else "（行情数据不可用，按台账与泛检索自行判断当日热点）"),
            key_themes="、".join(KEY_THEMES),
        )

        for attempt in range(3):
            raw = self._chat_safe(system, user, max_tokens=self.max_tokens)
            if not raw:
                logger.warning(f"[NewsCollector] chat empty, full retry ({attempt + 1}/3)")
                continue
            items = self._parse_items(raw)
            if items:
                return items
            if "couldn't generate a response" in raw:
                logger.warning(f"[NewsCollector] agent error, full retry ({attempt + 1}/3)")
                continue
            # JSON 畸形：会话内修复一次（参考 risk_screener._retry_json_fix）
            logger.warning("[NewsCollector] JSON 解析为空，会话内修复重试")
            retry_raw = self._retry_json_fix(system, user, raw)
            if retry_raw:
                items = self._parse_items(retry_raw)
                if items:
                    return items
        return []

    def _llm_collect_supplement(self, collect_date: str, missed_themes: List[str], batch_size: int = 5) -> List[Dict[str, Any]]:
        """第二轮定向补采：针对第一轮遗漏的主题分批定向搜索确认。失败不影响主流程。

        分批原因：一次让 OpenClaw agent 核查过多主题会导致 tool 循环过载，
        返回 "Agent couldn't generate a response" 错误文本而非 JSON（2026-07-29 实测）。
        每批解析为空时原样重试一次，仍空则跳过该批。
        """
        system = NEWS_SYSTEM.replace("{collect_date}", collect_date)
        all_items: List[Dict[str, Any]] = []
        batches = [missed_themes[i:i + batch_size] for i in range(0, len(missed_themes), batch_size)]
        for batch in batches:
            user = NEWS_USER_SUPPLEMENT.format(missed_themes="、".join(batch))
            raw = self._chat_safe(system, user)
            items = self._parse_items(raw) if raw else []
            if not items:
                logger.warning(f"[NewsCollector] supplement batch empty, retry once: {'、'.join(batch)}")
                raw = self._chat_safe(system, user)
                items = self._parse_items(raw) if raw else []
            logger.info(f"[NewsCollector] supplement batch {'、'.join(batch)}: {len(items)} items")
            all_items.extend(items)
        return all_items

    def _chat_safe(self, system: str, user: str, max_tokens: int = 4096) -> str:
        """容错版 chat：异常返回空串而非抛出（用于补采等非主流程调用）。"""
        try:
            return self.client.chat(
                system=system,
                user=user,
                response_format_json=True,
                temperature=self.temperature,
                max_tokens=min(self.max_tokens, max_tokens),
            )
        except Exception as e:
            logger.error(f"[NewsCollector] chat failed (non-fatal): {e}")
            return ""

    # ---------- 覆盖核查 ----------
    def _extract_hot_themes(self, concepts: List[Dict], top_gain: int = 15, top_loss: int = 8) -> List[str]:
        """从 dc_concept 提取最近交易日涨跌幅领先板块名（异动板块 = 最可能有消息驱动）。"""
        valid = [c for c in concepts if c.get("pct_change") is not None and c.get("name")]
        gainers = sorted(valid, key=lambda c: c["pct_change"], reverse=True)[:top_gain]
        losers = sorted(valid, key=lambda c: c["pct_change"])[:top_loss]
        out, seen = [], set()
        for c in gainers + losers:
            nm = c["name"]
            if nm not in seen:
                seen.add(nm)
                out.append(nm)
        return out

    def _find_missed_themes(self, items: List[NewsItem], hot_themes: List[str]) -> List[str]:
        """覆盖核查：找出第一轮未覆盖、值得定向补采的主题。

        两类来源：
        1) 当日异动板块（hot_themes）中没有任何一条资讯提及的——有资金异动大概率有消息驱动；
        2) 重点产业主题台账（KEY_THEMES）中完全未出现的——补采轮定向搜索确认当日增量。
        匹配做归一化：去掉"概念/板块"后缀后子串匹配（dc_concept 常带后缀，LLM 输出一般不带）。
        """
        corpus = " ".join(
            f"{it.sector} {' '.join(it.related_sectors)} {it.title} {it.summary}" for it in items
        )
        corpus_norm = corpus.replace("概念", "").replace("板块", "")

        def _covered(theme: str) -> bool:
            t = theme.replace("概念", "").replace("板块", "")
            return (theme in corpus) or (t and t in corpus_norm)

        # 分组配额（总上限 15，按优先级排序）：
        # 1) claimed：coverage_check 标 covered 但 items 语料未覆盖——LLM 声称有增量
        #    却没输出条目（2026-07-29 实测光刻机即此情况），必须补采落地，配额 4；
        # 2) key：重点台账保底——产业主题防遗漏是主要覆盖保障，配额 8；
        # 3) hot：当日异动板块无任何资讯提及——v4 起异动板块降级为背景参考，配额 5。
        claimed = [
            str(c.get("theme", "")).strip()
            for c in self.last_coverage
            if isinstance(c, dict) and c.get("status") == "covered"
        ]
        claimed_missed: List[str] = []
        for t in claimed:
            if t and t not in claimed_missed and not _covered(t):
                claimed_missed.append(t)
        claimed_missed = claimed_missed[:4]

        key_missed = [
            t for t in KEY_THEMES
            if t and t not in claimed_missed and not _covered(t)
        ][:8]
        hot_missed = [
            t for t in hot_themes
            if t and t not in claimed_missed and t not in key_missed and not _covered(t)
        ][:5]
        return (claimed_missed + key_missed + hot_missed)[:15]

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
        try:
            data = parse_json_response(raw)
        except ValueError as e:
            logger.error(f"[NewsCollector] JSON parse failed (after repair): {e}")
            return []
        if isinstance(data, dict):
            # 覆盖审计：合并 LLM 自查的 coverage_check（covered/no_news）。
            # 同名 theme 以后返回的为准（补采轮核查结果更新第一轮），不同名保留。
            cov = data.get("coverage_check")
            if isinstance(cov, list) and cov:
                merged = {
                    c.get("theme"): c
                    for c in self.last_coverage
                    if isinstance(c, dict) and c.get("theme")
                }
                for c in cov:
                    if isinstance(c, dict) and c.get("theme"):
                        merged[c["theme"]] = c
                self.last_coverage = list(merged.values())
            items = data.get("items")
            # 过滤非 dict 元素（修复重试后 LLM 可能返回字符串列表，2026-07-30 实测
            # 导致 _build_item 报 'str' object has no attribute 'get'）
            return [x for x in items if isinstance(x, dict)] if isinstance(items, list) else []
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
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
                stock_implications=_parse_implications(d.get("stock_implications")),
                related_sectors=[str(x).strip() for x in (d.get("related_sectors") or []) if str(x).strip()][:10],
                related_stocks=[str(x).strip() for x in (d.get("related_stocks") or []) if str(x).strip()][:10],
                source=str(d.get("source", "")).strip()[:200],
                collect_date=collect_date,
            )
        except Exception as e:
            logger.error(f"[NewsCollector] build item failed: {e}")
            return None

    # ---------- 板块/个股映射 ----------
    def _map_to_market(
        self,
        items: List[NewsItem],
        collect_date: str,
        concepts: Optional[List[Dict]] = None,
        stock_basic: Optional[Dict] = None,
        market_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        """用 dc_concept（板块行情）+ stock_basic + 盘面数据映射，结果挂在 item 上。

        concepts/stock_basic/market_ctx 可由调用方前置加载后传入复用（collect 主流程），
        为 None 时自行加载（兼容独立调用）。
        代表性个股优先级：盘面领涨成分股（实盘验证）> LLM 命名 > 行业模糊补足。
        """
        if concepts is None:
            concepts = self._load_concepts()        # [{theme_code, name, pct_change, hot, lead_stock, lead_stock_code}]
        if stock_basic is None:
            stock_basic = self._load_stock_basic()  # {name: vt_symbol, industry: {industry: [(vt_symbol,name)]}}
        if market_ctx is None:
            market_ctx = self._build_market_ctx()

        dc_index = market_ctx.get("dc_index", {})
        quotes = market_ctx.get("quotes", {})
        trade_date = market_ctx.get("trade_date")

        for item in items:
            # 1) 板块名 → dc_concept（取 sector 与 related_sectors 中首个命中）
            cand_sectors = [item.sector] + list(item.related_sectors)
            hit = self._match_concept(cand_sectors, concepts)

            market_leaders: List[Dict[str, Any]] = []
            if hit:
                item.concept_code = hit.get("theme_code")
                item.concept_pct_change = hit.get("pct_change")
                item.concept_hot = hit.get("hot")

                # 盘面领涨：dc_index 桥接东财 BK 代码 → dc_member 成分 × 当日涨幅 Top
                idx_row = self._match_dc_index(hit.get("name"), dc_index)
                if idx_row and idx_row.get("bk_code"):
                    members = self._load_concept_members(idx_row["bk_code"], trade_date)
                    ranked = sorted(
                        members,
                        key=lambda x: (quotes.get(x[0]) is not None, quotes.get(x[0]) or -999),
                        reverse=True,
                    )
                    # Top 8：覆盖当日板块涨停梯队（7-28 光刻机 8 只涨停含张江高科/海立股份，
                    # 取 Top 5 会被截断——用户反馈即此情况）
                    for code, name in ranked[:8]:
                        chg = quotes.get(code)
                        if chg is None:
                            continue
                        market_leaders.append({
                            "vt_symbol": _to_vt_symbol(code),
                            "name": name,
                            "pct_chg": chg,
                        })

                # 领涨股：盘面涨幅第一优先（实盘验证口径），退化为 dc_concept 官方口径
                if market_leaders:
                    item.lead_stock = market_leaders[0]["name"]
                    item.lead_stock_code = market_leaders[0]["vt_symbol"]
                else:
                    item.lead_stock = hit.get("lead_stock")
                    item.lead_stock_code = _to_vt_symbol(hit.get("lead_stock_code"))

            # 2) 代表性个股
            mapped: Dict[str, Dict[str, Any]] = {}  # vt_symbol -> {vt_symbol, name, pct_chg}

            # 2a) 盘面领涨成分股置前（实盘验证口径，最可靠）
            for ml in market_leaders:
                mapped.setdefault(ml["vt_symbol"], ml)

            # 2b) 领涨股
            if item.lead_stock_code:
                mapped.setdefault(item.lead_stock_code, {
                    "vt_symbol": item.lead_stock_code,
                    "name": item.lead_stock or "",
                    "pct_chg": None,
                })

            # 2c) LLM 自主推导的受影响个股（stock_implications，核心产出）→
            #     stock_basic 匹配 vt_symbol，direction 一并记录供前端着色
            for imp in item.stock_implications:
                vs = self._match_stock_name(imp["name"], stock_basic["by_name"])
                if vs:
                    mapped.setdefault(vs[0], {
                        "vt_symbol": vs[0],
                        "name": vs[1],
                        "pct_chg": None,
                        "direction": imp["direction"],
                    })

            # 2d) LLM 给的个股名 → stock_basic.name
            for nm in item.related_stocks:
                vs = self._match_stock_name(nm, stock_basic["by_name"])
                if vs:
                    mapped.setdefault(vs[0], {"vt_symbol": vs[0], "name": vs[1], "pct_chg": None})

            # 2e) 板块名 → industry 模糊匹配，仅在盘面/LLM 命中很少时补足
            # （industry 是大类口径，会带入无关股稀释代表性；ST 股不适合做代表，排除）
            if len(mapped) < 6:
                industry_stocks = self._match_industry(cand_sectors, stock_basic["by_industry"])
                for vs, nm in industry_stocks:
                    if len(mapped) >= 10:
                        break
                    if "ST" in (nm or ""):
                        continue
                    mapped.setdefault(vs, {"vt_symbol": vs, "name": nm, "pct_chg": None})

            item.mapped_stocks = list(mapped.values())[:12]

    # ---------- 盘面数据（dc_index 桥接 + 成分股 + 全市场涨跌幅）----------
    def _build_market_ctx(self) -> Dict[str, Any]:
        """构建盘面映射上下文：dc_index 桥接表 + 当日全市场涨跌幅。

        trade_date 与 dc_concept 最新交易日对齐（_load_concepts 设置）；
        缺失时盘面领涨功能自动降级为 dc_concept 官方口径。
        """
        return {
            "dc_index": self._load_dc_index(),
            "quotes": self._load_daily_quotes(self._concepts_trade_date),
            "trade_date": self._concepts_trade_date,
        }

    def _load_dc_index(self) -> Dict[str, Dict]:
        """东财板块指数（BK 代码 + 名称），用于 dc_concept(000xxx) → dc_member(BKxxxx) 桥接。"""
        out: Dict[str, Dict] = {}
        try:
            from vnpy.trader.setting import SETTINGS
            import tushare as ts
            pro = ts.pro_api(SETTINGS["datafeed.password"])
            df = pro.dc_index()
            if df is None or df.empty:
                return out
            latest = sorted(df["trade_date"].unique())[-1]
            df = df[df["trade_date"] == latest]
            for _, r in df.iterrows():
                name = str(r.get("name", "")).strip()
                if name:
                    out[name] = {"bk_code": str(r.get("ts_code", "")).strip()}
        except Exception as e:
            logger.error(f"[NewsCollector] load dc_index failed: {e}")
        return out

    def _match_dc_index(self, concept_name: Optional[str], dc_index: Dict[str, Dict]) -> Optional[Dict]:
        """dc_concept 板块名 → dc_index（BK 代码）桥接。名称口径不同
        （如 dc_concept「光刻机」 vs dc_index「光刻机(胶)」），先精确后包含。"""
        if not concept_name:
            return None
        if concept_name in dc_index:
            return dc_index[concept_name]
        for name, row in dc_index.items():
            if concept_name in name or name in concept_name:
                return row
        return None

    def _load_concept_members(self, bk_code: str, trade_date: Optional[str]) -> List[Tuple[str, str]]:
        """东财概念板块成分股 [(con_code, name)]。

        注意：dc_member 必须同时传 ts_code+trade_date，否则 ts_code 过滤不生效，
        返回被 8000 行上限截断（2026-07-29 实测）。带实例缓存，同板块不重复请求。
        """
        if not bk_code or not trade_date:
            return []
        if bk_code in self._members_cache:
            return self._members_cache[bk_code]
        out: List[Tuple[str, str]] = []
        try:
            from vnpy.trader.setting import SETTINGS
            import tushare as ts
            pro = ts.pro_api(SETTINGS["datafeed.password"])
            df = pro.dc_member(ts_code=bk_code, trade_date=trade_date)
            if df is not None and not df.empty:
                df = df.drop_duplicates("con_code")
                for _, r in df.iterrows():
                    code = str(r.get("con_code", "")).strip()
                    name = str(r.get("name", "")).strip()
                    if code and name:
                        out.append((code, name))
        except Exception as e:
            logger.error(f"[NewsCollector] load dc_member {bk_code} failed: {e}")
        self._members_cache[bk_code] = out
        return out

    def _load_daily_quotes(self, trade_date: Optional[str]) -> Dict[str, float]:
        """全市场日频涨跌幅 {ts_code: pct_chg}，用于成分股盘面排序。"""
        out: Dict[str, float] = {}
        if not trade_date:
            return out
        try:
            from vnpy.trader.setting import SETTINGS
            import tushare as ts
            pro = ts.pro_api(SETTINGS["datafeed.password"])
            df = pro.daily(trade_date=trade_date)
            if df is None or df.empty:
                return out
            for _, r in df.iterrows():
                chg = _to_float(r.get("pct_chg"))
                if chg is not None:
                    out[str(r["ts_code"]).strip()] = chg
        except Exception as e:
            logger.error(f"[NewsCollector] load daily quotes failed: {e}")
        return out

    # 概念匹配
    def _match_concept(self, cand_sectors: List[str], concepts: List[Dict]) -> Optional[Dict]:
        """板块名 → dc_concept 行。

        复合板块名（如"半导体设备/光刻机"）拆分关键词后**从后往前**优先精确匹配：
        中文习惯"大类/细分"，细分在后才是资讯主体；否则会被大类板块（"半导体"）
        的包含匹配截胡（2026-07-29 实测光刻机条目错配到 000058.DC「半导体」）。
        """
        def _split(sec: str) -> List[str]:
            return [k for k in re.split(r"[、/／\s()（）]+", sec) if len(k) >= 2]

        for sec in cand_sectors:
            if not sec:
                continue
            # 1) 整体精确
            for c in concepts:
                if c["name"] == sec:
                    return c
            # 2) 关键词精确（从后往前，细分优先）
            for k in reversed(_split(sec)):
                for c in concepts:
                    if c["name"] == k:
                        return c
            # 3) 整体包含
            for c in concepts:
                if sec in c["name"] or c["name"] in sec:
                    return c
            # 4) 关键词包含（从后往前）
            for k in reversed(_split(sec)):
                for c in concepts:
                    if k in c["name"]:
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
            # 取最新交易日（记录供 daily quotes / dc_member 对齐口径）
            latest = sorted(df["trade_date"].unique())[-1]
            self._concepts_trade_date = str(latest)
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
        """从 MySQL stock_basic 加载名称/行业映射（委托模块级 load_stock_basic，供 advisor 等复用）。"""
        return load_stock_basic()


def load_stock_basic() -> Dict:
    """从 MySQL stock_basic 加载 {name: (vt_symbol, name)} 与 {industry: [(vt_symbol,name)]}。

    模块级函数：NewsCollector 与 core/llm/advisor.py 共用。
    """
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
        logger.error(f"[load_stock_basic] failed: {e}")
    return {"by_name": by_name, "by_industry": by_industry}


# =============================================================================
# 工具
# =============================================================================
def _parse_implications(raw: Any) -> List[Dict[str, str]]:
    """解析 LLM 输出的 stock_implications（受影响个股推导，本模块核心产出）。

    容错：非 list / 非 dict 元素跳过；direction 只保留 bullish/bearish；
    name 为空跳过；每条 logic 截断 100 字；最多 8 条。
    """
    out: List[Dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for x in raw:
        if not isinstance(x, dict):
            continue
        name = str(x.get("name", "")).strip()
        direction = str(x.get("direction", "")).lower().strip()
        logic = str(x.get("logic", "")).strip()[:100]
        if not name or direction not in ("bullish", "bearish"):
            continue
        out.append({"name": name, "direction": direction, "logic": logic})
        if len(out) >= 8:
            break
    return out


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
def save_news(
    items: List[NewsItem],
    collect_date: str,
    output_dir: Optional[str] = None,
    coverage: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """落盘到 core/alpha_db/news/{date}.json（每日一份，列表）。

    同一日多次采集：按 (sector, title) 去重后追加，保留历史条目。
    coverage：LLM 输出的 coverage_check（覆盖审计），存入 meta 供核查覆盖完整性。
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
        "coverage_check": coverage or [],
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
        filepath = save_news(items, collect_date, coverage=collector.last_coverage)
        return {
            "status": "ok",
            "collect_date": collect_date,
            "count": len(items),
            "filepath": filepath,
            "items": [_item_to_dict(it) for it in items],
        }
    finally:
        _collect_lock.release()
