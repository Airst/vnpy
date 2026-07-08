# News v1 — 资讯采集与板块映射（后端）

## 背景

现有工程有完整的 LLM 因子挖掘（gp_mining_llm / HypothesisGenerator）与 LLM 个股评级（risk_screener）链路，但缺少"最新行业资讯"维度。交易者需要一张能反映**当日板块强弱、轮动方向、市场情绪**的资讯页面。来源见 goals.md。

## 方案

### 改什么
1. 新增 `core/llm/news_collector.py`：LLM 资讯采集器。
2. 新增 `core/news_service.py`：资讯读取 + 板块/个股映射。
3. 在 `core/main_controller.py` 注册定时任务 + FastAPI 接口。

### 为什么有效
- **连接复用**：gp llm 连接方式 = OpenAI SDK + 百炼 token-plan endpoint + DASHSCOPE_API_KEY（见 `core/alpha/hypothesis_generator.py`）。资讯采集沿用同一 client 构造，仅模型换成支持 web search 的 `qwen3.7-max` 并打开 `enable_search`。glm-5.2 已验证不支持 enable_search。
- **聚焦板块/轮动/情绪**：prompt 强制 LLM 只输出"会移动板块、驱动轮动、扭转情绪"的资讯，并给出影响分析、轮动含义、情绪方向。不做逐股覆盖。
- **板块落地**：LLM 给的板块名/个股名是自然语言，需映射到真实标的才能服务交易者。用 MySQL `dc_member`（概念板块成员）+ `stock_basic`（个股名称）做模糊匹配，得到概念板块代码与代表性 vt_symbol。
- **落盘复用**：参考 `llm_tasks` 的 per-file JSON 模式，每日一份 `core/alpha_db/news/{date}.json`，列表结构，便于前端按日读取、历史回看。

### 资讯条目 schema（LLM 输出）
```json
{
  "collect_date": "2026-07-08",
  "items": [
    {
      "sector": "半导体设备",
      "title": "国产半导体设备订单超预期",
      "summary": "……一句话摘要……",
      "impact": "利好，设备厂订单确认加速业绩兑现",
      "impact_type": "positive",
      "rotation": "资金从高位 AI 算力切向设备材料，高低切换",
      "sentiment": "positive",
      "timeliness": "high",
      "info_date": "2026-07-08",
      "related_sectors": ["半导体材料", "先进封装"],
      "related_stocks": ["北方华创", "中微公司"],
      "source": "财联社/公司公告"
    }
  ]
}
```

### 板块/个股映射（后处理）
- 板块名 → `dc_member` 最新成员：按概念 `name` 模糊匹配，取命中概念的成员并集，给代表性个股 vt_symbol（按 dc_daily 近端成交额/换手取 head）。
- LLM 给的 `related_stocks` 名称 → `stock_basic.name` 精确/模糊匹配 → ts_code → vt_symbol。
- 映射结果挂在 item 上：`mapped_sector_code`、`mapped_stocks: [{vt_symbol, name}]`，不覆盖原始文本。

### 定时任务
- 复用 main_controller 现有 `schedule` 库与 `scheduler()` async 模式。
- 默认两个采集点：盘前 09:00（隔夜资讯）、盘后 15:30（盘中资讯）。可配置。
- 采集在后台线程执行，状态写入 `_news_task_status`，前端可轮询。

### 接口
- `GET /api/news?date=&sector=&sentiment=&limit=` 列表
- `GET /api/news/sectors` 板块聚合（distinct + 计数）
- `GET /api/news/status` 采集任务状态
- `POST /api/news/collect` 手动触发（可选 date）

## 改动范围
- 新增：`core/llm/news_collector.py`、`core/news_service.py`、`core/alpha_db/news/`
- 修改：`core/main_controller.py`（定时任务 + 接口）、`core/llm/__init__.py`（导出）

## 风险与副作用
- web search 结果有时效漂移（模型可能报错日期）→ 强制 info_date 校验，异常时仍保留并标注 timeliness=low。
- LLM 板块名与 tushare 概念名口径不一致 → 模糊匹配 + 兜底，映射失败不阻断落盘，仅 mapped 字段为空。
- enable_search 耗时长（10-30s/次）→ 后台线程 + 超时控制，不阻塞 API。
- 不与现有因子/评级链路耦合，无回测影响。

## 验证标准
- 手动 POST /api/news/collect 能产出 {date}.json，字段完整。
- GET /api/news 返回真实资讯且 mapped_stocks 非空（至少部分条目）。
- 定时任务注册后到点触发（或手动触发等价路径验证）。

## 结果
通过（2026-07-08 验证）。POST /api/news/collect 成功采集 8 条新资讯（合并去重后 14 条），info_date 14/14 ≤ 今日、11/14 为前一日；板块映射 12/14 命中 dc_concept（含 pct_change），14/14 含代表性个股 vt_symbol；定时任务 news_scheduler 已注册（09:00/15:30）。详见 verification_log.md。
