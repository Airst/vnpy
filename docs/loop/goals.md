# 当前迭代目标

> 每轮迭代开始时（Step 3）更新本文件，覆盖上一轮内容。验收标准必须可量化。

## 本轮目标

- **版本**：News v1（股票资讯采集与展示）
- **目标**：在现有工程上构建"股票最新资讯采集与展示"功能。后台定时任务驱动 LLM 搜集最新时效的行业热点资讯，**聚焦影响板块、影响轮动、影响市场情绪**的消息加以分析，提炼板块/投资价值相关资讯，展示到页面上，可服务于一个 A 股投资交易者。
- **改动范围**：新增 LLM 资讯采集器、定时任务、FastAPI 接口、前端资讯页面；复用现有 LLM 连接方式（gp llm：OpenAI SDK + 百炼 + DASHSCOPE_API_KEY）与数据库（MySQL stock_basic / dc_member / dc_daily + alpha_db JSON 落盘）。

## 范围界定（关键）

- **不是逐股覆盖**：LLM 不关注所有个股，而是关注**会移动板块、驱动轮动、扭转市场情绪**的消息资讯并加以分析。
- 资讯粒度 = 板块/主题级，每条资讯给出：板块、标题、摘要、**影响分析**（利好/利空/中性 + 逻辑）、轮动含义、情绪方向、时效性、关联板块与代表性个股（用于落地，非穷举）。
- 个股仅在"该资讯的直接受益/受损标的"层面出现，作为板块分析的佐证。

## 验收标准

| 指标 | 目标 |
|:---|:---|
| 资讯采集 | 定时任务可触发，LLM 返回结构化 JSON 且字段完整、可解析 |
| 时效性 | 资讯带 info_date，与采集日同期（web search 能拿到近 1-3 日内信息） |
| 板块映射 | 板块名能映射到 dc_member 概念板块与代表性个股 vt_symbol |
| 接口 | GET /api/news、/api/news/sectors、/api/news/status 可用，POST /api/news/collect 可手动触发 |
| 页面 | 资讯页面可渲染真实数据，支持按板块/情绪过滤，有"立即采集"按钮 |
| 可服务性 | 一个 A 股交易者打开页面能据此判断当日板块强弱/轮动方向/情绪 |

## 方案索引

- [x] design/news_v1_collect.md
- [x] design/news_v1_frontend.md

## 备注

- LLM 连接参考 gp llm（HypothesisGenerator）：OpenAI SDK + 百炼 token-plan endpoint + DASHSCOPE_API_KEY。
- glm-5.2 不支持 web search（已验证返回 invalid_parameter），资讯采集改用同 endpoint 下支持 enable_search 的 qwen3.7-max，连接方式不变，仅模型与 enable_search 不同。
- 落盘参考 llm_tasks 的 per-stock JSON 模式：core/alpha_db/news/{date}.json（每日一份，列表追加）。
