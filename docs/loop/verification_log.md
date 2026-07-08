# 验证记录流水

> 按时间倒序记录每轮验证（Step 5）的输入/输出/判定。最新在最上。单次回测结论不可信（随机种子敏感），看趋势。

## 记录格式

```
## YYYY-MM-DD vX.{子版本} — {一句话主题}
- **基线**：Sharpe / 年化 / MaxDD
- **本次**：Sharpe / 年化 / MaxDD
- **非牛市对比**：基线 vs 本次
- **判定**：通过 / 失败
- **结论**：保留 / 回退 + 一句话原因
- **去向**：沉淀到 knowledge/... 或 problems/...
- **关联**：design/xxx.md, iterations/vX_xxx.md, commit hash
```

---

<!-- 新记录追加在此分隔线下方 -->

## 2026-07-08 News v1 — 股票资讯采集与展示（端到端验收）
- **基线**：无（新功能，首次构建）
- **本次**：定时任务 + LLM 联网采集 + 板块/个股映射 + FastAPI 接口 + 前端资讯页面
- **验证输入**：
  - 采集器：core/llm/news_collector.py（gp llm 连接：OpenAI SDK + 百炼 + DASHSCOPE_API_KEY；模型 qwen3.7-max + enable_search；glm-5.2 不支持 search 已记录）
  - 接口：/api/news、/api/news/sectors、/api/news/dates、/api/news/history、/api/news/status、POST /api/news/collect
  - 前端：core/web_ui/src/components/NewsDashboard.jsx + App.jsx /news 路由
  - 定时：main_controller news_scheduler（09:00 / 15:30，已注册并启动）
- **验证输出**：
  - 手动 POST /api/news/collect → 后台线程 → LLM web search → 映射 → 落盘。status 轮询：running→false，last_count=8，message="采集完成: ok"
  - 落盘：core/alpha_db/news/2026-07-08.json，14 条（6+8 合并去重）
  - info_date 校验：14/14 ≤ 今日；11/14 为昨日(2026-07-07)，时效达标
  - 板块映射：12/14 命中 dc_concept（含 concept_pct_change）；14/14 含 mapped_stocks（真实 vt_symbol）
  - 情绪分布：利好10 / 利空2 / 中性2（覆盖负面，非只报喜）
  - 时效：high 10 / medium 4
  - 前端：headless chrome 渲染 /news 页面，DOM 含"立即采集/资讯条数/利好/利空/代表性个股/板块当日涨跌幅"及真实板块名；截图确认卡片渲染（板块 Tag、情绪色标、高时效徽标、标题、摘要、影响分析、轮动含义、关联板块、代表性个股 vt_symbol）
- **判定**：通过
- **结论**：保留。功能满足"可服务于一个 A 股投资交易者"的验收：交易者打开页面可看到当日影响板块/轮动/情绪的资讯、板块当日涨跌幅、领涨股与代表性标的，并据此判断板块强弱与轮动方向。
- **去向**：problems.md 三项（glm 不支持 search / 日期漂移 / 板块口径）均标记解决或兜底达标；不沉淀回测知识（非量化因子迭代）。
- **关联**：design/news_v1_collect.md, design/news_v1_frontend.md, goals.md
- **备注**：运行环境——同机 8000 端口被 sibling 工程 /home/airst/Workspace/vnpy 的 start_vnpy_rs.sh 看门狗占用并自重启；验证用 8001 端口跑 vnpy2。用户若要在主端口 8000 使用，需停掉该看门狗与 sibling 服务后用 main.py 启动 vnpy2。

（待填）
