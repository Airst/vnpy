# 开放问题清单

> 识别但本轮不解决的问题记入此处，避免遗忘或误塞进当前轮。每条标注来源轮次与状态。

## 问题列表

### glm-5.2 不支持 web search
- **来源轮次**：News v1 / 2026-07-08
- **现象**：gp llm 默认模型 glm-5.2，开启 enable_search 返回 `This model does not support enable_search.`
- **初步归因**：glm-5.2 走百炼 token-plan 代理，未挂搜索插件。
- **去向**：资讯采集改用同 endpoint 下 qwen3.7-max + enable_search（已验证可用），连接方式不变。已转化为本轮方案，非开放问题。
- **状态**：已解决

### LLM 报告的日期漂移
- **来源轮次**：News v1 / 2026-07-08
- **现象**：web search 模型偶尔把"今天"报成训练截止日或错日期。
- **初步归因**：搜索结果时间戳不一致。
- **拟探索方向**：prompt 显式注入 collect_date 并要求 info_date 不晚于 collect_date；后端校验。
- **状态**：已解决（验证 14/14 条 info_date ≤ 今日，11/14 为采集日前一日）

### 板块名与 tushare 概念口径不一致
- **来源轮次**：News v1
- **现象**：LLM 说的"半导体设备"在 dc_member 里可能是"半导体概念"/"半导体设备"等多个概念。
- **拟探索方向**：模糊匹配 + 多概念并集；映射失败不阻断，mapped 字段留空。
- **状态**：已解决（改用 dc_concept API，其自带概念 name；模糊匹配 12/14 命中，含 concept_pct_change；dc_member 的 BK 码与 dc_concept 数字码不互通，故不用 dc_member 取成员，改由 stock_basic.industry + LLM 命名个股补足代表性标的）

### 运行端口被 sibling 工程看门狗占用
- **来源轮次**：News v1 / 2026-07-08
- **现象**：8000 端口被 /home/airst/Workspace/vnpy/main.py（sibling 工程）占用，start_vnpy_rs.sh 看门狗会在被杀后自重启，导致 vnpy2 无法绑定 8000。
- **拟探索方向**：本期验证用 8001 端口。用户在主端口使用需先停 sibling 看门狗。
- **状态**：待办（环境层面，非功能问题）

### （示例，可删除）
### GP 因子信号空间饱和
- **来源轮次**：V15 系列
- **现象**：新一轮 GP 候选大量集中于已有信号维度变体，无增量信息
- **拟探索方向**：GP 算子扩展（财务数据终端）/ 更换搜索空间
- **状态**：待办
