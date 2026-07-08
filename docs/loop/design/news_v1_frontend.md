# News v1 — 资讯展示页面（前端）

## 背景
后端产出结构化板块资讯（见 news_v1_collect.md）。需要一个前端页面让 A 股交易者据此判断当日板块强弱/轮动/情绪。

## 方案
- 新增 `core/web_ui/src/components/NewsDashboard.jsx`。
- 在 `App.jsx` 增加 `/news` 路由与侧边栏菜单项（图标 `ReadOutlined`/`NotificationOutlined`）。
- 复用现有 AntD + recharts 技术栈与布局风格。

### 页面结构
- 顶栏：采集日期选择、板块过滤、情绪过滤、`立即采集` 按钮、采集状态提示。
- 资讯卡片列表（按 timeliness + info_date 排序）：
  - 头部：板块 Tag + 情绪色标（利好绿/利空红/中性灰）+ 时效性徽标 + info_date。
  - 标题 + 摘要。
  - 影响分析（impact）+ 轮动含义（rotation）。
  - 关联板块 Tag 群 + 代表性个股（mapped_stocks，可点击后续跳个股——本期仅展示）。
- 空状态/加载态/错误态。

### 数据来源
- `GET /api/news` → 列表
- `GET /api/news/sectors` → 板块过滤下拉
- `POST /api/news/collect` → 立即采集
- `GET /api/news/status` → 轮询采集进度

## 改动范围
- 新增：`core/web_ui/src/components/NewsDashboard.jsx`
- 修改：`core/web_ui/src/App.jsx`（路由 + 菜单）
- 构建：`cd core/web_ui && npm run build`

## 风险与副作用
- 仅新增页面，不改动既有路由组件。
- 构建产物 dist 需更新，后端 StaticFiles 自动加载。

## 验证标准
- 访问 /news 渲染真实资讯卡片，过滤生效，立即采集可触发并刷新。
- 验收：交易者看完能说出当日强势板块、轮动方向、情绪。

## 结果
通过（2026-07-08 验证）。headless chrome 渲染 /news，DOM 与截图确认：顶栏过滤（采集日期/板块/情绪 + 立即采集）、汇总（资讯条数 6、利好 5、利空 0、中性 1）、资讯卡片（板块 Tag、情绪色标、高时效徽标、标题、摘要、影响分析、轮动含义、关联板块、代表性个股 vt_symbol）。验收：交易者可据此判断板块强弱/轮动方向/情绪。详见 verification_log.md。
