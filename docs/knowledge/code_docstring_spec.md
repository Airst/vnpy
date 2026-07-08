# 代码自描述规范

每个核心代码文件的头部必须包含模块级 docstring，格式如下：

```python
"""
<文件标题>

== 版本演进 ==
V8 → V9: <变更摘要>
V9 → V10: <变更摘要>
...

== 当前状态 ==
<当前版本、关键配置、因子数、指标等>

== 设计决策 ==
- <为什么选择这个方案>
- <关键参数的选择理由>

== 失败记录 ==
- <失败实验1>: <原因简述>
- <失败实验2>: <原因简述>
"""
```

## 规则

1. 每次迭代修改代码文件时，必须同步更新该文件的模块级 docstring
2. 失败记录必须保留，避免后续重复尝试已证明无效的方向
3. docstring 应简洁，每条记录控制在 1~2 行；详细实验数据放 `docs/iterations/`
4. 需要维护 docstring 的核心文件：`v*_factor_calculator.py`、`mlp_signals.py`、`mlp_model.py`、`engine.py`、`risk_controller.py`、`multifactor_strategy.py`

## 与文档空间的关系

- docstring 记**单文件的演进与失败**——贴近代码，便于修改时即时查阅。
- `docs/iterations/{版本}.md` 记**单版本的全量实验数据**——跨文件的综合视图。
- `docs/loop/verification_log.md` 记**跨版本的验证判定流水**。
- `docs/knowledge/research_principles.md` 记**提炼后的硬约束准则**。
- 失败实验若提炼为通用准则，应同时出现在代码 docstring（贴近现场）与本文件引用的 research_principles（全局约束）中。
