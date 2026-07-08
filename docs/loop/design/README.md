# 设计方案目录

每个设计方案单独成文，文件名 `{版本}_{主题}.md`（如 `v15_5b_dedup.md`、`v18_dynamic_position.md`）。

## 设计文档模板

```markdown
# {版本} — {主题}

## 背景
（来自 Step 1 的问题识别，链接 verification_log / problems）

## 方案
- **改什么**：
- **为什么有效**：（理论支撑 / 知识库依据）
- **预期效果**：
- **验证标准**：（可量化）

## 改动范围
（文件 / 模块 / 因子 / 参数）

## 风险与副作用
（对照 docs/knowledge/research_principles.md 检查是否违反已有结论）

## 结果
（验证后回填：通过/失败，关键数据，去向）
```

## 与其他文档的关系

- **设计方案**记"为什么这么做"——本目录。
- **实验数据**记"做出来怎样"——`docs/iterations/{版本}.md`。
- **验证判定**记"结论是什么"——`docs/loop/verification_log.md`。
- **沉淀规律**记"学到什么"——`docs/knowledge/research_principles.md`。
