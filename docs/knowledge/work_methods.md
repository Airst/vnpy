# 工作方法

1. **坚守量化研究准则**：每次迭代决策前对照 `docs/knowledge/research_principles.md` 检查，避免重复踩已知的坑
2. **持续沉淀经验教训**：每轮迭代结束后，将新发现的规律更新到 `docs/knowledge/research_principles.md`
3. **代码自描述优先**：迭代信息写入代码文件 docstring（规范见 `docs/knowledge/code_docstring_spec.md`），而非塞进 AGENTS.md
4. **严格遵循迭代流程**：按 `docs/loop/process.md` 的 6 步闭环执行，不跳步、不合并多个方案到一轮迭代
5. **迭代失败时止步等待**：记录结论后询问用户是否回退，等待用户指令，不自行发起下一轮
6. **知识库驱动决策**：迭代前阅读 `docs/knowledge/` 和代码文件 docstring 中的失败记录
7. **闭环状态显式化**：每轮迭代在 `docs/loop/goals.md` 写清目标与验收标准，验证后在 `docs/loop/verification_log.md` 留痕，非本轮问题入 `docs/loop/problems.md`
