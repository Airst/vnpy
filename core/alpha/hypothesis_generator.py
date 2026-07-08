"""
LLM 因子假设生成器

使用 GLM 模型分析知识库，生成因子假设 + 模型超参数建议。
支持因子与模型联合优化（RD-Agent 核心能力的借用）。
"""

import json
import os
from typing import Optional

from openai import OpenAI

from core.alpha.knowledge_base import KnowledgeBase


VALID_TERMINALS = [
    "O", "H", "L", "C", "V", "TR",
    "BL", "SL", "BE", "SE", "NMF",
    "CM5", "CM20", "CMX", "CHR",
    "MR",
    "PE", "PB", "PS",
    "MV", "CMV",
    "TRF",
    "BSM", "SSM", "BMD", "SMD",
]

VALID_OPERATORS = {
    "unary_ts": ["ts_mean", "ts_std", "ts_max", "ts_min", "ts_delta", "ts_rank", "ts_decay_linear"],
    "binary_ts": ["ts_corr", "ts_cov"],
    "cross_sectional": ["cs_rank", "cs_zscore"],
    "unary_arith": ["abs", "neg", "log"],
    "binary_arith": ["add", "sub", "mul", "div"],
}

VALID_WINDOWS = [3, 5, 10, 20]


FACTOR_HYPOTHESIS_PROMPT = """你是一位 A 股量化因子研究员。基于以下知识库，提出 {n_candidates} 个新的因子假设。

## 知识库

### 已验证因子（{n_validated} 个）
{validated_factors}

### 失败记录（{n_rejected} 个，仅列前 20 个）
{rejected_factors}

### 已覆盖信息维度（{n_dimensions} 个）
{covered_dimensions}

### 当前模型配置
- 模型: Factor Attention (d_token={d_token}, n_heads={n_heads}, n_attn_layers={n_attn_layers}, d_ffn={d_ffn})
- 训练窗口: {train_window} 天 ({train_size}训练 + {valid_window}验证)
- 重训周期: {retrain_period} 天
- 损失函数: MSE
- 特点: Attention 自动实现动态因子加权

### 模型失败记录
{model_failures}

### 量化研究准则（前 15 条）
{criteria}

## 可用终端节点
{terminals}

## 可用算子
{operators}

窗口参数: {windows}

## 生成要求

1. 每个因子必须使用上述终端节点和算子
2. 表达式格式: `op1(arg1, arg2, window)` 嵌套，如 `ts_corr(add(SL, TR), cs_zscore(BL), 5)`
3. 禁止重复已 rejected 因子的结构模式（见失败记录）
4. 必须提供独立信息维度（与现有 validated 因子相关 < 0.5）
5. 需有经济理论支撑（知情者交易/行为金融/估值回归/板块共振等）
6. 窗口参数限 {{3, 5, 10, 20}}

## 因子+模型联合优化

除了因子假设，还可以提出模型超参数调整建议：
- d_token, n_heads, n_attn_layers, d_ffn 的调整
- 训练窗口/验证窗口/重训周期的调整
- 损失函数的选择（注意：准则 8/11 指出损失函数改造已 3 次失败）

## 输出格式

输出一个 JSON 对象，包含名为 'factors' 的键，其值为因子假设数组：
```json
{{
  "factors": [
    {{
      "factor_id": "llm_001",
      "expr": "ts_corr(sub(SMD, V), cs_zscore(BMD), 5)",
      "hypothesis": "中单卖出与大单买入行为差异",
      "theory": "行为金融学散户机构博弈",
      "independence": "与gp_038差异：引入V维度",
      "expected_dimension": "散户/中单",
      "is_new_dimension": false
    }}
  ]
}}
```

注意：hypothesis 和 theory 字段不超过 30 字。不输出 model_suggestion。
只输出 JSON 对象，不要其他文字。"""


MODEL_SUGGESTION_SECTION = """
## 模型超参数建议（可选）

当前模型配置：
- d_token: {d_token} (准则：32不足，64最优，128无额外收益)
- n_attn_layers: {n_attn_layers} (准则：2层过拟合，1层最优)
- d_ffn: {d_ffn} (准则：128→256导致过拟合)
- train_window: {train_window} (准则：短窗口丧失regime多样性)
- retrain_period: {retrain_period} (准则：30天失败，45天最优)

如果某个因子假设需要特定模型配置才能发挥效果，在 model_suggestion 中说明。
"""


class HypothesisGenerator:
    """LLM 假设生成器"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "glm-5.2",
    ):
        self.api_key = (
            api_key
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL")
            or "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model

        if not self.api_key:
            raise ValueError(
                "API key not set. "
                "For 百炼 DashScope: export DASHSCOPE_API_KEY=your_key "
                "(get from https://bailian.console.aliyun.com/)"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(
        self,
        kb: KnowledgeBase,
        n_candidates: int = 10,
    ) -> list:
        return self.generate_with_feedback(kb, n_candidates, feedback=[])

    def generate_with_feedback(
        self,
        kb: KnowledgeBase,
        n_candidates: int = 10,
        feedback: list = None,
    ) -> list:
        prompt = self._build_prompt(kb, n_candidates)

        if feedback:
            feedback_text = self._format_feedback(feedback)
            prompt += f"\n\n## 上一轮反馈\n{feedback_text}"
        else:
            prompt += "\n\n这是第一轮，没有历史反馈。"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一位 A 股量化因子研究员。"
                        "输出一个 JSON 对象，包含名为 'factors' 的键，值为因子假设数组。"
                        "每个因子的 hypothesis 和 theory 字段控制在 30 字以内。"
                        "不要输出多余文字。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=16384,
        )

        raw = response.choices[0].message.content
        return self._parse_response(raw)

    def _format_feedback(self, feedback: list) -> str:
        lines = []
        for fb in feedback:
            round_num = fb.get("round", "?")
            accepted = fb.get("accepted", 0)
            total = fb.get("translated", fb.get("total_hypotheses", 0))
            lines.append(f"### 第 {round_num} 轮结果")
            lines.append(f"- 候选: {total}, 通过: {accepted}, 拒绝: {total - accepted}")

            if fb.get("result") == "no_valid_expressions":
                lines.append(f"- 问题: {fb.get('detail', '表达式解析失败')}")
                lines.append("- 建议: 严格使用可用算子和终端节点格式")

            new_exprs = fb.get("new_factor_exprs", [])
            if new_exprs:
                lines.append(f"- 新发现因子:")
                for expr in new_exprs:
                    lines.append(f"  - {expr}")
            else:
                lines.append("- 本轮无新因子通过验证")
                lines.append("- 请尝试完全不同的信息维度和算子组合")

        lines.append("\n### 下一轮要求")
        lines.append("- 避免重复已尝试的结构模式")
        lines.append("- 优先探索未覆盖的信息维度")
        lines.append("- 确保与已验证因子的相关性 < 0.5")

        return "\n".join(lines)

    def _build_prompt(self, kb: KnowledgeBase, n_candidates: int) -> str:
        validated = kb.validated_factors_text()
        rejected = kb.rejected_factors_text()[:2000]
        dimensions_text = "\n".join(
            f"  {i+1}. {d['name']}: {', '.join(d['factors'])} — {d['description']}"
            for i, d in enumerate(kb.covered_dimensions)
        )
        criteria_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(kb.criteria[:15]))
        model_failures_text = "\n".join(f"  - {f}" for f in kb.model.failures)
        terminals_text = ", ".join(VALID_TERMINALS)
        operators_text = "\n".join(
            f"  {k}: {', '.join(v)}" for k, v in VALID_OPERATORS.items()
        )
        windows_text = str(VALID_WINDOWS)

        return FACTOR_HYPOTHESIS_PROMPT.format(
            n_candidates=n_candidates,
            n_validated=len([f for f in kb.factors if f.status == "validated"]),
            n_rejected=len([f for f in kb.factors if f.status == "rejected"]),
            n_dimensions=len(kb.covered_dimensions),
            validated_factors=validated,
            rejected_factors=rejected,
            covered_dimensions=dimensions_text,
            d_token=kb.model.d_token,
            n_heads=kb.model.n_heads,
            n_attn_layers=kb.model.n_attn_layers,
            d_ffn=kb.model.d_ffn,
            train_window=kb.model.train_window,
            train_size=kb.model.train_window - kb.model.valid_window,
            valid_window=kb.model.valid_window,
            retrain_period=kb.model.retrain_period,
            model_failures=model_failures_text,
            criteria=criteria_text,
            terminals=terminals_text,
            operators=operators_text,
            windows=windows_text,
        )

    def _parse_response(self, raw: str) -> list:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                if "factors" in data:
                    return data["factors"]
                if "hypotheses" in data:
                    return data["hypotheses"]
                return [data] if "expr" in data else list(data.values())
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            start = raw.find("[")
            end = raw.rfind("]")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    pass
            print(f"[HypothesisGen] Failed to parse response:\n{raw[:500]}")
            return []
