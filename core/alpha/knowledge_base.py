"""
LLM 因子挖掘的知识库构建模块

从 GP 注册表、迭代文档、模型 docstring、量化准则中提取结构化知识，
供 LLM 假设生成器检索和使用。
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class FactorKnowledge:
    """单个 GP 因子的结构化知识"""
    factor_id: str
    expr: str
    status: str
    ic: float = 0.0
    icir: float = 0.0
    turnover: float = 0.0
    note: str = ""
    discovered_at: str = ""
    information_dimension: str = ""
    reject_category: str = ""


@dataclass
class ModelKnowledge:
    """模型超参数与失败记录"""
    model_type: str = ""
    d_token: int = 0
    n_heads: int = 0
    n_attn_layers: int = 0
    d_ffn: int = 0
    train_window: int = 0
    valid_window: int = 0
    retrain_period: int = 0
    batch_size: int = 0
    lr: float = 0.0
    weight_decay: float = 0.0
    early_stop: int = 0
    seed: int = 0
    failures: list = field(default_factory=list)


@dataclass
class QuantCriteria:
    """量化研究准则"""
    criteria: list = field(default_factory=list)


@dataclass
class KnowledgeBase:
    """完整知识库"""
    factors: list = field(default_factory=list)
    model: ModelKnowledge = field(default_factory=ModelKnowledge)
    criteria: list = field(default_factory=list)
    covered_dimensions: list = field(default_factory=list)

    def summary(self) -> str:
        validated = [f for f in self.factors if f.status == "validated"]
        rejected = [f for f in self.factors if f.status == "rejected"]
        return (
            f"Factors: {len(self.factors)} total "
            f"({len(validated)} validated, {len(rejected)} rejected)\n"
            f"Model: {self.model.model_type}, d_token={self.model.d_token}, "
            f"window={self.model.train_window}+{self.model.valid_window}\n"
            f"Criteria: {len(self.criteria)} rules\n"
            f"Dimensions: {len(self.covered_dimensions)} covered"
        )

    def to_dict(self) -> dict:
        return {
            "factors": [asdict(f) for f in self.factors],
            "model": asdict(self.model),
            "criteria": self.criteria,
            "covered_dimensions": self.covered_dimensions,
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(f), f, ensure_ascii=False, indent=2)

    def validated_factors_text(self) -> str:
        lines = []
        for f in self.factors:
            if f.status == "validated":
                lines.append(
                    f"  {f.factor_id}: {f.expr}\n"
                    f"    IC={f.ic:.4f}, ICIR={f.icir:.4f}, "
                    f"turnover={f.turnover:.3f}\n"
                    f"    note: {f.note}"
                )
        return "\n".join(lines)

    def rejected_factors_text(self) -> str:
        lines = []
        for f in self.factors:
            if f.status == "rejected":
                lines.append(
                    f"  {f.factor_id}: {f.expr}\n"
                    f"    reject_reason: {f.note}"
                )
        return "\n".join(lines)


DIMENSION_MAP = {
    "gp_001": "换手率结构",
    "gp_002": "换手率结构",
    "gp_006": "资金流买卖",
    "gp_009": "资金流买卖",
    "gp_010": "资金流买卖",
    "gp_012": "资金流买卖",
    "gp_014": "资金流买卖",
    "gp_017": "超大单",
    "gp_018": "超大单",
    "gp_019": "超大单",
    "gp_038": "散户/中单",
    "gp_041": "估值×成交量",
    "gp_042": "估值×成交量",
    "gp_052": "估值×资金流",
    "gp_055": "BE波动+SMD+BL",
}

REJECT_CATEGORIES = {
    "同质化": "homogeneous",
    "dedup": "homogeneous",
    "无信息增益": "no_info_gain",
    "无增量": "no_info_gain",
    "过拟合": "overfitting",
    "attention稀释": "attention_dilution",
    "auto-rejected": "low_ic",
    "ICIR": "low_ic",
}


def _classify_reject(note: str) -> str:
    for keyword, category in REJECT_CATEGORIES.items():
        if keyword in note:
            return category
    return "other"


def build_factor_knowledge(registry_path: str) -> list:
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    factors = []
    for rec in registry.get("factors", []):
        dim = DIMENSION_MAP.get(rec["id"], "")
        note = rec.get("note", "")
        factors.append(FactorKnowledge(
            factor_id=rec["id"],
            expr=rec["expr"],
            status=rec["status"],
            ic=rec.get("mining_metrics", {}).get("ic", 0.0),
            icir=rec.get("mining_metrics", {}).get("icir", 0.0),
            turnover=rec.get("mining_metrics", {}).get("turnover", 0.0),
            note=note,
            discovered_at=rec.get("discovered_at", ""),
            information_dimension=dim,
            reject_category=_classify_reject(note) if rec["status"] == "rejected" else "",
        ))
    return factors


def build_model_knowledge() -> ModelKnowledge:
    model = ModelKnowledge(
        model_type="factor_attention",
        d_token=64,
        n_heads=4,
        n_attn_layers=1,
        d_ffn=128,
        train_window=700,
        valid_window=100,
        retrain_period=45,
        batch_size=2048,
        lr=0.001,
        weight_decay=0.002,
        early_stop=40,
        seed=42,
        failures=[
            "IC-Loss: 改善非牛市但损害牛市(Sharpe 1.10→0.70)",
            "混合损失(MSE+IC): 梯度冲突，效果最差",
            "多任务学习(1d/10d/20d): Sharpe 1.20→0.86，辅助损失梯度冲突",
            "2层Attention: 过拟合，不如1层",
            "d_ffn 128→256: Sharpe 1.36→1.01，FFN加大导致过拟合",
            "Input feature dropout(0.15): Sharpe 1.36→0.96",
            "3-seed ensemble跨重训不稳定，已回退为单次训练",
            "400天训练窗口: 丧失regime多样性导致Q4暴跌",
            "30天重训周期: Sharpe 1.79→1.36",
            "Gate Network: Sharpe 1.76→0.84，功能冗余+过拟合",
        ],
    )
    return model


def build_criteria_list() -> list:
    return [
        "因子理论支撑：新增因子必须有量化研究理论支撑",
        "结构性提升：补全当前因子体系的结构性缺失",
        "禁止唯IC论：IC绝对值不能作为评估因子有效性的唯一标准",
        "因子与框架匹配性检查：引入学术因子前必须验证数据粒度/频率/调仓周期/选股宇宙匹配",
        "失败反思：因子效果不佳时先分析原因，而非首先回退",
        "不硬编码市场观点：让模型从原子因子中自主学习",
        "标签设计优先于因子工程",
        "交互因子方法成立但受底层因子约束",
        "学术因子不等于A股有效因子",
        "模型层与风控层职责分离",
        "参考知识库：迭代前必须阅读 docs/knowledge/",
        "损失函数改造是双刃剑：IC-Loss改善非牛市但损害牛市",
        "训练采样策略不可轻易改变",
        "因子有效性与损失函数深度耦合",
        "多任务学习在截面选股框架下失败（3次失败，该方向暂停）",
        "Factor Self-Attention有效：模型结构改变 > 损失函数改造",
        "Attention框架下弱因子重测需提供独立信息维度",
        "Gate Network在Factor Attention框架下失败",
        "动量崩溃检测在A股截面选股中无效",
        "GP因子需要去重：结构同质化因子稀释attention权重",
        "验证集长度影响early stopping可靠性：100天比50天更稳定",
        "Multi-seed ensemble显著降低OOS variance",
        "季频数据在日频截面框架下信号过滤效率低",
        "GP因子信号空间在当前算子体系下已趋于饱和",
        "执行层复杂仓位调节是Sharpe杀手：等权入场+score阈值清仓是最优简单规则",
        "A股牛市轮动反对集中持仓",
        "减半仓在A股日频框架下产生双重摩擦",
        "pyramid加仓与排名退出存在结构性互斥",
    ]


def build_covered_dimensions() -> list:
    return [
        {
            "name": "换手率结构",
            "factors": ["gp_001", "gp_002"],
            "description": "换手率的截面排名与标准化，捕捉流动性异常",
        },
        {
            "name": "资金流买卖",
            "factors": ["gp_006", "gp_009", "gp_010", "gp_012", "gp_014"],
            "description": "大单买卖的时序波动率与相关性，知情者交易信号",
        },
        {
            "name": "超大单",
            "factors": ["gp_017", "gp_018", "gp_019"],
            "description": "超大单买卖的时序协方差，机构建仓/减仓信号",
        },
        {
            "name": "散户/中单",
            "factors": ["gp_038"],
            "description": "中小单买卖行为差异，零售情绪反向信号",
        },
        {
            "name": "估值×成交量",
            "factors": ["gp_041", "gp_042"],
            "description": "PB估值与成交量/换手率的交互，价值反转信号",
        },
        {
            "name": "估值×资金流",
            "factors": ["gp_052"],
            "description": "PS估值与资金流的相关性，盈利质量维度",
        },
        {
            "name": "BE波动+SMD+BL组合",
            "factors": ["gp_055"],
            "description": "超大单卖出波动与中单卖出/大单买入的协方差",
        },
    ]


def build_knowledge_base(registry_path: Optional[str] = None) -> KnowledgeBase:
    if registry_path is None:
        registry_path = str(
            PROJECT_ROOT / "core" / "alpha" / "gp_factors.json"
        )

    factors = build_factor_knowledge(registry_path)
    model = build_model_knowledge()
    criteria = build_criteria_list()
    dimensions = build_covered_dimensions()

    kb = KnowledgeBase(
        factors=factors,
        model=model,
        criteria=criteria,
        covered_dimensions=dimensions,
    )
    return kb


if __name__ == "__main__":
    kb = build_knowledge_base()
    print(kb.summary())
    print("\n--- Validated Factors ---")
    print(kb.validated_factors_text()[:500])
    print("\n--- Rejected Factors (first 500 chars) ---")
    print(kb.rejected_factors_text()[:500])
