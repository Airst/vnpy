"""
遗传编程(GP)因子挖掘模块 — 自动发现短周期 Alpha 因子

== 入口脚本 ==
独立运行: python gp_mining.py -v{版本号} (支持 --pop/--gen/--max-factors 等超参数)
注册表管理: python gp_mining.py -v{版本号} --status/--accept/--reject/--test/--note

== 设计目标 ==
补充当前慢因子体系（120d/60d/20d 为主），发现 3-20 天衰减的短周期因子，
增加模型信号的时效性和变异性，缓解"分数横线→入场时机差"问题。

== 因子生命周期管理 ==
JSON 注册表 (v2 schema) 管理因子全生命周期：
- discovered: 新挖掘，待验证
- testing: 正在验证中（已进入训练流程）
- validated: 验证通过，参与模型训练
- rejected: 验证失败，永不使用，但保留用于去重

挖掘增量追加，不覆盖已有因子。rejected 因子的表达式参与去重防止重复挖掘。

== 设计决策 ==
- 复用 factor_calculator.py 中的 GPU 张量算子（ts_mean/ts_std/cs_rank 等）
- 适应度函数: Rank IC (Spearman) 在滚动 OOS 窗口上评估
- 窗口参数限制在 {3, 5, 10, 20} 强制短周期特征
- 相关性去重: 与现有因子池相关 > 0.7 的因子丢弃
- 复杂度惩罚 (parsimony): 树深度 > 5 的表达式被惩罚
- GPU 加速因子计算
- 向量化 Rank IC: 全矩阵 argsort 替代逐天循环
- 并行 fitness: ThreadPoolExecutor 8线程并行评估

== 算子体系 ==
终端节点 (Terminals): O, H, L, C, V, TR (量价原子数据)
                      BL, SL, BE, SE, NMF (资金流向 — 知情者交易信号)
                      CM5, CM20, CMX, CHR (概念板块 — 板块共振信号)
                      MR (市场收益 — 逆大盘/beta不对称信号)
                      PE, PB, PS (估值因子)
                      MV, CMV (市值 — 总市值/流通市值)
                      TRF (自由流通换手率)
                      BSM, SSM, BMD, SMD (散户/中单委托 — 零售行为信号)
一元时序 (Unary TS): ts_mean, ts_std, ts_max, ts_min, ts_delta, ts_rank, ts_decay_linear
二元时序 (Binary TS): ts_corr, ts_cov
截面 (Cross-Sectional): cs_rank, cs_zscore
算术 (Arithmetic): add, sub, mul, div, abs, neg, log
窗口参数 (Windows): 3, 5, 10, 20
"""

import os
import random
import copy
import json
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import torch

from core.alpha.factor_calculator import (
    device, ts_mean, ts_std, ts_max, ts_min, ts_delta, ts_rank,
    ts_decay_linear, ts_corr, ts_cov, cs_rank, cs_zscore,
    ts_sum, ts_argmax, ts_argmin, ts_slope, ts_abs
)


# ============================================================
# Expression Tree Representation
# ============================================================

WINDOWS = [3, 5, 10, 20]
TERMINALS = ['O', 'H', 'L', 'C', 'V', 'TR',
             'BL', 'SL', 'BE', 'SE', 'NMF',  # MoneyFlow: 大单/超大单/净流
             'CM5', 'CM20', 'CMX', 'CHR',  # Concept: 板块动量/热度
             'MR',  # Market Return: 大盘日收益
             'PE', 'PB', 'PS',  # 估值: 市盈率/市净率/市销率
             'MV', 'CMV',  # 市值: 总市值/流通市值
             'TRF',  # 自由流通换手率
             'BSM', 'SSM', 'BMD', 'SMD']  # 散户/中单: 小单买卖/中单买卖

@dataclass
class Node:
    """Expression tree node."""
    op: str
    children: List['Node'] = field(default_factory=list)
    value: Optional[str] = None  # For terminal nodes
    window: Optional[int] = None  # For ts operators

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def to_expr(self) -> str:
        """Convert tree to human-readable expression string."""
        if self.op == 'terminal':
            return self.value
        if self.op in UNARY_ARITHMETIC:
            return f"{self.op}({self.children[0].to_expr()})"
        if self.op in BINARY_ARITHMETIC:
            return f"{self.op}({self.children[0].to_expr()}, {self.children[1].to_expr()})"
        if self.op in UNARY_TS:
            return f"{self.op}({self.children[0].to_expr()}, {self.window})"
        if self.op in BINARY_TS:
            return f"{self.op}({self.children[0].to_expr()}, {self.children[1].to_expr()}, {self.window})"
        if self.op in CROSS_SECTIONAL:
            return f"{self.op}({self.children[0].to_expr()})"
        return f"UNKNOWN({self.op})"

    def clone(self) -> 'Node':
        return copy.deepcopy(self)


# Operator categories
UNARY_TS = ['ts_mean', 'ts_std', 'ts_max', 'ts_min', 'ts_delta', 'ts_rank', 'ts_decay_linear']
BINARY_TS = ['ts_corr', 'ts_cov']
CROSS_SECTIONAL = ['cs_rank', 'cs_zscore']
UNARY_ARITHMETIC = ['abs', 'neg', 'log']
BINARY_ARITHMETIC = ['add', 'sub', 'mul', 'div']

ALL_OPS = UNARY_TS + BINARY_TS + CROSS_SECTIONAL + UNARY_ARITHMETIC + BINARY_ARITHMETIC


def _arity(op: str) -> int:
    if op in UNARY_TS or op in CROSS_SECTIONAL or op in UNARY_ARITHMETIC:
        return 1
    if op in BINARY_TS or op in BINARY_ARITHMETIC:
        return 2
    return 0


# ============================================================
# Random Tree Generation
# ============================================================

def random_terminal() -> Node:
    return Node(op='terminal', value=random.choice(TERMINALS))


def random_tree(max_depth: int = 4, current_depth: int = 0) -> Node:
    """Generate a random expression tree with grow method."""
    if current_depth >= max_depth - 1:
        return random_terminal()

    # Bias towards operators at shallow depths, terminals at deep
    if current_depth == 0 or random.random() < 0.7:
        op = random.choice(ALL_OPS)
        arity = _arity(op)
        children = [random_tree(max_depth, current_depth + 1) for _ in range(arity)]
        window = random.choice(WINDOWS) if op in UNARY_TS + BINARY_TS else None
        return Node(op=op, children=children, window=window)
    else:
        return random_terminal()


# ============================================================
# Expression Evaluation (GPU)
# ============================================================

# Operator dispatch table
_TS_DISPATCH = {
    'ts_mean': ts_mean,
    'ts_std': ts_std,
    'ts_max': ts_max,
    'ts_min': ts_min,
    'ts_delta': ts_delta,
    'ts_rank': ts_rank,
    'ts_decay_linear': ts_decay_linear,
}

_TS_BINARY_DISPATCH = {
    'ts_corr': ts_corr,
    'ts_cov': ts_cov,
}

_CS_DISPATCH = {
    'cs_rank': cs_rank,
    'cs_zscore': cs_zscore,
}


def _safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Division with zero protection."""
    return a / (b + 1e-8 * torch.sign(b + 1e-10))


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    """Log with negative protection."""
    return torch.log(torch.abs(x) + 1e-8)


_ARITH_DISPATCH = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': _safe_div,
    'abs': lambda x: torch.abs(x),
    'neg': lambda x: -x,
    'log': _safe_log,
}


def evaluate_tree(node: Node, data: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Evaluate expression tree on GPU tensors.
    
    Args:
        node: Expression tree root
        data: Dict mapping terminal names to (num_stocks, max_len) tensors
    
    Returns:
        (num_stocks, max_len) tensor or None if evaluation fails
    """
    try:
        if node.op == 'terminal':
            return data[node.value]

        if node.op in UNARY_TS:
            child_val = evaluate_tree(node.children[0], data)
            if child_val is None:
                return None
            return _TS_DISPATCH[node.op](child_val, node.window)

        if node.op in BINARY_TS:
            left = evaluate_tree(node.children[0], data)
            right = evaluate_tree(node.children[1], data)
            if left is None or right is None:
                return None
            return _TS_BINARY_DISPATCH[node.op](left, right, node.window)

        if node.op in CROSS_SECTIONAL:
            child_val = evaluate_tree(node.children[0], data)
            if child_val is None:
                return None
            return _CS_DISPATCH[node.op](child_val)

        if node.op in UNARY_ARITHMETIC:
            child_val = evaluate_tree(node.children[0], data)
            if child_val is None:
                return None
            return _ARITH_DISPATCH[node.op](child_val)

        if node.op in BINARY_ARITHMETIC:
            left = evaluate_tree(node.children[0], data)
            right = evaluate_tree(node.children[1], data)
            if left is None or right is None:
                return None
            return _ARITH_DISPATCH[node.op](left, right)

        return None

    except Exception:
        return None


def _is_valid_tensor(x: Optional[torch.Tensor]) -> bool:
    """Check if tensor is valid (no all-NaN, no all-same, no inf)."""
    if x is None:
        return False
    if torch.isinf(x).any():
        return False
    valid_mask = ~torch.isnan(x)
    if valid_mask.sum() < x.numel() * 0.3:  # >70% NaN
        return False
    valid_vals = x[valid_mask]
    if valid_vals.std() < 1e-10:  # constant
        return False
    return True


# ============================================================
# Fitness Evaluation
# ============================================================

def compute_rank_ic(factor: torch.Tensor, label: torch.Tensor, eval_days: int = 0) -> Tuple[float, float, float]:
    """
    Compute daily Rank IC (Spearman correlation) between factor and label.
    Vectorized GPU implementation — no Python per-day loop.
    
    Args:
        factor: (num_stocks, num_days) tensor
        label: (num_stocks, num_days) tensor (forward returns)
        eval_days: If >0, only use last N days. 0 = use all days.
    
    Returns:
        (mean_ic, ic_ir, direction_ratio) tuple
    """
    num_stocks, num_days = factor.shape

    # Sample every 10th day for speed
    start_t = max(0, num_days - eval_days) if eval_days > 0 else 0
    step = 10
    f_sampled = factor[:, start_t::step]  # (stocks, n_eval)
    l_sampled = label[:, start_t::step]

    # Valid mask: both factor and label are not NaN
    valid = ~(torch.isnan(f_sampled) | torch.isnan(l_sampled))  # (stocks, n_eval)
    valid_counts = valid.sum(dim=0)  # (n_eval,)

    # Need at least 30 valid stocks per day, and at least 10 valid days total
    day_mask = valid_counts >= 30  # (n_eval,)
    n_valid_days = day_mask.sum().item()
    if n_valid_days < 10:
        return 0.0, 0.0, 0.0

    # Replace invalid with NaN for ranking, then fill with 0 rank contribution
    f_filled = f_sampled.clone()
    l_filled = l_sampled.clone()
    f_filled[~valid] = float('-inf')  # will get rank 0
    l_filled[~valid] = float('-inf')

    # Rank along stock dimension (dim=0): (stocks, n_eval)
    f_rank = f_filled.argsort(dim=0).argsort(dim=0).float()
    l_rank = l_filled.argsort(dim=0).argsort(dim=0).float()

    # Zero out invalid entries so they don't contribute
    f_rank[~valid] = 0.0
    l_rank[~valid] = 0.0

    # Demean per day (only over valid entries)
    f_sum = f_rank.sum(dim=0)  # (n_eval,)
    l_sum = l_rank.sum(dim=0)
    f_mean = f_sum / (valid_counts.float() + 1e-8)  # (n_eval,)
    l_mean = l_sum / (valid_counts.float() + 1e-8)

    f_centered = f_rank - f_mean.unsqueeze(0)  # broadcast (stocks, n_eval)
    l_centered = l_rank - l_mean.unsqueeze(0)
    f_centered[~valid] = 0.0
    l_centered[~valid] = 0.0

    # IC per day = dot(f, l) / (norm(f) * norm(l))
    numerator = (f_centered * l_centered).sum(dim=0)  # (n_eval,)
    f_norm = (f_centered ** 2).sum(dim=0).sqrt()
    l_norm = (l_centered ** 2).sum(dim=0).sqrt()
    denom = f_norm * l_norm
    denom = torch.clamp(denom, min=1e-10)

    ics_tensor = numerator / denom  # (n_eval,)

    # Apply day_mask
    ics_valid = ics_tensor[day_mask].cpu().numpy()

    mean_ic = float(np.mean(ics_valid))
    std_ic = float(np.std(ics_valid)) + 1e-8
    ic_ir = mean_ic / std_ic

    # Sub-period direction consistency
    n_periods = min(5, len(ics_valid) // 5)
    if n_periods < 3:
        return mean_ic, ic_ir, 0.5

    chunk_size = len(ics_valid) // n_periods
    sign_overall = np.sign(mean_ic)
    same_sign_count = 0
    for i in range(n_periods):
        chunk = ics_valid[i * chunk_size: (i + 1) * chunk_size]
        chunk_mean = np.mean(chunk)
        if np.sign(chunk_mean) == sign_overall:
            same_sign_count += 1

    direction_ratio = same_sign_count / n_periods

    return mean_ic, ic_ir, direction_ratio


def compute_turnover(factor: torch.Tensor, top_pct: float = 0.1, eval_days: int = 200) -> float:
    """
    Compute average daily turnover of top-ranked stocks.
    High turnover = short-period factor (desirable for our goal).
    
    Returns:
        Average daily turnover ratio (0 to 1).
    """
    num_stocks, num_days = factor.shape
    start_t = max(0, num_days - eval_days)
    f_slice = factor[:, start_t::5]  # Sample every 5th day
    n_eval = f_slice.shape[1]
    
    turnovers = []
    prev_top_set = None
    
    for t in range(n_eval):
        f_col = f_slice[:, t]
        valid = ~torch.isnan(f_col)
        n_valid = valid.sum().item()
        if n_valid < 30:
            prev_top_set = None
            continue

        n_top = max(int(n_valid * top_pct), 1)
        top_indices = f_col.clone()
        top_indices[~valid] = float('-inf')
        _, top_idx = top_indices.topk(n_top)
        current_set = set(top_idx.cpu().tolist())

        if prev_top_set is not None:
            overlap = len(current_set & prev_top_set)
            turnover = 1.0 - overlap / max(len(current_set), 1)
            turnovers.append(turnover)

        prev_top_set = current_set

    return np.mean(turnovers) if turnovers else 0.0


def fitness(node: Node, data: Dict[str, torch.Tensor], label: torch.Tensor,
            existing_factors: Optional[List[torch.Tensor]] = None,
            max_corr: float = 0.7,
            recent_start_idx: int = 0) -> float:
    """
    Compute fitness score for a GP individual.
    
    Fitness = (|ICIR_full| * 0.4 + |ICIR_recent| * 0.6) * turnover_bonus * consistency_bonus - parsimony_penalty
    
    When recent_start_idx > 0, also evaluates IC on the recent period (e.g., 2026)
    and requires it to be non-negligible.
    
    Higher is better.
    """
    factor = evaluate_tree(node, data)
    if not _is_valid_tensor(factor):
        return -999.0

    mean_ic, ic_ir, direction_ratio = compute_rank_ic(factor, label)

    # Reject factors with negligible IC
    if abs(mean_ic) < 0.01:
        return -999.0

    # Reject factors with poor direction consistency (< 60% sub-periods agree)
    if direction_ratio < 0.6:
        return -999.0

    # Recent period IC (2026-focused)
    recent_ic_ir = 0.0
    if recent_start_idx > 0:
        num_days = label.shape[1]
        recent_eval_days = num_days - recent_start_idx
        if recent_eval_days >= 20:
            recent_ic, recent_ic_ir, recent_dir = compute_rank_ic(
                factor, label, eval_days=recent_eval_days
            )
            # Hard gate: recent IC must be non-negligible and same sign as full IC
            if abs(recent_ic) < 0.01:
                return -999.0
            if np.sign(recent_ic) != np.sign(mean_ic):
                return -999.0

    # Parsimony penalty
    depth = node.depth()
    size = node.size()
    parsimony = max(0, (depth - 5)) * 0.05 + max(0, (size - 12)) * 0.01

    # Correlation penalty with existing factors (check before expensive turnover)
    if existing_factors:
        sample_days = min(40, factor.shape[1])
        f_flat = factor[:, -sample_days:].reshape(-1)
        for ef in existing_factors:
            e_flat = ef[:, -sample_days:].reshape(-1)
            valid = ~(torch.isnan(f_flat) | torch.isnan(e_flat))
            if valid.sum() < 100:
                continue
            corr = torch.corrcoef(torch.stack([f_flat[valid], e_flat[valid]]))[0, 1].item()
            if abs(corr) > max_corr:
                return -999.0

    # Turnover bonus: reward short-period factors
    turnover = compute_turnover(factor)
    turnover_bonus = 1.0 + turnover  # Range [1.0, 2.0]

    # Direction consistency bonus (0.6->1.0, 0.8->1.2, 1.0->1.4)
    consistency_bonus = 1.0 + (direction_ratio - 0.6) * 1.0

    if recent_start_idx > 0 and recent_ic_ir != 0.0:
        combined_icir = abs(ic_ir) * 0.4 + abs(recent_ic_ir) * 0.6
    else:
        combined_icir = abs(ic_ir)
    score = combined_icir * turnover_bonus * consistency_bonus - parsimony
    return score


# ============================================================
# Genetic Operations
# ============================================================

def _get_all_nodes(node: Node) -> List[Tuple[Node, Optional[Node], int]]:
    """Get all nodes with parent info. Returns [(node, parent, child_idx)]."""
    result = [(node, None, -1)]
    for i, child in enumerate(node.children):
        result.append((child, node, i))
        for sub_node, sub_parent, sub_idx in _get_all_nodes(child):
            if sub_parent is not None:
                result.append((sub_node, sub_parent, sub_idx))
    return result


def crossover(parent1: Node, parent2: Node) -> Tuple[Node, Node]:
    """Subtree crossover between two parents."""
    child1 = parent1.clone()
    child2 = parent2.clone()

    nodes1 = _get_all_nodes(child1)
    nodes2 = _get_all_nodes(child2)

    if len(nodes1) < 2 or len(nodes2) < 2:
        return child1, child2

    # Pick random non-root nodes
    _, p1, idx1 = random.choice(nodes1[1:])
    _, p2, idx2 = random.choice(nodes2[1:])

    if p1 is None or p2 is None:
        return child1, child2

    # Swap subtrees
    temp = p1.children[idx1]
    p1.children[idx1] = p2.children[idx2]
    p2.children[idx2] = temp

    # Enforce depth limit
    if child1.depth() > 7:
        child1 = parent1.clone()
    if child2.depth() > 7:
        child2 = parent2.clone()

    return child1, child2


def mutate(node: Node, prob: float = 0.2) -> Node:
    """Point mutation: randomly replace a subtree."""
    child = node.clone()

    if random.random() > prob:
        return child

    nodes = _get_all_nodes(child)
    if len(nodes) < 2:
        return random_tree(max_depth=4)

    _, parent, idx = random.choice(nodes[1:])
    if parent is None:
        return random_tree(max_depth=4)

    # Replace with a new random subtree
    remaining_depth = 7 - parent.children[idx].depth()
    new_subtree = random_tree(max_depth=max(2, min(4, remaining_depth)))
    parent.children[idx] = new_subtree

    if child.depth() > 7:
        return node.clone()

    return child


def tournament_select(population: List[Tuple[Node, float]], k: int = 5) -> Node:
    """Tournament selection."""
    tournament = random.sample(population, min(k, len(population)))
    winner = max(tournament, key=lambda x: x[1])
    return winner[0].clone()


# ============================================================
# Main GP Mining Engine
# ============================================================

class GPFactorMiner:
    """
    Genetic Programming Factor Miner with lifecycle registry management.
    
    Discovers short-period alpha factors via evolutionary search on GPU tensors.
    Manages factor lifecycle: discovered → testing → validated/rejected.
    """

    VALID_STATUSES = ('discovered', 'testing', 'validated', 'rejected')

    def __init__(
        self,
        population_size: int = 500,
        n_generations: int = 50,
        tournament_size: int = 7,
        crossover_prob: float = 0.6,
        mutation_prob: float = 0.3,
        max_tree_depth: int = 5,
        min_ic: float = 0.02,
        min_icir: float = 0.3,
        max_corr_with_pool: float = 0.7,
        max_factors: int = 10,
        seed: int = 42,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_tree_depth = max_tree_depth
        self.min_ic = min_ic
        self.min_icir = min_icir
        self.max_corr_with_pool = max_corr_with_pool
        self.max_factors = max_factors
        self.seed = seed

        self.discovered_factors: List[Tuple[Node, float, float, float]] = []  # (tree, ic, icir, turnover)
        self._factor_ids: List[str] = []  # parallel to discovered_factors, stable IDs

    # ============================================================
    # Registry I/O (v2 schema with lifecycle management)
    # ============================================================

    @staticmethod
    def load_registry(path: str) -> dict:
        """Load the full registry from JSON, auto-migrating v1 format if needed."""
        p = Path(path)
        if not p.exists():
            return {"version": 2, "next_id": 1, "factors": []}
        with open(p) as f:
            data = json.load(f)
        # v1 detection: top-level is a list
        if isinstance(data, list):
            return GPFactorMiner._migrate_v1(data)
        return data

    @staticmethod
    def save_registry(path: str, registry: dict):
        """Atomically write registry to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)

    @staticmethod
    def _migrate_v1(records: list) -> dict:
        """Migrate v1 flat list to v2 registry. Existing factors become validated."""
        factors = []
        for i, rec in enumerate(records):
            factors.append({
                "id": f"gp_{i+1:03d}",
                "expr": rec["expr"],
                "tree": rec["tree"],
                "status": "validated",
                "mining_metrics": {
                    "ic": rec.get("ic", 0),
                    "icir": rec.get("icir", 0),
                    "turnover": rec.get("turnover", 0),
                },
                "discovered_at": "2026-05-08",
                "note": "migrated from v1",
            })
        return {"version": 2, "next_id": len(records) + 1, "factors": factors}

    def load(self, path: str, status_filter: Optional[List[str]] = None) -> bool:
        """
        Load factors from registry into discovered_factors for computation.
        
        Args:
            path: Path to gp_factors.json
            status_filter: If provided, only load factors with these statuses.
                          e.g. ["validated"] for production, ["validated", "testing"] for validation run.
                          None = load all (for mining dedup).
        """
        registry = self.load_registry(path)
        factors = registry.get("factors", [])
        if not factors:
            return False

        self.discovered_factors = []
        self._factor_ids = []
        for rec in factors:
            if status_filter and rec["status"] not in status_filter:
                continue
            tree = self._deserialize_tree(rec["tree"])
            metrics = rec.get("mining_metrics", {})
            self.discovered_factors.append((
                tree,
                metrics.get("ic", 0),
                metrics.get("icir", 0),
                metrics.get("turnover", 0),
            ))
            self._factor_ids.append(rec["id"])

        if self.discovered_factors:
            filter_desc = f" (filter={status_filter})" if status_filter else ""
            print(f"[GP Miner] Loaded {len(self.discovered_factors)} factors from {path}{filter_desc}")
            return True
        return False

    def append_discovered(self, path: str, new_factors: List[Tuple[Node, float, float, float]]):
        """
        Append newly mined factors to registry as 'discovered' status.
        Deduplicates against ALL existing factors (any status) by expression string.
        """
        registry = self.load_registry(path)
        existing_exprs = {f["expr"] for f in registry["factors"]}

        added = 0
        for tree, ic, icir, turnover in new_factors:
            expr = tree.to_expr()
            if expr in existing_exprs:
                continue
            fid = f"gp_{registry['next_id']:03d}"
            registry["next_id"] += 1
            registry["factors"].append({
                "id": fid,
                "expr": expr,
                "tree": self._serialize_tree(tree),
                "status": "discovered",
                "mining_metrics": {"ic": ic, "icir": icir, "turnover": turnover},
                "discovered_at": str(date.today()),
                "note": "",
            })
            existing_exprs.add(expr)
            added += 1

        self.save_registry(path, registry)
        print(f"[GP Miner] Appended {added} new factors (skipped {len(new_factors) - added} duplicates)")

    @staticmethod
    def set_status(path: str, ids: List[str], status: str, note: str = ""):
        """
        Change status of factors by ID.
        
        Args:
            path: Registry file path
            ids: List of factor IDs (e.g. ["gp_003", "gp_004"])
            status: Target status (discovered/testing/validated/rejected)
            note: Optional note to attach
        """
        if status not in GPFactorMiner.VALID_STATUSES:
            print(f"[GP Miner] Error: invalid status '{status}'. Valid: {GPFactorMiner.VALID_STATUSES}")
            return

        registry = GPFactorMiner.load_registry(path)
        id_set = set(ids)
        updated = 0
        for factor in registry["factors"]:
            if factor["id"] in id_set:
                factor["status"] = status
                if note:
                    factor["note"] = note
                updated += 1
                id_set.discard(factor["id"])

        if id_set:
            print(f"[GP Miner] Warning: IDs not found: {id_set}")
        if updated:
            GPFactorMiner.save_registry(path, registry)
            print(f"[GP Miner] Updated {updated} factors → status='{status}'")

    @staticmethod
    def print_status(path: str):
        """Print a summary of the factor registry."""
        registry = GPFactorMiner.load_registry(path)
        factors = registry.get("factors", [])

        if not factors:
            print("[GP Registry] Empty — no factors registered.")
            return

        # Group by status
        by_status: Dict[str, list] = {}
        for f in factors:
            by_status.setdefault(f["status"], []).append(f)

        print(f"\n[GP Registry] {len(factors)} factors total:")
        print("-" * 60)
        for status in ('validated', 'testing', 'discovered', 'rejected'):
            group = by_status.get(status, [])
            if not group:
                continue
            print(f"\n  {status.upper()} ({len(group)}):")
            for f in group:
                metrics = f.get("mining_metrics", {})
                ic = metrics.get("icir", 0)
                note = f.get("note", "")
                note_str = f"  | {note}" if note else ""
                print(f"    {f['id']}: {f['expr']:<45} ICIR={ic:.2f}{note_str}")
        print()

    # ============================================================
    # GP Evolution Engine
    # ============================================================

    def mine(
        self,
        padded_raw: torch.Tensor,
        col_map: Dict[str, int],
        label: torch.Tensor,
        existing_factor_tensors: Optional[List[torch.Tensor]] = None,
        registry_path: Optional[str] = None,
        recent_start_idx: int = 0,
    ) -> List[Tuple[str, Node]]:
        """
        Run GP evolution to discover new factors.
        
        Args:
            padded_raw: (num_stocks, max_len, num_features) GPU tensor
            col_map: Column name -> index mapping
            label: (num_stocks, max_len) forward return label
            existing_factor_tensors: List of existing factor tensors for deduplication
            registry_path: If provided, load all existing exprs for dedup (prevents re-mining rejected)
            
        Returns:
            List of (factor_name, expression_tree) tuples
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Load existing expressions for dedup (all statuses including rejected)
        self._existing_exprs: set = set()
        if registry_path:
            registry = self.load_registry(registry_path)
            self._existing_exprs = {f["expr"] for f in registry.get("factors", [])}
            if self._existing_exprs:
                print(f"[GP Miner] Loaded {len(self._existing_exprs)} existing expressions for dedup")

        # Prepare terminal data
        data = self._prepare_terminals(padded_raw, col_map)
        
        print(f"[GP Miner] Starting evolution: pop={self.population_size}, gen={self.n_generations}")
        print(f"[GP Miner] Data shape: {padded_raw.shape[0]} stocks, {padded_raw.shape[1]} days")
        print(f"[GP Miner] Targeting short-period factors (windows: {WINDOWS})")

        # Initialize population
        population = []
        init_trees = [random_tree(max_depth=self.max_tree_depth) for _ in range(self.population_size)]
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fitness, t, data, label, existing_factor_tensors, self.max_corr_with_pool, recent_start_idx): t for t in init_trees}
            for future in as_completed(futures):
                t = futures[future]
                population.append((t, future.result()))

        best_ever_score = -999.0
        stagnation = 0
        factor_pool = list(existing_factor_tensors) if existing_factor_tensors else []

        for gen in range(self.n_generations):
            # Sort by fitness
            population.sort(key=lambda x: x[1], reverse=True)
            gen_best = population[0][1]

            if gen_best > best_ever_score:
                best_ever_score = gen_best
                stagnation = 0
            else:
                stagnation += 1

            if gen % 10 == 0 or gen == self.n_generations - 1:
                valid_count = sum(1 for _, s in population if s > -999)
                print(f"[GP Miner] Gen {gen:3d}: best={gen_best:.4f}, "
                      f"valid={valid_count}/{self.population_size}, "
                      f"discovered={len(self.discovered_factors)}")

            # Check if top individual qualifies as a new factor
            if gen_best > 0:
                top_tree = population[0][0]
                self._try_add_factor(top_tree, data, label, factor_pool, recent_start_idx)

            # Early stopping
            if stagnation > 15:
                print(f"[GP Miner] Early stopping at gen {gen} (stagnation={stagnation})")
                break

            if len(self.discovered_factors) >= self.max_factors:
                print(f"[GP Miner] Reached max factors ({self.max_factors}), stopping.")
                break

            # Create next generation — batch generate offspring first
            new_population = []

            # Elitism: keep top 5%
            elite_size = max(5, self.population_size // 20)
            for i in range(elite_size):
                new_population.append(population[i])

            # Batch generate all offspring trees
            offspring_trees = []
            while len(offspring_trees) + len(new_population) < self.population_size:
                if random.random() < self.crossover_prob:
                    p1 = tournament_select(population, self.tournament_size)
                    p2 = tournament_select(population, self.tournament_size)
                    c1, c2 = crossover(p1, p2)
                    c1 = mutate(c1, self.mutation_prob)
                    c2 = mutate(c2, self.mutation_prob)
                    offspring_trees.append(c1)
                    if len(offspring_trees) + len(new_population) < self.population_size:
                        offspring_trees.append(c2)
                else:
                    p = tournament_select(population, self.tournament_size)
                    c = mutate(p, prob=0.5)
                    offspring_trees.append(c)

            # Parallel fitness evaluation
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(fitness, t, data, label, factor_pool, self.max_corr_with_pool, recent_start_idx): t for t in offspring_trees}
                for future in as_completed(futures):
                    t = futures[future]
                    new_population.append((t, future.result()))

            population = new_population

        # Final report
        print(f"\n[GP Miner] === Mining Complete ===")
        print(f"[GP Miner] Discovered {len(self.discovered_factors)} factors:")
        results = []
        for i, (tree, ic, icir, turnover) in enumerate(self.discovered_factors):
            name = f"gp_alpha_{i:03d}"
            expr = tree.to_expr()
            print(f"  {name}: IC={ic:.4f}, ICIR={icir:.4f}, turnover={turnover:.3f}")
            print(f"    expr: {expr}")
            results.append((name, tree))

        return results

    # ============================================================
    # Factor Computation (used during training)
    # ============================================================

    def compute_factors(
        self, padded_raw: torch.Tensor, col_map: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loaded GP factors on new data.
        Uses stable IDs from registry when available.
        
        Returns:
            Dict of factor_name -> (num_stocks, max_len) tensor
        """
        if not self.discovered_factors:
            return {}

        data = self._prepare_terminals(padded_raw, col_map)
        result = {}

        for i, (tree, _, _, _) in enumerate(self.discovered_factors):
            # Use stable ID if available, fallback to index-based name
            name = self._factor_ids[i] if i < len(self._factor_ids) else f"gp_alpha_{i:03d}"
            factor = evaluate_tree(tree, data)
            if _is_valid_tensor(factor):
                result[name] = factor

        return result

    # ============================================================
    # Internal Helpers
    # ============================================================

    def _prepare_terminals(self, padded_raw: torch.Tensor, col_map: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """Extract terminal tensors from padded raw data."""
        terminal_map = {
            'O': 'open',
            'H': 'high',
            'L': 'low',
            'C': 'close',
            'V': 'volume',
            'TR': 'turnover_rate',
            # MoneyFlow — 知情者交易信号
            'BL': 'buy_lg_amount',    # 大单买入金额
            'SL': 'sell_lg_amount',   # 大单卖出金额
            'BE': 'buy_elg_amount',   # 超大单买入金额
            'SE': 'sell_elg_amount',  # 超大单卖出金额
            'NMF': 'net_mf_amount',   # 净资金流入金额
            # Concept — 板块共振信号
            'CM5': 'concept_mom_5d',       # 概念板块5日动量
            'CM20': 'concept_mom_20d',     # 概念板块20日动量
            'CMX': 'concept_mom_20d_max',  # 最强概念板块动量
            'CHR': 'concept_hot_ratio',    # 概念热度比例
            # 估值
            'PE': 'pe',
            'PB': 'pb',
            'PS': 'ps',
            # 市值
            'MV': 'total_mv',
            'CMV': 'circ_mv',
            # 自由流通换手率
            'TRF': 'turnover_rate_f',
            # 散户/中单行为
            'BSM': 'buy_sm_amount',
            'SSM': 'sell_sm_amount',
            'BMD': 'buy_md_amount',
            'SMD': 'sell_md_amount',
        }
        data = {}
        for term_name, col_name in terminal_map.items():
            if col_name in col_map:
                data[term_name] = padded_raw[:, :, col_map[col_name]]
            else:
                data[term_name] = torch.zeros(
                    padded_raw.shape[0], padded_raw.shape[1],
                    device=padded_raw.device, dtype=padded_raw.dtype
                )

        # Computed terminal: Market Return (截面均值日收益, 广播到所有股票)
        C = data['C']
        C_prev = torch.roll(C, shifts=1, dims=1)
        C_prev[:, 0] = C[:, 0]
        ret_1 = C / (C_prev + 1e-8) - 1
        ret_1_clean = torch.nan_to_num(ret_1, nan=0.0)
        valid_mask = ~torch.isnan(ret_1)
        valid_cnt = valid_mask.sum(dim=0).clamp(min=1)
        mkt_ret = ret_1_clean.sum(dim=0) / valid_cnt  # (max_len,)
        data['MR'] = mkt_ret.unsqueeze(0).expand_as(C)  # 广播到每只股票

        return data

    def _try_add_factor(
        self, tree: Node, data: Dict[str, torch.Tensor],
        label: torch.Tensor, factor_pool: List[torch.Tensor],
        recent_start_idx: int = 0,
    ):
        """Try to add a factor to the discovered pool after validation."""
        # Expression deduplication (against current session + registry)
        expr_str = tree.to_expr()
        if expr_str in self._existing_exprs:
            return
        for existing_tree, _, _, _ in self.discovered_factors:
            if existing_tree.to_expr() == expr_str:
                return

        factor = evaluate_tree(tree, data)
        if not _is_valid_tensor(factor):
            return

        mean_ic, ic_ir, direction_ratio = compute_rank_ic(factor, label)

        # Quality gate
        if abs(mean_ic) < self.min_ic:
            return
        if abs(ic_ir) < self.min_icir:
            return
        if direction_ratio < 0.6:
            return

        # Recent period IC gate (2026-focused)
        if recent_start_idx > 0:
            num_days = label.shape[1]
            recent_eval_days = num_days - recent_start_idx
            if recent_eval_days >= 20:
                recent_ic, recent_icir, recent_dir = compute_rank_ic(
                    factor, label, eval_days=recent_eval_days
                )
                if abs(recent_ic) < 0.01:
                    return
                if np.sign(recent_ic) != np.sign(mean_ic):
                    return

        # Correlation check with discovered pool (stricter: 0.5 between GP factors)
        if factor_pool:
            sample_days = min(40, factor.shape[1])
            f_flat = factor[:, -sample_days:].reshape(-1)
            for ef in factor_pool:
                e_flat = ef[:, -sample_days:].reshape(-1)
                valid = ~(torch.isnan(f_flat) | torch.isnan(e_flat))
                if valid.sum() < 100:
                    continue
                corr = torch.corrcoef(torch.stack([f_flat[valid], e_flat[valid]]))[0, 1].item()
                if abs(corr) > 0.5:
                    return

        turnover = compute_turnover(factor)
        self.discovered_factors.append((tree, mean_ic, ic_ir, turnover))
        factor_pool.append(factor)
        print(f"[GP Miner] +++ New factor discovered! IC={mean_ic:.4f}, ICIR={ic_ir:.4f}, "
              f"dir={direction_ratio:.2f}, turnover={turnover:.3f}")
        print(f"           expr: {expr_str}")

    def _serialize_tree(self, node: Node) -> dict:
        return {
            'op': node.op,
            'value': node.value,
            'window': node.window,
            'children': [self._serialize_tree(c) for c in node.children],
        }

    def _deserialize_tree(self, d: dict) -> Node:
        children = [self._deserialize_tree(c) for c in d.get('children', [])]
        return Node(op=d['op'], children=children, value=d.get('value'), window=d.get('window'))

    # Legacy compatibility
    def save(self, path: str):
        """Legacy save — redirects to append_discovered for backward compat."""
        self.append_discovered(path, self.discovered_factors)

    # ============================================================
    # Automatic Validation Gate
    # ============================================================

    @staticmethod
    def validate_discovered(
        path: str,
        padded_raw: torch.Tensor,
        col_map: Dict[str, int],
        label: torch.Tensor,
        min_rolling_ic: float = 0.03,
        n_windows: int = 5,
        window_size: int = 200,
        recent_start_idx: int = 0,
    ) -> Tuple[List[str], List[str]]:
        """
        Auto-validate 'discovered' factors using rolling IC.
        
        Computes IC in multiple non-overlapping windows. Factors that pass
        the threshold in majority of windows are promoted to 'testing';
        others are rejected.
        
        Args:
            path: Registry JSON path
            padded_raw: (num_stocks, max_len, features) GPU tensor
            col_map: Column name -> index mapping
            label: (num_stocks, max_len) forward return label
            min_rolling_ic: Minimum |IC| to pass (default 0.03)
            n_windows: Number of rolling windows to evaluate
            window_size: Days per window
            
        Returns:
            (accepted_ids, rejected_ids) tuple
        """
        registry = GPFactorMiner.load_registry(path)
        discovered = [f for f in registry["factors"] if f["status"] == "discovered"]
        
        if not discovered:
            print("[GP Validate] No discovered factors to validate.")
            return [], []

        # Prepare terminal data
        terminal_map = {
            'O': 'open', 'H': 'high', 'L': 'low', 
            'C': 'close', 'V': 'volume', 'TR': 'turnover_rate',
            'BL': 'buy_lg_amount', 'SL': 'sell_lg_amount',
            'BE': 'buy_elg_amount', 'SE': 'sell_elg_amount',
            'NMF': 'net_mf_amount',
            'CM5': 'concept_mom_5d', 'CM20': 'concept_mom_20d',
            'CMX': 'concept_mom_20d_max', 'CHR': 'concept_hot_ratio',
            'PE': 'pe', 'PB': 'pb', 'PS': 'ps',
            'MV': 'total_mv', 'CMV': 'circ_mv',
            'TRF': 'turnover_rate_f',
            'BSM': 'buy_sm_amount', 'SSM': 'sell_sm_amount',
            'BMD': 'buy_md_amount', 'SMD': 'sell_md_amount',
        }
        data = {}
        for term_name, col_name in terminal_map.items():
            if col_name in col_map:
                data[term_name] = padded_raw[:, :, col_map[col_name]]
            else:
                data[term_name] = torch.zeros(
                    padded_raw.shape[0], padded_raw.shape[1],
                    device=padded_raw.device, dtype=padded_raw.dtype
                )

        # Computed terminal: Market Return
        C = data['C']
        C_prev = torch.roll(C, shifts=1, dims=1)
        C_prev[:, 0] = C[:, 0]
        ret_1 = C / (C_prev + 1e-8) - 1
        ret_1_clean = torch.nan_to_num(ret_1, nan=0.0)
        valid_mask = ~torch.isnan(ret_1)
        valid_cnt = valid_mask.sum(dim=0).clamp(min=1)
        mkt_ret = ret_1_clean.sum(dim=0) / valid_cnt
        data['MR'] = mkt_ret.unsqueeze(0).expand_as(C)

        num_days = padded_raw.shape[1]
        total_eval_days = n_windows * window_size
        if total_eval_days > num_days:
            n_windows = max(2, num_days // window_size)
            total_eval_days = n_windows * window_size

        print(f"[GP Validate] Validating {len(discovered)} factors "
              f"({n_windows} windows x {window_size} days, min |IC| = {min_rolling_ic})")

        accepted_ids = []
        rejected_ids = []

        for rec in discovered:
            tree = GPFactorMiner._deserialize_tree_static(rec["tree"])
            factor = evaluate_tree(tree, data)
            
            if not _is_valid_tensor(factor):
                rejected_ids.append(rec["id"])
                print(f"  {rec['id']}: REJECTED (invalid tensor)")
                continue

            # Compute IC in each window
            window_ics = []
            for w in range(n_windows):
                end_t = num_days - w * window_size
                start_t = end_t - window_size
                if start_t < 0:
                    break
                
                f_win = factor[:, start_t:end_t]
                l_win = label[:, start_t:end_t]
                
                # Compute daily IC within this window (sample every 5th day)
                ics = []
                for t in range(0, window_size, 5):
                    f_col = f_win[:, t]
                    l_col = l_win[:, t]
                    valid = ~(torch.isnan(f_col) | torch.isnan(l_col))
                    n_valid = valid.sum().item()
                    if n_valid < 30:
                        continue
                    f_valid = f_col[valid]
                    l_valid = l_col[valid]
                    f_rank = f_valid.argsort().argsort().float()
                    l_rank = l_valid.argsort().argsort().float()
                    f_rank = f_rank - f_rank.mean()
                    l_rank = l_rank - l_rank.mean()
                    denom = f_rank.norm() * l_rank.norm()
                    if denom < 1e-10:
                        continue
                    ic = (f_rank * l_rank).sum() / denom
                    ics.append(ic.item())
                
                if ics:
                    window_ics.append(np.mean(ics))

            if len(window_ics) < 2:
                rejected_ids.append(rec["id"])
                print(f"  {rec['id']}: REJECTED (insufficient data)")
                continue

            mean_ic = np.mean([abs(ic) for ic in window_ics])
            # Check: majority of windows must pass threshold
            pass_count = sum(1 for ic in window_ics if abs(ic) >= min_rolling_ic)
            pass_ratio = pass_count / len(window_ics)

            if mean_ic >= min_rolling_ic and pass_ratio >= 0.6:
                # 2026-specific IC gate
                if recent_start_idx > 0:
                    recent_eval_days = num_days - recent_start_idx
                    if recent_eval_days >= 20:
                        f_recent = factor[:, recent_start_idx:]
                        l_recent = label[:, recent_start_idx:]
                        recent_ics = []
                        for t in range(0, recent_eval_days, 5):
                            f_col = f_recent[:, t]
                            l_col = l_recent[:, t]
                            valid = ~(torch.isnan(f_col) | torch.isnan(l_col))
                            if valid.sum() < 30:
                                continue
                            f_valid = f_col[valid]
                            l_valid = l_col[valid]
                            f_rank = f_valid.argsort().argsort().float()
                            l_rank = l_valid.argsort().argsort().float()
                            f_rank = f_rank - f_rank.mean()
                            l_rank = l_rank - l_rank.mean()
                            denom = f_rank.norm() * l_rank.norm()
                            if denom < 1e-10:
                                continue
                            recent_ics.append(((f_rank * l_rank).sum() / denom).item())
                        if recent_ics:
                            recent_mean_ic = np.mean(recent_ics)
                            if abs(recent_mean_ic) < 0.02:
                                rejected_ids.append(rec["id"])
                                print(f"  {rec['id']}: REJECTED (2026 |IC|={abs(recent_mean_ic):.4f} < 0.02)")
                                continue
                accepted_ids.append(rec["id"])
                print(f"  {rec['id']}: ACCEPTED (mean |IC|={mean_ic:.4f}, "
                      f"pass={pass_count}/{len(window_ics)})")
            else:
                rejected_ids.append(rec["id"])
                print(f"  {rec['id']}: REJECTED (mean |IC|={mean_ic:.4f}, "
                      f"pass={pass_count}/{len(window_ics)})")

        # Update registry
        if accepted_ids:
            GPFactorMiner.set_status(path, accepted_ids, "testing",
                                     note=f"auto-validated: rolling IC >= {min_rolling_ic}")
        if rejected_ids:
            GPFactorMiner.set_status(path, rejected_ids, "rejected",
                                     note=f"auto-rejected: rolling IC < {min_rolling_ic}")

        return accepted_ids, rejected_ids

    @staticmethod
    def _deserialize_tree_static(d: dict) -> 'Node':
        """Static version of _deserialize_tree for use without instance."""
        children = [GPFactorMiner._deserialize_tree_static(c) for c in d.get('children', [])]
        return Node(op=d['op'], children=children, value=d.get('value'), window=d.get('window'))
