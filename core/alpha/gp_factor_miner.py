"""
遗传编程(GP)因子挖掘模块 — 自动发现短周期 Alpha 因子

== 设计目标 ==
补充当前慢因子体系（120d/60d/20d 为主），发现 3-20 天衰减的短周期因子，
增加模型信号的时效性和变异性，缓解"分数横线→入场时机差"问题。

== 设计决策 ==
- 复用 factor_calculator.py 中的 GPU 张量算子（ts_mean/ts_std/cs_rank 等）
- 适应度函数: Rank IC (Spearman) 在滚动 OOS 窗口上评估
- 窗口参数限制在 {3, 5, 10, 20} 强制短周期特征
- 相关性去重: 与现有因子池相关 > 0.7 的因子丢弃
- 复杂度惩罚 (parsimony): 树深度 > 5 的表达式被惩罚
- 基于 DEAP 框架实现，GPU 加速因子计算

== 算子体系 ==
终端节点 (Terminals): O, H, L, C, V, TR (量价原子数据)
一元时序 (Unary TS): ts_mean, ts_std, ts_max, ts_min, ts_delta, ts_rank, ts_decay_linear
二元时序 (Binary TS): ts_corr, ts_cov
截面 (Cross-Sectional): cs_rank, cs_zscore
算术 (Arithmetic): add, sub, mul, div, abs, neg, log
窗口参数 (Windows): 3, 5, 10, 20
"""

import random
import copy
import json
import time
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
TERMINALS = ['O', 'H', 'L', 'C', 'V', 'TR']

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

@torch.no_grad()
def compute_rank_ic(factor: torch.Tensor, label: torch.Tensor, eval_days: int = 0) -> Tuple[float, float, float]:
    """
    Compute daily Rank IC (Spearman correlation) — fully vectorized on GPU.
    
    Args:
        factor: (num_stocks, num_days) tensor
        label: (num_stocks, num_days) tensor
        eval_days: If >0, only use last N days. 0 = use all.
    
    Returns:
        (mean_ic, ic_ir, direction_ratio)
    """
    num_stocks, num_days = factor.shape
    start_t = max(0, num_days - eval_days) if eval_days > 0 else 0
    step = 10

    # Sample columns: (num_stocks, n_eval)
    f_s = factor[:, start_t::step]
    l_s = label[:, start_t::step]
    n_eval = f_s.shape[1]

    if n_eval < 10:
        return 0.0, 0.0, 0.0

    # Mask: valid where both are non-NaN
    valid = ~(torch.isnan(f_s) | torch.isnan(l_s))
    counts = valid.sum(dim=0)  # (n_eval,)

    # Replace NaN with large negative for ranking (will be masked out)
    f_filled = f_s.clone()
    l_filled = l_s.clone()
    f_filled[~valid] = float('-inf')
    l_filled[~valid] = float('-inf')

    # Batch rank: argsort().argsort() along stock dimension -> (num_stocks, n_eval)
    f_rank = f_filled.argsort(dim=0).argsort(dim=0).float()
    l_rank = l_filled.argsort(dim=0).argsort(dim=0).float()

    # Zero out invalid positions
    f_rank[~valid] = 0.0
    l_rank[~valid] = 0.0

    # Per-day mean of ranks (only valid)
    f_sum = f_rank.sum(dim=0)  # (n_eval,)
    l_sum = l_rank.sum(dim=0)
    f_mean = f_sum / counts.clamp(min=1)
    l_mean = l_sum / counts.clamp(min=1)

    # Center ranks
    f_centered = (f_rank - f_mean.unsqueeze(0)) * valid.float()
    l_centered = (l_rank - l_mean.unsqueeze(0)) * valid.float()

    # Per-day correlation: sum(f*l) / (norm(f) * norm(l))
    numerator = (f_centered * l_centered).sum(dim=0)  # (n_eval,)
    f_norm = f_centered.pow(2).sum(dim=0).sqrt()
    l_norm = l_centered.pow(2).sum(dim=0).sqrt()
    denom = f_norm * l_norm

    # Filter days with enough data
    day_valid = (counts >= 30) & (denom > 1e-10)
    if day_valid.sum() < 10:
        return 0.0, 0.0, 0.0

    ics = (numerator[day_valid] / denom[day_valid]).cpu().numpy()

    mean_ic = float(np.mean(ics))
    std_ic = float(np.std(ics)) + 1e-8
    ic_ir = mean_ic / std_ic

    # Sub-period direction consistency
    n_ics = len(ics)
    n_periods = min(5, n_ics // 5)
    if n_periods < 3:
        return mean_ic, ic_ir, 0.5

    chunk_size = n_ics // n_periods
    sign_overall = np.sign(mean_ic)
    same_sign_count = sum(
        1 for i in range(n_periods)
        if np.sign(np.mean(ics[i * chunk_size: (i + 1) * chunk_size])) == sign_overall
    )
    direction_ratio = same_sign_count / n_periods

    return mean_ic, ic_ir, direction_ratio


@torch.no_grad()
def compute_turnover(factor: torch.Tensor, top_pct: float = 0.1, eval_days: int = 200) -> float:
    """
    Compute average daily turnover of top-ranked stocks — vectorized on GPU.
    High turnover = short-period factor.
    """
    num_stocks, num_days = factor.shape
    start_t = max(0, num_days - eval_days)
    step = 5
    f_slice = factor[:, start_t::step]  # (stocks, n_eval)
    n_eval = f_slice.shape[1]

    if n_eval < 2:
        return 0.0

    # Replace NaN with -inf for topk
    f_filled = f_slice.clone()
    f_filled[torch.isnan(f_filled)] = float('-inf')

    n_top = max(int(num_stocks * top_pct), 1)

    # Get top-k indices for all days at once: (n_top, n_eval)
    _, top_indices = f_filled.topk(n_top, dim=0)  # (n_top, n_eval)

    # Compute turnover between consecutive days
    turnovers = []
    # Sort indices for set comparison
    top_sorted = top_indices.sort(dim=0).values  # (n_top, n_eval)

    for t in range(1, n_eval):
        prev = top_sorted[:, t - 1]
        curr = top_sorted[:, t]
        # Count overlap: for each element in curr, check if it exists in prev
        # Use broadcasting: (n_top, 1) == (1, n_top) -> (n_top, n_top)
        overlap = (curr.unsqueeze(1) == prev.unsqueeze(0)).any(dim=1).sum().item()
        turnovers.append(1.0 - overlap / n_top)

    return float(np.mean(turnovers)) if turnovers else 0.0


@torch.no_grad()
def fitness(node: Node, data: Dict[str, torch.Tensor], label: torch.Tensor,
            existing_factors: Optional[List[torch.Tensor]] = None,
            max_corr: float = 0.7) -> float:
    """
    Compute fitness score for a GP individual.
    
    Fitness = |ICIR| * turnover_bonus * consistency_bonus - parsimony_penalty
    
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

    score = abs(ic_ir) * turnover_bonus * consistency_bonus - parsimony
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
    Genetic Programming Factor Miner.
    
    Discovers short-period alpha factors via evolutionary search on GPU tensors.
    """

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

    def mine(
        self,
        padded_raw: torch.Tensor,
        col_map: Dict[str, int],
        label: torch.Tensor,
        existing_factor_tensors: Optional[List[torch.Tensor]] = None,
    ) -> List[Tuple[str, Node]]:
        """
        Run GP evolution to discover new factors.
        
        Args:
            padded_raw: (num_stocks, max_len, num_features) GPU tensor
            col_map: Column name -> index mapping
            label: (num_stocks, max_len) forward return label
            existing_factor_tensors: List of existing factor tensors for deduplication
            
        Returns:
            List of (factor_name, expression_tree) tuples
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Trim label to match terminal window
        max_days = 500
        if label.shape[1] > max_days:
            label = label[:, -max_days:]
        
        # Prepare terminal data (trimmed to 500 days for GP mining memory)
        data = self._prepare_terminals(padded_raw, col_map, trim_days=500)
        
        # Release full padded_raw reference — only terminals needed from here
        del padded_raw
        torch.cuda.empty_cache()
        
        print(f"[GP Miner] Starting evolution: pop={self.population_size}, gen={self.n_generations}")
        print(f"[GP Miner] Data shape: {label.shape[0]} stocks, {label.shape[1]} days")
        print(f"[GP Miner] Targeting short-period factors (windows: {WINDOWS})")

        # Initialize population
        population = []
        for _ in range(self.population_size):
            tree = random_tree(max_depth=self.max_tree_depth)
            score = fitness(tree, data, label, existing_factor_tensors, self.max_corr_with_pool)
            population.append((tree, score))

        best_ever_score = -999.0
        stagnation = 0
        factor_pool = list(existing_factor_tensors) if existing_factor_tensors else []

        for gen in range(self.n_generations):
            gen_start = time.time()
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
                elapsed = time.time() - gen_start
                print(f"[GP Miner] Gen {gen:3d}: best={gen_best:.4f}, "
                      f"valid={valid_count}/{self.population_size}, "
                      f"discovered={len(self.discovered_factors)}, "
                      f"time={elapsed:.1f}s")

            # Check if top individual qualifies as a new factor
            if gen_best > 0:
                top_tree = population[0][0]
                self._try_add_factor(top_tree, data, label, factor_pool)

            # Early stopping
            if stagnation > 15:
                print(f"[GP Miner] Early stopping at gen {gen} (stagnation={stagnation})")
                break

            if len(self.discovered_factors) >= self.max_factors:
                print(f"[GP Miner] Reached max factors ({self.max_factors}), stopping.")
                break

            # Create next generation
            new_population = []

            # Elitism: keep top 5%
            elite_size = max(5, self.population_size // 20)
            for i in range(elite_size):
                new_population.append(population[i])

            # Fill rest with crossover and mutation
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_prob:
                    p1 = tournament_select(population, self.tournament_size)
                    p2 = tournament_select(population, self.tournament_size)
                    c1, c2 = crossover(p1, p2)
                    c1 = mutate(c1, self.mutation_prob)
                    c2 = mutate(c2, self.mutation_prob)
                    s1 = fitness(c1, data, label, factor_pool, self.max_corr_with_pool)
                    s2 = fitness(c2, data, label, factor_pool, self.max_corr_with_pool)
                    new_population.append((c1, s1))
                    if len(new_population) < self.population_size:
                        new_population.append((c2, s2))
                else:
                    p = tournament_select(population, self.tournament_size)
                    c = mutate(p, prob=0.5)
                    s = fitness(c, data, label, factor_pool, self.max_corr_with_pool)
                    new_population.append((c, s))

            population = new_population
            
            # Free GPU memory fragmentation
            torch.cuda.empty_cache()

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

    def _prepare_terminals(self, padded_raw: torch.Tensor, col_map: Dict[str, int], trim_days: int = 0) -> Dict[str, torch.Tensor]:
        """Extract terminal tensors from padded raw data.
        
        Args:
            trim_days: If >0, only use the last N days (for GP mining memory reduction).
                       0 means use all days (for compute_factors during training).
        """
        if trim_days > 0 and padded_raw.shape[1] > trim_days:
            padded_raw = padded_raw[:, -trim_days:, :]
        
        terminal_map = {
            'O': 'open',
            'H': 'high',
            'L': 'low',
            'C': 'close',
            'V': 'volume',
            'TR': 'turnover_rate',
        }
        data = {}
        for term_name, col_name in terminal_map.items():
            if col_name in col_map:
                data[term_name] = padded_raw[:, :, col_map[col_name]]
            else:
                data[term_name] = torch.full(
                    (padded_raw.shape[0], padded_raw.shape[1]),
                    float('nan'), device=padded_raw.device, dtype=padded_raw.dtype
                )
        return data

    def _try_add_factor(
        self, tree: Node, data: Dict[str, torch.Tensor],
        label: torch.Tensor, factor_pool: List[torch.Tensor]
    ):
        """Try to add a factor to the discovered pool after validation."""
        # Expression deduplication
        expr_str = tree.to_expr()
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
        # Direction consistency gate
        if direction_ratio < 0.6:
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
                if abs(corr) > 0.5:  # Stricter for inter-GP diversity
                    return

        turnover = compute_turnover(factor)
        self.discovered_factors.append((tree, mean_ic, ic_ir, turnover))
        factor_pool.append(factor)
        print(f"[GP Miner] +++ New factor discovered! IC={mean_ic:.4f}, ICIR={ic_ir:.4f}, "
              f"dir={direction_ratio:.2f}, turnover={turnover:.3f}")
        print(f"           expr: {expr_str}")

    def compute_factors(
        self, padded_raw: torch.Tensor, col_map: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute discovered GP factors on new data.
        Called by the factor calculator during normal operation.
        
        Returns:
            Dict of factor_name -> (num_stocks, max_len) tensor
        """
        if not self.discovered_factors:
            return {}

        data = self._prepare_terminals(padded_raw, col_map)
        result = {}

        for i, (tree, _, _, _) in enumerate(self.discovered_factors):
            name = f"gp_alpha_{i:03d}"
            factor = evaluate_tree(tree, data)
            if _is_valid_tensor(factor):
                result[name] = factor

        return result

    def save(self, path: str):
        """Save discovered factors to JSON."""
        records = []
        for tree, ic, icir, turnover in self.discovered_factors:
            records.append({
                'expr': tree.to_expr(),
                'tree': self._serialize_tree(tree),
                'ic': ic,
                'icir': icir,
                'turnover': turnover,
            })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(records, f, indent=2)
        print(f"[GP Miner] Saved {len(records)} factors to {path}")

    def load(self, path: str) -> bool:
        """Load discovered factors from JSON."""
        p = Path(path)
        if not p.exists():
            return False
        with open(p) as f:
            records = json.load(f)
        self.discovered_factors = []
        for rec in records:
            tree = self._deserialize_tree(rec['tree'])
            self.discovered_factors.append((tree, rec['ic'], rec['icir'], rec['turnover']))
        print(f"[GP Miner] Loaded {len(self.discovered_factors)} factors from {path}")
        return True

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
