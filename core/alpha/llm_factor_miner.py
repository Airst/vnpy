"""
LLM 驱动的因子挖掘引擎

使用 LLM（GLM 模型）生成因子假设，转译为表达式树后，
通过现有 GP 评估管线（GPU 张量计算 + Rank IC + 去重）验证。
"""

import gc
import sys
import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import polars as pl

from core.alpha.gp_factor_miner import (
    Node,
    evaluate_tree,
    compute_rank_ic,
    compute_turnover,
    _is_valid_tensor,
    GPFactorMiner,
)
from core.alpha.knowledge_base import KnowledgeBase, build_knowledge_base
from core.alpha.hypothesis_generator import HypothesisGenerator
from core.alpha.expression_translator import translate_batch
from core.alpha.factor_calculator import device, ts_delay, cs_rank
from core.alpha.mlp_signals import MLPSignals
from core.alpha.engine import AlphaEngine
from core.selector.selector import FundamentalSelector


GP_REGISTRY_PATH = "core/alpha/gp_factors.json"


class LLMFactorMiner:
    """LLM 驱动的因子挖掘引擎"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        model: str = "glm-5.2",
        min_ic: float = 0.02,
        min_icir: float = 0.3,
        max_corr: float = 0.5,
        n_candidates: int = 20,
    ):
        self.gen = HypothesisGenerator(api_key=api_key, base_url=base_url, model=model)
        self.min_ic = min_ic
        self.min_icir = min_icir
        self.max_corr = max_corr
        self.n_candidates = n_candidates

        self.kb = build_knowledge_base(GP_REGISTRY_PATH)
        self.discovered_factors: List[Tuple[Node, float, float, float]] = []
        self.translated: List[Tuple[dict, Node]] = []
        self.existing_exprs: set = set()

        registry = GPFactorMiner.load_registry(GP_REGISTRY_PATH)
        for f in registry.get("factors", []):
            self.existing_exprs.add(f["expr"])

    def mine(
        self,
        padded_raw: torch.Tensor,
        col_map: Dict[str, int],
        label: torch.Tensor,
        existing_tensors: List[torch.Tensor],
        recent_start_idx: int = 0,
        max_rounds: int = 5,
        target_count: int = 10,
    ) -> List[Tuple[str, Node]]:
        print(f"[LLM Miner] Knowledge base loaded: {self.kb.summary()}")
        print(f"[LLM Miner] Iterative mining: max_rounds={max_rounds}, target={target_count} factors")

        existing_pool = list(existing_tensors)
        data = self._prepare_terminals(padded_raw, col_map)
        round_feedback: List[dict] = []

        for round_idx in range(max_rounds):
            print(f"\n{'='*60}")
            print(f"[LLM Miner] Round {round_idx + 1}/{max_rounds}")
            print(f"[LLM Miner] Discovered so far: {len(self.discovered_factors)}/{target_count}")
            print(f"{'='*60}")

            if len(self.discovered_factors) >= target_count:
                print(f"[LLM Miner] Target reached ({target_count} factors). Stopping.")
                break

            hypotheses = self.gen.generate_with_feedback(
                self.kb, n_candidates=self.n_candidates, feedback=round_feedback
            )
            print(f"[LLM Miner] Received {len(hypotheses)} hypotheses")

            translated = translate_batch(hypotheses)
            print(f"[LLM Miner] Translated {len(translated)}/{len(hypotheses)} to expression trees")
            self.translated.extend(translated)

            if not translated:
                print("[LLM Miner] No valid expressions this round.")
                round_feedback.append({
                    "round": round_idx + 1,
                    "result": "no_valid_expressions",
                    "detail": "LLM output could not be parsed into expression trees",
                })
                continue

            new_this_round = 0
            for hyp, tree in translated:
                expr_str = tree.to_expr()
                if expr_str in self.existing_exprs:
                    print(f"  [skip] {hyp.get('factor_id', '?')}: duplicate expression")
                    continue

                is_dup = False
                for existing_tree, _, _, _ in self.discovered_factors:
                    if existing_tree.to_expr() == expr_str:
                        is_dup = True
                        break
                if is_dup:
                    print(f"  [skip] {hyp.get('factor_id', '?')}: duplicate in session")
                    continue

                accepted = self._evaluate_and_add(
                    tree, data, label, existing_pool, recent_start_idx, hyp
                )
                if accepted:
                    new_this_round += 1
                    self.existing_exprs.add(expr_str)

            print(f"\n[LLM Miner] Round {round_idx + 1} summary: {new_this_round} new factors")

            round_feedback.append({
                "round": round_idx + 1,
                "total_hypotheses": len(hypotheses),
                "translated": len(translated),
                "accepted": new_this_round,
                "rejected": len(translated) - new_this_round,
                "discovered_so_far": len(self.discovered_factors),
                "new_factor_exprs": [
                    t.to_expr() for t, _, _, _ in self.discovered_factors[-new_this_round:]
                ] if new_this_round > 0 else [],
            })

            if round_idx + 1 < max_rounds and new_this_round == 0:
                print("[LLM Miner] No new factors this round. Will try different direction next round.")

        if self.discovered_factors:
            gp_helper = GPFactorMiner()
            gp_helper.append_discovered(GP_REGISTRY_PATH, self.discovered_factors)
            print(f"\n[LLM Miner] === Mining Complete ===")
            print(f"[LLM Miner] Found {len(self.discovered_factors)} new factors in {min(round_idx + 1, max_rounds)} rounds:")
            for tree, ic, icir, turnover in self.discovered_factors:
                print(f"  IC={ic:.4f}, ICIR={icir:.4f}, turnover={turnover:.3f}")
                print(f"  expr: {tree.to_expr()}")
        else:
            print("\n[LLM Miner] No factors passed validation gates after all rounds.")

        GPFactorMiner.print_status(GP_REGISTRY_PATH)
        return [(t.to_expr(), t) for t, _, _, _ in self.discovered_factors]

    def _evaluate_and_add(
        self,
        tree: Node,
        data: Dict[str, torch.Tensor],
        label: torch.Tensor,
        factor_pool: List[torch.Tensor],
        recent_start_idx: int,
        hyp: dict,
    ) -> bool:
        factor = evaluate_tree(tree, data)
        if not _is_valid_tensor(factor):
            print(f"  [reject] {hyp.get('factor_id', '?')}: invalid tensor")
            return False

        mean_ic, ic_ir, direction_ratio = compute_rank_ic(factor, label)

        if abs(mean_ic) < self.min_ic:
            print(f"  [reject] {hyp.get('factor_id', '?')}: IC={mean_ic:.4f} < {self.min_ic}")
            return False
        if abs(ic_ir) < self.min_icir:
            print(f"  [reject] {hyp.get('factor_id', '?')}: ICIR={ic_ir:.4f} < {self.min_icir}")
            return False
        if direction_ratio < 0.6:
            print(f"  [reject] {hyp.get('factor_id', '?')}: direction_ratio={direction_ratio:.2f} < 0.6")
            return False

        if recent_start_idx > 0:
            num_days = label.shape[1]
            recent_eval_days = num_days - recent_start_idx
            if recent_eval_days >= 20:
                recent_ic, recent_icir, recent_dir = compute_rank_ic(
                    factor, label, eval_days=recent_eval_days
                )
                if abs(recent_ic) < 0.01:
                    print(f"  [reject] {hyp.get('factor_id', '?')}: recent IC too low")
                    return False
                if np.sign(recent_ic) != np.sign(mean_ic):
                    print(f"  [reject] {hyp.get('factor_id', '?')}: recent IC sign mismatch")
                    return False

        if factor_pool:
            sample_days = min(40, factor.shape[1])
            f_flat = factor[:, -sample_days:].reshape(-1)
            for ef in factor_pool:
                e_flat = ef[:, -sample_days:].reshape(-1)
                valid = ~(torch.isnan(f_flat) | torch.isnan(e_flat))
                if valid.sum() < 100:
                    continue
                corr = torch.corrcoef(torch.stack([f_flat[valid], e_flat[valid]]))[0, 1].item()
                if abs(corr) > self.max_corr:
                    print(
                        f"  [reject] {hyp.get('factor_id', '?')}: "
                        f"corr={corr:.3f} > {self.max_corr} with existing factor"
                    )
                    return False

        turnover = compute_turnover(factor)
        self.discovered_factors.append((tree, mean_ic, ic_ir, turnover))
        factor_pool.append(factor)

        print(f"  [accept] {hyp.get('factor_id', '?')}: IC={mean_ic:.4f}, ICIR={ic_ir:.4f}, "
              f"turnover={turnover:.3f}")
        print(f"           expr: {tree.to_expr()}")
        if hyp.get("hypothesis"):
            print(f"           hypothesis: {hyp['hypothesis']}")

        return True

    @staticmethod
    def _prepare_terminals(padded_raw: torch.Tensor, col_map: Dict[str, int]) -> Dict[str, torch.Tensor]:
        terminal_map = {
            "O": "open", "H": "high", "L": "low", "C": "close", "V": "volume",
            "TR": "turnover_rate",
            "BL": "buy_lg_amount", "SL": "sell_lg_amount",
            "BE": "buy_elg_amount", "SE": "sell_elg_amount",
            "NMF": "net_mf_amount",
            "CM5": "concept_mom_5d", "CM20": "concept_mom_20d",
            "CMX": "concept_mom_20d_max", "CHR": "concept_hot_ratio",
            "PE": "pe", "PB": "pb", "PS": "ps",
            "MV": "total_mv", "CMV": "circ_mv",
            "TRF": "turnover_rate_f",
            "BSM": "buy_sm_amount", "SSM": "sell_sm_amount",
            "BMD": "buy_md_amount", "SMD": "sell_md_amount",
        }
        data = {}
        for short, long_name in terminal_map.items():
            if long_name in col_map:
                data[short] = padded_raw[:, :, col_map[long_name]]
        C = data.get("C")
        if C is not None:
            C_prev = ts_delay(C, 1)
            ret_1 = C / (C_prev + 1e-8) - 1
            ret_1_clean = torch.nan_to_num(ret_1, nan=0.0)
            valid_mask = ~torch.isnan(ret_1)
            valid_cnt = valid_mask.sum(dim=0).clamp(min=1)
            mkt_ret = ret_1_clean.sum(dim=0) / valid_cnt
            data["MR"] = mkt_ret.unsqueeze(0).expand_as(C)
        return data
