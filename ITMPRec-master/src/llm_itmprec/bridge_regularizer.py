# -*- coding: utf-8 -*-

import json
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


class BridgeRankingRegularizer:
    """Pairwise item-target regularizer from LLM bridge/risk scores.

    The cot cache stores 1-based item ids. In this codebase, the evaluation
    item embedding table is addressed as weight[item_id], so the 1-based ids
    can be used directly for item embedding indices.
    """

    def __init__(
        self,
        cot_cache_path: str,
        item_size: int,
        sample_size: int = 256,
        margin: float = 0.05,
        min_score_gap: float = 2.0,
        pair_mode: str = "rank_all",
        pos_transition_min: float = 3.0,
        pos_risk_max: float = 3.0,
        neg_transition_max: float = 2.0,
        neg_risk_min: float = 4.0,
        top_k: int = 8,
        bottom_k: int = 12,
    ) -> None:
        self.cot_cache_path = cot_cache_path
        self.item_size = int(item_size)
        self.sample_size = int(sample_size)
        self.margin = float(margin)
        self.min_score_gap = float(min_score_gap)
        self.pair_mode = pair_mode
        self.pos_transition_min = float(pos_transition_min)
        self.pos_risk_max = float(pos_risk_max)
        self.neg_transition_max = float(neg_transition_max)
        self.neg_risk_min = float(neg_risk_min)
        self.top_k = int(top_k)
        self.bottom_k = int(bottom_k)
        self.device = torch.device("cpu")
        self.target_idx, self.good_idx, self.bad_idx, self.pair_weight = self._load_pairs(cot_cache_path)

    def _load_pairs(self, path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grouped: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                target = int(record["target_item"])
                candidate = int(record["candidate_item"])
                if target <= 0 or candidate <= 0 or target >= self.item_size or candidate >= self.item_size:
                    continue
                transition = float(record.get("transition_score", 0.0))
                risk = float(record.get("risk_score", 0.0))
                acceptance = float(record.get("acceptance_score", 0.0))
                bridge_score = transition + 0.25 * acceptance - risk
                grouped[target].append((candidate, bridge_score, transition, risk, acceptance))

        pair_targets: List[int] = []
        pair_goods: List[int] = []
        pair_bads: List[int] = []
        pair_weights: List[float] = []
        for target, scored_items in grouped.items():
            if self.pair_mode == "extreme":
                positives = [
                    row
                    for row in scored_items
                    if row[2] >= self.pos_transition_min and row[3] <= self.pos_risk_max
                ]
                negatives = [
                    row
                    for row in scored_items
                    if row[2] <= self.neg_transition_max and row[3] >= self.neg_risk_min
                ]
                for good_item, good_score, good_transition, good_risk, _ in positives:
                    for bad_item, bad_score, bad_transition, bad_risk, _ in negatives:
                        if good_item == bad_item:
                            continue
                        score_gap = good_score - bad_score
                        if score_gap < self.min_score_gap:
                            continue
                        risk_gap = max(bad_risk - good_risk, 0.0)
                        transition_gap = max(good_transition - bad_transition, 0.0)
                        pair_targets.append(target)
                        pair_goods.append(good_item)
                        pair_bads.append(bad_item)
                        pair_weights.append(min((score_gap + risk_gap + transition_gap) / 8.0, 2.0))
                continue

            scored_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
            if self.pair_mode == "top_bottom":
                positives = scored_items[: self.top_k]
                negatives = scored_items[-self.bottom_k :]
                for good_item, good_score, _, _, _ in positives:
                    for bad_item, bad_score, _, _, _ in negatives:
                        if good_item == bad_item:
                            continue
                        score_gap = good_score - bad_score
                        if score_gap < self.min_score_gap:
                            continue
                        pair_targets.append(target)
                        pair_goods.append(good_item)
                        pair_bads.append(bad_item)
                        pair_weights.append(min(score_gap / 5.0, 2.0))
                continue

            for good_pos, (good_item, good_score, _, _, _) in enumerate(scored_items):
                for bad_item, bad_score, _, _, _ in scored_items[good_pos + 1 :]:
                    score_gap = good_score - bad_score
                    if score_gap < self.min_score_gap:
                        continue
                    pair_targets.append(target)
                    pair_goods.append(good_item)
                    pair_bads.append(bad_item)
                    pair_weights.append(min(score_gap / 5.0, 2.0))

        if not pair_targets:
            raise ValueError(f"No bridge ranking pairs built from {path}; lower min_score_gap.")

        return (
            torch.tensor(pair_targets, dtype=torch.long),
            torch.tensor(pair_goods, dtype=torch.long),
            torch.tensor(pair_bads, dtype=torch.long),
            torch.tensor(pair_weights, dtype=torch.float32),
        )

    @property
    def num_pairs(self) -> int:
        return int(self.target_idx.numel())

    def to(self, device: torch.device) -> "BridgeRankingRegularizer":
        self.device = torch.device(device)
        self.target_idx = self.target_idx.to(self.device)
        self.good_idx = self.good_idx.to(self.device)
        self.bad_idx = self.bad_idx.to(self.device)
        self.pair_weight = self.pair_weight.to(self.device)
        return self

    def __call__(self, model) -> torch.Tensor:
        if self.num_pairs <= self.sample_size:
            pair_idx = torch.arange(self.num_pairs, device=self.device)
        else:
            pair_idx = torch.randint(0, self.num_pairs, (self.sample_size,), device=self.device)

        target = self.target_idx[pair_idx]
        good = self.good_idx[pair_idx]
        bad = self.bad_idx[pair_idx]
        weight = self.pair_weight[pair_idx]

        item_emb = F.normalize(model.item_embeddings.weight, p=2, dim=1)
        target_emb = item_emb[target]
        good_sim = torch.sum(target_emb * item_emb[good], dim=-1)
        bad_sim = torch.sum(target_emb * item_emb[bad], dim=-1)
        return (F.relu(self.margin - (good_sim - bad_sim)) * weight).mean()


def attach_bridge_regularization(trainer, args) -> None:
    weight = float(getattr(args, "bridge_reg_weight", 0.0) or 0.0)
    path = getattr(args, "bridge_reg_path", None)
    if weight <= 0.0 or not path:
        return

    regularizer = BridgeRankingRegularizer(
        cot_cache_path=path,
        item_size=args.item_size,
        sample_size=getattr(args, "bridge_reg_samples", 256),
        margin=getattr(args, "bridge_reg_margin", 0.05),
        min_score_gap=getattr(args, "bridge_reg_min_score_gap", 2.0),
        pair_mode=getattr(args, "bridge_reg_pair_mode", "rank_all"),
        pos_transition_min=getattr(args, "bridge_reg_pos_transition_min", 3.0),
        pos_risk_max=getattr(args, "bridge_reg_pos_risk_max", 3.0),
        neg_transition_max=getattr(args, "bridge_reg_neg_transition_max", 2.0),
        neg_risk_min=getattr(args, "bridge_reg_neg_risk_min", 4.0),
        top_k=getattr(args, "bridge_reg_top_k", 8),
        bottom_k=getattr(args, "bridge_reg_bottom_k", 12),
    ).to(trainer.device)

    original_cross_entropy = trainer.cross_entropy

    def cross_entropy_with_bridge(seq_out, pos_ids, neg_ids):
        rec_loss = original_cross_entropy(seq_out, pos_ids, neg_ids)
        return rec_loss + weight * regularizer(trainer.model)

    trainer.cross_entropy = cross_entropy_with_bridge
    trainer.bridge_regularizer = regularizer
    print(
        "Bridge ranking regularization enabled: "
        f"pairs={regularizer.num_pairs}, weight={weight}, "
        f"sample_size={regularizer.sample_size}, margin={regularizer.margin}, "
        f"min_score_gap={regularizer.min_score_gap}, pair_mode={regularizer.pair_mode}"
    )
