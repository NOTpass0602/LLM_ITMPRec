from typing import Dict, Iterable, List, Optional, Tuple

from .config import LIRConfig
from .cot_bridge_planner import CachedCoTBridgePlanner
from .profile_store import ProfileStore
from .replanner import FeedbackAwareReplanner
from .semantic_embedding_store import SemanticEmbeddingStore
from .semantic_scorer import SemanticScorer, minmax


class LIRReranker:
    """Reranks ITMPRec top-K candidates with textual profiles and feedback state."""

    def __init__(
        self,
        config: LIRConfig,
        profile_store: ProfileStore,
        planner: Optional[CachedCoTBridgePlanner] = None,
        replanner: Optional[FeedbackAwareReplanner] = None,
    ):
        self.config = config
        self.profile_store = profile_store
        self.planner = planner or CachedCoTBridgePlanner()
        self.replanner = replanner or FeedbackAwareReplanner()
        self.scorer = SemanticScorer()
        self.embedding_store = SemanticEmbeddingStore(config.profile_embedding_path) if config.profile_embedding_path else None
        self._acceptability_cache = {}
        self._bridge_cache = {}
        self._item_similarity_cache = {}

    def rerank(
        self,
        user_id: int,
        history: Iterable[int],
        target_item: int,
        candidate_items: List[int],
        base_scores: List[float],
    ) -> Tuple[int, Dict[str, float]]:
        if not candidate_items:
            return 0, {"acceptability": 0.0, "bridge": 0.0, "transition": 0.0, "risk": 0.0}

        user_profile = self.profile_store.user_profile(user_id, history)
        target_profile = self.profile_store.item_profile(target_item)
        normalized_base = minmax(base_scores)
        base_margin = self._top_margin(normalized_base)
        state = self.replanner.state(user_id, target_item)

        best_idx = 0
        best_score = float("-inf")
        best_meta = None
        for idx, item_id in enumerate(candidate_items):
            cand_profile = self.profile_store.item_profile(item_id)
            accept_key = (int(user_id), int(item_id))
            if accept_key not in self._acceptability_cache:
                embedding_score = self.embedding_store.user_item_score(user_id, item_id) if self.embedding_store else None
                self._acceptability_cache[accept_key] = (
                    embedding_score if embedding_score is not None else self.scorer.cosine(user_profile, cand_profile)
                )
            acceptability = self._acceptability_cache[accept_key]

            bridge_key = (int(item_id), int(target_item))
            if bridge_key not in self._bridge_cache:
                embedding_score = self.embedding_store.item_item_score(item_id, target_item) if self.embedding_store else None
                self._bridge_cache[bridge_key] = (
                    embedding_score if embedding_score is not None else self.scorer.cosine(cand_profile, target_profile)
                )
            bridge = self._bridge_cache[bridge_key]
            transition = min(acceptability, bridge)
            cot_score, cot_risk, cot_transition = self.planner.score(user_id, target_item + 1, item_id + 1) if self.config.use_cot else (0.0, 0.0, 0.0)
            rejected_penalty = self._rejection_penalty(item_id, target_item, state)
            score = normalized_base[idx]
            if self.config.use_profiles:
                score += self.config.alpha * state.accept_weight * acceptability
                score += self.config.beta * state.bridge_weight * bridge
                score += self.config.gamma * state.transition_weight * transition
            if self.config.use_cot:
                if self.config.cot_use_bonus and cot_transition >= self.config.cot_min_transition and cot_risk <= self.config.cot_max_risk:
                    score += self.config.gamma * cot_score
                if self.config.cot_use_risk_penalty:
                    score -= self.config.delta * cot_risk
            if self.config.use_replanning:
                score -= self.config.delta * state.penalty_weight * rejected_penalty
                if self._should_use_cot_replan_filter(state, base_margin):
                    score -= self.config.delta * state.penalty_weight * self._cot_replan_penalty(cot_risk, cot_transition)
            if score > best_score:
                best_idx = idx
                best_score = score
                best_meta = {
                    "acceptability": acceptability,
                    "bridge": bridge,
                    "transition": transition,
                    "risk": cot_risk,
                    "cot_transition": cot_transition,
                    "score": score,
                }

        return int(candidate_items[best_idx]), best_meta or {}

    def _top_margin(self, values: List[float]) -> float:
        if len(values) < 2:
            return 1.0
        top_two = sorted(values, reverse=True)[:2]
        return float(top_two[0] - top_two[1])

    def _should_use_cot_replan_filter(self, state, base_margin: float) -> bool:
        if not (self.config.use_cot and self.config.cot_replan_filter):
            return False
        if self.config.cot_replan_proactive:
            return base_margin <= self.config.cot_replan_margin
        if not state.rejected_items:
            return False
        if state.consecutive_rejects < self.config.cot_replan_min_rejects:
            return False
        if base_margin > self.config.cot_replan_margin:
            return False
        return True

    def _rejection_penalty(self, item_id: int, target_item: int, state) -> float:
        rejected_items = state.rejected_items
        if not self.config.use_replanning or not rejected_items or int(item_id) == int(target_item):
            return 0.0
        if item_id in rejected_items:
            return 1.0

        max_similarity = 0.0
        for rejected_item in rejected_items:
            key = (int(item_id), int(rejected_item))
            if key not in self._item_similarity_cache:
                embedding_score = self.embedding_store.item_item_score(item_id, rejected_item) if self.embedding_store else None
                if embedding_score is None:
                    cand_profile = self.profile_store.item_profile(item_id)
                    rejected_profile = self.profile_store.item_profile(rejected_item)
                    embedding_score = self.scorer.cosine(cand_profile, rejected_profile)
                self._item_similarity_cache[key] = max(0.0, float(embedding_score))
            max_similarity = max(max_similarity, self._item_similarity_cache[key])
        if max_similarity < 0.70:
            return 0.0
        return state.semantic_penalty_weight * ((max_similarity - 0.70) / 0.30)

    def _cot_replan_penalty(self, cot_risk: float, cot_transition: float) -> float:
        if cot_risk < self.config.cot_replan_risk_min:
            return 0.0
        if cot_transition > self.config.cot_replan_transition_max:
            return 0.0
        risk_span = max(1e-12, 1.0 - self.config.cot_replan_risk_min)
        transition_span = max(1e-12, self.config.cot_replan_transition_max)
        risk_strength = (cot_risk - self.config.cot_replan_risk_min) / risk_span
        transition_strength = (self.config.cot_replan_transition_max - cot_transition) / transition_span
        return max(0.0, min(1.0, 0.5 * risk_strength + 0.5 * transition_strength))
