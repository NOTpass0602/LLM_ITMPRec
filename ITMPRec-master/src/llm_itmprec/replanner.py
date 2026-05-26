from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple


@dataclass
class UserPlanState:
    accept_weight: float = 1.0
    bridge_weight: float = 1.0
    transition_weight: float = 1.0
    penalty_weight: float = 1.0
    semantic_penalty_weight: float = 0.35
    consecutive_rejects: int = 0
    rejected_items: Set[int] = field(default_factory=set)


class FeedbackAwareReplanner:
    """Maintains per-user, per-target reranking weights from complex feedback."""

    def __init__(self):
        self.states: Dict[Tuple[int, int], UserPlanState] = defaultdict(UserPlanState)

    def state(self, user_id: int, target_item: int) -> UserPlanState:
        return self.states[(int(user_id), int(target_item))]

    def update(self, user_id: int, target_item: int, recommended_item: int, action: str) -> None:
        state = self.state(user_id, target_item)
        if action == "click":
            state.consecutive_rejects = 0
            state.bridge_weight = min(1.6, state.bridge_weight + 0.08)
            state.transition_weight = min(1.6, state.transition_weight + 0.08)
        elif action == "reject":
            state.consecutive_rejects += 1
            state.rejected_items.add(int(recommended_item))
            state.accept_weight = min(1.8, state.accept_weight + 0.18)
            state.bridge_weight = max(0.55, state.bridge_weight - 0.10)
            state.penalty_weight = min(2.0, state.penalty_weight + 0.20)
            state.semantic_penalty_weight = min(0.9, state.semantic_penalty_weight + 0.06)
        else:
            state.accept_weight = min(1.5, state.accept_weight + 0.08)
