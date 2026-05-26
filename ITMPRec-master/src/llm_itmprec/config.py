from dataclasses import dataclass
from typing import Optional


@dataclass
class LIRConfig:
    alpha: float = 0.20
    beta: float = 0.20
    gamma: float = 0.20
    delta: float = 0.30
    candidate_k: int = 50
    use_profiles: bool = True
    use_cot: bool = True
    use_feedback: bool = True
    use_replanning: bool = True
    profile_embedding_path: Optional[str] = None
    cot_min_transition: float = 0.50
    cot_max_risk: float = 0.50
    cot_use_bonus: bool = True
    cot_use_risk_penalty: bool = False
    cot_replan_filter: bool = False
    cot_replan_risk_min: float = 0.75
    cot_replan_transition_max: float = 0.25
    cot_replan_min_rejects: int = 1
    cot_replan_margin: float = 0.0
    cot_replan_proactive: bool = False
