from dataclasses import dataclass


@dataclass
class ComplexFeedback:
    action: str
    preference_strength: float
    critique: str
    interest_shift: float


class ComplexFeedbackSimulator:
    """Rule-based proxy for GRSU-style multi-grained feedback."""

    def observe(self, clicked: bool, acceptability: float, bridge: float, transition: float, risk: float = 0.0) -> ComplexFeedback:
        preference = 0.45 * acceptability + 0.35 * bridge + 0.20 * transition - 0.25 * risk
        preference = max(0.0, min(1.0, preference))
        if clicked:
            return ComplexFeedback("click", max(0.6, preference), "accepted bridge item", bridge * 0.1)

        weak_user_fit = acceptability < 0.24
        weak_transition = transition < 0.28
        weak_overall = preference < 0.35
        if risk > 0.75 or (weak_overall and (weak_user_fit or weak_transition)):
            return ComplexFeedback("reject", preference, "candidate is too abrupt or mismatched", -0.05)
        return ComplexFeedback("skip", preference, "candidate is plausible but not attractive enough", 0.0)
