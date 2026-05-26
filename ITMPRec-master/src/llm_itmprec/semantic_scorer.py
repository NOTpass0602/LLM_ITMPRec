import math
import re
from collections import Counter
from typing import Dict, Iterable, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class SemanticScorer:
    """Small deterministic text scorer for cached LLM/RLMRec-style profiles."""

    def __init__(self):
        self._cache: Dict[str, Counter] = {}

    def _tokens(self, text: str) -> Counter:
        if text not in self._cache:
            toks = [tok.lower() for tok in TOKEN_RE.findall(text or "")]
            self._cache[text] = Counter(toks)
        return self._cache[text]

    def cosine(self, left: str, right: str) -> float:
        a = self._tokens(left)
        b = self._tokens(right)
        if not a or not b:
            return 0.0
        dot = sum(a[token] * b.get(token, 0) for token in a)
        norm_a = math.sqrt(sum(value * value for value in a.values()))
        norm_b = math.sqrt(sum(value * value for value in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def bridge_scores(self, user_profile: str, candidate_profile: str, target_profile: str) -> Tuple[float, float, float]:
        acceptability = self.cosine(user_profile, candidate_profile)
        target_bridge = self.cosine(candidate_profile, target_profile)
        transition = min(acceptability, target_bridge)
        return acceptability, target_bridge, transition


def minmax(values: Iterable[float]) -> list:
    vals = [float(v) for v in values]
    if not vals:
        return []
    low = min(vals)
    high = max(vals)
    if high - low < 1e-12:
        return [0.0 for _ in vals]
    return [(v - low) / (high - low) for v in vals]

