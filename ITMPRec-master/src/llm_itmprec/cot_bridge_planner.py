import json
import os
from typing import Dict, Optional, Tuple


class CachedCoTBridgePlanner:
    """Reads optional CoT bridge judgments generated offline and cached as JSONL."""

    def __init__(self, cache_path: Optional[str] = None):
        self.cache: Dict[Tuple[int, int, int], dict] = {}
        if cache_path and os.path.exists(cache_path):
            self._load(cache_path)

    def _load(self, cache_path: str) -> None:
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (
                    int(record.get("user_id", 0)),
                    int(record["target_item"]),
                    int(record["candidate_item"]),
                )
                self.cache[key] = record

    def score(self, user_id: int, target_item: int, candidate_item: int) -> Tuple[float, float, float]:
        record = self.cache.get((int(user_id), int(target_item), int(candidate_item)))
        if not record:
            record = self.cache.get((0, int(target_item), int(candidate_item)))
        if not record:
            return 0.0, 0.0, 0.0
        acceptance = _norm_1_to_5(record.get("acceptance_score", 0))
        transition = _norm_1_to_5(record.get("transition_score", 0))
        risk = _norm_1_to_5(record.get("risk_score", 0))
        return (acceptance + transition) / 2.0, risk, transition

    def describe(self) -> int:
        return len(self.cache)


def _norm_1_to_5(value) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, (raw - 1.0) / 4.0))
