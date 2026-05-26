import json
import os
from typing import Any, Dict, Iterable, Optional


class ProfileStore:
    """Loads cached textual profiles while keeping deterministic fallbacks."""

    def __init__(self, profile_path: Optional[str] = None):
        self.users: Dict[int, str] = {}
        self.items: Dict[int, str] = {}
        if profile_path and os.path.exists(profile_path):
            self._load(profile_path)

    def _load(self, profile_path: str) -> None:
        if profile_path.endswith(".jsonl"):
            with open(profile_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        else:
            with open(profile_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            records = payload if isinstance(payload, list) else payload.get("records", [])

        for record in records:
            kind = record.get("type") or record.get("kind")
            entity_id = record.get("id", record.get("user_id", record.get("item_id")))
            profile = record.get("profile") or record.get("text") or record.get("description")
            if entity_id is None or not profile:
                continue
            entity_id = int(entity_id)
            if kind == "user" or "user_id" in record:
                self.users[entity_id] = str(profile)
            elif kind == "item" or "item_id" in record:
                self.items[entity_id] = str(profile)

    def user_profile(self, user_id: int, history: Optional[Iterable[int]] = None) -> str:
        if user_id in self.users:
            return self.users[user_id]
        if history:
            recent = " ".join(f"item_{int(item_id)}" for item_id in list(history)[-10:])
            return f"user_{user_id} recent interactions {recent}"
        return f"user_{user_id}"

    def item_profile(self, item_id_zero_based: int) -> str:
        item_id = int(item_id_zero_based) + 1
        return self.items.get(item_id, f"item_{item_id}")

    def describe(self) -> Dict[str, Any]:
        return {"users": len(self.users), "items": len(self.items)}

