from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


class SemanticEmbeddingStore:
    """Loads offline profile embeddings and provides cosine scores."""

    def __init__(self, embedding_path: Optional[str] = None):
        self.path = embedding_path
        self.user_emb: Dict[int, torch.Tensor] = {}
        self.item_emb: Dict[int, torch.Tensor] = {}
        self.model_name = ""
        if embedding_path:
            self._load(embedding_path)

    def _load(self, embedding_path: str) -> None:
        payload = torch.load(embedding_path, map_location="cpu")
        self.model_name = str(payload.get("model_name", ""))

        user_ids = [int(uid) for uid in payload.get("user_ids", [])]
        item_ids = [int(iid) for iid in payload.get("item_ids", [])]
        user_emb = self._normalize(payload.get("user_emb"))
        item_emb = self._normalize(payload.get("item_emb"))

        if user_emb is not None:
            for idx, user_id in enumerate(user_ids):
                self.user_emb[user_id] = user_emb[idx]
        if item_emb is not None:
            for idx, item_id in enumerate(item_ids):
                self.item_emb[item_id] = item_emb[idx]

    def _normalize(self, emb: Any) -> Optional[torch.Tensor]:
        if emb is None:
            return None
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb, dtype=torch.float32)
        emb = emb.float().cpu()
        return F.normalize(emb, p=2, dim=-1)

    def user_item_score(self, user_id: int, item_id_zero_based: int) -> Optional[float]:
        user_vec = self.user_emb.get(int(user_id))
        item_vec = self.item_emb.get(int(item_id_zero_based) + 1)
        return self._dot(user_vec, item_vec)

    def item_item_score(self, left_item_zero_based: int, right_item_zero_based: int) -> Optional[float]:
        left_vec = self.item_emb.get(int(left_item_zero_based) + 1)
        right_vec = self.item_emb.get(int(right_item_zero_based) + 1)
        return self._dot(left_vec, right_vec)

    def _dot(self, left: Optional[torch.Tensor], right: Optional[torch.Tensor]) -> Optional[float]:
        if left is None or right is None:
            return None
        return float(torch.dot(left, right).item())

    def describe(self) -> Dict[str, Any]:
        return {
            "path": self.path or "",
            "model_name": self.model_name,
            "users": len(self.user_emb),
            "items": len(self.item_emb),
        }

