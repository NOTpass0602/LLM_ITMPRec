import argparse
import json
import os

import torch
import torch.nn.functional as F


def load_semantic_items(path: str, expected_items: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    item_ids = [int(item_id) for item_id in payload["item_ids"]]
    item_emb = payload["item_emb"].float()
    dim = item_emb.shape[1]
    aligned = torch.zeros(expected_items, dim, dtype=torch.float32)
    seen = set()
    for row_idx, item_id in enumerate(item_ids):
        zero_idx = item_id - 1
        if 0 <= zero_idx < expected_items:
            aligned[zero_idx] = item_emb[row_idx]
            seen.add(zero_idx)
    if len(seen) != expected_items:
        missing = expected_items - len(seen)
        raise RuntimeError(f"semantic item embedding misses {missing} / {expected_items} items")
    return aligned


def ridge_project(source: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    source = F.normalize(source.float(), p=2, dim=1)
    target = F.normalize(target.float(), p=2, dim=1)
    eye = torch.eye(source.shape[1], dtype=torch.float32)
    lhs = source.T @ source + ridge * eye
    rhs = source.T @ target
    weight = torch.linalg.solve(lhs, rhs)
    return F.normalize(source @ weight, p=2, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_item_emb", required=True)
    parser.add_argument("--semantic_profile_emb", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--semantic_weight", type=float, default=0.2)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--metadata_path", default=None)
    args = parser.parse_args()

    graph = torch.load(args.graph_item_emb, map_location="cpu").float()
    semantic = load_semantic_items(args.semantic_profile_emb, graph.shape[0])
    semantic_projected = ridge_project(semantic, graph, args.ridge)

    graph = F.normalize(graph, p=2, dim=1)
    fused = F.normalize((1.0 - args.semantic_weight) * graph + args.semantic_weight * semantic_projected, p=2, dim=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(fused, args.output_path)

    metadata = {
        "graph_item_emb": args.graph_item_emb,
        "semantic_profile_emb": args.semantic_profile_emb,
        "output_path": args.output_path,
        "semantic_weight": args.semantic_weight,
        "ridge": args.ridge,
        "num_items": int(fused.shape[0]),
        "hidden_size": int(fused.shape[1]),
        "mean_cos_graph_semantic_projected": float((graph * semantic_projected).sum(dim=1).mean().item()),
        "mean_cos_graph_fused": float((graph * fused).sum(dim=1).mean().item()),
    }
    if args.metadata_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.metadata_path)), exist_ok=True)
        with open(args.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
