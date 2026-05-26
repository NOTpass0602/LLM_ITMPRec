import argparse
import json
import os
import sys

import torch


def load_profiles(profile_path):
    users = {}
    items = {}
    with open(profile_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("type") or record.get("kind") or record.get("entity_type")
            entity_id = record.get("id", record.get("user_id", record.get("item_id", record.get("entity_id"))))
            profile = record.get("profile") or record.get("text") or record.get("description")
            if entity_id is None or not profile:
                continue
            entity_id = int(entity_id)
            if kind == "user" or "user_id" in record:
                users[entity_id] = str(profile)
            elif kind == "item" or "item_id" in record:
                items[entity_id] = str(profile)
    return users, items


def encode_texts(model, texts, batch_size):
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    return emb.detach().cpu().float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers is not installed.", file=sys.stderr)
        print(
            r"Install it with: E:\anaconda\envs\itmprec-rtx4080\python.exe -m pip install sentence-transformers==2.7.0",
            file=sys.stderr,
        )
        return 2

    users, items = load_profiles(args.profile_path)
    if not users or not items:
        raise RuntimeError(f"No usable user/item profiles found in {args.profile_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading embedding model: {args.model_name} on {device}")
    model = SentenceTransformer(args.model_name, device=device)

    user_ids = sorted(users)
    item_ids = sorted(items)
    print(f"Encoding {len(user_ids)} user profiles and {len(item_ids)} item profiles")

    user_emb = encode_texts(model, [users[user_id] for user_id in user_ids], args.batch_size)
    item_emb = encode_texts(model, [items[item_id] for item_id in item_ids], args.batch_size)

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "model_name": args.model_name,
            "profile_path": args.profile_path,
            "user_ids": user_ids,
            "item_ids": item_ids,
            "user_emb": user_emb,
            "item_emb": item_emb,
        },
        args.output_path,
    )
    print(f"Saved semantic embeddings to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
