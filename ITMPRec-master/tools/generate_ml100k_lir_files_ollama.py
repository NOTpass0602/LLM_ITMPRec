import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


GENRE_NAMES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def load_item_metadata(raw_root: Path) -> Dict[int, dict]:
    items = {}
    with (raw_root / "u.item").open("r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            raw_id = int(parts[0])
            title = parts[1]
            release_date = parts[2]
            genre_flags = [int(v) for v in parts[5:24]]
            genres = [name for name, flag in zip(GENRE_NAMES, genre_flags) if flag] or ["unknown"]
            year = release_date[-4:] if release_date and "-" in release_date else ""
            items[raw_id] = {"title": title, "year": year, "genres": genres}
    return items


def load_raw_ratings(raw_root: Path) -> List[Tuple[int, int, int, int]]:
    rows = []
    with (raw_root / "u.data").open("r", encoding="latin-1") as f:
        for line in f:
            rows.append(tuple(map(int, line.split())))
    return rows


def build_maps(raw_ratings: List[Tuple[int, int, int, int]]):
    user_map = {}
    item_map = {}
    for raw_user, raw_item, _rating, timestamp in sorted(raw_ratings, key=lambda x: (x[0], x[3])):
        user_map.setdefault(raw_user, len(user_map) + 1)
        item_map.setdefault(raw_item, len(item_map) + 1)
    inv_user_map = {internal: raw for raw, internal in user_map.items()}
    inv_item_map = {internal: raw for raw, internal in item_map.items()}
    return user_map, item_map, inv_user_map, inv_item_map


def load_sequences(data_file: Path) -> Dict[int, List[int]]:
    seqs = {}
    with data_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = [int(x) for x in line.strip().split()]
            if parts:
                seqs[parts[0]] = parts[1:]
    return seqs


def genre_phrase(genres: Iterable[str]) -> str:
    genres = list(genres)
    if not genres:
        return "mixed genres"
    if len(genres) == 1:
        return genres[0]
    return ", ".join(genres[:-1]) + ", and " + genres[-1]


def item_label(item_id: int, item_meta: Dict[int, dict], inv_item_map: Dict[int, int]) -> str:
    raw_id = inv_item_map.get(item_id)
    meta = item_meta.get(raw_id)
    if not meta:
        return f"Movie {item_id}"
    year = f" ({meta['year']})" if meta.get("year") and meta["year"] not in meta["title"] else ""
    return f"{meta['title']}{year}, genres: {genre_phrase(meta['genres'])}"


def safe_json_from_text(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


class OllamaClient:
    def __init__(self, url: str, model: str, timeout: int, temperature: float, num_ctx: int, retries: int, sleep: float):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.retries = retries
        self.sleep = sleep

    def generate_json(self, prompt: str) -> dict:
        endpoint = f"{self.url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(endpoint, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                parsed = safe_json_from_text(data.get("response", ""))
                if parsed:
                    return parsed
                last_error = RuntimeError("empty or invalid JSON response")
            except Exception as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(self.sleep * (attempt + 1))
        raise RuntimeError(f"Ollama request failed: {last_error}")


def load_existing_profile_ids(path: Path):
    users = set()
    items = set()
    if not path.exists():
        return users, items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("type")
            entity_id = int(record.get("id"))
            if kind == "user":
                users.add(entity_id)
            elif kind == "item":
                items.add(entity_id)
    return users, items


def load_existing_cot_keys(path: Path):
    keys = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            keys.add((int(record.get("user_id", 0)), int(record["target_item"]), int(record["candidate_item"])))
    return keys


def load_pair_specs(pair_path: Path, max_pairs: int = 0) -> List[dict]:
    rows = []
    with pair_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append({
                "target_item": int(record["target_item"]),
                "candidate_item": int(record["candidate_item"]),
                "count": int(record.get("count", 0)),
            })
    rows.sort(key=lambda r: (-r["count"], r["target_item"], r["candidate_item"]))
    return rows[:max_pairs] if max_pairs > 0 else rows


def fallback_user_profile(user_id: int, history: List[int], item_meta, inv_item_map) -> str:
    counter = Counter()
    for item_id in history:
        raw_item = inv_item_map.get(item_id)
        for genre in item_meta.get(raw_item, {}).get("genres", ["unknown"]):
            counter[genre] += 1
    top = [genre for genre, _ in counter.most_common(4)]
    return f"The user tends to prefer {genre_phrase(top)} movies based on {len(history)} historical interactions."


def fallback_item_profile(item_id: int, item_meta, inv_item_map) -> str:
    raw_id = inv_item_map.get(item_id)
    meta = item_meta.get(raw_id, {"title": f"Movie {item_id}", "year": "", "genres": ["unknown"]})
    year = f" released in {meta['year']}" if meta.get("year") else ""
    return f"{meta['title']} is a {genre_phrase(meta['genres'])} movie{year}."


def generate_profiles(client: OllamaClient, out_path: Path, sequences, item_meta, inv_item_map, max_users: int, max_items: int):
    done_users, done_items = load_existing_profile_ids(out_path)
    user_ids = sorted(sequences)
    item_ids = sorted(inv_item_map)
    if max_users > 0:
        user_ids = user_ids[:max_users]
    if max_items > 0:
        item_ids = item_ids[:max_items]

    with out_path.open("a", encoding="utf-8") as f:
        for idx, user_id in enumerate(user_ids, start=1):
            if user_id in done_users:
                continue
            history = sequences[user_id]
            recent = history[-20:]
            history_lines = "\n".join(f"- {item_label(item_id, item_meta, inv_item_map)}" for item_id in recent)
            prompt = f"""You generate concise movie recommendation profiles.
Return JSON only with this schema: {{"profile": "..."}}

User id: {user_id}
Recent watched movies:
{history_lines}

Write one English sentence describing this user's movie preferences. Mention genres and style, not item ids."""
            try:
                payload = client.generate_json(prompt)
                profile = str(payload.get("profile") or "").strip() or fallback_user_profile(user_id, history, item_meta, inv_item_map)
            except Exception as exc:
                print(f"user {user_id} failed, fallback: {exc}")
                profile = fallback_user_profile(user_id, history, item_meta, inv_item_map)
            f.write(json.dumps({"type": "user", "id": user_id, "profile": profile}, ensure_ascii=False) + "\n")
            f.flush()
            if idx % 25 == 0:
                print(f"profiles users: {idx}/{len(user_ids)}")

        for idx, item_id in enumerate(item_ids, start=1):
            if item_id in done_items:
                continue
            raw_id = inv_item_map.get(item_id)
            meta = item_meta.get(raw_id, {"title": f"Movie {item_id}", "year": "", "genres": ["unknown"]})
            prompt = f"""You generate concise movie item profiles for recommendation.
Return JSON only with this schema: {{"profile": "..."}}

Movie title: {meta['title']}
Release year: {meta.get('year') or 'unknown'}
Genres: {', '.join(meta['genres'])}

Write one English sentence describing the movie and the kind of users likely to enjoy it."""
            try:
                payload = client.generate_json(prompt)
                profile = str(payload.get("profile") or "").strip() or fallback_item_profile(item_id, item_meta, inv_item_map)
            except Exception as exc:
                print(f"item {item_id} failed, fallback: {exc}")
                profile = fallback_item_profile(item_id, item_meta, inv_item_map)
            f.write(json.dumps({"type": "item", "id": item_id, "profile": profile}, ensure_ascii=False) + "\n")
            f.flush()
            if idx % 50 == 0:
                print(f"profiles items: {idx}/{len(item_ids)}")


def load_profiles(path: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    users = {}
    items = {}
    if not path.exists():
        return users, items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "user":
                users[int(record["id"])] = str(record["profile"])
            elif record.get("type") == "item":
                items[int(record["id"])] = str(record["profile"])
    return users, items


def clamp_score(value) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return 3
    return max(1, min(5, score))


def fallback_cot_scores(user_profile: str, target_profile: str, candidate_profile: str) -> dict:
    def toks(text):
        return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))

    u = toks(user_profile)
    t = toks(target_profile)
    c = toks(candidate_profile)
    accept = len(u & c) / (len(u | c) or 1)
    bridge = len(t & c) / (len(t | c) or 1)
    return {
        "acceptance_score": clamp_score(1 + 4 * accept),
        "transition_score": clamp_score(1 + 4 * bridge),
        "risk_score": clamp_score(5 - 4 * ((accept + bridge) / 2)),
    }


def generate_cot(client: OllamaClient, out_path: Path, profile_path: Path, sequences, item_meta, inv_item_map, max_targets: int, candidates_per_target: int, max_users: int):
    user_profiles, item_profiles = load_profiles(profile_path)
    if not user_profiles or not item_profiles:
        raise RuntimeError(f"profiles are required before cot generation: {profile_path}")

    item_popularity = Counter(item for items in sequences.values() for item in items)
    popular_items = [item for item, _ in item_popularity.most_common()]
    target_items = popular_items[:max_targets]
    user_ids = sorted(sequences)
    if max_users > 0:
        user_ids = user_ids[:max_users]

    done = load_existing_cot_keys(out_path)
    with out_path.open("a", encoding="utf-8") as f:
        total = len(target_items) * min(candidates_per_target, max(0, len(popular_items) - 1)) * len(user_ids)
        seen = 0
        for target in target_items:
            candidates = [item for item in popular_items if item != target][:candidates_per_target]
            target_profile = item_profiles.get(target) or fallback_item_profile(target, item_meta, inv_item_map)
            for candidate in candidates:
                candidate_profile = item_profiles.get(candidate) or fallback_item_profile(candidate, item_meta, inv_item_map)
                for user_id in user_ids:
                    key = (user_id, target, candidate)
                    seen += 1
                    if key in done:
                        continue
                    user_profile = user_profiles.get(user_id) or fallback_user_profile(user_id, sequences[user_id], item_meta, inv_item_map)
                    prompt = f"""You score whether a candidate movie can bridge a user toward a target movie.
Return JSON only with integer scores from 1 to 5:
{{"acceptance_score": 1, "transition_score": 1, "risk_score": 1}}

Definitions:
- acceptance_score: how likely the user accepts the candidate movie.
- transition_score: how naturally the candidate movie leads toward the target movie.
- risk_score: how risky or mismatched the candidate recommendation is.

User profile:
{user_profile}

Target movie:
{target_profile}

Candidate movie:
{candidate_profile}

Score carefully. Use only integers 1, 2, 3, 4, or 5."""
                    try:
                        payload = client.generate_json(prompt)
                        scores = {
                            "acceptance_score": clamp_score(payload.get("acceptance_score")),
                            "transition_score": clamp_score(payload.get("transition_score")),
                            "risk_score": clamp_score(payload.get("risk_score")),
                        }
                    except Exception as exc:
                        print(f"cot {key} failed, fallback: {exc}")
                        scores = fallback_cot_scores(user_profile, target_profile, candidate_profile)
                    record = {
                        "user_id": user_id,
                        "target_item": target,
                        "candidate_item": candidate,
                        **scores,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    if seen % 100 == 0:
                        print(f"cot progress: {seen}/{total}")


def generate_pair_cot(client: OllamaClient, out_path: Path, profile_path: Path, pair_path: Path, item_meta, inv_item_map, max_pairs: int):
    _user_profiles, item_profiles = load_profiles(profile_path)
    if not item_profiles:
        raise RuntimeError(f"item profiles are required before pair cot generation: {profile_path}")

    pair_specs = load_pair_specs(pair_path, max_pairs=max_pairs)
    done = load_existing_cot_keys(out_path)
    with out_path.open("a", encoding="utf-8") as f:
        total = len(pair_specs)
        for seen, pair in enumerate(pair_specs, start=1):
            target = int(pair["target_item"])
            candidate = int(pair["candidate_item"])
            key = (0, target, candidate)
            if key in done:
                continue

            target_profile = item_profiles.get(target) or fallback_item_profile(target, item_meta, inv_item_map)
            candidate_profile = item_profiles.get(candidate) or fallback_item_profile(candidate, item_meta, inv_item_map)
            prompt = f"""You score whether a candidate movie can serve as a useful bridge toward a target movie in a target-oriented recommender.
Return JSON only with integer scores from 1 to 5:
{{"acceptance_score": 1, "transition_score": 1, "risk_score": 1}}

Definitions:
- acceptance_score: general audience acceptability of the candidate as an intermediate recommendation.
- transition_score: how naturally the candidate can guide a user toward the target movie over several recommendation steps.
- risk_score: how risky, abrupt, or misleading the candidate is as a bridge.

Important scoring guidance:
- A bridge movie does NOT need to be similar to the target movie. It can be useful because it shares an actor, era, mood, audience, theme, popularity level, narrative style, or a plausible taste transition.
- Do not give high risk only because the candidate and target have different genres. Different genres can still be good bridges when the candidate is broadly acceptable or creates a gradual transition.
- Use the full 1-5 scale. Reserve transition_score=1 and risk_score=5 for cases that are truly poor, misleading, or almost impossible as a transition.
- If the candidate is a reasonable stepping stone but not a perfect match, prefer transition_score=3 and risk_score=2 or 3 rather than marking it as failure.

Target movie:
{target_profile}

Candidate movie:
{candidate_profile}

Score carefully. Use only integers 1, 2, 3, 4, or 5."""
            try:
                payload = client.generate_json(prompt)
                scores = {
                    "acceptance_score": clamp_score(payload.get("acceptance_score")),
                    "transition_score": clamp_score(payload.get("transition_score")),
                    "risk_score": clamp_score(payload.get("risk_score")),
                }
            except Exception as exc:
                print(f"pair cot {key} failed, fallback: {exc}")
                scores = fallback_cot_scores("", target_profile, candidate_profile)
            record = {
                "user_id": 0,
                "target_item": target,
                "candidate_item": candidate,
                "count": int(pair.get("count", 0)),
                **scores,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if seen % 50 == 0 or seen == total:
                print(f"pair cot progress: {seen}/{total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["profiles", "cot", "pair_cot", "both"], default="profiles")
    parser.add_argument("--raw_root", type=Path, default=Path("../ml-100k/ml-100k"))
    parser.add_argument("--data_dir", type=Path, default=Path("data/ml100k"))
    parser.add_argument("--profile_path", type=Path, default=None)
    parser.add_argument("--cot_path", type=Path, default=None)
    parser.add_argument("--pair_path", type=Path, default=None)
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--ollama_url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num_ctx", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--max_users", type=int, default=0, help="0 means all users for profiles; for cot, prefer a small number first.")
    parser.add_argument("--max_items", type=int, default=0, help="0 means all items for profiles.")
    parser.add_argument("--max_targets", type=int, default=5)
    parser.add_argument("--candidates_per_target", type=int, default=10)
    parser.add_argument("--max_pairs", type=int, default=0)
    args = parser.parse_args()

    profile_path = args.profile_path or (args.data_dir / "lir_profiles_ollama.jsonl")
    cot_path = args.cot_path or (args.data_dir / "cot_bridge_cache_ollama.jsonl")

    item_meta = load_item_metadata(args.raw_root)
    raw_ratings = load_raw_ratings(args.raw_root)
    _user_map, _item_map, _inv_user_map, inv_item_map = build_maps(raw_ratings)
    sequences = load_sequences(args.data_dir / "ml100k.txt")
    client = OllamaClient(args.ollama_url, args.model, args.timeout, args.temperature, args.num_ctx, args.retries, args.retry_sleep)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in {"profiles", "both"}:
        print(f"writing profiles to {profile_path}")
        generate_profiles(client, profile_path, sequences, item_meta, inv_item_map, args.max_users, args.max_items)
    if args.mode in {"cot", "both"}:
        print(f"writing cot scores to {cot_path}")
        generate_cot(client, cot_path, profile_path, sequences, item_meta, inv_item_map, args.max_targets, args.candidates_per_target, args.max_users)
    if args.mode == "pair_cot":
        pair_path = args.pair_path or (args.data_dir / "lir_bridge_pairs_top50.jsonl")
        print(f"writing pair cot scores to {cot_path}")
        generate_pair_cot(client, cot_path, profile_path, pair_path, item_meta, inv_item_map, args.max_pairs)

    print("done")


if __name__ == "__main__":
    main()
