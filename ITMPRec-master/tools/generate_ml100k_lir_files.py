import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


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


def load_item_metadata(raw_root):
    item_file = raw_root / "u.item"
    items = {}
    with item_file.open("r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            raw_id = int(parts[0])
            title = parts[1]
            release_date = parts[2]
            genre_flags = [int(v) for v in parts[5:24]]
            genres = [name for name, flag in zip(GENRE_NAMES, genre_flags) if flag]
            if not genres:
                genres = ["unknown"]
            year = ""
            if release_date and "-" in release_date:
                year = release_date[-4:]
            items[raw_id] = {"title": title, "year": year, "genres": genres}
    return items


def load_internal_sequences(data_file):
    sequences = {}
    with data_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = [int(x) for x in line.strip().split()]
            if not parts:
                continue
            sequences[parts[0]] = parts[1:]
    return sequences


def load_raw_ratings(raw_root):
    raw_ratings = []
    with (raw_root / "u.data").open("r", encoding="latin-1") as f:
        for line in f:
            raw_user, raw_item, rating, timestamp = map(int, line.split())
            raw_ratings.append((raw_user, raw_item, rating, timestamp))
    return raw_ratings


def build_item_map(raw_ratings):
    item_map = {}
    user_map = {}
    for raw_user, raw_item, _rating, timestamp in sorted(raw_ratings, key=lambda x: (x[0], x[3])):
        user_map.setdefault(raw_user, len(user_map) + 1)
        item_map.setdefault(raw_item, len(item_map) + 1)
    inv_item_map = {internal: raw for raw, internal in item_map.items()}
    return user_map, item_map, inv_item_map


def genre_phrase(genres):
    if not genres:
        return "mixed genres"
    if len(genres) == 1:
        return genres[0]
    return ", ".join(genres[:-1]) + ", and " + genres[-1]


def write_profiles(out_path, sequences, item_meta, inv_item_map):
    item_popularity = Counter()
    user_genres = {}
    for user_id, item_ids in sequences.items():
        counter = Counter()
        for item_id in item_ids:
            item_popularity[item_id] += 1
            raw_item = inv_item_map.get(item_id)
            for genre in item_meta.get(raw_item, {}).get("genres", ["unknown"]):
                counter[genre] += 1
        user_genres[user_id] = counter

    with out_path.open("w", encoding="utf-8") as f:
        for user_id in sorted(sequences):
            total = sum(user_genres[user_id].values()) or 1
            top_genres = [genre for genre, _count in user_genres[user_id].most_common(4)]
            profile = (
                f"The user tends to prefer {genre_phrase(top_genres)} movies, "
                f"based on {len(sequences[user_id])} historical interactions. "
                f"The strongest genre share is {user_genres[user_id].most_common(1)[0][0] if top_genres else 'unknown'} "
                f"with {user_genres[user_id].most_common(1)[0][1] / total:.0%} of observed genre signals."
            )
            f.write(json.dumps({"type": "user", "id": user_id, "profile": profile}, ensure_ascii=False) + "\n")

        for item_id in sorted(inv_item_map):
            raw_item = inv_item_map[item_id]
            meta = item_meta.get(raw_item, {"title": f"Movie {item_id}", "year": "", "genres": ["unknown"]})
            year_text = f" released in {meta['year']}" if meta.get("year") else ""
            profile = (
                f"{meta['title']} is a {genre_phrase(meta['genres'])} movie{year_text}. "
                f"It appears in {item_popularity[item_id]} user histories in the converted ML-100K data."
            )
            f.write(json.dumps({"type": "item", "id": item_id, "profile": profile}, ensure_ascii=False) + "\n")


def score_to_1_5(value):
    return max(1, min(5, int(round(value))))


def write_cot_cache(out_path, sequences, item_meta, inv_item_map, max_targets, candidates_per_target):
    item_popularity = Counter()
    user_genres = {}
    for user_id, item_ids in sequences.items():
        counter = Counter()
        for item_id in item_ids:
            item_popularity[item_id] += 1
            raw_item = inv_item_map.get(item_id)
            for genre in item_meta.get(raw_item, {}).get("genres", ["unknown"]):
                counter[genre] += 1
        user_genres[user_id] = counter

    popular_items = [item for item, _count in item_popularity.most_common()]
    target_items = popular_items[:max_targets]
    max_pop = max(item_popularity.values()) if item_popularity else 1

    with out_path.open("w", encoding="utf-8") as f:
        for target in target_items:
            target_raw = inv_item_map.get(target)
            target_genres = set(item_meta.get(target_raw, {}).get("genres", ["unknown"]))
            candidate_pool = [item for item in popular_items if item != target][:candidates_per_target]
            for candidate in candidate_pool:
                cand_raw = inv_item_map.get(candidate)
                cand_genres = set(item_meta.get(cand_raw, {}).get("genres", ["unknown"]))
                overlap = len(target_genres & cand_genres)
                union = len(target_genres | cand_genres) or 1
                genre_sim = overlap / union
                popularity_score = math.log1p(item_popularity[candidate]) / math.log1p(max_pop)

                for user_id in sorted(sequences):
                    pref_counter = user_genres[user_id]
                    pref_total = sum(pref_counter.values()) or 1
                    cand_pref = sum(pref_counter[g] for g in cand_genres) / pref_total
                    target_pref = sum(pref_counter[g] for g in target_genres) / pref_total

                    acceptance = score_to_1_5(1 + 4 * (0.60 * cand_pref + 0.40 * popularity_score))
                    transition = score_to_1_5(1 + 4 * (0.75 * genre_sim + 0.25 * min(cand_pref, target_pref)))
                    risk = score_to_1_5(1 + 4 * (1 - (0.50 * cand_pref + 0.50 * genre_sim)))
                    record = {
                        "user_id": user_id,
                        "target_item": target,
                        "candidate_item": candidate,
                        "acceptance_score": acceptance,
                        "transition_score": transition,
                        "risk_score": risk,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=Path, default=Path("../ml-100k/ml-100k"))
    parser.add_argument("--data_dir", type=Path, default=Path("data/ml100k"))
    parser.add_argument("--max_targets", type=int, default=50)
    parser.add_argument("--candidates_per_target", type=int, default=50)
    args = parser.parse_args()

    item_meta = load_item_metadata(args.raw_root)
    raw_ratings = load_raw_ratings(args.raw_root)
    _user_map, _item_map, inv_item_map = build_item_map(raw_ratings)
    sequences = load_internal_sequences(args.data_dir / "ml100k.txt")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.data_dir / "lir_profiles.jsonl"
    cot_path = args.data_dir / "cot_bridge_cache.jsonl"
    write_profiles(profile_path, sequences, item_meta, inv_item_map)
    write_cot_cache(cot_path, sequences, item_meta, inv_item_map, args.max_targets, args.candidates_per_target)
    print(f"wrote {profile_path}")
    print(f"wrote {cot_path}")


if __name__ == "__main__":
    main()
