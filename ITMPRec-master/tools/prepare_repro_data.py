import argparse
import ast
import csv
from collections import defaultdict
from pathlib import Path


def write_sequence_dataset(name, interactions, out_root):
    """Write ITMPRec files. Interactions are one-based (user, item, rating, order)."""
    data_dir = out_root / "ITMPRec-master" / "data" / name
    data_dir.mkdir(parents=True, exist_ok=True)

    by_user = defaultdict(list)
    popularity = defaultdict(int)
    for user, item, rating, order in sorted(interactions, key=lambda x: (x[0], x[3])):
        by_user[user].append(item)
        popularity[item] += 1

    with open(data_dir / f"{name}.txt", "w", encoding="utf-8") as f:
        for user in sorted(by_user):
            items = " ".join(str(item) for item in by_user[user])
            f.write(f"{user} {items}\n")

    with open(data_dir / "data.txt", "w", encoding="utf-8") as f:
        for user, item, rating, order in sorted(interactions, key=lambda x: (x[0], x[3])):
            if name in {"steam", "douban_movie"}:
                f.write(f"{user},{item},{order}\n")
            else:
                f.write(f"{user} {item}\n")

    with open(data_dir / "target_items.txt", "w", encoding="utf-8") as f:
        for item, _ in sorted(popularity.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"{item}\n")

    return data_dir, by_user


def write_graphau_dataset(name, interactions, graphau_root):
    """Write GraphAU files. GraphAU expects zero-based comma CSV: user,item,rating."""
    out_dir = graphau_root / "datasets" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    by_user = defaultdict(list)
    for user, item, rating, order in sorted(interactions, key=lambda x: (x[0], x[3])):
        by_user[user - 1].append((item - 1, rating, order))

    train_rows, val_rows, test_rows = [], [], []
    for user in sorted(by_user):
        seq = by_user[user]
        if len(seq) >= 3:
            train, val, test = seq[:-2], seq[-2:-1], seq[-1:]
        elif len(seq) == 2:
            train, val, test = seq[:1], seq[1:], []
        else:
            train, val, test = seq, [], []
        train_rows.extend((user, item, rating) for item, rating, _ in train)
        val_rows.extend((user, item, rating) for item, rating, _ in val)
        test_rows.extend((user, item, rating) for item, rating, _ in test)

    for filename, rows in [("train.txt", train_rows), ("val.txt", val_rows), ("test.txt", test_rows)]:
        with open(out_dir / filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    return out_dir


def build_ml100k(raw_root):
    raw_file = raw_root / "ml-100k" / "ml-100k" / "u.data"
    user_map, item_map = {}, {}
    interactions = []
    with open(raw_file, "r", encoding="latin-1") as f:
        for line in f:
            raw_user, raw_item, rating, timestamp = map(int, line.split())
            user = user_map.setdefault(raw_user, len(user_map) + 1)
            item = item_map.setdefault(raw_item, len(item_map) + 1)
            interactions.append((user, item, rating, timestamp))
    return interactions


def build_lastfm(raw_root):
    raw_file = raw_root / "hetrec2011-lastfm-2k" / "user_taggedartists-timestamps.dat"
    earliest = {}
    with open(raw_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            raw_user, raw_item, raw_tag, timestamp = line.rstrip("\n").split("\t")
            key = (int(raw_user), int(raw_item))
            timestamp = int(timestamp)
            if key not in earliest or timestamp < earliest[key]:
                earliest[key] = timestamp

    user_map, item_map = {}, {}
    interactions = []
    for (raw_user, raw_item), timestamp in sorted(earliest.items(), key=lambda kv: (kv[0][0], kv[1])):
        user = user_map.setdefault(raw_user, len(user_map) + 1)
        item = item_map.setdefault(raw_item, len(item_map) + 1)
        interactions.append((user, item, 5, timestamp))
    return interactions


def build_steam(raw_root):
    raw_file = raw_root / "australian_users_items.json" / "australian_users_items.json"
    user_map, item_map = {}, {}
    interactions = []
    order = 0
    with open(raw_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            record = ast.literal_eval(line)
            raw_user = record.get("user_id")
            user = user_map.setdefault(raw_user, len(user_map) + 1)
            for item_info in record.get("items", []):
                if int(item_info.get("playtime_forever", 0)) <= 0:
                    continue
                raw_item = item_info.get("item_id")
                item = item_map.setdefault(raw_item, len(item_map) + 1)
                order += 1
                interactions.append((user, item, 5, order))
    return interactions


def build_douban_movie(raw_root):
    raw_file = raw_root / "archive" / "douban_dataset(text information)" / "moviereviews_cleaned.txt"
    user_map, item_map = {}, {}
    interactions = []
    with open(raw_file, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_id, row in enumerate(reader, start=1):
            raw_user = row["user_id"].strip('"')
            raw_item = row["movie_id"].strip('"')
            try:
                rating = int(float(row["rating"].strip('"')))
            except ValueError:
                continue
            if rating <= 0:
                continue
            user = user_map.setdefault(raw_user, len(user_map) + 1)
            item = item_map.setdefault(raw_item, len(item_map) + 1)
            interactions.append((user, item, rating, row_id))
    return interactions


BUILDERS = {
    "ml100k": build_ml100k,
    "lastfm": build_lastfm,
    "steam": build_steam,
    "douban_movie": build_douban_movie,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=Path, default=Path(".."))
    parser.add_argument("--project_root", type=Path, default=Path("."))
    parser.add_argument("--graphau_data_root", type=Path, default=Path("data_graphau"))
    parser.add_argument("--datasets", nargs="+", default=["ml100k", "lastfm"])
    args = parser.parse_args()

    for dataset in args.datasets:
        interactions = BUILDERS[dataset](args.raw_root)
        data_dir, by_user = write_sequence_dataset(dataset, interactions, args.project_root)
        graphau_dir = write_graphau_dataset(dataset, interactions, args.graphau_data_root)
        item_count = len({item for _, item, _, _ in interactions})
        print(
            f"{dataset}: users={len(by_user)} items={item_count} interactions={len(interactions)} "
            f"itmprec={data_dir} graphau={graphau_dir}"
        )


if __name__ == "__main__":
    main()
