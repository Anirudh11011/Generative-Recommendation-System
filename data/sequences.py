"""
data/sequences.py  —  Phase 0 (final step)

Turns the raw 5-core reviews into per-user chronological sequences of ITEM ROW-INDICES
(the same indices used by data/item_embeddings.pt, and later by the semantic-ID tensor).

Output: data/user_sequences.pkl
    { reviewerID (str) : [item_idx, item_idx, ...] }   # oldest -> newest

The train/val/test split is NOT stored here. Phase 2's loader does leave-one-out on load:
    test target   = seq[-1]   (predicted from seq[:-1])
    val   target  = seq[-2]   (predicted from seq[:-2])
    train         = all next-token steps within seq[:-2]

Run from the PROJECT ROOT (so the data/ package resolves) with your venv activated:
    source .venv/bin/activate
    python -m data.sequences
or simply:
    python data/sequences.py
"""

import gzip
import json
import pickle
import statistics
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — resolved relative to THIS file, so it works from any working dir.
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent          # .../generative-recsys/data
RAW_DIR = DATA_DIR / "raw"
REVIEWS_PATH = RAW_DIR / "reviews_Beauty_5.json.gz"  # 5-core interactions
ID_MAPS_PATH = DATA_DIR / "item_id_maps.pkl"         # written by embeddings.py
OUT_PATH = DATA_DIR / "user_sequences.pkl"

# Minimum interactions per user AFTER dropping items that have no embedding.
# Need >=3 for leave-one-out (train/val/test); we keep 5 to match 5-core intent.
# If the stats below show you're losing too many users, lower this to 3.
MIN_ITEMS_PER_USER = 5


def load_asin_to_idx(path: Path) -> dict:
    """
    Robustly extract an {asin -> row_index} mapping from item_id_maps.pkl,
    regardless of exactly how embeddings.py structured it.

    NOTE: if this raises, open a Python shell and inspect the file:
        import pickle; print(pickle.load(open(path, "rb")))
    then adjust this function to match its actual structure.
    """
    with open(path, "rb") as f:
        maps = pickle.load(f)

    if isinstance(maps, dict):
        # Case A: nested dict with a named sub-map
        for key in ("asin_to_idx", "asin2idx", "asin_to_index", "item2idx", "asin_to_row"):
            if key in maps and isinstance(maps[key], dict):
                return maps[key]
        # Peek at a value to infer direction
        sample_val = next(iter(maps.values()))
        # Case B: already {asin(str) -> idx(int)}
        if isinstance(sample_val, int):
            return maps
        # Case C: {idx(int) -> asin(str)} — invert it
        if isinstance(sample_val, str):
            return {asin: idx for idx, asin in maps.items()}

    raise ValueError(
        f"Could not interpret {path}. Inspect it manually and adjust load_asin_to_idx()."
    )


def main():
    print(f"Loading id maps from {ID_MAPS_PATH.name} ...")
    asin_to_idx = load_asin_to_idx(ID_MAPS_PATH)
    valid_asins = set(asin_to_idx.keys())
    print(f"  items with embeddings: {len(valid_asins):,}")

    # -----------------------------------------------------------------------
    # 1. Stream the reviews, bucketing (timestamp, asin) per user.
    #    We keep the timestamp so we can sort chronologically next.
    # -----------------------------------------------------------------------
    print(f"Reading reviews from {REVIEWS_PATH.name} ...")
    interactions = defaultdict(list)
    total_reviews = 0
    dropped_missing_meta = 0

    with gzip.open(REVIEWS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            asin = r["asin"]
            total_reviews += 1
            if asin not in valid_asins:
                dropped_missing_meta += 1          # no embedding -> unusable later
                continue
            interactions[r["reviewerID"]].append((r["unixReviewTime"], asin))

    # -----------------------------------------------------------------------
    # 2. Sort each user chronologically, map asin -> row index, drop short users.
    #    sorted() is stable, so same-day ties keep original file order.
    # -----------------------------------------------------------------------
    user_sequences = {}
    dropped_short_users = 0

    for user, items in interactions.items():
        items.sort(key=lambda ta: ta[0])                    # oldest -> newest
        seq = [asin_to_idx[asin] for (_, asin) in items]
        if len(seq) >= MIN_ITEMS_PER_USER:
            user_sequences[user] = seq
        else:
            dropped_short_users += 1

    # -----------------------------------------------------------------------
    # 3. Save + sanity stats.
    # -----------------------------------------------------------------------
    with open(OUT_PATH, "wb") as f:
        pickle.dump(user_sequences, f)

    lengths = [len(s) for s in user_sequences.values()]
    used_items = {idx for s in user_sequences.values() for idx in s}
    kept_interactions = sum(lengths)

    print("\n" + "=" * 60)
    print("PHASE 0 — sequences.py summary")
    print("=" * 60)
    print(f"total reviews read           : {total_reviews:,}")
    print(f"dropped (no metadata/embed)  : {dropped_missing_meta:,} "
          f"({dropped_missing_meta / total_reviews:.1%})")
    print(f"users before length filter   : {len(interactions):,}")
    print(f"users dropped (< {MIN_ITEMS_PER_USER} items)     : {dropped_short_users:,}")
    print(f"users kept                   : {len(user_sequences):,}")
    print(f"interactions kept            : {kept_interactions:,}")
    print(f"distinct items used          : {len(used_items):,} / {len(valid_asins):,}")
    if lengths:
        print(f"seq length  min / median / mean / max : "
              f"{min(lengths)} / {statistics.median(lengths):.0f} / "
              f"{statistics.mean(lengths):.1f} / {max(lengths)}")
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()