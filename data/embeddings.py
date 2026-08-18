"""
data/embeddings.py — Phase 0, step 2

Build item content embeddings for the RQ-VAE tokenizer.
Reads the 5-core reviews (to know which items actually appear in interactions)
and the metadata (for item text), builds one TIGER-style sentence per item,
encodes with all-MiniLM-L6-v2, and saves:
  data/item_embeddings.pt   (num_items, 384) float tensor
  data/item_id_maps.pkl     asin <-> row-index mappings
Run from the project root:  python data/embeddings.py
"""

import gzip, json, ast, pickle
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data")
REVIEWS_FILE = RAW_DIR / "reviews_Beauty_5.json.gz"   # adjust names if different
META_FILE    = RAW_DIR / "meta_Beauty.json.gz"
MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def parse_gz(path, strict_json=True):
    """One dict per line. Reviews are real JSON; metadata is python-dict
    syntax (single quotes, True/False/None) -> use ast.literal_eval."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line) if strict_json else ast.literal_eval(line))
    return pd.DataFrame(rows)


def build_sentence(row):
    """TIGER-style: title + brand + categories + price -> one string."""
    parts = []
    title = row.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(f"Title: {title.strip()}")
    brand = row.get("brand")
    if isinstance(brand, str) and brand.strip():
        parts.append(f"Brand: {brand.strip()}")
    cats = row.get("categories")           # nested list, e.g. [["Beauty","Makeup","Eyes"]]
    if isinstance(cats, list) and cats:
        flat = [c for path in cats if isinstance(path, list)
                  for c in path if isinstance(c, str)]
        if flat:
            parts.append("Categories: " + ", ".join(dict.fromkeys(flat)))  # dedupe, keep order
    price = row.get("price")
    if isinstance(price, (int, float)) and price == price:   # price == price is False for NaN
        parts.append(f"Price: {price}")
    return ". ".join(parts)


def main():
    print(f"Device: {DEVICE}")

    # 1. items that actually appear in interactions
    reviews = parse_gz(REVIEWS_FILE, strict_json=True)
    review_asins = set(reviews["asin"].unique())
    print(f"Unique items in 5-core reviews: {len(review_asins)}")

    # 2. metadata, restricted to those items
    meta = parse_gz(META_FILE, strict_json=False)
    meta = meta.drop_duplicates(subset="asin", keep="first")
    meta = meta[meta["asin"].isin(review_asins)].reset_index(drop=True)
    print(f"Items with metadata: {len(meta)}")
    print(f"Items in reviews but MISSING metadata: {len(review_asins - set(meta['asin']))}")

    # 3. one sentence per item, drop any that come out empty
    meta["sentence"] = meta.apply(build_sentence, axis=1)
    meta = meta[meta["sentence"].str.len() > 0].reset_index(drop=True)
    print(f"Items with a usable sentence: {len(meta)}")

    asins = meta["asin"].tolist()
    sentences = meta["sentence"].tolist()

    # 4. encode (frozen model, no grad)
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    embeddings = model.encode(
        sentences, batch_size=256, convert_to_tensor=True,
        show_progress_bar=True, normalize_embeddings=False,
    ).cpu()
    print(f"Embeddings shape: {tuple(embeddings.shape)}")   # -> (num_items, 384)

    # 5. mappings + save (row i of the tensor  <->  asins[i])
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    idx_to_asin = {i: a for a, i in asin_to_idx.items()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, OUT_DIR / "item_embeddings.pt")
    with open(OUT_DIR / "item_id_maps.pkl", "wb") as f:
        pickle.dump({"asin_to_idx": asin_to_idx, "idx_to_asin": idx_to_asin}, f)
    print("Saved item_embeddings.pt and item_id_maps.pkl")

    for s in sentences[:3]:            # eyeball a few sentences
        print("  •", s[:120])


if __name__ == "__main__":
    main()