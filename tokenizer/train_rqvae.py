"""
tokenizer/train_rqvae.py
------------------------
Phase 1, final step: train the RQ-VAE on the real item embeddings, then assign a
semantic ID to every item and save it for Phase 2.

Run:  python tokenizer/train_rqvae.py

Reads:   data/item_embeddings.pt        (num_items, 384)  from Phase 0
Writes:  checkpoints/rqvae.pt           trained model weights
         data/item_semantic_ids.pt      (num_items, 4)   the semantic IDs
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from .rqvae import RQVAE
except ImportError:
    from rqvae import RQVAE

# ---- config (mirror of compact-context; move into config.py later) ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

EMB_PATH = DATA_DIR / "item_embeddings.pt"
OUT_IDS  = DATA_DIR / "item_semantic_ids.pt"
OUT_CKPT = CKPT_DIR / "rqvae.pt"

INPUT_DIM, HIDDEN_DIM = 384, 256
NUM_CODEBOOKS, CODEBOOK_SIZE = 3, 256
BETA = 0.25
LR, EPOCHS, BATCH_SIZE = 1e-3, 100, 512
LOG_EVERY = 10
REVIVE_EVERY = 20          # periodically reset dead codes; 0 disables
NORMALIZE = True           # L2-normalise embeddings (well-conditions k-means)
SEED = 0


def l2_normalize(x, eps=1e-8):
    return x / (x.norm(dim=1, keepdim=True) + eps)


@torch.no_grad()
def usage_and_revive(model, embeddings, revive=False):
    """Walk the codebooks stage by stage on the real latents. Report how many
    of the 256 codes each codebook actually uses; optionally revive dead ones
    by resetting them onto random real residuals."""
    model.eval()
    residual = model.encoder(embeddings)          # (N, hidden)
    usages = []
    for cb in model.quantizer.codebooks:
        idx = torch.cdist(residual, cb.weight).argmin(dim=1)   # (N,)
        # count on CPU (small; sidesteps MPS quirks in bincount/isin)
        counts = torch.bincount(idx.cpu(), minlength=cb.num_embeddings)
        usages.append(int((counts > 0).sum()))

        if revive:
            dead = torch.nonzero(counts == 0, as_tuple=False).squeeze(1)
            if dead.numel() > 0:
                pick = torch.randint(0, residual.shape[0], (dead.numel(),),
                                     device=residual.device)
                cb.weight.data[dead.to(residual.device)] = (
                    residual[pick] + 0.01 * torch.randn_like(residual[pick])
                )

        residual = residual - cb(idx)             # move to next stage's residual
    model.train()
    return usages


def assign_semantic_ids(codes):
    """codes: (N, 3) tensor of the raw codebook tokens.
    Append a 4th 'disambiguation counter' token so items that landed on the same
    3-token prefix still get unique IDs. Returns (N, 4) tensor + collision stats."""
    codes = codes.cpu().tolist()
    seen, fourth = {}, []
    for c in codes:
        key = tuple(c)
        n = seen.get(key, 0)
        fourth.append(n)          # 0 for the first item at this prefix, 1 for the next, ...
        seen[key] = n + 1

    ids4 = torch.cat(
        [torch.tensor(codes, dtype=torch.long),
         torch.tensor(fourth, dtype=torch.long).unsqueeze(1)],
        dim=1,
    )
    n_collisions = sum(v - 1 for v in seen.values() if v > 1)
    max_bucket = max(seen.values())
    unique_prefixes = len(seen)
    return ids4, n_collisions, max_bucket, unique_prefixes


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    # --- load + normalise real embeddings ---
    embeddings = torch.load(EMB_PATH, map_location="cpu")
    if isinstance(embeddings, dict):                      # in case it was saved wrapped
        embeddings = embeddings.get("embeddings", next(iter(embeddings.values())))
    embeddings = embeddings.float()
    if NORMALIZE:
        embeddings = l2_normalize(embeddings)
    N, D = embeddings.shape
    print(f"loaded {N} embeddings, dim {D}")

    # --- build model + k-means init (both on CPU, BEFORE moving to MPS) ---
    model = RQVAE(INPUT_DIM, HIDDEN_DIM,
                  num_codebooks=NUM_CODEBOOKS, codebook_size=CODEBOOK_SIZE, beta=BETA)
    print("k-means init on real embeddings (CPU)...")
    model.init_codebooks_kmeans(embeddings)

    # --- now move model + data onto the GPU together ---
    model = model.to(device)
    emb_dev = embeddings.to(device)

    loader = DataLoader(TensorDataset(emb_dev), batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # --- training loop ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot = tot_r = tot_v = 0.0
        for (batch,) in loader:
            opt.zero_grad()
            out = model(batch)
            out["loss"].backward()
            opt.step()
            tot   += out["loss"].item()
            tot_r += out["recon_loss"].item()
            tot_v += out["vq_loss"].item()
        nb = len(loader)

        if epoch == 1 or epoch % LOG_EVERY == 0:
            usages = usage_and_revive(model, emb_dev, revive=False)
            print(f"epoch {epoch:3d} | loss {tot/nb:.4f} | recon {tot_r/nb:.4f} "
                  f"| vq {tot_v/nb:.4f} | codes used/256 {usages}")

        if REVIVE_EVERY and epoch % REVIVE_EVERY == 0:
            usage_and_revive(model, emb_dev, revive=True)

    torch.save(model.state_dict(), OUT_CKPT)
    print("saved model ->", OUT_CKPT)

    # --- assign semantic IDs to every item ---
    codes = model.get_codes(emb_dev)                     # (N, 3)
    ids4, n_col, max_bucket, uniq3 = assign_semantic_ids(codes)

    ct = codes.cpu()
    uniq1 = len(torch.unique(ct[:, 0]))
    uniq2 = len({tuple(c) for c in ct[:, :2].tolist()})

    print("\n--- semantic IDs ---")
    print(f"  unique 1-token prefixes : {uniq1}")
    print(f"  unique 2-token prefixes : {uniq2}")
    print(f"  unique 3-token prefixes : {uniq3} / {N}")
    print(f"  colliding items         : {n_col} ({100*n_col/N:.2f}%)")
    print(f"  largest collision bucket: {max_bucket} items share one 3-token ID")

    torch.save(ids4, OUT_IDS)
    print("saved semantic IDs ->", OUT_IDS, "shape", tuple(ids4.shape))


if __name__ == "__main__":
    main()