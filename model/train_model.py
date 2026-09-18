"""
model/train_model.py — train the decoder-only transformer.

Run from project root:  python -m model.train_model

Training signal: next-token cross-entropy over the flattened train sequences
(dense — every non-PAD position supervised, teacher-forced).
Checkpoint selection: a cheap per-epoch VAL loss = teacher-forced CE on just the
held-out val item's 4 tokens (given the user's train history as context). This is
a next-ITEM proxy for Phase 4's Recall/NDCG, good enough to pick the right epoch.
"""

import math
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import vocab
from model.transformer import RecTransformer
from data.build_token_sequences import flatten_history   # reuse EXACT Step-1 logic

# ----------------------------- config -----------------------------
EPOCHS        = 100
BATCH_SIZE    = 64
LR            = 1e-3
WEIGHT_DECAY  = 0.01
WARMUP_FRAC   = 0.03          # fraction of total steps spent warming up
GRAD_CLIP     = 1.0
CKPT_EVERY    = 10
MAX_SEQ_LEN   = 200
SEED          = 42

TRAIN_PATH    = "data/train_token_sequences.pkl"
USER_SEQS     = "data/user_sequences.pkl"
SEM_IDS_PATH  = "data/item_semantic_ids.pt"
CKPT_DIR      = "checkpoints"

device = "mps" if torch.backends.mps.is_available() else "cpu"


# ----------------------------- data -----------------------------
class SeqDataset(Dataset):
    """Holds a list of variable-length token sequences (python lists of ints)."""
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, i):
        return self.sequences[i]


def pad_batch(sequences, pad_value=vocab.PAD):
    """List of int-lists -> (padded LongTensor (B, Lmax), lengths (B,))."""
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    Lmax = int(lengths.max())
    padded = torch.full((len(sequences), Lmax), pad_value, dtype=torch.long)
    for i, s in enumerate(sequences):
        padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return padded, lengths


def build_val_sequences():
    """Per user, val_seq = flatten_history(full_history[:-1]) — i.e. train history
    PLUS the held-out val item as the final 4 tokens. We'll score only those 4."""
    sem_ids = torch.load(SEM_IDS_PATH)
    user_seqs = pickle.load(open(USER_SEQS, "rb"))
    val = []
    for full_history in user_seqs.values():
        if len(full_history) < 3:            # need >=1 train item + val item; <5 filter guarantees this
            continue
        val.append(flatten_history(full_history[:-1], sem_ids, max_len=MAX_SEQ_LEN))
    return val


# ----------------------------- loss helpers -----------------------------
def train_loss_fn(model, padded):
    """Dense next-token CE over all non-PAD positions."""
    inp, tgt = padded[:, :-1], padded[:, 1:]              # shift by one
    logits = model(inp)                                    # (B, L-1, V)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt.reshape(-1),
        ignore_index=vocab.PAD,                            # <-- masks PAD targets
    )


def val_loss_fn(model, padded, lengths):
    """CE on ONLY the val item's 4 tokens (the last 4 real target positions).
    We keep those 4 and set every other target position to PAD so ignore_index
    averages over exactly the held-out item."""
    inp, tgt = padded[:, :-1], padded[:, 1:].clone()      # (B, L-1)
    tlen = lengths - 1                                     # true target length per row
    ar = torch.arange(tgt.size(1), device=tgt.device)[None, :]   # (1, L-1)
    keep = (ar >= (tlen[:, None] - 4)) & (ar < tlen[:, None])   # last 4 real positions
    tgt = torch.where(keep, tgt, torch.full_like(tgt, vocab.PAD))
    logits = model(inp)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt.reshape(-1),
        ignore_index=vocab.PAD,
    )


# ----------------------------- train -----------------------------
def main():
    import os
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    train_seqs = pickle.load(open(TRAIN_PATH, "rb"))["sequences"]
    val_seqs   = build_val_sequences()
    print(f"device: {device} | train seqs: {len(train_seqs)} | val seqs: {len(val_seqs)}")

    train_loader = DataLoader(SeqDataset(train_seqs), batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=pad_batch, num_workers=0)
    val_loader   = DataLoader(SeqDataset(val_seqs), batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=pad_batch, num_workers=0)

    model = RecTransformer(max_seq_len=MAX_SEQ_LEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_FRAC * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)             # linear warmup 0 -> 1
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))        # cosine decay 1 -> 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ---- train ----
        model.train()
        running = 0.0
        for padded, _ in train_loader:
            padded = padded.to(device)
            opt.zero_grad()
            loss = train_loss_fn(model, padded)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)   # stabilizes updates
            opt.step()
            sched.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        # ---- val ----
        model.eval()
        vrunning = 0.0
        with torch.no_grad():
            for padded, lengths in val_loader:
                padded, lengths = padded.to(device), lengths.to(device)
                vrunning += val_loss_fn(model, padded, lengths).item()
        val_loss = vrunning / len(val_loader)

        lr_now = sched.get_last_lr()[0]
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| lr {lr_now:.2e} | {time.time()-t0:.1f}s")

        # ---- checkpoints ----
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_loss": val_loss}, f"{CKPT_DIR}/model_best.pt")
            print(f"           ^ new best val {val_loss:.4f} -> model_best.pt")
        if epoch % CKPT_EVERY == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "val_loss": val_loss}, f"{CKPT_DIR}/model_epoch{epoch}.pt")

    torch.save({"epoch": EPOCHS, "model": model.state_dict(),
                "val_loss": val_loss}, f"{CKPT_DIR}/model_final.pt")
    print(f"done. best val {best_val:.4f}. Phase 4 should load model_best.pt")


if __name__ == "__main__":
    main()