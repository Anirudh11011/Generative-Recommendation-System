"""
inference/beam_search.py

Trie-constrained beam search (Option A: decode all 4 tokens, mask -> log_softmax).

Run quick test from project root:
    python -m inference.beam_search
"""
import pickle
import time

import torch

import vocab
from inference.trie import SemanticIDTrie

MAX_SEQ_LEN = 200
TOKENS_PER_ITEM = 4
# context = [BOS] + k*(4 item tokens + SEP) = 5k+1 tokens; must leave room for 4 decoded tokens
MAX_CONTEXT_ITEMS = (MAX_SEQ_LEN - TOKENS_PER_ITEM - 1) // (TOKENS_PER_ITEM + 1)  # = 39


def build_context(history, item_tokens):
    """[BOS] item [SEP] item [SEP] ... item [SEP]  -- ends in SEP so the model predicts an item next.
    Keeps the most recent MAX_CONTEXT_ITEMS items so context + 4 decoded tokens fits in max_seq_len."""
    ctx = [vocab.BOS]
    for idx in history[-MAX_CONTEXT_ITEMS:]:
        ctx += item_tokens[idx]
        ctx.append(vocab.SEP)
    return ctx


@torch.no_grad()
def beam_search(model, context, trie, beam_width=50, device="cpu"):
    """
    Returns list of (item_idx, log_prob), best first, length == beam_width.
    Each beam = (generated_tokens, cumulative_log_prob).
    """
    beams = [([], 0.0)]

    for step in range(trie.depth):
        # All beams have the same length at a given step -> no padding needed.
        seqs = torch.tensor([context + prefix for prefix, _ in beams], device=device)  # (nb, L)
        out = model(seqs)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[:, -1, :]                                  # (nb, V) -- only the LAST row matters
        V = logits.size(-1)

        # Trie mask: 0 for legal next tokens, -inf for everything else. Built on CPU, one transfer.
        mask = torch.full((len(beams), V), float("-inf"))
        for i, (prefix, _) in enumerate(beams):
            mask[i, trie.get_valid_tokens(prefix)] = 0.0

        # MASK FIRST, THEN log_softmax -> renormalized over legal tokens; forced steps cost 0.
        logp = torch.log_softmax(logits + mask.to(device), dim=-1)  # (nb, V)

        prev = torch.tensor([s for _, s in beams], device=device).unsqueeze(1)  # (nb, 1)
        scores = (logp + prev).view(-1)                                         # (nb*V,)

        # Never select -inf (illegal) candidates.
        k = min(beam_width, int(torch.isfinite(scores).sum().item()))
        top_scores, top_idx = scores.topk(k)

        new_beams = []
        for s, flat_idx in zip(top_scores.tolist(), top_idx.tolist()):
            b, tok = divmod(flat_idx, V)           # which beam it came from, which token was added
            new_beams.append((beams[b][0] + [tok], s))
        beams = new_beams

    return [(trie.get_item(prefix), s) for prefix, s in beams]


def load_model(device, path="checkpoints/model_best.pt"):
    from model.transformer import RecTransformer
    model = RecTransformer(vocab_size=vocab.VOCAB_SIZE, d_model=128, n_heads=4,
                           n_layers=4, dropout=0.1, max_seq_len=MAX_SEQ_LEN)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Accept either a raw state_dict or a dict wrapping it under a common key.
    state = ckpt
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print(f"loaded checkpoint from epoch {ckpt['epoch']}")
    model.load_state_dict(state)
    return model.to(device).eval()   # eval() turns dropout OFF -- essential at inference


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(device)
    trie = SemanticIDTrie.from_file()

    sem_ids = torch.load("data/item_semantic_ids.pt")
    item_tokens = [[int(t) for t in vocab.item_to_tokens(r.tolist())] for r in sem_ids]

    with open("data/user_sequences.pkl", "rb") as f:
        user_seqs = pickle.load(f)

    users = list(user_seqs.keys())[:20]
    hits, t0 = 0, time.time()

    for u in users:
        hist = user_seqs[u]
        ctx = build_context(hist[:-1], item_tokens)    # TEST context: train history + val item
        target = hist[-1]                              # TEST item
        assert len(ctx) + TOKENS_PER_ITEM <= MAX_SEQ_LEN

        recs = beam_search(model, ctx, trie, beam_width=50, device=device)
        items = [i for i, _ in recs]
        scores = [s for _, s in recs]

        # Sanity checks
        assert None not in items, "beam produced a non-item path (trie mask broken)"
        assert len(items) == 50 and len(set(items)) == 50, "expected 50 distinct items"
        assert all(a >= b for a, b in zip(scores, scores[1:])), "scores not sorted"

        rank = items.index(target) + 1 if target in items else None
        hits += rank is not None
        print(f"user {str(u)[:14]:>14} | ctx {len(ctx):>3} tok | target rank: {rank} "
              f"| top score {scores[0]:.2f} | 50th {scores[-1]:.2f}")

    per_user = (time.time() - t0) / len(users)
    print(f"\n{hits}/{len(users)} targets in top-50")
    print(f"{per_user:.3f}s/user -> ~{per_user * len(user_seqs) / 60:.1f} min for all "
          f"{len(user_seqs)} users")