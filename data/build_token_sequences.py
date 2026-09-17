"""
Phase 2, Step 1 — flatten each user's item history into a token sequence.

Loads Phase-0 user sequences + Phase-1 semantic IDs, applies the vocab offsets,
adds BOS/SEP delimiters, truncates to the most-recent items that fit, and saves
the flattened TRAINING sequences. Pure data prep — CPU only, no MPS.

Run from project root:  python -m data.build_token_sequences
"""

import pickle
import torch
import vocab

MAX_SEQ_LEN = 200
SEM_IDS_PATH   = "data/item_semantic_ids.pt"
USER_SEQS_PATH = "data/user_sequences.pkl"
OUT_PATH       = "data/train_token_sequences.pkl"


def flatten_history(item_indices, sem_ids, max_len=MAX_SEQ_LEN):
    """[BOS] item0(4 tok) [SEP] item1(4 tok) [SEP] ...  truncated to most-recent
    items at ITEM boundaries. k items cost 1 + 4k + (k-1) = 5k tokens."""
    max_items = max_len // 5                 # 200 // 5 = 40
    recent = list(item_indices)[-max_items:] # keep most recent history
    seq = [vocab.BOS]
    for pos, item in enumerate(recent):
        if pos > 0:
            seq.append(vocab.SEP)
        seq.extend(vocab.item_to_tokens(sem_ids[item]))
    return seq


def main():
    sem_ids = torch.load(SEM_IDS_PATH)              # (12101, 4) raw local indices
    user_sequences = pickle.load(open(USER_SEQS_PATH, "rb"))  # {uid: [full history]}

    # --- sanity: confirm the semantic IDs on disk are RAW (not already offset) ---
    assert sem_ids.shape[1] == 4, f"expected 4 tokens/item, got {sem_ids.shape}"
    assert int(sem_ids[:, :3].max()) < vocab.CODEBOOK_SIZE, \
        "codebook tokens exceed 256 — are these already offset?"
    real_counter = int(sem_ids[:, 3].max()) + 1
    assert real_counter <= vocab.COUNTER_SIZE, \
        f"counter needs {real_counter} slots; bump vocab.COUNTER_SIZE"
    print(f"vocab: {vocab.VOCAB_SIZE} tokens | PAD={vocab.PAD} BOS={vocab.BOS} "
          f"SEP={vocab.SEP} | counter uses {real_counter}/{vocab.COUNTER_SIZE}")

    # --- build flattened TRAIN sequences ---
    # Leave-one-out: last item = test, 2nd-last = val, REST = train history.
    # The split is a load-time convention; the pkl stores full chronological lists.
    user_ids, sequences, n_truncated, n_skipped = [], [], 0, 0
    for uid, full_history in user_sequences.items():
        train_items = full_history[:-2]             # drop val + test
        if len(train_items) < 1:                    # can't happen given <5 filter
            n_skipped += 1
            continue
        if len(train_items) > MAX_SEQ_LEN // 5:
            n_truncated += 1
        user_ids.append(uid)
        sequences.append(flatten_history(train_items, sem_ids))

    # --- sanity: every token in range, length stats, one decoded example ---
    all_tok = torch.tensor([t for s in sequences for t in s])
    assert int(all_tok.min()) >= 0 and int(all_tok.max()) < vocab.VOCAB_SIZE
    lens = torch.tensor([len(s) for s in sequences])
    print(f"users: {len(sequences)} | skipped: {n_skipped} | truncated: {n_truncated} "
          f"({100*n_truncated/len(sequences):.2f}%)")
    print(f"seq-len tokens  min/median/mean/max = "
          f"{int(lens.min())}/{int(lens.median())}/{lens.float().mean():.1f}/{int(lens.max())}")

    ex = sequences[0][:14]
    print("example:", ex)
    print("levels :", [vocab.token_level(t) for t in ex],
          "  (None = BOS/SEP/PAD)")

    pickle.dump({"user_ids": user_ids, "sequences": sequences},
                open(OUT_PATH, "wb"))
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()