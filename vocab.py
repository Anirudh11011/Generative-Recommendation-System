"""
vocab.py — single source of truth for how semantic IDs become token IDs.

Each item is 4 tokens: 3 RQ-VAE codebook levels (256 codes each) + 1 collision
counter. The transformer looks every token up in ONE embedding table, so each
level lives in its own disjoint slice of a global vocab — codebook-1 code 5 and
codebook-2 code 5 must be DIFFERENT ids or the model can't tell the level apart.

Layout:
    level 0 (codebook 1)  ->  [  0 .. 255]   offset   0
    level 1 (codebook 2)  ->  [256 .. 511]   offset 256
    level 2 (codebook 3)  ->  [512 .. 767]   offset 512
    level 3 (counter)     ->  [768 .. 782]   offset 768   (only ~15 values used)
    PAD 783   BOS 784   SEP 785
    VOCAB_SIZE = 786

Phase 2 (flattening + model) and Phase 3 (trie + beam search) both import this,
so the offset logic is defined exactly once.
"""

CODEBOOK_SIZE = 256          # codes per RQ-VAE codebook (levels 0, 1, 2)
COUNTER_SIZE  = 15           # distinct 4th-token values (largest collision bucket)
                             # asserted against the real data in the build script

LEVEL_OFFSETS = [0, 256, 512, 768]   # start of each level's slice; 3 = counter

_SPECIALS_START = LEVEL_OFFSETS[3] + COUNTER_SIZE   # 768 + 15 = 783
PAD = _SPECIALS_START + 0    # 783
BOS = _SPECIALS_START + 1    # 784
SEP = _SPECIALS_START + 2    # 785
VOCAB_SIZE = _SPECIALS_START + 3   # 786


def item_to_tokens(sem_id):
    """Raw semantic ID (4 local indices) -> 4 global token ids.
    (5, 12, 200, 0)  ->  [5, 268, 712, 768]"""
    return [LEVEL_OFFSETS[level] + int(code) for level, code in enumerate(sem_id)]


def token_level(token_id):
    """Which level a token came from (0-3), or None for a special. Debug aid."""
    if token_id >= _SPECIALS_START:
        return None
    for level in range(4):
        hi = LEVEL_OFFSETS[level] + (COUNTER_SIZE if level == 3 else CODEBOOK_SIZE)
        if LEVEL_OFFSETS[level] <= token_id < hi:
            return level
    return None