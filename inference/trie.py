"""
inference/trie.py

Prefix trie over every catalog item's 4-token semantic ID, in GLOBAL vocab
(offset) space -- the same token space the transformer reads and writes.

Beam search asks the trie one question at every decode step:
    "Given the tokens generated so far, which next tokens still lead to a real item?"
Everything else gets masked to -inf, so the model can only spell actual items.

Run sanity checks from project root:
    python -m inference.trie
"""
from collections import deque

import torch

import vocab


class TrieNode:
    # __slots__ keeps ~12k*4 nodes memory-light (no per-instance __dict__)
    __slots__ = ("children", "item_idx")

    def __init__(self):
        self.children = {}      # token_id -> TrieNode
        self.item_idx = None    # set only on leaves (depth == 4)


class SemanticIDTrie:
    def __init__(self, item_token_ids):
        """
        item_token_ids: list where row i = the 4 GLOBAL token ids of item i.
        Row index == item index (same indexing as item_embeddings.pt).
        """
        self.root = TrieNode()
        self.depth = len(item_token_ids[0])
        self.num_items = 0
        for item_idx, tokens in enumerate(item_token_ids):
            self.insert(tokens, item_idx)

    def insert(self, tokens, item_idx):
        assert len(tokens) == self.depth, f"expected {self.depth} tokens, got {len(tokens)}"
        node = self.root
        for t in tokens:
            node = node.children.setdefault(t, TrieNode())
        if node.item_idx is not None:
            # Phase 1 collision handling should make this impossible.
            raise ValueError(
                f"Duplicate semantic ID {tokens}: items {node.item_idx} and {item_idx}"
            )
        node.item_idx = item_idx
        self.num_items += 1

    def _walk(self, prefix):
        node = self.root
        for t in prefix:
            node = node.children.get(t)
            if node is None:
                return None
        return node

    def get_valid_tokens(self, prefix):
        """Tokens that can legally follow `prefix`. Empty list = dead prefix."""
        node = self._walk(prefix)
        return [] if node is None else list(node.children.keys())

    def get_item(self, tokens):
        """Full 4-token path -> item index (None if not a real item)."""
        node = self._walk(tokens)
        return None if node is None else node.item_idx

    @classmethod
    def from_file(cls, path="data/item_semantic_ids.pt"):
        sem_ids = torch.load(path)  # (num_items, 4) RAW local ids
        item_tokens = [
            [int(t) for t in vocab.item_to_tokens(row.tolist())]
            for row in sem_ids
        ]
        return cls(item_tokens)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def _fanout_by_depth(trie):
    """For each depth, list of child-counts of every node at that depth."""
    stats = [[] for _ in range(trie.depth)]
    queue = deque([(trie.root, 0)])
    while queue:
        node, d = queue.popleft()
        if d < trie.depth:
            stats[d].append(len(node.children))
            for child in node.children.values():
                queue.append((child, d + 1))
    return stats


if __name__ == "__main__":
    sem_ids = torch.load("data/item_semantic_ids.pt")
    trie = SemanticIDTrie.from_file()
    n = sem_ids.shape[0]

    # 1. Every item inserted exactly once
    print(f"items in trie: {trie.num_items} (expected {n})")
    assert trie.num_items == n

    # 2. Round-trip: every item's path resolves back to itself
    for i in range(n):
        toks = [int(t) for t in vocab.item_to_tokens(sem_ids[i].tolist())]
        assert trie.get_item(toks) == i, f"round-trip failed for item {i}"
    print("round-trip OK for all items")

    # 3. Every token at depth d lives in level d's vocab range (matches vocab.py)
    level_ranges = [(0, 256), (256, 512), (512, 768), (768, 783)]
    queue = deque([(trie.root, 0)])
    while queue:
        node, d = queue.popleft()
        for tok, child in node.children.items():
            lo, hi = level_ranges[d]
            assert lo <= tok < hi, f"token {tok} at depth {d} outside [{lo},{hi})"
            queue.append((child, d + 1))
    print("level ranges OK")

    # 4. Fan-out per depth -- this is what beam search will "see" each step
    for d, counts in enumerate(_fanout_by_depth(trie)):
        forced = sum(c == 1 for c in counts)
        print(
            f"depth {d}: {len(counts):>6} nodes | children min/mean/max = "
            f"{min(counts)}/{sum(counts)/len(counts):.2f}/{max(counts)} | "
            f"forced (1 child): {forced/len(counts):.1%}"
        )

    # 5. Dead prefixes return nothing
    assert trie.get_valid_tokens([vocab.PAD]) == []
    assert trie.get_valid_tokens([vocab.SEP]) == []
    print("dead-prefix OK")

    # 6. Walk one example
    toks = [int(t) for t in vocab.item_to_tokens(sem_ids[0].tolist())]
    print(f"\nitem 0 path: {toks}")
    for k in range(len(toks)):
        print(f"  prefix {toks[:k]} -> {len(trie.get_valid_tokens(toks[:k]))} valid next tokens")