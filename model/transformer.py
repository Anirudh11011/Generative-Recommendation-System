"""
model/transformer.py — decoder-only (GPT-style) transformer, from scratch.

Input : (batch, L)      integer token ids  (right-padded with PAD in training)
Output: (batch, L, V)   logits over the V-token vocab, one distribution per position;
        position t's row predicts the token at position t+1.

Built from nn.Linear / nn.Embedding / nn.LayerNorm on purpose — no
nn.TransformerDecoder — so the whole mechanism is exposed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import vocab   # VOCAB_SIZE, PAD


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t may attend only to 0..t."""

    def __init__(self, d_model, n_heads, dropout, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly into heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads              # 128 / 4 = 32

        # One fused projection makes Q, K, V at once, then we split it three ways.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)          # recombines the heads
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Lower-triangular mask, built once. register_buffer => moves with
        # .to(device) and is saved in state_dict, but is NOT a trained parameter.
        # Shape (1,1,L,L) so it broadcasts over (batch, heads, L, L).
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):                                # x: (B, L, D)
        B, L, D = x.shape

        q, k, v = self.qkv(x).split(D, dim=2)           # each (B, L, D)

        # Split each into heads: (B, L, D) -> (B, heads, L, head_dim). Putting
        # 'heads' before 'L' makes each head an independent (L, head_dim) problem.
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Scores: how much each position attends to each other position.
        # (B,h,L,hd) @ (B,h,hd,L) -> (B,h,L,L). Scale by sqrt(head_dim) so the
        # softmax doesn't saturate as head_dim grows.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask: where the triangular mask is 0 (columns j > row i, i.e. the
        # future), set the score to -inf so softmax gives those positions weight 0.
        scores = scores.masked_fill(self.causal_mask[:, :, :L, :L] == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)                # (B,h,L,L), each row sums to 1
        attn = self.attn_dropout(attn)

        y = attn @ v                                    # (B,h,L,L)@(B,h,L,hd)->(B,h,L,hd)

        y = y.transpose(1, 2).contiguous().view(B, L, D)   # merge heads -> (B,L,D)
        y = self.resid_dropout(self.out(y))
        return y


class TransformerBlock(nn.Module):
    """Pre-norm block: x + attn(norm(x)), then x + ffn(norm(x))."""

    def __init__(self, d_model, n_heads, dropout, max_seq_len, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),      # 128 -> 512
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),      # 512 -> 128
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Residual connections: each sublayer LEARNS a correction added onto x
        # rather than replacing it. norm BEFORE the sublayer (pre-norm) trains stably.
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RecTransformer(nn.Module):
    def __init__(self, vocab_size=vocab.VOCAB_SIZE, d_model=128, n_heads=4,
                 n_layers=4, dropout=0.1, max_seq_len=200):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)   # id -> vector
        self.pos_emb = nn.Embedding(max_seq_len, d_model)  # position -> vector (learned)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)                # final norm
        self.head = nn.Linear(d_model, vocab_size, bias=False)   # -> logits

        # Weight tying: share the embedding matrix with the output projection.
        # Standard in LMs; fewer params, and it aligns the input/output token spaces.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):                                # idx: (B, L) int64
        B, L = idx.shape
        assert L <= self.max_seq_len, f"seq len {L} exceeds max {self.max_seq_len}"

        pos = torch.arange(L, device=idx.device)           # (L,)
        x = self.tok_emb(idx) + self.pos_emb(pos)          # (B,L,D); pos broadcasts over batch
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        return self.head(x)                                # (B, L, vocab_size)


if __name__ == "__main__":
    # Smoke test — run this before training. Verifies shapes and the causal mask.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = RecTransformer().to(device).eval()             # eval() disables dropout

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device} | params: {n_params:,}")

    x = torch.randint(0, vocab.VOCAB_SIZE, (2, 15), device=device)   # fake (B=2, L=15)
    logits = model(x)
    print("input :", tuple(x.shape))
    print("logits:", tuple(logits.shape), "| expected (2, 15, 786)")

    # Causal check: changing tokens from position 10 onward must NOT change the
    # logits at positions 0..9. If it does, the mask is leaking the future.
    with torch.no_grad():
        x2 = x.clone()
        x2[:, 10:] = (x2[:, 10:] + 1) % vocab.VOCAB_SIZE
        same = torch.allclose(model(x)[:, :10], model(x2)[:, :10], atol=1e-5)
    print("causal ok:", same, "(early logits unaffected by later tokens)")