"""
tokenizer/quantizer.py
----------------------
Phase 1, step 1: the ResidualQuantizer, built from scratch.

This is the core of the semantic-ID tokenizer. It takes a dense vector z
(e.g. the encoder's 256-dim output) and turns it into a tuple of small
integers -- one per codebook -- by repeatedly snapping the *residual* to the
nearest centroid.

It does NOT include the encoder or decoder (that's step 2, the full RQ-VAE).
Here we only build and test the quantization mechanism itself.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):
    def __init__(self, num_codebooks=3, codebook_size=256, dim=256, beta=0.25):
        """
        num_codebooks : how many quantization stages (3 -> 3-token semantic IDs)
        codebook_size : centroids per codebook (256 -> each token is 0..255)
        dim           : dimensionality of the vectors being quantized
        beta          : weight on the commitment loss term
        """
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta = beta

        # One codebook per stage. Each is a (codebook_size, dim) table of learnable
        # centroids -- nn.Embedding is just a lookup table of vectors, which is
        # exactly what a codebook is. We start with small random values; in the
        # real run these get overwritten by k-means init before training.
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, dim) for _ in range(num_codebooks)
        ])
        for cb in self.codebooks:
            nn.init.normal_(cb.weight, mean=0.0, std=0.1)

    def forward(self, z):
        """
        z : (batch, dim) dense vectors to quantize.

        Returns:
          z_q     : (batch, dim) quantized vector (sum of chosen centroids),
                    with the straight-through estimator applied so gradients
                    flow back into z as if quantization were the identity.
          codes   : (batch, num_codebooks) integer tokens = the semantic IDs.
          vq_loss : scalar; the codebook + commitment loss (NOT reconstruction;
                    that lives in the full RQ-VAE in step 2).
        """
        residual = z                      # stage 1 quantizes z itself
        z_q = torch.zeros_like(z)         # running sum of chosen centroids
        codes = []
        codebook_loss = 0.0
        commitment_loss = 0.0

        for cb in self.codebooks:
            centroids = cb.weight          # (codebook_size, dim)

            # --- nearest-centroid lookup (argmin over squared L2 distance) ---
            # distance from each residual to every centroid: (batch, codebook_size)
            dist = torch.cdist(residual, centroids)   # euclidean; argmin is what matters
            idx = dist.argmin(dim=1)                  # (batch,) chosen token per row
            chosen = cb(idx)                          # (batch, dim) the centroid vectors

            # --- losses for THIS stage (computed on the residual) ---
            # codebook loss: pull centroids toward the residuals assigned to them
            #   (residual detached -> gradient only moves the centroid)
            # commitment loss: pull the residual toward its centroid
            #   (centroid detached -> gradient only moves the encoder output)
            codebook_loss   += F.mse_loss(chosen, residual.detach())
            commitment_loss += F.mse_loss(residual, chosen.detach())

            # accumulate and move to the next, finer residual
            z_q = z_q + chosen
            residual = residual - chosen
            codes.append(idx)

        codes = torch.stack(codes, dim=1)             # (batch, num_codebooks)
        vq_loss = codebook_loss + self.beta * commitment_loss

        # --- straight-through estimator ---
        # forward value is the quantized z_q; backward pretends it equals z,
        # so gradients reach the encoder despite the non-differentiable argmin.
        z_q = z + (z_q - z).detach()

        return z_q, codes, vq_loss


# --- quick self-test on dummy data (no real data / no MPS needed yet) --------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, DIM = 8, 256
    rq = ResidualQuantizer(num_codebooks=3, codebook_size=256, dim=DIM, beta=0.25)

    z = torch.randn(B, DIM, requires_grad=True)   # pretend encoder outputs
    z_q, codes, vq_loss = rq(z)

    print("input z     :", tuple(z.shape))
    print("quantized zq:", tuple(z_q.shape))
    print("codes       :", tuple(codes.shape), "dtype", codes.dtype)
    print("codes[0]    :", codes[0].tolist(), "  <- a semantic ID (3 tokens, each 0..255)")
    print("token range :", int(codes.min()), "..", int(codes.max()))
    print("vq_loss     :", float(vq_loss))

    # confirm the straight-through estimator lets gradients reach z
    vq_loss.backward()
    print("grad reaches z:", z.grad is not None and torch.isfinite(z.grad).all().item())