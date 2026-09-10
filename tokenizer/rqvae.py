"""
tokenizer/rqvae.py
------------------
Phase 1, step 2: the full RQ-VAE that wraps the ResidualQuantizer.

Pipeline (all shapes for a batch of B items):
    x        (B, 384)   MiniLM embedding
      -> encoder ->   z    (B, 256)
      -> quantizer -> z_q  (B, 256),  codes (B, 3),  vq_loss
      -> decoder ->   x_hat(B, 384)   reconstruction of x

Loss = reconstruction MSE (x_hat vs x)  +  vq_loss (from the quantizer).

Also provides k-means codebook initialisation, which must run on CPU/numpy
BEFORE the model is moved onto the Apple-Silicon GPU (MPS).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# works whether imported as a package (tokenizer.rqvae) or run standalone
try:
    from .quantizer import ResidualQuantizer
except ImportError:
    from quantizer import ResidualQuantizer


class RQVAE(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, mid_dim=512,
                 num_codebooks=3, codebook_size=256, beta=0.25):
        super().__init__()

        # encoder: 384 -> 512 -> 256  (squeeze MiniLM embedding into latent space)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, hidden_dim),
        )

        # the module you built in step 1; it operates in the 256-dim latent space
        self.quantizer = ResidualQuantizer(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            dim=hidden_dim,
            beta=beta,
        )

        # decoder: 256 -> 512 -> 384  (mirror of the encoder, back to embedding space)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)                      # (B, 256)
        z_q, codes, vq_loss = self.quantizer(z)  # (B, 256), (B, 3), scalar
        x_hat = self.decoder(z_q)                # (B, 384)

        recon_loss = F.mse_loss(x_hat, x)        # rebuild the original embedding
        loss = recon_loss + vq_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "codes": codes,
            "x_hat": x_hat,
        }

    @torch.no_grad()
    def get_codes(self, x):
        """Inference: item embeddings -> semantic IDs (B, num_codebooks)."""
        z = self.encoder(x)
        _, codes, _ = self.quantizer(z)
        return codes

    @torch.no_grad()
    def init_codebooks_kmeans(self, embeddings, n_init=10, seed=0):
        """
        Seed each codebook with k-means centroids instead of random noise, so no
        code starts 'dead'. Runs stage by stage on the RESIDUALS, mirroring how the
        quantizer works at run time.

        Must be called on CPU (scikit-learn is numpy-only) BEFORE model.to('mps').
        """
        from sklearn.cluster import KMeans

        self.eval()
        # encode all items once; k-means clusters these latent vectors
        z = self.encoder(embeddings).cpu().numpy()   # (N, 256)
        residual = z

        for stage, cb in enumerate(self.quantizer.codebooks):
            k = cb.num_embeddings                     # 256
            km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(residual)
            centers = km.cluster_centers_.astype(np.float32)   # (256, 256)

            # overwrite this codebook's table with the centroids (in place)
            cb.weight.data.copy_(torch.from_numpy(centers))

            # subtract chosen centroid -> residual for the NEXT stage's k-means
            residual = residual - centers[km.labels_]
            print(f"  codebook {stage}: k-means done, "
                  f"residual norm now {np.linalg.norm(residual):.1f}")

        print("k-means codebook init complete.")


# --- self-test on dummy data (CPU here; on your Mac this runs on MPS) ---------
if __name__ == "__main__":
    torch.manual_seed(0)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    N = 500
    fake_embeddings = torch.randn(N, 384)          # pretend MiniLM outputs

    model = RQVAE(input_dim=384, hidden_dim=256, num_codebooks=3, codebook_size=256)

    # 1) k-means init runs on CPU (numpy), BEFORE moving to the GPU
    print("k-means init (CPU):")
    model.init_codebooks_kmeans(fake_embeddings, n_init=3)

    # 2) now move model + data to the device together
    model = model.to(device)
    x = fake_embeddings.to(device)

    # 3) one forward pass
    out = model(x)
    print("\nforward pass:")
    print("  loss       :", round(out["loss"].item(), 4))
    print("  recon_loss :", round(out["recon_loss"].item(), 4))
    print("  vq_loss    :", round(out["vq_loss"].item(), 4))
    print("  codes shape:", tuple(out["codes"].shape))
    print("  codes[0]   :", out["codes"][0].tolist())

    # 4) confirm training works end to end
    out["loss"].backward()
    enc_grad = model.encoder[0].weight.grad
    print("\nbackward pass: encoder receives gradient:",
          enc_grad is not None and torch.isfinite(enc_grad).all().item())

    # 5) codebook utilisation: how many of the 256 codes get used in stage 1?
    codes_all = model.get_codes(x)
    used = len(torch.unique(codes_all[:, 0]))
    print(f"stage-1 codes used: {used}/256")