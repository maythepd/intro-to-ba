"""
Pure PyTorch Graph Attention Network (GAT) & Self-Supervised Graph Autoencoder.
Zero external C++ dependencies (no torch_geometric binary needed).
Encodes customer network neighborhood topologies into dense latent representations (z_gnn)
using self-supervised feature masked reconstruction and topological link alignment.
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GraphAttentionLayer(nn.Module):
    """
    Multi-Head Graph Attention Layer implemented entirely in standard PyTorch.
    Performs vectorized message passing using edge_index and scatter_add.
    Features:
      - Multi-head edge attention mechanism
      - Scaled residual skip connection
      - LayerNorm and Mish activation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 2,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads

        self.lin_self = nn.Linear(in_features, out_features * heads)
        self.lin_neigh = nn.Linear(in_features, out_features * heads)
        self.attn = nn.Linear(in_features * 2, heads)

        self.norm = nn.LayerNorm(out_features * heads)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Mish()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (N, in_features)
        edge_index: (2, E)
        """
        N = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]

        # 1. Transform self node features
        h_self = self.lin_self(x)  # (N, out_features * heads)

        # 2. Extract edge endpoint features
        x_src = x[src]  # (E, in_features)
        x_dst = x[dst]  # (E, in_features)

        # 3. Compute multi-head edge attention scores
        edge_cat = torch.cat([x_src, x_dst], dim=-1)  # (E, in_features * 2)
        raw_attn = F.leaky_relu(self.attn(edge_cat), negative_slope=0.2)  # (E, heads)

        # Numerically stable edge attention softmax per destination node
        attn_exp = torch.exp(raw_attn - raw_attn.max(dim=0, keepdim=True).values)
        denom = torch.zeros((N, self.heads), device=x.device).scatter_add_(0, dst.unsqueeze(1).expand(-1, self.heads), attn_exp) + 1e-8
        norm_attn = attn_exp / denom[dst]  # (E, heads)
        norm_attn = self.dropout(norm_attn)

        # 4. Message computation
        msg = self.lin_neigh(x_src).view(-1, self.heads, self.out_features)  # (E, heads, out_features)
        weighted_msg = (msg * norm_attn.unsqueeze(-1)).view(-1, self.heads * self.out_features)

        # 5. Scatter aggregate messages to destination nodes
        total_out_dim = self.heads * self.out_features
        agg = torch.zeros((N, total_out_dim), device=x.device).scatter_add_(
            0, dst.unsqueeze(1).expand(-1, total_out_dim), weighted_msg
        )

        # 6. Residual combination, normalization, and activation
        out = self.norm(self.act(h_self + agg))
        return out


class SelfSupervisedGNNAutoencoder(nn.Module):
    """
    Self-Supervised Graph Autoencoder for Customer Representation Learning.
    Architecture:
      Encoder: 2-layer Graph Attention Network mapping (N, in_features) -> (N, latent_dim).
      Feature Decoder: MLP reconstructing masked node features.
      Topology Predictor: Cosine dot-product preserving edge connectivity.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        heads: int = 2,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # GAT Backbone
        self.layer1 = GraphAttentionLayer(in_features, hidden_dim, heads=heads, dropout=dropout)
        self.layer2 = GraphAttentionLayer(hidden_dim * heads, latent_dim, heads=1, dropout=dropout)

        # Latent projection head
        self.to_latent = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
        )

        # Feature Reconstruction Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, in_features),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encodes graph (x, edge_index) into dense customer latent representations."""
        h1 = self.layer1(x, edge_index)
        h2 = self.layer2(h1, edge_index)
        z = self.to_latent(h2)
        return z

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        mask_ratio: float = 0.20,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with feature masking.
        Returns:
          z: (N, latent_dim) latent representation
          x_recon: (N, in_features) reconstructed node features
          mask: (N, in_features) boolean mask indicating masked features
        """
        if self.training and mask_ratio > 0.0:
            mask = torch.rand_like(x) < mask_ratio
            x_corrupted = x.clone()
            x_corrupted[mask] = 0.0
        else:
            mask = torch.zeros_like(x, dtype=torch.bool)
            x_corrupted = x

        z = self.encode(x_corrupted, edge_index)
        x_recon = self.decoder(z)
        return z, x_recon, mask


class GNNTrainer:
    """
    Self-Supervised Trainer for Customer GNN Autoencoder.
    Trains strictly without class labels (100% unsupervised, zero target leakage).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        heads: int = 2,
        lr: float = 0.005,
        dropout: float = 0.20,
        epochs: int = 20,
        checkpoint_path: Optional[Union[str, Path]] = None,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.heads = heads
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.random_state = random_state
        self.verbose = verbose

        self.device = get_device()
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = SelfSupervisedGNNAutoencoder(
            in_features=in_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            heads=heads,
            dropout=dropout,
        ).to(self.device)

        self.history_: Dict[str, List[float]] = {"train_loss": [], "recon_mse": []}

    def fit(self, x: torch.Tensor, edge_index: torch.Tensor) -> "GNNTrainer":
        """
        Fits GNN Autoencoder on the training graph via masked feature reconstruction.
        """
        t0 = time.time()
        self.model.train()
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion_recon = nn.MSELoss()

        if self.verbose:
            print(
                f"[INFO] Training Self-Supervised GNN Autoencoder: "
                f"nodes={x.size(0)}, edges={edge_index.size(1)}, latent_dim={self.latent_dim}, epochs={self.epochs}"
            )

        for ep in range(1, self.epochs + 1):
            optimizer.zero_grad()
            z, x_recon, mask = self.model(x, edge_index, mask_ratio=0.25)

            # Feature reconstruction loss
            loss_recon = criterion_recon(x_recon, x)

            # Topological link consistency: connected nodes should have aligned representations
            src = edge_index[0]
            dst = edge_index[1]
            z_src = F.normalize(z[src], p=2, dim=-1)
            z_dst = F.normalize(z[dst], p=2, dim=-1)
            topo_sim = (z_src * z_dst).sum(dim=-1)
            loss_topo = (1.0 - topo_sim).mean()

            total_loss = loss_recon + 0.10 * loss_topo
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            self.history_["train_loss"].append(round(total_loss.item(), 4))
            self.history_["recon_mse"].append(round(loss_recon.item(), 4))

        if self.verbose:
            print(
                f"[OK] GNN training completed in {time.time() - t0:.2f}s! "
                f"Final Loss: {total_loss.item():.4f} (Recon MSE: {loss_recon.item():.4f})"
            )

        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": self.model.state_dict(),
                    "metadata": {
                        "in_features": self.in_features,
                        "latent_dim": self.latent_dim,
                        "hidden_dim": self.hidden_dim,
                        "heads": self.heads,
                    },
                },
                self.checkpoint_path,
            )

        return self

    def transform(self, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
        """Transforms node features through GNN encoder into dense latents."""
        self.model.eval()
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        with torch.no_grad():
            z = self.model.encode(x, edge_index)
        return z.detach().cpu().numpy().astype(np.float32)
