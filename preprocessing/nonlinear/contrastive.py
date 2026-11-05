"""
Supervised Contrastive Metric Learning Representation Module.
Pulls churned customers together and pushes retained apart on the unit hypersphere S^(k-1).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import checkpoint_exists, get_device, load_checkpoint, save_checkpoint


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)."""

    def __init__(self, temperature: float = 0.10):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        sim_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0,
        )
        mask = mask * logits_mask

        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss


class ContrastiveTabNet(nn.Module):
    """Contrastive Metric Projection Network."""

    def __init__(
        self,
        in_features: int,
        latent_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.20,
    ):
        super().__init__()
        dims = hidden_dims or [64, 32]
        layers: List[nn.Module] = []
        prev = in_features
        for h in dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.extend([
            nn.Linear(prev, latent_dim),
            nn.BatchNorm1d(latent_dim),
        ])
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ContrastiveTabularTransformer(BaseEstimator, TransformerMixin):
    """
    Supervised Contrastive Metric Learning Transformer.
    Pulls samples of the same class together on the unit hypersphere.
    Supports checkpoint save/load to reuse representations without retraining.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        temperature: float = 0.10,
        dropout: float = 0.20,
        epochs: int = 25,
        batch_size: int = 256,
        lr: float = 1e-3,
        random_state: int = 42,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.temperature = temperature
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        self.scaler = StandardScaler()
        self.model: Optional[ContrastiveTabNet] = None
        self.device = get_device()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ContrastiveTabularTransformer":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_clean = np.nan_to_num(X, nan=0.0)
        X_s = self.scaler.fit_transform(X_clean).astype(np.float32)
        n_features = X_s.shape[1]

        self.model = ContrastiveTabNet(
            in_features=n_features,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        # Check if checkpoint exists
        if self.checkpoint_path and checkpoint_exists(self.checkpoint_path):
            self.model, meta = load_checkpoint(self.model, self.checkpoint_path, device=self.device)
            print(f"[INFO] Loaded pretrained ContrastiveTabNet from checkpoint: {self.checkpoint_path}")
            return self

        print(f"[INFO] Training ContrastiveTabNet from scratch (epochs={self.epochs}, latent_dim={self.latent_dim})...")
        y_t = torch.tensor(y, dtype=torch.long)

        criterion = SupConLoss(temperature=self.temperature)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        dataset = TensorDataset(torch.tensor(X_s), y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                z = self.model(bx)
                loss = criterion(z, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

        # Save checkpoint
        if self.checkpoint_path:
            save_checkpoint(
                self.model,
                self.checkpoint_path,
                metadata={"in_features": n_features, "latent_dim": self.latent_dim},
            )
            print(f"[OK] Saved ContrastiveTabNet checkpoint to: {self.checkpoint_path}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted or loaded yet.")
        self.model.eval()
        X_clean = np.nan_to_num(X, nan=0.0)
        X_s = self.scaler.transform(X_clean).astype(np.float32)
        with torch.no_grad():
            bx = torch.tensor(X_s).to(self.device)
            z = self.model(bx)
            z_norm = F.normalize(z, dim=1)
            return z_norm.cpu().numpy()
