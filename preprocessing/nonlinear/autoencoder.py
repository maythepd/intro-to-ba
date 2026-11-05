"""
Self-Supervised Tabular Denoising Autoencoder (DAE) Module.
Learns intrinsic manifold representations and extracts bottleneck latent vectors z.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import (
    checkpoint_exists,
    get_device,
    is_checkpoint_compatible,
    load_checkpoint,
    save_checkpoint,
)


class TabularDAE(nn.Module):
    """Deep Denoising Autoencoder with symmetric bottleneck architecture."""

    def __init__(
        self,
        in_features: int,
        latent_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        dims = hidden_dims or [64, 32]
        
        # Build encoder layers
        enc_layers: List[nn.Module] = []
        prev = in_features
        for h in dims:
            enc_layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.SiLU(),
                nn.Dropout(0.15),
            ])
            prev = h
        enc_layers.extend([
            nn.Linear(prev, latent_dim),
            nn.BatchNorm1d(latent_dim),
        ])
        self.encoder = nn.Sequential(*enc_layers)

        # Build decoder layers
        dec_layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(dims):
            dec_layers.extend([
                nn.SiLU(),
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
            ])
            prev = h
        dec_layers.extend([
            nn.SiLU(),
            nn.Linear(prev, in_features),
        ])
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class TabularDenoisingAutoencoderTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer that learns data manifold reconstructions
    and outputs the latent bottleneck embeddings.
    Supports checkpoint save/load to avoid retraining.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        noise_level: float = 0.15,
        epochs: int = 25,
        batch_size: int = 256,
        lr: float = 1e-3,
        random_state: int = 42,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.noise_level = noise_level
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        self.scaler = StandardScaler()
        self.model: Optional[TabularDAE] = None
        self.device = get_device()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TabularDenoisingAutoencoderTransformer":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_clean = np.nan_to_num(X, nan=0.0)
        X_s = self.scaler.fit_transform(X_clean).astype(np.float32)
        n_features = X_s.shape[1]

        self.model = TabularDAE(
            in_features=n_features,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        # Check if checkpoint exists and is compatible with current input feature dimension
        if self.checkpoint_path and checkpoint_exists(self.checkpoint_path):
            if is_checkpoint_compatible(self.checkpoint_path, in_features=n_features):
                self.model, meta = load_checkpoint(self.model, self.checkpoint_path, device=self.device)
                print(f"[INFO] Loaded pretrained TabularDAE from checkpoint: {self.checkpoint_path}")
                return self
            else:
                print(f"[WARN] Checkpoint '{self.checkpoint_path}' feature dimension does not match current input ({n_features} features). Training fresh model...")

        print(f"[INFO] Training TabularDAE from scratch (epochs={self.epochs}, latent_dim={self.latent_dim})...")
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.tensor(X_s))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for (bx,) in loader:
                bx = bx.to(self.device)
                noise = torch.randn_like(bx) * self.noise_level
                corrupted_bx = bx + noise

                optimizer.zero_grad()
                recon, _ = self.model(corrupted_bx)
                loss = criterion(recon, bx)
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
            print(f"[OK] Saved TabularDAE checkpoint to: {self.checkpoint_path}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted or loaded yet.")
        self.model.eval()
        X_clean = np.nan_to_num(X, nan=0.0)
        X_s = self.scaler.transform(X_clean).astype(np.float32)
        with torch.no_grad():
            bx = torch.tensor(X_s).to(self.device)
            _, z = self.model(bx)
            return z.cpu().numpy()
