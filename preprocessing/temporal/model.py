"""
Deep Temporal Sequence Autoencoder for Customer Event Trajectories.
Self-Supervised Representation Learning using Bidirectional GRU Encoder-Decoder.
Compresses multi-snapshot behavioral sequences into dense, leak-free latent embeddings.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BiGRUTemporalAutoencoder(nn.Module):
    """
    Bidirectional GRU Temporal Sequence Autoencoder.
    
    Encoder: Bi-GRU over T snapshots -> latent representation z in R^{latent_dim}.
    Decoder: Uni-directional GRU reconstructing the T x D trajectory from z.
    """

    def __init__(
        self,
        in_features: int = 10,
        seq_len: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        latent_dim: int = 16,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Bidirectional Encoder
        self.encoder_gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Projection to latent bottleneck z
        encoder_out_dim = hidden_dim * 2  # Bidirectional
        self.to_latent = nn.Sequential(
            nn.Linear(encoder_out_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

        # Decoder initialization from latent z
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
        )

        # Reconstruction Decoder GRU
        self.decoder_gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim * 2,
            num_layers=1,
            batch_first=True,
        )

        # Projection head to reconstructed snapshot features
        self.recon_head = nn.Linear(hidden_dim * 2, in_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes sequence x (batch, seq_len, in_features) into latent z (batch, latent_dim).
        """
        _, h_n = self.encoder_gru(x)
        # h_n shape: (num_layers * 2, batch, hidden_dim)
        # Concatenate forward and backward final layer states
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=-1)
        return self.to_latent(h_cat)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: returns reconstructed sequence (batch, seq_len, in_features) and latent z.
        """
        z = self.encode(x)
        batch_size = x.size(0)

        # Decoder initial hidden state
        h_dec = self.from_latent(z).unsqueeze(0)  # (1, batch, hidden_dim * 2)

        # Teacher forcing / sequence decoding
        dec_out, _ = self.decoder_gru(x, h_dec)
        recon = self.recon_head(dec_out)
        return recon, z


class TemporalAutoencoderTrainer:
    """
    Self-Supervised Trainer for BiGRUTemporalAutoencoder.
    Guarantees 100% Zero Target Leakage since training uses only unsupervised event sequences.
    """

    def __init__(
        self,
        in_features: int = 10,
        seq_len: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        latent_dim: int = 16,
        epochs: int = 25,
        batch_size: int = 256,
        lr: float = 1e-3,
        dropout: float = 0.20,
        val_size: float = 0.15,
        random_state: int = 42,
        checkpoint_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        force_retrain: bool = False,
    ):
        self.in_features = in_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.val_size = val_size
        self.random_state = random_state
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.verbose = verbose
        self.force_retrain = force_retrain

        self.device = get_device()
        self.model: Optional[BiGRUTemporalAutoencoder] = None
        self.history_: Dict[str, List[float]] = {"train_mse": [], "val_mse": []}
        self.metrics_: Dict[str, Any] = {}

    def fit(self, tensor: np.ndarray) -> "TemporalAutoencoderTrainer":
        """
        Trains temporal sequence autoencoder on 3D sequence tensor (N, seq_len, in_features).
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_samples, seq_len, in_features = tensor.shape
        self.seq_len = seq_len
        self.in_features = in_features

        self.model = BiGRUTemporalAutoencoder(
            in_features=in_features,
            seq_len=seq_len,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(self.device)

        # Check existing checkpoint unless force_retrain is active
        if not self.force_retrain and self.checkpoint_path and self.checkpoint_path.exists():
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(ckpt["state_dict"])
                self.history_ = ckpt.get("metadata", {}).get("history", {})
                self.metrics_ = ckpt.get("metadata", {}).get("metrics", {})
                if self.verbose:
                    print(f"[INFO] Loaded pretrained BiGRUTemporalAutoencoder from: {self.checkpoint_path}")
                return self
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Failed to load checkpoint ({e}). Training from scratch...")

        if self.verbose:
            print(
                f"[INFO] Training BiGRUTemporalAutoencoder (epochs={self.epochs}, "
                f"batch_size={self.batch_size}, latent_dim={self.latent_dim}, device={self.device})..."
            )

        # Unsupervised train / validation split
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        val_count = int(n_samples * self.val_size)
        val_idx = indices[:val_count]
        tr_idx = indices[val_count:]

        X_tr = torch.from_numpy(tensor[tr_idx]).float()
        X_va = torch.from_numpy(tensor[val_idx]).float()

        tr_loader = DataLoader(TensorDataset(X_tr), batch_size=self.batch_size, shuffle=True)
        va_loader = DataLoader(TensorDataset(X_va), batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_tr_loss = 0.0
            total_tr_samples = 0

            pbar = tqdm(
                tr_loader,
                desc=f"[Epoch {epoch:02d}/{self.epochs:02d}]",
                disable=not self.verbose,
                leave=False,
                ncols=100,
                unit="batch",
            )
            for (bx,) in pbar:
                bx = bx.to(self.device)
                optimizer.zero_grad()
                recon, _ = self.model(bx)
                loss = criterion(recon, bx)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_loss = loss.item()
                total_tr_loss += batch_loss * bx.size(0)
                total_tr_samples += bx.size(0)

                running_loss = total_tr_loss / max(1, total_tr_samples)
                pbar.set_postfix({"recon_mse": f"{running_loss:.4f}"})

            ep_tr_loss = total_tr_loss / max(1, total_tr_samples)

            # Validation pass
            self.model.eval()
            total_va_loss = 0.0
            total_va_samples = 0
            with torch.no_grad():
                for (bx,) in va_loader:
                    bx = bx.to(self.device)
                    recon, _ = self.model(bx)
                    loss = criterion(recon, bx)
                    total_va_loss += loss.item() * bx.size(0)
                    total_va_samples += bx.size(0)

            ep_va_loss = total_va_loss / max(1, total_va_samples)

            self.history_["train_mse"].append(round(ep_tr_loss, 4))
            self.history_["val_mse"].append(round(ep_va_loss, 4))

            if self.verbose:
                bar_str = "━" * 22
                print(
                    f"[Epoch {epoch:02d}/{self.epochs:02d}] {bar_str} 100% | "
                    f"train_recon_mse: {ep_tr_loss:.4f} | val_recon_mse: {ep_va_loss:.4f}"
                )

        self.metrics_ = {
            "Model": "BiGRUTemporalAutoencoder",
            "Final_Train_Recon_MSE": round(ep_tr_loss, 4),
            "Final_Val_Recon_MSE": round(ep_va_loss, 4),
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "seq_len": self.seq_len,
            "in_features": self.in_features,
            "epochs": self.epochs,
        }

        if self.verbose:
            print(
                f"\n[OK] Temporal Sequence Autoencoder trained successfully! "
                f"Validation Recon MSE: {ep_va_loss:.4f} across {self.latent_dim} latent dimensions.\n"
            )

        # Save checkpoint
        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": self.model.state_dict(),
                    "metadata": {
                        "metrics": self.metrics_,
                        "history": self.history_,
                        "latent_dim": self.latent_dim,
                        "in_features": self.in_features,
                        "seq_len": self.seq_len,
                    },
                },
                self.checkpoint_path,
            )

        return self

    def transform(self, tensor: np.ndarray) -> np.ndarray:
        """
        Transforms sequence tensor (N, seq_len, in_features) into latent embeddings (N, latent_dim).
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")

        self.model.eval()
        X_all = torch.from_numpy(tensor).float()
        loader = DataLoader(TensorDataset(X_all), batch_size=self.batch_size, shuffle=False)

        latents: List[np.ndarray] = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                z = self.model.encode(bx)
                latents.append(z.cpu().numpy())

        return np.vstack(latents).astype(np.float32)

