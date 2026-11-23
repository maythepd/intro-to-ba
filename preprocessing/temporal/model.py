"""
Deep Temporal Sequence Autoencoder for Customer Event Trajectories.
Self-Supervised & Manifold Representation Learning using Bidirectional GRU.
Pluggable strategies:
  1. unsupervised: Pure self-supervised sequence reconstruction (zero target leakage baseline)
  2. asymmetric_arcface: Hyperspherical metric learning with cost-asymmetric angular margins
  3. hyperbolic_poincare: Poincaré ball embedding with exponential volume hierarchy
  4. optimal_transport: Dynamic trajectory velocity field, kinetic energy, and flow alignment
  5. neyman_pearson: Constrained optimization bounding False Positive Rate (FPR <= alpha)
"""

from pathlib import Path
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .strategies import (
    BaseTemporalStrategy,
    get_temporal_strategy,
    AsymmetricArcFaceStrategy,
    HyperbolicPoincareStrategy,
    OptimalTransportStrategy,
    NeymanPearsonStrategy,
    UnsupervisedStrategy,
    TemporalAttentionStrategy,
)
from .modules import TemporalAttentionAutoencoder


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
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=-1)
        return self.to_latent(h_cat)

    def encode_endpoints(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes sequence x and returns latent z, plus endpoint representations z_t1 and z_t6.
        """
        out_gru, h_n = self.encoder_gru(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=-1)
        z = self.to_latent(h_cat)
        z_t1 = self.to_latent(out_gru[:, 0, :])
        z_t6 = self.to_latent(out_gru[:, -1, :])
        return z, z_t1, z_t6

    def encode_with_geometry(
        self,
        x: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Backward compatibility helper for legacy callers.
        """
        z, z_t1, z_t6 = self.encode_endpoints(x)
        drift_norm = torch.norm(z_t6 - z_t1, p=2, dim=-1)
        geom: Dict[str, torch.Tensor] = {"temporal_drift_velocity": drift_norm}

        if prototypes is not None:
            norm_z = F.normalize(z, p=2, dim=-1)
            norm_w = F.normalize(prototypes, p=2, dim=-1)
            cos_sim = torch.matmul(norm_z, norm_w.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(cos_sim)

            geom["temporal_angular_loyal_dist"] = theta[:, 0]
            geom["temporal_angular_churn_dist"] = theta[:, 1]
            geom["temporal_angular_risk_ratio"] = theta[:, 0] / (theta[:, 1] + 1e-4)

        return z, geom

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: returns reconstructed sequence (batch, seq_len, in_features) and latent z.
        """
        z = self.encode(x)
        h_dec = self.from_latent(z).unsqueeze(0)
        dec_out, _ = self.decoder_gru(x, h_dec)
        recon = self.recon_head(dec_out)
        return recon, z


class TemporalAutoencoderTrainer:
    """
    Temporal Sequence Representation Trainer with Modular Strategy Pattern.
    Delegates metric learning and manifold geometry to pluggable BaseTemporalStrategy instances.
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
        manifold_strategy: str = "unsupervised",
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        # Direct keyword args for backwards compatibility:
        lambda_angular: float = 0.30,
        margin_churn: float = 0.20,
        margin_loyal: float = 0.00,
        angular_scale: float = 4.0,
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

        self.manifold_strategy = manifold_strategy.lower()

        # Build resolved strategy kwargs
        s_kwargs: Dict[str, Any] = dict(strategy_kwargs or {})
        if self.manifold_strategy == "asymmetric_arcface":
            s_kwargs.setdefault("scale", angular_scale)
            s_kwargs.setdefault("margin_churn", margin_churn)
            s_kwargs.setdefault("margin_loyal", margin_loyal)
            s_kwargs.setdefault("lambda_angular", lambda_angular)

        self.strategy_kwargs = s_kwargs
        self.strategy: BaseTemporalStrategy = get_temporal_strategy(self.manifold_strategy, **s_kwargs)

        self.device = get_device()
        self.model: Optional[BiGRUTemporalAutoencoder] = None
        self.strategy_head: Optional[nn.Module] = None
        self.prototypes_: Optional[torch.Tensor] = None
        self.history_: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_recon": []}
        self.metrics_: Dict[str, Any] = {}

    def fit(self, tensor: np.ndarray, labels: Optional[np.ndarray] = None) -> "TemporalAutoencoderTrainer":
        """
        Trains temporal sequence representation on 3D sequence tensor (N, seq_len, in_features).
        Delegates auxiliary metric learning to the configured strategy.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_samples, seq_len, in_features = tensor.shape
        self.seq_len = seq_len
        self.in_features = in_features

        # Strictly decouple architecture selection:
        # TemporalAttentionAutoencoder is ONLY used for 'temporal_attention' strategy!
        # The 4 specialized manifold strategies (asymmetric_arcface, hyperbolic_poincare,
        # optimal_transport, neyman_pearson) and unsupervised use the clean BiGRUTemporalAutoencoder backbone.
        if self.manifold_strategy == "temporal_attention":
            self.model = TemporalAttentionAutoencoder(
                in_features=in_features,
                seq_len=seq_len,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                latent_dim=self.latent_dim,
                dropout=self.dropout,
                conv_kernels=self.strategy_kwargs.get("conv_kernels", [2, 3]),
                noise_std=self.strategy_kwargs.get("noise_std", 0.02),
                noise_decay=self.strategy_kwargs.get("noise_decay", 0.95),
                use_feature_attention=self.strategy_kwargs.get("use_feature_attention", True),
                use_multiscale_conv=self.strategy_kwargs.get("use_multiscale_conv", True),
            ).to(self.device)
        else:
            self.model = BiGRUTemporalAutoencoder(
                in_features=in_features,
                seq_len=seq_len,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                latent_dim=self.latent_dim,
                dropout=self.dropout,
            ).to(self.device)

        # Instantiate strategy auxiliary loss head
        self.strategy_head = self.strategy.build_loss_head(latent_dim=self.latent_dim, device=self.device)

        # Check existing checkpoint unless force_retrain is active
        if not self.force_retrain and self.checkpoint_path and self.checkpoint_path.exists():
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                saved_model = ckpt.get("metadata", {}).get("metrics", {}).get("Model")
                current_model = self.model.__class__.__name__

                saved_keys = set(ckpt.get("state_dict", {}).keys())
                current_keys = set(self.model.state_dict().keys())
                is_mismatch = (saved_model and saved_model != current_model) or (saved_keys and current_keys and saved_keys != current_keys)

                # Check architecture compatibility before calling load_state_dict
                if is_mismatch:
                    if self.verbose:
                        mismatch_info = f"{saved_model} -> {current_model}" if (saved_model and saved_model != current_model) else "state_dict key mismatch"
                        print(
                            f"[INFO] Architecture changed ({mismatch_info}). "
                            f"Retraining {current_model} from scratch for strategy [{self.manifold_strategy.upper()}]..."
                        )
                else:
                    self.model.load_state_dict(ckpt["state_dict"])
                    if "strategy_head_state_dict" in ckpt and self.strategy_head is not None:
                        try:
                            self.strategy_head.load_state_dict(ckpt["strategy_head_state_dict"])
                        except Exception:
                            pass
                    elif "arcface_state_dict" in ckpt and self.strategy_head is not None:
                        try:
                            self.strategy_head.load_state_dict(ckpt["arcface_state_dict"])
                        except Exception:
                            pass

                    if hasattr(self.strategy_head, "weight"):
                        self.prototypes_ = self.strategy_head.weight.data.clone()
                    elif "prototypes" in ckpt:
                        self.prototypes_ = ckpt["prototypes"].to(self.device)

                    self.history_ = ckpt.get("metadata", {}).get("history", {})
                    self.metrics_ = ckpt.get("metadata", {}).get("metrics", {})
                    if self.verbose:
                        print(
                            f"[INFO] Loaded pretrained {current_model} [{self.manifold_strategy.upper()}] "
                            f"from: {self.checkpoint_path}"
                        )
                    return self
            except Exception as e:
                if self.verbose:
                    print(f"[INFO] Checkpoint incompatible with current architecture ({e}). Retraining from scratch...")

        model_name = self.model.__class__.__name__
        if self.verbose:
            print(
                f"[INFO] Training {model_name} [{self.manifold_strategy.upper()}], "
                f"epochs={self.epochs}, batch_size={self.batch_size}, latent_dim={self.latent_dim}, device={self.device}..."
            )

        # Train / Validation split
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        val_count = int(n_samples * self.val_size)
        val_idx = indices[:val_count]
        tr_idx = indices[val_count:]

        X_tr = torch.from_numpy(tensor[tr_idx]).float()
        X_va = torch.from_numpy(tensor[val_idx]).float()

        has_labels = labels is not None and self.strategy_head is not None
        if has_labels:
            y_arr = np.asarray(labels)
            y_tr = torch.from_numpy(y_arr[tr_idx]).long()
            y_va = torch.from_numpy(y_arr[val_idx]).long()
            tr_dataset = TensorDataset(X_tr, y_tr)
            va_dataset = TensorDataset(X_va, y_va)
        else:
            tr_dataset = TensorDataset(X_tr)
            va_dataset = TensorDataset(X_va)

        tr_loader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True)
        va_loader = DataLoader(va_dataset, batch_size=self.batch_size, shuffle=False)

        trainable_params = list(self.model.parameters())
        if self.strategy_head is not None:
            trainable_params += list(self.strategy_head.parameters())

        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=1e-4)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            if self.strategy_head is not None:
                self.strategy_head.train()

            total_tr_loss = 0.0
            total_tr_recon = 0.0
            total_tr_samples = 0

            pbar = tqdm(
                tr_loader,
                desc=f"[Epoch {epoch:02d}/{self.epochs:02d}]",
                disable=not self.verbose,
                leave=False,
                ncols=100,
                unit="batch",
            )
            for batch in pbar:
                bx = batch[0].to(self.device)
                by = batch[1].to(self.device) if has_labels else None

                optimizer.zero_grad()
                recon, z = self.model(bx)

                loss, loss_dict = self.strategy.compute_loss(
                    recon=recon,
                    bx=bx,
                    z=z,
                    by=by,
                    head=self.strategy_head,
                )

                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                batch_sz = bx.size(0)
                total_tr_loss += loss.item() * batch_sz
                recon_item = loss_dict.get("recon_mse", loss.item())
                total_tr_recon += recon_item * batch_sz
                total_tr_samples += batch_sz

                pbar.set_postfix({"loss": f"{loss.item():.4f}", "recon": f"{recon_item:.4f}"})

            ep_tr_loss = total_tr_loss / max(1, total_tr_samples)
            ep_tr_recon = total_tr_recon / max(1, total_tr_samples)

            # Validation pass
            self.model.eval()
            if self.strategy_head is not None:
                self.strategy_head.eval()

            total_va_loss = 0.0
            total_va_recon = 0.0
            total_va_samples = 0
            with torch.no_grad():
                for batch in va_loader:
                    bx = batch[0].to(self.device)
                    by = batch[1].to(self.device) if has_labels else None

                    recon, z = self.model(bx)
                    va_loss, va_dict = self.strategy.compute_loss(
                        recon=recon,
                        bx=bx,
                        z=z,
                        by=by,
                        head=self.strategy_head,
                    )

                    batch_sz = bx.size(0)
                    total_va_loss += va_loss.item() * batch_sz
                    total_va_recon += va_dict.get("recon_mse", va_loss.item()) * batch_sz
                    total_va_samples += batch_sz

            ep_va_loss = total_va_loss / max(1, total_va_samples)
            ep_va_recon = total_va_recon / max(1, total_va_samples)

            if hasattr(self.model, "step_epoch"):
                self.model.step_epoch()

            self.history_["train_loss"].append(round(ep_tr_loss, 4))
            self.history_["val_loss"].append(round(ep_va_loss, 4))
            self.history_["train_recon"].append(round(ep_tr_recon, 4))

            if self.verbose:
                bar_str = "━" * 22
                print(
                    f"[Epoch {epoch:02d}/{self.epochs:02d}] {bar_str} 100% | "
                    f"train_loss: {ep_tr_loss:.4f} (recon={ep_tr_recon:.4f}) | val_loss: {ep_va_loss:.4f} (recon={ep_va_recon:.4f})"
                )

        if self.strategy_head is not None and hasattr(self.strategy_head, "weight"):
            self.prototypes_ = self.strategy_head.weight.data.clone()

        self.metrics_ = {
            "Model": self.model.__class__.__name__,
            "Manifold_Strategy": self.manifold_strategy,
            "Final_Train_Loss": round(ep_tr_loss, 4),
            "Final_Val_Loss": round(ep_va_loss, 4),
            "Final_Val_Recon_MSE": round(ep_va_recon, 4),
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "seq_len": self.seq_len,
            "in_features": self.in_features,
            "epochs": self.epochs,
        }

        if self.verbose:
            print(
                f"\n[OK] Temporal Sequence Representation [{self.manifold_strategy.upper()}] trained successfully! "
                f"Val Loss: {ep_va_loss:.4f} across {self.latent_dim} latent dimensions.\n"
            )

        # Save checkpoint
        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_payload = {
                "state_dict": self.model.state_dict(),
                "metadata": {
                    "metrics": self.metrics_,
                    "history": self.history_,
                    "latent_dim": self.latent_dim,
                    "in_features": self.in_features,
                    "seq_len": self.seq_len,
                    "strategy": self.strategy.get_save_metadata(),
                },
            }
            if self.strategy_head is not None:
                save_payload["strategy_head_state_dict"] = self.strategy_head.state_dict()
                if hasattr(self.strategy_head, "weight"):
                    save_payload["prototypes"] = self.strategy_head.weight.data.cpu()
            torch.save(save_payload, self.checkpoint_path)

        return self

    def transform(self, tensor: np.ndarray) -> np.ndarray:
        """
        Transforms sequence tensor (N, seq_len, in_features) into latent embeddings (N, latent_dim).
        """
        latents, _ = self.transform_with_geometry(tensor)
        return latents

    def transform_with_geometry(self, tensor: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Transforms sequence tensor into latent embeddings and manifold geometric features
        as defined by the configured strategy.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")

        self.model.eval()
        if self.strategy_head is not None:
            self.strategy_head.eval()

        X_all = torch.from_numpy(tensor).float()
        loader = DataLoader(TensorDataset(X_all), batch_size=self.batch_size, shuffle=False)

        latents: List[np.ndarray] = []
        geom_accum: Dict[str, List[np.ndarray]] = {}

        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                z, z_t1, z_t6 = self.model.encode_endpoints(bx)
                batch_latents, batch_geom = self.strategy.compute_features(
                    z=z,
                    bx=bx,
                    z_t1=z_t1,
                    z_t6=z_t6,
                    head=self.strategy_head,
                )
                latents.append(batch_latents)
                for k, v in batch_geom.items():
                    if k not in geom_accum:
                        geom_accum[k] = []
                    geom_accum[k].append(v)

        stacked_latents = np.vstack(latents).astype(np.float32)
        stacked_geom = {k: np.concatenate(v).astype(np.float32) for k, v in geom_accum.items()}
        return stacked_latents, stacked_geom
