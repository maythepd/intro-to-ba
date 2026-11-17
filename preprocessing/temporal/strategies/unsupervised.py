"""
Unsupervised Sequence Reconstruction Strategy (Pure Self-Supervised Baseline).
Optimizes pure reconstruction MSE over multi-period event trajectories.
Zero target leakage, 100% unsupervised representation learning.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from .base import BaseTemporalStrategy


class UnsupervisedStrategy(BaseTemporalStrategy):
    """
    Standard Bidirectional GRU Sequence Autoencoder.
    Preserves maximum variance of the behavioral trajectory without class distortion.
    """

    def __init__(self, **kwargs):
        super().__init__(name="unsupervised", **kwargs)
        self.criterion = nn.MSELoss()

    def build_loss_head(self, latent_dim: int, device: torch.device) -> Optional[nn.Module]:
        return None

    def compute_loss(
        self,
        recon: torch.Tensor,
        bx: torch.Tensor,
        z: torch.Tensor,
        by: Optional[torch.Tensor] = None,
        head: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_recon = self.criterion(recon, bx)
        return loss_recon, {"recon_mse": loss_recon.item(), "total_loss": loss_recon.item()}

    def compute_features(
        self,
        z: torch.Tensor,
        bx: torch.Tensor,
        z_t1: torch.Tensor,
        z_t6: torch.Tensor,
        head: Optional[nn.Module] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        latents = z.detach().cpu().numpy().astype(np.float32)
        drift = torch.norm(z_t6 - z_t1, p=2, dim=-1).detach().cpu().numpy().astype(np.float32)
        geom = {"temporal_drift_velocity": drift}
        return latents, geom

    def get_feature_names(self, latent_dim: int) -> List[str]:
        z_cols = [f"temporal_z{i}" for i in range(latent_dim)]
        return z_cols + ["temporal_drift_velocity"]

