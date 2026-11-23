"""
Strategy: Temporal Attention Representation Learning (Dual-Attention & Multi-Scale Conv1D).
Implements the 4-pillar deep learning architecture inspired by BAREFOOTS:
  1. Feature Attention & Temporal Attention Pooling
  2. Multi-Scale Temporal Convolutions (k=2,3) + BiGRU Backbone + Scaled Skip
  3. Scheduled Gaussian Noise Injection, Mish activations, LayerNorm, and Gradient Safety
  4. Huber Directional Correlation Hybrid Loss
Maintains 100% unsupervised manifold integrity (zero target leakage, orthogonal to downstream XGBoost).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from .base import BaseTemporalStrategy
from ..modules.losses import HuberDirectionalCorrelationLoss


class TemporalAttentionStrategy(BaseTemporalStrategy):
    """
    Temporal Attention Representation Learning Strategy.
    Combines hierarchical dual-attention, multi-scale 1D convolutions, scheduled noise,
    and hybrid trend-preserving reconstruction loss.
    """

    def __init__(
        self,
        delta: float = 1.0,
        lambda_direction: float = 0.10,
        lambda_correlation: float = 0.05,
        use_feature_attention: bool = True,
        use_multiscale_conv: bool = True,
        conv_kernels: Optional[List[int]] = None,
        noise_std: float = 0.02,
        noise_decay: float = 0.95,
        **kwargs,
    ):
        super().__init__(
            name="temporal_attention",
            delta=delta,
            lambda_direction=lambda_direction,
            lambda_correlation=lambda_correlation,
            use_feature_attention=use_feature_attention,
            use_multiscale_conv=use_multiscale_conv,
            conv_kernels=conv_kernels or [2, 3],
            noise_std=noise_std,
            noise_decay=noise_decay,
            **kwargs,
        )
        self.delta = delta
        self.lambda_direction = lambda_direction
        self.lambda_correlation = lambda_correlation
        self.use_feature_attention = use_feature_attention
        self.use_multiscale_conv = use_multiscale_conv
        self.conv_kernels = conv_kernels or [2, 3]
        self.noise_std = noise_std
        self.noise_decay = noise_decay

        self.criterion = HuberDirectionalCorrelationLoss(
            delta=self.delta,
            lambda_direction=self.lambda_direction,
            lambda_correlation=self.lambda_correlation,
        )

    def build_loss_head(self, latent_dim: int, device: torch.device) -> Optional[nn.Module]:
        """Self-supervised representation learning does not require auxiliary supervised heads."""
        return None

    def compute_loss(
        self,
        recon: torch.Tensor,
        bx: torch.Tensor,
        z: torch.Tensor,
        by: Optional[torch.Tensor] = None,
        head: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss, metrics = self.criterion(recon, bx)
        return loss, metrics

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
        geom = {
            "temporal_drift_velocity": drift,
        }
        return latents, geom

    def get_feature_names(self, latent_dim: int) -> List[str]:
        z_cols = [f"temporal_z{i}" for i in range(latent_dim)]
        return z_cols + ["temporal_drift_velocity"]

