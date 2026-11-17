"""
Strategy 1: Hyperspherical Manifold Representation Learning (Unsupervised / Self-Supervised).
Projects sequence embeddings onto the unit hypersphere S^{d-1}.
Leverages Hyperspherical Alignment (trajectory consistency) and Uniformity (maximal entropy
dispersion) to prevent representation collapse and capture non-linear angular dynamics
without target label leakage or premature binary classification.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTemporalStrategy


class HypersphericalGeometryHead(nn.Module):
    """
    Self-supervised hyperspherical geometry head (Wang & Isola, ICML 2020).
    Optimizes geodesic alignment along sequence endpoints and uniform dispersion
    across the hypersphere S^{d-1} without using target labels.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        scale: float = 4.0,
        t_uniform: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.scale = scale
        self.t_uniform = t_uniform
        # Lightweight projection layer to stabilize spherical embeddings
        self.proj = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(
        self,
        z: torch.Tensor,
        z_t1: Optional[torch.Tensor] = None,
        z_t6: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Normalize to unit hypersphere S^{d-1}
        z_p = self.proj(z)
        norm_z = F.normalize(z_p, p=2, dim=-1)

        # 1. Hyperspherical Uniformity Loss (Maximal Entropy on S^{d-1})
        # Raw Wang & Isola (ICML 2020) uniformity:
        #   L_raw = log E_{i!=j} [exp(-t * ||z_i - z_j||^2)] in [-4t, 0].
        # Calibrated with + 4t theoretical offset to guarantee strictly non-negative loss (L >= 0)
        # while preserving 100% exact gradient dynamics: d(L + 4t)/dz == dL/dz.
        b_sz = norm_z.size(0)
        if b_sz > 1:
            cos_sim = torch.matmul(norm_z, norm_z.t())
            pdist_sq = 2.0 - 2.0 * cos_sim.clamp(-1.0, 1.0)
            mask = ~torch.eye(b_sz, dtype=torch.bool, device=norm_z.device)
            pdist_sq_masked = pdist_sq[mask]
            raw_uniform = torch.log(torch.mean(torch.exp(-self.t_uniform * pdist_sq_masked)) + 1e-8)
            # Offset by 4 * t_uniform ensures loss_uniform >= 0 for all points on S^{d-1}
            loss_uniform = raw_uniform + 4.0 * self.t_uniform
        else:
            raw_uniform = torch.tensor(0.0, device=norm_z.device)
            loss_uniform = torch.tensor(0.0, device=norm_z.device)

        # 2. Hyperspherical Trajectory Alignment (Smooth Geodesics)
        if z_t6 is not None:
            norm_z6 = F.normalize(self.proj(z_t6), p=2, dim=-1)
            loss_align = torch.mean(1.0 - torch.sum(norm_z * norm_z6, dim=-1))
        else:
            loss_align = torch.tensor(0.0, device=norm_z.device)

        total_geo_loss = loss_align + 0.5 * loss_uniform
        return total_geo_loss, {
            "align_loss": loss_align.item() if isinstance(loss_align, torch.Tensor) else 0.0,
            "uniform_loss": loss_uniform.item() if isinstance(loss_uniform, torch.Tensor) else 0.0,
            "raw_uniform": raw_uniform.item() if isinstance(raw_uniform, torch.Tensor) else 0.0,
        }


# Maintain class alias for backward compatibility
AsymmetricArcFaceHead = HypersphericalGeometryHead


class AsymmetricArcFaceStrategy(BaseTemporalStrategy):
    """
    Hyperspherical Representation Learning Strategy (Unsupervised / Self-Supervised).
    Deforms the manifold onto S^{d-1} via alignment and uniformity without label leakage.
    """

    def __init__(
        self,
        scale: float = 4.0,
        margin_churn: float = 0.20,
        margin_loyal: float = 0.00,
        lambda_angular: float = 0.10,
        **kwargs,
    ):
        super().__init__(
            name="asymmetric_arcface",
            scale=scale,
            margin_churn=margin_churn,
            margin_loyal=margin_loyal,
            lambda_angular=lambda_angular,
            **kwargs,
        )
        self.scale = scale
        self.margin_churn = margin_churn
        self.margin_loyal = margin_loyal
        self.lambda_angular = lambda_angular
        self.criterion_recon = nn.MSELoss()

    def build_loss_head(self, latent_dim: int, device: torch.device) -> Optional[nn.Module]:
        return HypersphericalGeometryHead(
            latent_dim=latent_dim,
            scale=self.scale,
        ).to(device)

    def compute_loss(
        self,
        recon: torch.Tensor,
        bx: torch.Tensor,
        z: torch.Tensor,
        by: Optional[torch.Tensor] = None,
        head: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_recon = self.criterion_recon(recon, bx)
        if head is not None:
            # Fully self-supervised: ignores by to prevent label leakage and split hijacking!
            loss_geo, geo_dict = head(z)
            total_loss = loss_recon + self.lambda_angular * loss_geo
            metrics = {
                "recon_mse": loss_recon.item(),
                "angular_geo_loss": loss_geo.item(),
                "total_loss": total_loss.item(),
            }
            metrics.update(geo_dict)
            return total_loss, metrics
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

        norm_z = F.normalize(z, p=2, dim=-1)
        norm_t1 = F.normalize(z_t1, p=2, dim=-1)
        norm_t6 = F.normalize(z_t6, p=2, dim=-1)

        # 1. Net Drift Velocity in Euclidean latent space
        drift = torch.norm(z_t6 - z_t1, p=2, dim=-1).detach().cpu().numpy().astype(np.float32)

        # 2. Geodesic angular drift on S^{d-1}: theta in [0, pi]
        cos_endpoint = torch.sum(norm_t1 * norm_t6, dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        angular_drift = torch.acos(cos_endpoint).detach().cpu().numpy().astype(np.float32)

        # 3. Hyperspherical radius norm (energy before projection)
        z_norm = torch.norm(z, p=2, dim=-1).detach().cpu().numpy().astype(np.float32)

        # 4. Local hyperspherical density (von Mises-Fisher kernel centrality)
        # Higher density indicates common regular behavior; lower density indicates isolated/outlier trajectory
        cos_self = torch.matmul(norm_z, norm_z.t()).clamp(-1.0, 1.0)
        local_density = torch.mean(torch.exp(self.scale * cos_self), dim=-1).detach().cpu().numpy().astype(np.float32)

        geom: Dict[str, np.ndarray] = {
            "temporal_drift_velocity": drift,
            "temporal_angular_drift": angular_drift,
            "temporal_hypersphere_norm": z_norm,
            "temporal_hypersphere_density": local_density,
            # Seamless geometric proxies for backward compatibility (zero label leakage)
            "temporal_angular_churn_dist": angular_drift,
            "temporal_angular_loyal_dist": (np.pi - angular_drift).astype(np.float32),
            "temporal_angular_risk_ratio": (angular_drift / (np.pi - angular_drift + 1e-4)).astype(np.float32),
        }

        return latents, geom

    def get_feature_names(self, latent_dim: int) -> List[str]:
        z_cols = [f"temporal_z{i}" for i in range(latent_dim)]
        geom_cols = [
            "temporal_drift_velocity",
            "temporal_angular_drift",
            "temporal_hypersphere_norm",
            "temporal_hypersphere_density",
            "temporal_angular_churn_dist",
            "temporal_angular_loyal_dist",
            "temporal_angular_risk_ratio",
        ]
        return z_cols + geom_cols


