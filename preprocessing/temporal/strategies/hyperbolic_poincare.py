"""
Strategy 2: Hyperbolic Representation Learning in Poincaré Ball Manifold (Unsupervised / Self-Supervised).
Embeds sequential behaviors into the Poincaré Ball B_c^d, where volume grows
exponentially with radius. Stable, regular behaviors cluster compactly near the origin (root),
while anomalous shifts and complex trajectories naturally navigate the hyperbolic hierarchy
without supervised class prototypes or target label leakage.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTemporalStrategy


def poincare_project(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Projects Euclidean vectors into the open Poincaré ball with curvature c."""
    sqrt_c = math.sqrt(c)
    max_norm = (1.0 - eps) / sqrt_c
    norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(1e-10)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)


def poincare_dist(u: torch.Tensor, v: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Computes geodesic distance between u and v in Poincaré Ball:
    d_H(u, v) = (1/sqrt(c)) * arcosh(1 + 2 * c * ||u - v||^2 / ((1 - c||u||^2)(1 - c||v||^2)))
    """
    sqrt_c = math.sqrt(c)
    sqdist = torch.sum((u - v) ** 2, dim=-1)
    u_sqnorm = torch.sum(u ** 2, dim=-1).clamp_max((1.0 - eps) / c)
    v_sqnorm = torch.sum(v ** 2, dim=-1).clamp_max((1.0 - eps) / c)

    den = (1.0 - c * u_sqnorm) * (1.0 - c * v_sqnorm)
    den = den.clamp_min(eps)
    delta = 1.0 + 2.0 * c * sqdist / den
    delta = delta.clamp_min(1.0 + eps)
    # arcosh(x) = ln(x + sqrt(x^2 - 1))
    dist = torch.log(delta + torch.sqrt((delta ** 2 - 1.0).clamp_min(eps))) / sqrt_c
    return dist


def poincare_origin_dist(u: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Geodesic distance from u to origin 0:
    d_H(u, 0) = (2 / sqrt(c)) * artanh(sqrt(c) * ||u||)
    """
    sqrt_c = math.sqrt(c)
    norm = torch.norm(u, p=2, dim=-1).clamp_max((1.0 - eps) / sqrt_c)
    # artanh(y) = 0.5 * ln((1 + y) / (1 - y))
    y = (sqrt_c * norm).clamp(0.0, 1.0 - eps)
    dist = (1.0 / sqrt_c) * torch.log((1.0 + y) / (1.0 - y))
    return dist


class PoincareUnsupervisedHead(nn.Module):
    """
    Self-supervised Poincaré geometry head.
    Regularizes trajectory curvature and hierarchical radial dispersion in B_c^d
    without requiring class labels or supervised prototypes.
    """

    def __init__(self, latent_dim: int, curvature: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.curvature = curvature
        # Learnable reference origin anchor (stable equilibrium representation)
        self.origin_anchor = nn.Parameter(torch.zeros(latent_dim), requires_grad=False)

    def forward(
        self,
        z: torch.Tensor,
        z_t1: Optional[torch.Tensor] = None,
        z_t6: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        u = poincare_project(z, c=self.curvature)
        d_origin = poincare_origin_dist(u, c=self.curvature)
        loss_center = torch.mean(d_origin)

        # Geodesic smoothness if endpoints are provided
        if z_t1 is not None and z_t6 is not None:
            u1 = poincare_project(z_t1, c=self.curvature)
            u6 = poincare_project(z_t6, c=self.curvature)
            d_traj = poincare_dist(u1, u6, c=self.curvature)
            loss_traj = torch.mean(d_traj)
        else:
            loss_traj = torch.tensor(0.0, device=z.device)

        loss_total = 0.5 * loss_center + 0.5 * loss_traj
        return loss_total, {
            "poincare_center_loss": loss_center.item(),
            "poincare_traj_loss": loss_traj.item() if isinstance(loss_traj, torch.Tensor) else 0.0,
        }


# Maintain alias for backward compatibility
PoincarePrototypeHead = PoincareUnsupervisedHead


class HyperbolicPoincareStrategy(BaseTemporalStrategy):
    """
    Hyperbolic Poincaré Ball Representation Learning Strategy (Unsupervised / Self-Supervised).
    Enforces exponential hierarchy: loyal users stay in dense core,
    while anomalous behavioral drift traverses toward the hyperbolic boundary without label leakage.
    """

    def __init__(
        self,
        curvature: float = 1.0,
        lambda_poincare: float = 0.20,
        **kwargs,
    ):
        super().__init__(
            name="hyperbolic_poincare",
            curvature=curvature,
            lambda_poincare=lambda_poincare,
            **kwargs,
        )
        self.curvature = curvature
        self.lambda_poincare = lambda_poincare
        self.criterion_recon = nn.MSELoss()

    def build_loss_head(self, latent_dim: int, device: torch.device) -> Optional[nn.Module]:
        return PoincareUnsupervisedHead(latent_dim=latent_dim, curvature=self.curvature).to(device)

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
            # Completely unsupervised: ignores target labels by!
            loss_p, p_dict = head(z)
            total_loss = loss_recon + self.lambda_poincare * loss_p
            metrics = {
                "recon_mse": loss_recon.item(),
                "poincare_geo_loss": loss_p.item(),
                "total_loss": total_loss.item(),
            }
            metrics.update(p_dict)
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
        drift = torch.norm(z_t6 - z_t1, p=2, dim=-1).detach().cpu().numpy().astype(np.float32)

        # Projected Poincaré coordinates in B_c^d
        u = poincare_project(z, c=self.curvature)
        u1 = poincare_project(z_t1, c=self.curvature)
        u6 = poincare_project(z_t6, c=self.curvature)

        u_norm = torch.norm(u, p=2, dim=-1).detach().cpu().numpy().astype(np.float32)
        d_origin = poincare_origin_dist(u, c=self.curvature).detach().cpu().numpy().astype(np.float32)
        d_geodesic = poincare_dist(u1, u6, c=self.curvature).detach().cpu().numpy().astype(np.float32)

        # Conformal metric factor lambda_u = 2 / (1 - c * ||u||^2)
        u_sq = torch.sum(u ** 2, dim=-1).clamp_max((1.0 - 1e-5) / self.curvature)
        conformal_factor = (2.0 / (1.0 - self.curvature * u_sq)).detach().cpu().numpy().astype(np.float32)

        geom: Dict[str, np.ndarray] = {
            "temporal_drift_velocity": drift,
            "temporal_poincare_norm": u_norm,
            "temporal_poincare_origin_dist": d_origin,
            "temporal_poincare_geodesic_drift": d_geodesic,
            "temporal_poincare_conformal_factor": conformal_factor,
            # Seamless geometric proxies for backward compatibility (zero label leakage)
            "temporal_poincare_loyal_dist": (1.0 / (1.0 + d_origin)).astype(np.float32),
            "temporal_poincare_churn_dist": d_origin,
        }

        return latents, geom

    def get_feature_names(self, latent_dim: int) -> List[str]:
        z_cols = [f"temporal_z{i}" for i in range(latent_dim)]
        geom_cols = [
            "temporal_poincare_norm",
            "temporal_poincare_origin_dist",
            "temporal_poincare_geodesic_drift",
            "temporal_poincare_conformal_factor",
            "temporal_poincare_loyal_dist",
            "temporal_poincare_churn_dist",
            "temporal_drift_velocity",
        ]
        return z_cols + geom_cols


