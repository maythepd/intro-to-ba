"""
Strategy 3: Optimal Transport & Dynamic Trajectory Kinetics (Unsupervised / Self-Supervised).
Models customer behavioral evolution as dynamical trajectories on the manifold.
Extracts kinetic action, trajectory acceleration (curvature), geodesic length, and tortuosity
to distinguish directed ballistic drift from random equilibrium fluctuations
without supervised class prototypes or target label leakage.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTemporalStrategy


class TrajectoryKineticsHead(nn.Module):
    """
    Self-supervised trajectory dynamics & kinetic action head (Benamou-Brenier optimal transport).
    Regularizes trajectory kinetic action (Hamilton's principle of least action)
    without requiring target labels.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(
        self,
        z: torch.Tensor,
        z_t1: Optional[torch.Tensor] = None,
        z_t6: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # If explicit endpoints are provided, compute net ballistic kinetic action
        if z_t1 is not None and z_t6 is not None:
            v_drift = z_t6 - z_t1
            loss_action = 0.5 * torch.mean(torch.sum(v_drift ** 2, dim=-1))
        else:
            # Fallback to latent norm kinetic regularization
            loss_action = 0.5 * torch.mean(torch.sum(z ** 2, dim=-1))

        return loss_action, {"kinetic_action_loss": loss_action.item()}


# Maintain alias for backward compatibility
TrajectoryVelocityHead = TrajectoryKineticsHead


class OptimalTransportStrategy(BaseTemporalStrategy):
    """
    Optimal Transport & Dynamic Trajectory Kinetics Strategy (Unsupervised / Self-Supervised).
    Quantifies the physics of behavioral movement across time:
    kinetic action, geodesic path length, acceleration, and tortuosity.
    """

    def __init__(
        self,
        lambda_drift: float = 0.05,
        **kwargs,
    ):
        super().__init__(
            name="optimal_transport",
            lambda_drift=lambda_drift,
            **kwargs,
        )
        self.lambda_drift = lambda_drift
        self.criterion_recon = nn.MSELoss()

    def build_loss_head(self, latent_dim: int, device: torch.device) -> Optional[nn.Module]:
        return TrajectoryKineticsHead(latent_dim=latent_dim).to(device)

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
            # Unsupervised kinetic action: ignores target label by to eliminate split hijacking!
            loss_kinetics, k_dict = head(z)
            total_loss = loss_recon + self.lambda_drift * loss_kinetics
            metrics = {
                "recon_mse": loss_recon.item(),
                "kinetic_loss": loss_kinetics.item(),
                "total_loss": total_loss.item(),
            }
            metrics.update(k_dict)
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

        # 1. Net Ballistic Drift Displacement: D = ||z_t6 - z_t1||
        v_drift = z_t6 - z_t1
        drift_norm = torch.norm(v_drift, p=2, dim=-1).detach().cpu().numpy().astype(np.float32)

        # 2. Geodesic Path Length: L = ||z_t6 - z|| + ||z - z_t1||
        seg1 = torch.norm(z - z_t1, p=2, dim=-1)
        seg2 = torch.norm(z_t6 - z, p=2, dim=-1)
        geodesic_length = (seg1 + seg2).detach().cpu().numpy().astype(np.float32)

        # 3. Tortuosity (Straightness Ratio): tau = D / (L + eps) in (0, 1]
        # tau ~ 1: Highly directed ballistic path (monotonic degradation)
        # tau << 1: Brownian random wandering around equilibrium
        tortuosity = (drift_norm / (geodesic_length + 1e-5)).clip(0.0, 1.0).astype(np.float32)

        # 4. Kinetic Action / Dynamic Energy: S = 0.5 * D^2
        kinetic_action = (0.5 * (drift_norm ** 2)).astype(np.float32)

        # 5. Trajectory Curvature / Acceleration Norm: a = (z_t6 - z) - (z - z_t1)
        accel = (z_t6 - z) - (z - z_t1)
        accel_norm = torch.norm(accel, p=2, dim=-1).detach().cpu().numpy().astype(np.float32)

        geom: Dict[str, np.ndarray] = {
            "temporal_drift_velocity": drift_norm,
            "temporal_geodesic_length": geodesic_length,
            "temporal_tortuosity": tortuosity,
            "temporal_kinetic_action": kinetic_action,
            "temporal_acceleration_norm": accel_norm,
            # Seamless geometric proxies for backward compatibility (zero label leakage)
            "temporal_ot_churn_alignment": tortuosity,
            "temporal_ot_loyal_alignment": (1.0 - tortuosity).astype(np.float32),
        }

        return latents, geom

    def get_feature_names(self, latent_dim: int) -> List[str]:
        z_cols = [f"temporal_z{i}" for i in range(latent_dim)]
        geom_cols = [
            "temporal_drift_velocity",
            "temporal_geodesic_length",
            "temporal_tortuosity",
            "temporal_kinetic_action",
            "temporal_acceleration_norm",
            "temporal_ot_churn_alignment",
            "temporal_ot_loyal_alignment",
        ]
        return z_cols + geom_cols


