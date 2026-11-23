"""
Strategy 4: Neyman-Pearson Level-Set & Quantile Envelope Representation Learning (Unsupervised / Self-Supervised).
Estimates a minimum-volume statistical level-set envelope at controlled risk level alpha (FPR <= alpha)
without target labels (Scott & Nowak 2005; Tong et al. 2018).
Identifies high-energy, atypical behavioral trajectories via unsupervised pinball quantile regression,
preventing premature binary classification and label leakage.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTemporalStrategy


class NeymanPearsonQuantileHead(nn.Module):
    """
    Self-supervised Neyman-Pearson quantile envelope head.
    Learns a statistical level-set energy threshold tau_alpha at quantile (1 - alpha_fpr)
    via pinball loss on behavioral reconstruction energy without class labels.
    """

    def __init__(
        self,
        latent_dim: int,
        alpha_fpr: float = 0.10,
        tau: float = 0.50,
        gamma: float = 0.10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.alpha_fpr = alpha_fpr
        self.gamma = gamma
        # Learnable level-set threshold parameter tau_alpha initialized to tau
        self.tau_alpha = nn.Parameter(torch.tensor(float(tau)))

    def forward(
        self,
        energy: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Pinball loss for quantile q = 1 - alpha_fpr
        # L_q(u) = max(q * u, (q - 1) * u) = max(q * (E - tau), -(1 - q) * (E - tau))
        q = 1.0 - self.alpha_fpr
        u = energy - self.tau_alpha
        pinball_loss = torch.mean(torch.max(q * u, (q - 1.0) * u))

        # Soft risk probability score in [0, 1]
        risk_score = torch.sigmoid(u / self.gamma)
        return pinball_loss, {
            "pinball_quantile_loss": pinball_loss.item(),
            "tau_alpha": self.tau_alpha.item(),
        }


# Maintain alias for backward compatibility
NeymanPearsonHead = NeymanPearsonQuantileHead


class NeymanPearsonStrategy(BaseTemporalStrategy):
    """
    Neyman-Pearson Level-Set & Quantile Envelope Strategy (Unsupervised / Self-Supervised).
    Binds the empirical quantile envelope to risk budget alpha without target label leakage.
    """

    def __init__(
        self,
        alpha_fpr: float = 0.10,
        lambda_np: float = 0.10,
        tau: float = 0.50,
        gamma: float = 0.10,
        **kwargs,
    ):
        super().__init__(
            name="neyman_pearson",
            alpha_fpr=alpha_fpr,
            lambda_np=lambda_np,
            tau=tau,
            gamma=gamma,
            **kwargs,
        )
        self.alpha_fpr = alpha_fpr
        self.lambda_np = lambda_np
        self.tau = tau
        self.gamma = gamma
        self.criterion_recon = nn.MSELoss()

    def build_loss_head(self, latent_dim: int, device: torch.device) -> Optional[nn.Module]:
        return NeymanPearsonQuantileHead(
            latent_dim=latent_dim,
            alpha_fpr=self.alpha_fpr,
            tau=self.tau,
            gamma=self.gamma,
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
            # Reconstruction energy per sample: E_i = MSE(bx_i, recon_i)
            # bx: (B, T, D), recon: (B, T, D)
            energy = torch.mean((bx - recon) ** 2, dim=(1, 2))
            loss_np, np_dict = head(energy)
            total_loss = loss_recon + self.lambda_np * loss_np
            metrics = {
                "recon_mse": loss_recon.item(),
                "np_levelset_loss": loss_np.item(),
                "total_loss": total_loss.item(),
            }
            metrics.update(np_dict)
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

        # Unsupervised sample-level energy: deviation between endpoints
        # Higher drift/energy means atypical trajectory
        traj_energy = torch.mean((z_t6 - z_t1) ** 2, dim=-1).detach().cpu().numpy().astype(np.float32)

        tau_val = float(head.tau_alpha.item()) if head is not None and hasattr(head, "tau_alpha") else float(self.tau)
        margin = (traj_energy - tau_val).astype(np.float32)

        # Smooth surrogate risk / novelty score in [0, 1]
        risk_score = 1.0 / (1.0 + np.exp(-margin / self.gamma))

        geom: Dict[str, np.ndarray] = {
            "temporal_drift_velocity": drift,
            "temporal_np_recon_energy": traj_energy,
            "temporal_np_margin": margin.astype(np.float32),
            "temporal_np_risk_score": risk_score.astype(np.float32),
        }

        return latents, geom

    def get_feature_names(self, latent_dim: int) -> List[str]:
        z_cols = [f"temporal_z{i}" for i in range(latent_dim)]
        geom_cols = [
            "temporal_drift_velocity",
            "temporal_np_recon_energy",
            "temporal_np_margin",
            "temporal_np_risk_score",
        ]
        return z_cols + geom_cols


