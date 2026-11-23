"""
Pillar 4 Deep Learning Loss Modules:
HuberDirectionalCorrelationLoss:
  1. Huber (Smooth-L1) Loss for robust value reconstruction (dampens outlier whale transactions)
  2. Direction-Aware Trend Loss (penalizes opposite trajectory momentum)
  3. Pearson Correlation Loss (enforces longitudinal curve shape preservation)
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberDirectionalCorrelationLoss(nn.Module):
    """
    Pillar 4: Hybrid Trend-Aware Reconstruction Loss for Customer Behavioral Sequences.
    Replaces brittle MSE with a robust, trend-preserving multi-objective:
        L_total = L_Huber + lambda_dir * L_direction + lambda_corr * L_correlation
    """

    def __init__(
        self,
        delta: float = 1.0,
        lambda_direction: float = 0.10,
        lambda_correlation: float = 0.05,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.delta = delta
        self.lambda_direction = lambda_direction
        self.lambda_correlation = lambda_correlation
        self.eps = eps

    def compute_huber_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Robust point-wise reconstruction using Huber / Smooth-L1 loss."""
        return F.smooth_l1_loss(recon, target, beta=self.delta)

    def compute_direction_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Penalizes reconstructions that predict the wrong direction of change across consecutive months:
            delta_target = target[:, t] - target[:, t-1]
            delta_recon  = recon[:, t]  - recon[:, t-1]
            penalty      = ReLU(- sign(delta_target) * delta_recon)
        Scale-invariant and denominator-free to eliminate division-by-zero or gradient spikes.
        """
        if recon.size(1) <= 1:
            return torch.tensor(0.0, device=recon.device, dtype=recon.dtype)

        d_target = target[:, 1:, :] - target[:, :-1, :]
        d_recon = recon[:, 1:, :] - recon[:, :-1, :]

        # Target change direction (-1, 0, +1)
        sign_target = torch.sign(d_target)

        # Soft sign violation penalty: active strictly when recon moves in opposite direction of target
        violation = F.relu(-sign_target * d_recon)
        return torch.mean(violation)

    def compute_correlation_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes Pearson correlation loss across the sequence snapshots:
            L_corr = 1.0 - mean(Pearson_r(recon, target))
        Encourages the model to match longitudinal trajectory profiles rather than static points.
        Numerically stabilized:
          1. Epsilon added inside square roots to prevent infinite gradients at zero variance:
             d/dx sqrt(x + eps) = 1 / (2 * sqrt(x + eps)) <= 1 / (2 * sqrt(eps)) < inf.
          2. Channels with static target values (target_var <= 1e-4) are masked out, as Pearson r
             is mathematically undefined for constant sequences.
        """
        if recon.size(1) <= 2:
            return torch.tensor(0.0, device=recon.device, dtype=recon.dtype)

        # Center across time dimension (dim=1)
        recon_mean = torch.mean(recon, dim=1, keepdim=True)
        target_mean = torch.mean(target, dim=1, keepdim=True)

        recon_centered = recon - recon_mean
        target_centered = target - target_mean

        # Covariance and variances along time dimension
        cov = torch.sum(recon_centered * target_centered, dim=1)
        recon_var = torch.sum(recon_centered ** 2, dim=1)
        target_var = torch.sum(target_centered ** 2, dim=1)

        # Mask channels with dynamic target variation (constant sequences have undefined correlation)
        valid_mask = target_var > 1e-4

        # Stabilized standard deviations (eps INSIDE sqrt ensures bounded gradients)
        recon_std = torch.sqrt(recon_var + self.eps)
        target_std = torch.sqrt(target_var + self.eps)
        denom = recon_std * target_std

        r = (cov / denom).clamp(-1.0, 1.0)

        # 1.0 - r: 0 when perfectly correlated, 2 when inverted
        if valid_mask.any():
            return torch.mean((1.0 - r)[valid_mask])
        return torch.tensor(0.0, device=recon.device, dtype=recon.dtype)

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_huber = self.compute_huber_loss(recon, target)
        loss_dir = self.compute_direction_loss(recon, target)
        loss_corr = self.compute_correlation_loss(recon, target)

        total_loss = (
            loss_huber
            + self.lambda_direction * loss_dir
            + self.lambda_correlation * loss_corr
        )

        metrics = {
            "huber_loss": loss_huber.item(),
            "direction_loss": loss_dir.item(),
            "correlation_loss": loss_corr.item(),
            "recon_mse": loss_huber.item(),
            "recon_loss": loss_huber.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics
