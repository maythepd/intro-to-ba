"""
Pillar 1-3 Deep Learning Building Blocks for Temporal Sequence Modeling:
1. FeatureAttentionLayer (Pillar 1: Dynamic channel gating across in_features)
2. TemporalAttentionPooling (Pillar 1: Dynamic temporal snapshot weighting across T steps)
3. MultiScaleTemporalConvBlock (Pillar 2: Parallel 1D Convolutions with kernels k=2,3 + residual skip)
4. ScheduledGaussianNoise (Pillar 3: Adaptive decaying noise for manifold regularization)
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MishActivation(nn.Module):
    """Fallback Mish activation function if torch.nn.Mish is unavailable."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "mish"):
            return F.mish(x)
        return x * torch.tanh(F.softplus(x))


class FeatureAttentionLayer(nn.Module):
    """
    Pillar 1: Feature Attention Layer (Channel Gating).
    Dynamically scores and weights the in_features channels (D=37) at each timestep:
        alpha_t = Sigmoid(W_2 * Mish(W_1 * x_t + b_1) + b_2)
        x_tilde = x_t * (0.5 + alpha_t)
    Allows the model to amplify high-signal behavior channels (login collapses,
    unresolved complaints, net flow deceleration) while attenuating noisy features.
    """

    def __init__(self, in_features: int, reduction_ratio: int = 2):
        super().__init__()
        self.in_features = in_features
        hidden_dim = max(in_features // reduction_ratio, 8)

        self.gate_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_features),
            MishActivation(),
            nn.Sigmoid(),
        )

        # Initialize output layer with zero bias so initial gating is balanced (~0.5)
        nn.init.zeros_(self.gate_net[2].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, In_Features)
        Returns:
            gated_x: Tensor of shape (Batch, Seq_Len, In_Features)
            gate_weights: Tensor of shape (Batch, Seq_Len, In_Features)
        """
        # gate_weights in (0, 1)
        gate_weights = self.gate_net(x)
        # Soft residual gating: prevents gradient vanishing if gates are initially small
        gated_x = x * (0.5 + gate_weights)
        return gated_x, gate_weights


class TemporalAttentionPooling(nn.Module):
    """
    Pillar 1: Temporal Attention Pooling Layer.
    Learns adaptive importance weights beta_1..beta_T over the sequence snapshots:
        e_t = v^T * tanh(W_h * h_t + b_h)
        beta_t = Softmax(e_t)
        context = sum_t (beta_t * h_t)
    Focuses the representation on decisive time periods (e.g. months 5-6 prior to survey
    or months with friction spikes) instead of uniform averaging.
    """

    def __init__(self, hidden_dim: int, attention_dim: Optional[int] = None):
        super().__init__()
        attn_dim = attention_dim or max(hidden_dim // 2, 16)
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_seq: Tensor of shape (Batch, Seq_Len, Hidden_Dim)
        Returns:
            context: (Batch, Hidden_Dim) weighted pooled representation
            weights: (Batch, Seq_Len) attention distribution across time steps
            entropy: (Batch,) attention entropy measuring concentration vs uniformity
        """
        # score: (Batch, Seq_Len, Attn_Dim) -> (Batch, Seq_Len, 1)
        scores = self.v(torch.tanh(self.proj(h_seq)))
        weights = F.softmax(scores, dim=1)  # (Batch, Seq_Len, 1)

        # Context vector: (Batch, Hidden_Dim)
        context = torch.sum(weights * h_seq, dim=1)

        # Attention distribution: (Batch, Seq_Len)
        weights_flat = weights.squeeze(-1)

        # Entropy H(beta) = - sum(beta * log(beta + eps))
        entropy = -torch.sum(weights_flat * torch.log(weights_flat + 1e-8), dim=-1)

        return context, weights_flat, entropy


class MultiScaleTemporalConvBlock(nn.Module):
    """
    Pillar 2: Multi-Scale 1D Convolutional Block.
    Runs parallel temporal convolutions with multiple receptive fields (e.g. k=2 for
    bi-monthly abrupt shocks, k=3 for quarterly trends) in parallel to RNNs.
    Outputs are fused through a learnable scaled residual skip connection.
    """

    def __init__(
        self,
        in_features: int,
        out_channels: int,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes or [2, 3]
        self.branches = nn.ModuleList()

        for k in self.kernel_sizes:
            # Receptive field padding along time dimension
            padding = (k - 1) // 2
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_features,
                    out_channels=out_channels,
                    kernel_size=k,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                MishActivation(),
                nn.Dropout(dropout),
            )
            self.branches.append(branch)

        total_conv_channels = out_channels * len(self.kernel_sizes)
        self.fusion = nn.Sequential(
            nn.Linear(total_conv_channels, out_channels),
            MishActivation(),
        )

        # Learnable residual skip scaling parameter alpha (Xavier-like initialization)
        self.alpha_skip = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, In_Features)
        Returns:
            conv_out: Tensor of shape (Batch, Seq_Len, Out_Channels)
        """
        B, T, D = x.shape
        # Permute to (Batch, In_Features, Seq_Len) for Conv1D
        x_perm = x.permute(0, 2, 1)

        branch_outs = []
        for k, branch in zip(self.kernel_sizes, self.branches):
            b_out = branch(x_perm)
            # Ensure exact length match with T if even kernel caused padding delta
            if b_out.shape[-1] != T:
                b_out = F.interpolate(b_out, size=T, mode="linear", align_corners=False)
            branch_outs.append(b_out)

        # Concat across channel dimension: (Batch, Total_Channels, Seq_Len)
        cat_out = torch.cat(branch_outs, dim=1)
        # Permute back: (Batch, Seq_Len, Total_Channels)
        cat_out = cat_out.permute(0, 2, 1)

        # Linear fusion projection
        fused = self.fusion(cat_out)
        return fused


class ScheduledGaussianNoise(nn.Module):
    """
    Pillar 3: Scheduled Gaussian Noise Injection.
    Injects decaying Gaussian noise during training:
        x' = x + N(0, sigma_e^2)
    where sigma_e = initial_std * (decay_rate ^ epoch).
    Smooths the learned manifold representation, encouraging representations to be
    continuous and robust to micro-variations. Active strictly during training mode.
    """

    def __init__(self, initial_std: float = 0.02, decay_rate: float = 0.95, min_std: float = 0.001):
        super().__init__()
        self.initial_std = initial_std
        self.decay_rate = decay_rate
        self.min_std = min_std
        self.current_std = initial_std

    def step_epoch(self) -> float:
        """Decays noise intensity at the end of an epoch."""
        self.current_std = max(self.current_std * self.decay_rate, self.min_std)
        return self.current_std

    def reset(self):
        """Resets current noise level to initial_std."""
        self.current_std = self.initial_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.current_std > 0:
            noise = torch.randn_like(x) * self.current_std
            return x + noise
        return x

