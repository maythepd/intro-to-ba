"""
TemporalAttentionAutoencoder:
Unified Deep Learning Architecture for Longitudinal Customer Event Sequences.
Integrates the 4 pillars inspired by BAREFOOTS:
  1. Feature Attention Gating (Channel-wise weighting across D=37 channels)
  2. Multi-Scale Temporal Convolutions (Parallel k=2,3 Conv1D) + BiGRU Backbone + Scaled Skip
  3. Scheduled Gaussian Noise Injection, Mish activations, LayerNorm, and Gradient Safety
  4. Temporal Attention Pooling across snapshots + Unsupervised Latent Representation
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_blocks import (
    FeatureAttentionLayer,
    TemporalAttentionPooling,
    MultiScaleTemporalConvBlock,
    ScheduledGaussianNoise,
    MishActivation,
)


class TemporalAttentionAutoencoder(nn.Module):
    """
    Advanced Temporal Sequence Autoencoder featuring:
    - Scheduled Gaussian Noise Injection
    - Feature Attention Gating across D input features
    - Dual Backbone (Multi-Scale Conv1D + Bidirectional GRU) with Scaled Residual Fusion
    - Temporal Attention Pooling across T snapshots
    - LayerNorm + Mish Bottleneck mapping to latent dimension z
    - Autoregressive / Recurrent Sequence Reconstruction Decoder
    """

    def __init__(
        self,
        in_features: int = 37,
        seq_len: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        latent_dim: int = 16,
        dropout: float = 0.20,
        conv_kernels: Optional[List[int]] = None,
        noise_std: float = 0.02,
        noise_decay: float = 0.95,
        use_feature_attention: bool = True,
        use_multiscale_conv: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.use_feature_attention = use_feature_attention
        self.use_multiscale_conv = use_multiscale_conv

        # 1. Scheduled Gaussian Noise (Pillar 3)
        self.noise_module = ScheduledGaussianNoise(
            initial_std=noise_std,
            decay_rate=noise_decay,
            min_std=0.001,
        )

        # 2. Feature Attention Layer (Pillar 1)
        if self.use_feature_attention:
            self.feature_attention = FeatureAttentionLayer(in_features=in_features)
        else:
            self.feature_attention = None

        # 3. Dual Backbone: Bi-GRU + Multi-Scale Conv1D (Pillar 2)
        self.encoder_gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        gru_out_dim = hidden_dim * 2  # Bidirectional

        if self.use_multiscale_conv:
            self.conv_block = MultiScaleTemporalConvBlock(
                in_features=in_features,
                out_channels=hidden_dim,
                kernel_sizes=conv_kernels or [2, 3],
                dropout=dropout,
            )
            # Projection to fuse Conv1D with BiGRU
            self.backbone_fusion = nn.Sequential(
                nn.Linear(gru_out_dim + hidden_dim, gru_out_dim),
                MishActivation(),
                nn.Dropout(dropout),
            )
        else:
            self.conv_block = None
            self.backbone_fusion = None

        # 4. Temporal Attention Pooling (Pillar 1)
        self.temporal_attention = TemporalAttentionPooling(hidden_dim=gru_out_dim)

        # 5. Bottleneck Projection to Latent z (Pillar 3: LayerNorm + Mish)
        self.to_latent = nn.Sequential(
            nn.Linear(gru_out_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            MishActivation(),
        )

        # 6. Decoder Modules
        dec_hidden = hidden_dim * 2
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, dec_hidden),
            MishActivation(),
        )
        self.decoder_gru = nn.GRU(
            input_size=in_features,
            hidden_size=dec_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.recon_head = nn.Sequential(
            nn.Linear(dec_hidden, in_features),
        )

        self._init_weights()

    def _init_weights(self):
        """Initializes linear weights with Xavier normal to avoid vanishing/exploding gradients."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def step_epoch(self) -> float:
        """Decays the noise standard deviation at epoch boundary."""
        return self.noise_module.step_epoch()

    def forward_encoder(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Passes input through noise, feature attention, dual backbone, and temporal attention.
        Returns:
            z: (Batch, Latent_Dim)
            h_seq: (Batch, Seq_Len, Hidden_Dim*2)
            attn_weights: (Batch, Seq_Len)
            aux_info: Dictionary containing attention entropy and feature gate statistics
        """
        # Step A: Noise injection
        x_noisy = self.noise_module(x)

        # Step B: Feature attention gating
        if self.feature_attention is not None:
            x_gated, gate_weights = self.feature_attention(x_noisy)
            mean_gate = torch.mean(gate_weights, dim=[1, 2])  # (Batch,)
        else:
            x_gated = x_noisy
            mean_gate = torch.ones(x.size(0), device=x.device)

        # Step C: Dual backbone execution
        gru_seq, _ = self.encoder_gru(x_gated)  # (Batch, Seq_Len, gru_out_dim)

        if self.conv_block is not None and self.backbone_fusion is not None:
            conv_seq = self.conv_block(x_gated)  # (Batch, Seq_Len, hidden_dim)
            fused_seq = self.backbone_fusion(torch.cat([gru_seq, conv_seq], dim=-1))
            # Residual connection with learnable alpha
            h_seq = fused_seq + self.conv_block.alpha_skip * gru_seq
        else:
            h_seq = gru_seq

        # Step D: Temporal attention pooling
        context, attn_weights, attn_entropy = self.temporal_attention(h_seq)

        # Step E: Latent bottleneck mapping
        z = self.to_latent(context)

        aux_info = {
            "temporal_attention_weights": attn_weights,
            "temporal_attention_entropy": attn_entropy,
            "temporal_feature_gate_mean": mean_gate,
        }

        return z, h_seq, attn_weights, aux_info

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes sequence tensor into latent vector z."""
        z, _, _, _ = self.forward_encoder(x)
        return z

    def encode_endpoints(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes sequence x and returns latent z, along with endpoint representations z_t1 and z_t6.
        """
        z, h_seq, _, _ = self.forward_encoder(x)
        z_t1 = self.to_latent(h_seq[:, 0, :])
        z_t6 = self.to_latent(h_seq[:, -1, :])
        return z, z_t1, z_t6

    def encode_with_geometry(
        self,
        x: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns latent vector z and auxiliary geometric/attention features.
        """
        z, h_seq, _, aux = self.forward_encoder(x)
        z_t1 = self.to_latent(h_seq[:, 0, :])
        z_t6 = self.to_latent(h_seq[:, -1, :])

        drift_norm = torch.norm(z_t6 - z_t1, p=2, dim=-1)

        geom: Dict[str, torch.Tensor] = {
            "temporal_drift_velocity": drift_norm,
            "temporal_attention_entropy": aux["temporal_attention_entropy"],
            "temporal_feature_gate_intensity": aux["temporal_feature_gate_mean"],
        }

        return z, geom

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full Autoencoder forward pass: returns reconstructed sequence and latent bottleneck z.
        """
        z, _, _, _ = self.forward_encoder(x)
        h_dec = self.from_latent(z).unsqueeze(0)  # (1, Batch, dec_hidden)
        dec_out, _ = self.decoder_gru(x, h_dec)
        recon = self.recon_head(dec_out)
        return recon, z

