"""
Deep Learning Operations & Attention Modules for Temporal Sequences.
Implements the 4 pillars from BAREFOOTS:
  1. FeatureAttentionLayer & TemporalAttentionPooling
  2. MultiScaleTemporalConvBlock
  3. ScheduledGaussianNoise & MishActivation
  4. HuberDirectionalCorrelationLoss & TemporalAttentionAutoencoder
"""

from .attention_blocks import (
    FeatureAttentionLayer,
    TemporalAttentionPooling,
    MultiScaleTemporalConvBlock,
    ScheduledGaussianNoise,
    MishActivation,
)
from .losses import HuberDirectionalCorrelationLoss
from .attention_model import TemporalAttentionAutoencoder

__all__ = [
    "FeatureAttentionLayer",
    "TemporalAttentionPooling",
    "MultiScaleTemporalConvBlock",
    "ScheduledGaussianNoise",
    "MishActivation",
    "HuberDirectionalCorrelationLoss",
    "TemporalAttentionAutoencoder",
]

