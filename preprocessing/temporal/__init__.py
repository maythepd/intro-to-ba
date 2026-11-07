"""
Temporal Behavioral Sequence Representation Package.
Extracts longitudinal event snapshots from raw transactions, logins, and complaints,
and generates deep self-supervised sequence embeddings via Bi-directional GRU Autoencoder.
"""

from .extractor import TemporalSnapshotExtractor
from .model import BiGRUTemporalAutoencoder, TemporalAutoencoderTrainer
from .manager import TemporalEmbeddingManager

__all__ = [
    "TemporalSnapshotExtractor",
    "BiGRUTemporalAutoencoder",
    "TemporalAutoencoderTrainer",
    "TemporalEmbeddingManager",
]

