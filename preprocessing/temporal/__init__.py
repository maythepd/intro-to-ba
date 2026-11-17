"""
Temporal Behavioral Sequence Representation Package.
Extracts longitudinal event snapshots from raw transactions, logins, and complaints,
and generates deep sequence representations via Bi-directional GRU Autoencoder
with pluggable manifold strategies (Unsupervised, Asymmetric ArcFace, Poincaré Ball,
Optimal Transport, Neyman-Pearson).
"""

from .extractor import TemporalSnapshotExtractor
from .raw_extractor import RawEventSequenceExtractor
from .model import BiGRUTemporalAutoencoder, TemporalAutoencoderTrainer
from .manager import TemporalEmbeddingManager
from .strategies import (
    BaseTemporalStrategy,
    get_temporal_strategy,
    UnsupervisedStrategy,
    AsymmetricArcFaceStrategy,
    HyperbolicPoincareStrategy,
    OptimalTransportStrategy,
    NeymanPearsonStrategy,
    STRATEGY_REGISTRY,
)

__all__ = [
    "TemporalSnapshotExtractor",
    "RawEventSequenceExtractor",
    "BiGRUTemporalAutoencoder",
    "TemporalAutoencoderTrainer",
    "TemporalEmbeddingManager",
    "BaseTemporalStrategy",
    "get_temporal_strategy",
    "UnsupervisedStrategy",
    "AsymmetricArcFaceStrategy",
    "HyperbolicPoincareStrategy",
    "OptimalTransportStrategy",
    "NeymanPearsonStrategy",
    "STRATEGY_REGISTRY",
]
