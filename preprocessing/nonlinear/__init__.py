"""
Non-Linear & Neural Representation Learning Sub-Package.

Provides modular architectures to transform tabular banking data into discriminative spaces:
  - NystroemKernelTransformer: Non-linear RBF kernel approximation into Hilbert space.
  - TabularDenoisingAutoencoderTransformer: Self-supervised DAE extracting manifold bottleneck vectors.
  - SupervisedTabularEmbeddingTransformer: Supervised deep bottleneck latent representation (Fisher-optimal).
  - ContrastiveTabularTransformer: Supervised contrastive metric learning on unit hypersphere.
  - NonlinearRepresentationManager: Unified orchestrator with checkpoint caching and feature augmentation.
"""

from .base import get_device, save_checkpoint, load_checkpoint, checkpoint_exists
from .kernel import NystroemKernelTransformer
from .autoencoder import TabularDAE, TabularDenoisingAutoencoderTransformer
from .supervised_latent import SupervisedTabNet, SupervisedTabularEmbeddingTransformer
from .contrastive import SupConLoss, ContrastiveTabNet, ContrastiveTabularTransformer
from .manager import NonlinearRepresentationManager
from .benchmark import run_representation_benchmark

__all__ = [
    "get_device",
    "save_checkpoint",
    "load_checkpoint",
    "checkpoint_exists",
    "NystroemKernelTransformer",
    "TabularDAE",
    "TabularDenoisingAutoencoderTransformer",
    "SupervisedTabNet",
    "SupervisedTabularEmbeddingTransformer",
    "SupConLoss",
    "ContrastiveTabNet",
    "ContrastiveTabularTransformer",
    "NonlinearRepresentationManager",
    "run_representation_benchmark",
]

