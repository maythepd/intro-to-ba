"""
Manifold Projection and Dimensionality Reduction Feature Extractors.
Provides:
  - PCAFeatureExtractor: Unsupervised orthogonal variance axes, reconstruction error, and Mahalanobis distance.
  - LDAFeatureExtractor: Supervised Fisher canonical projection, Bayesian posterior log-odds, and class centroid proximity.
"""

from .pca_features import PCAFeatureExtractor
from .lda_features import LDAFeatureExtractor

__all__ = ["PCAFeatureExtractor", "LDAFeatureExtractor"]
