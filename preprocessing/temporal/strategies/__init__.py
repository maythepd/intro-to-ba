"""
Temporal Sequence Representation Learning Strategies Package.
Provides pluggable manifold representation strategies:
  1. unsupervised: Pure self-supervised sequence reconstruction (zero target leakage baseline)
  2. asymmetric_arcface: Hyperspherical metric learning with cost-asymmetric angular margins
  3. hyperbolic_poincare: Poincaré ball embedding with exponential volume growth for hierarchical separation
  4. optimal_transport: Dynamic trajectory velocity field, kinetic energy, and alignment
  5. neyman_pearson: Constrained optimization bounding False Positive Rate (FPR <= alpha)
"""

from typing import Any, Dict

from .base import BaseTemporalStrategy
from .unsupervised import UnsupervisedStrategy
from .asymmetric_arcface import AsymmetricArcFaceStrategy
from .hyperbolic_poincare import HyperbolicPoincareStrategy
from .optimal_transport import OptimalTransportStrategy
from .neyman_pearson import NeymanPearsonStrategy
from .temporal_attention import TemporalAttentionStrategy

STRATEGY_REGISTRY = {
    "unsupervised": UnsupervisedStrategy,
    "asymmetric_arcface": AsymmetricArcFaceStrategy,
    "hyperbolic_poincare": HyperbolicPoincareStrategy,
    "optimal_transport": OptimalTransportStrategy,
    "neyman_pearson": NeymanPearsonStrategy,
    "temporal_attention": TemporalAttentionStrategy,
}


def get_temporal_strategy(strategy_name: str = "unsupervised", **kwargs) -> BaseTemporalStrategy:
    """
    Factory function to instantiate the requested temporal representation strategy.
    Falls back gracefully to 'unsupervised' if unknown strategy name is provided.
    """
    key = str(strategy_name).lower().strip()
    strategy_cls = STRATEGY_REGISTRY.get(key)
    if strategy_cls is None:
        valid_keys = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown temporal manifold strategy '{strategy_name}'. Valid options: {valid_keys}")
    return strategy_cls(**kwargs)


__all__ = [
    "BaseTemporalStrategy",
    "UnsupervisedStrategy",
    "AsymmetricArcFaceStrategy",
    "HyperbolicPoincareStrategy",
    "OptimalTransportStrategy",
    "NeymanPearsonStrategy",
    "TemporalAttentionStrategy",
    "STRATEGY_REGISTRY",
    "get_temporal_strategy",
]

