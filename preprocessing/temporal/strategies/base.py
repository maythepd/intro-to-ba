"""
Abstract Base Class for Temporal Sequence Representation Learning Strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class BaseTemporalStrategy(ABC):
    """
    Defines the contract for temporal sequence representation learning,
    manifold manipulation loss functions, and downstream feature generation.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs

    @abstractmethod
    def build_loss_head(self, latent_dim: int, device: torch.device) -> Optional[nn.Module]:
        """
        Instantiates auxiliary loss heads or learnable prototypes (if required).
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        recon: torch.Tensor,
        bx: torch.Tensor,
        z: torch.Tensor,
        by: Optional[torch.Tensor] = None,
        head: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes composite training loss and returns (total_loss, metric_dict).
        """
        pass

    @abstractmethod
    def compute_features(
        self,
        z: torch.Tensor,
        bx: torch.Tensor,
        z_t1: torch.Tensor,
        z_t6: torch.Tensor,
        head: Optional[nn.Module] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generates latent embeddings and manifold geometric features.
        Returns:
            latents: (B, latent_dim) array
            geom_dict: mapping from feature name to (B,) array
        """
        pass

    @abstractmethod
    def get_feature_names(self, latent_dim: int) -> List[str]:
        """
        Returns full list of feature column names produced by this strategy.
        """
        pass

    def get_save_metadata(self) -> Dict[str, Any]:
        """Returns metadata for checkpoint serialization."""
        return {"strategy_name": self.name, "strategy_params": self.params}

