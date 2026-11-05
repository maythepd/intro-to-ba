"""
Base interface and utilities for feature engineering transformers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd


class BaseFeatureTransformer(ABC):
    """
    Abstract base class for all feature engineering and transformation modules.
    Follows scikit-learn transformer conventions (fit, transform, fit_transform).
    """

    def __init__(self, name: str = "BaseTransformer", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.is_fitted = False
        self.created_features: List[str] = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseFeatureTransformer":
        """Fit transformer parameters on training data."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature transformation to input dataframe."""
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in a single pass."""
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        """Return list of feature names generated or managed by this transformer."""
        return self.created_features

    def get_params(self) -> Dict[str, Any]:
        """Return parameter configuration for inspection and reproduction."""
        return {"name": self.name, "enabled": self.enabled, "is_fitted": self.is_fitted}
