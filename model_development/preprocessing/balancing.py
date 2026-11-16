"""
Phase 5: Class imbalance remediation.
Provides minority oversampling strategies (SMOTE) strictly for the training partition.
"""

from typing import Any, Dict, Optional, Tuple
import pandas as pd
from imblearn.over_sampling import SMOTE


class ClassBalancer:
    """
    Handles class imbalance remediation on training partitions.
    Supports easy ablation (toggle enabled=False to benchmark without SMOTE).
    """

    def __init__(
        self,
        enabled: bool = True,
        method: str = "smote",
        random_state: int = 42,
        sampling_strategy: str = "auto",
        k_neighbors: int = 5,
    ):
        self.enabled = enabled
        self.method = method
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors

        self._sampler: Optional[SMOTE] = None

    def fit_resample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Resamples training data to mitigate class imbalance.
        If disabled or method is None, returns X and y unmodified.
        """
        if not self.enabled or self.method is None:
            return X, y

        if self.method.lower() == "smote":
            self._sampler = SMOTE(
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.k_neighbors,
            )
            X_res, y_res = self._sampler.fit_resample(X, y)
            # Retain pandas DataFrame structure and column headers
            X_res_df = pd.DataFrame(X_res, columns=X.columns)
            y_res_sr = pd.Series(y_res, name=y.name)
            return X_res_df, y_res_sr
        else:
            raise ValueError(f"Unsupported balancing method: {self.method}")

    def get_params(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "random_state": self.random_state,
            "sampling_strategy": self.sampling_strategy,
            "k_neighbors": self.k_neighbors,
        }
