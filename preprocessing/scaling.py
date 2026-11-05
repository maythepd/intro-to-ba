"""
Phase 2: Numerical feature scaling & standardization.
Applies RobustScaler (median / IQR) to heavy-tailed transactional and behavioral metrics,
and StandardScaler (mean / std) to symmetric demographic attributes.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from .base import BaseFeatureTransformer


class FeatureScaler(BaseFeatureTransformer):
    """
    Scales numerical features using RobustScaler and StandardScaler.
    Appends scaled features with prefix `robust_` or `std_`.
    """

    DEFAULT_ROBUST_FEATURES = [
        "recency_days",
        "monthly_transactions",
        "trans_amount_3m",
        "trans_trend_ratio",
        "app_logins_monthly",
    ]

    DEFAULT_STANDARD_FEATURES = [
        "age",
        "credit_score",
        "estimated_salary",
        "tenure",
    ]

    def __init__(
        self,
        enabled: bool = True,
        apply_robust_scaling: bool = True,
        apply_standard_scaling: bool = True,
        robust_features: Optional[List[str]] = None,
        standard_features: Optional[List[str]] = None,
        robust_prefix: str = "robust_",
        standard_prefix: str = "std_",
    ):
        super().__init__(name="FeatureScaler", enabled=enabled)
        self.apply_robust_scaling = apply_robust_scaling
        self.apply_standard_scaling = apply_standard_scaling
        self.robust_features = robust_features or list(self.DEFAULT_ROBUST_FEATURES)
        self.standard_features = standard_features or list(self.DEFAULT_STANDARD_FEATURES)
        self.robust_prefix = robust_prefix
        self.standard_prefix = standard_prefix

        self._robust_scaler: Optional[RobustScaler] = None
        self._standard_scaler: Optional[StandardScaler] = None
        self._fitted_robust_cols: List[str] = []
        self._fitted_standard_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureScaler":
        if not self.enabled:
            self.is_fitted = True
            return self

        # 1. Fit RobustScaler
        if self.apply_robust_scaling:
            self._fitted_robust_cols = [c for c in self.robust_features if c in X.columns]
            if self._fitted_robust_cols:
                self._robust_scaler = RobustScaler()
                self._robust_scaler.fit(X[self._fitted_robust_cols])

        # 2. Fit StandardScaler
        if self.apply_standard_scaling:
            self._fitted_standard_cols = [c for c in self.standard_features if c in X.columns]
            if self._fitted_standard_cols:
                self._standard_scaler = StandardScaler()
                self._standard_scaler.fit(X[self._fitted_standard_cols])

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()

        out = X.copy()
        created: List[str] = []

        # 1. Transform Robust features
        if self.apply_robust_scaling and self._robust_scaler is not None and self._fitted_robust_cols:
            scaled_vals = self._robust_scaler.transform(out[self._fitted_robust_cols])
            robust_col_names = [f"{self.robust_prefix}{c}" for c in self._fitted_robust_cols]
            df_scaled = pd.DataFrame(scaled_vals, columns=robust_col_names, index=out.index)
            out = pd.concat([out, df_scaled], axis=1)
            created.extend(robust_col_names)

        # 2. Transform Standard features
        if self.apply_standard_scaling and self._standard_scaler is not None and self._fitted_standard_cols:
            scaled_vals = self._standard_scaler.transform(out[self._fitted_standard_cols])
            std_col_names = [f"{self.standard_prefix}{c}" for c in self._fitted_standard_cols]
            df_scaled = pd.DataFrame(scaled_vals, columns=std_col_names, index=out.index)
            out = pd.concat([out, df_scaled], axis=1)
            created.extend(std_col_names)

        self.created_features = created
        return out

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "apply_robust_scaling": self.apply_robust_scaling,
            "apply_standard_scaling": self.apply_standard_scaling,
            "robust_features": self.robust_features,
            "standard_features": self.standard_features,
            "robust_prefix": self.robust_prefix,
            "standard_prefix": self.standard_prefix,
        })
        return params
