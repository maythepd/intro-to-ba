"""
Phase 4: Categorical encoding strategies.
Provides One-Hot Encoding and Bayesian m-estimate Smoothed Target Encoding.
Designed to prevent data leakage by learning priors on training data.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .base import BaseFeatureTransformer


class CategoricalEncoder(BaseFeatureTransformer):
    """
    Encodes categorical features via One-Hot Encoding and Target Encoding.
    
    Features engineered:
      - One-Hot Encoded columns: <col>_<category>
      - Smoothed Target Encoded columns: <col>_target_enc
        Formula: S_i = (n_i * mean_y_i + m * global_mean_y) / (n_i + m)
    """

    DEFAULT_CATEGORICAL_COLS = [
        "country",
        "gender",
        "preferred_channel",
        "primary_device",
    ]

    def __init__(
        self,
        enabled: bool = True,
        apply_ohe: bool = True,
        apply_target_enc: bool = True,
        categorical_cols: Optional[List[str]] = None,
        m_weight: float = 10.0,
        ohe_drop_first: bool = False,
    ):
        super().__init__(name="CategoricalEncoder", enabled=enabled)
        self.apply_ohe = apply_ohe
        self.apply_target_enc = apply_target_enc
        self.categorical_cols = categorical_cols or list(self.DEFAULT_CATEGORICAL_COLS)
        self.m_weight = m_weight
        self.ohe_drop_first = ohe_drop_first

        # State learned during fit
        self._target_enc_maps: Dict[str, Dict[Any, float]] = {}
        self._global_target_mean: float = 0.0
        self._ohe_categories: Dict[str, List[Any]] = {}
        self._expected_ohe_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CategoricalEncoder":
        if not self.enabled:
            self.is_fitted = True
            return self

        # 1. Learn OHE categories
        if self.apply_ohe:
            self._ohe_categories = {}
            expected_cols: List[str] = []
            for col in self.categorical_cols:
                if col in X.columns:
                    # Sort unique categories for deterministic column ordering
                    cats = sorted(X[col].dropna().unique().tolist())
                    if self.ohe_drop_first and len(cats) > 1:
                        cats = cats[1:]
                    self._ohe_categories[col] = cats
                    for cat in cats:
                        expected_cols.append(f"{col}_{cat}")
            self._expected_ohe_cols = expected_cols

        # 2. Learn Target Encoding mappings (requires y)
        if self.apply_target_enc and y is not None:
            self._global_target_mean = float(y.mean())
            self._target_enc_maps = {}

            # Join col and target temporarily for group stats
            df_fit = pd.DataFrame({"target": y})
            for col in self.categorical_cols:
                if col in X.columns:
                    df_fit[col] = X[col]
                    stats_grp = df_fit.groupby(col)["target"].agg(["count", "mean"])
                    smoothed = (
                        stats_grp["count"] * stats_grp["mean"] + self.m_weight * self._global_target_mean
                    ) / (stats_grp["count"] + self.m_weight)
                    self._target_enc_maps[col] = smoothed.to_dict()

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()

        out = X.copy()
        created: List[str] = []

        # 1. Target Encoding
        if self.apply_target_enc and self._target_enc_maps:
            for col, mapping in self._target_enc_maps.items():
                if col in out.columns:
                    enc_col_name = f"{col}_target_enc"
                    out[enc_col_name] = out[col].map(mapping).fillna(self._global_target_mean).astype(float)
                    created.append(enc_col_name)

        # 2. One-Hot Encoding
        if self.apply_ohe and self._ohe_categories:
            ohe_dfs: List[pd.DataFrame] = []
            for col, cats in self._ohe_categories.items():
                if col in out.columns:
                    for cat in cats:
                        col_name = f"{col}_{cat}"
                        col_series = (out[col] == cat).astype(int)
                        ohe_dfs.append(pd.DataFrame({col_name: col_series}, index=out.index))
                        created.append(col_name)

            if ohe_dfs:
                out = pd.concat([out] + ohe_dfs, axis=1)

        self.created_features = created
        return out

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "apply_ohe": self.apply_ohe,
            "apply_target_enc": self.apply_target_enc,
            "categorical_cols": self.categorical_cols,
            "m_weight": self.m_weight,
            "ohe_drop_first": self.ohe_drop_first,
        })
        return params
