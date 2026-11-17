"""
Phase 6: Feature selection & predictive importance ranking.
Calculates Mutual Information (non-linear dependency), Pearson Correlation (linear predictive power),
Hybrid MI-Pearson score, and Random Forest Gini Importance, with support for Top-K filtering.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from ..base import BaseFeatureTransformer


class FeatureSelector(BaseFeatureTransformer):
    """
    Ranks and selects features using Mutual Information, Pearson Correlation,
    Hybrid MI-Pearson score, and Random Forest importance.
    
    Ablation options:
      - enabled: Toggle feature selection on/off.
      - top_k: Select top K features (if None, all features are retained).
      - ranking_metric: 'MI_Score', 'Pearson', 'MI_Pearson_Hybrid', or 'RF_Importance'.
      - hybrid_pearson_weight: Weight given to Pearson vs MI in 'MI_Pearson_Hybrid' (default: 0.50).
    """

    SUPPORTED_METRICS = ["MI_Score", "Pearson", "MI_Pearson_Hybrid", "RF_Importance"]

    def __init__(
        self,
        enabled: bool = False,
        top_k: Optional[int] = None,
        ranking_metric: str = "MI_Score",
        hybrid_pearson_weight: float = 0.50,
        rf_n_estimators: int = 100,
        rf_max_depth: int = 8,
        random_state: int = 42,
        verbose: bool = True,
    ):
        super().__init__(name="FeatureSelector", enabled=enabled)
        self.top_k = top_k
        self.ranking_metric = ranking_metric
        self.hybrid_pearson_weight = float(np.clip(hybrid_pearson_weight, 0.0, 1.0))
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.random_state = random_state
        self.verbose = verbose

        self.importance_df: Optional[pd.DataFrame] = None
        self.selected_features_: List[str] = []

    def _compute_pearson(self, X: pd.DataFrame, y: pd.Series) -> List[float]:
        """Fast vectorized Pearson point-biserial correlation with binary target."""
        y_vals = np.asarray(y, dtype=np.float64)
        X_vals = np.asarray(X, dtype=np.float64)
        N = len(y_vals)
        std_y = np.std(y_vals)
        if std_y <= 1e-8 or N <= 1:
            return [0.0] * X.shape[1]

        y_norm = (y_vals - np.mean(y_vals)) / (std_y * N)
        x_mean = np.mean(X_vals, axis=0)
        x_std = np.std(X_vals, axis=0)
        x_std_safe = np.where(x_std > 1e-8, x_std, 1.0)
        X_norm = (X_vals - x_mean) / x_std_safe
        pearson_vec = np.abs(np.dot(y_norm, X_norm))
        pearson_vec = np.where(x_std > 1e-8, pearson_vec, 0.0)
        return np.nan_to_num(pearson_vec, nan=0.0).tolist()

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        compute_ranking: bool = False,
    ) -> "FeatureSelector":
        self.selected_features_ = list(X.columns)

        if y is None or (not self.enabled and not compute_ranking):
            self.is_fitted = True
            return self

        # Full ranking computation (for reporting and analysis)
        if compute_ranking:
            # 1. Mutual Information Score (Non-linear relationship)
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
            df_mi = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores})

            # 2. Pearson Correlation Score (Linear relationship)
            pearson_scores = self._compute_pearson(X, y)
            df_pearson = pd.DataFrame({"Feature": X.columns, "Pearson": pearson_scores})

            # 3. Hybrid MI-Pearson Score: normalized convex combination
            mi_arr = np.array(mi_scores)
            p_arr = np.array(pearson_scores)
            mi_min, mi_max = mi_arr.min(), mi_arr.max()
            p_min, p_max = p_arr.min(), p_arr.max()
            mi_norm = (mi_arr - mi_min) / (mi_max - mi_min + 1e-8)
            p_norm = (p_arr - p_min) / (p_max - p_min + 1e-8)
            w_p = self.hybrid_pearson_weight
            w_mi = 1.0 - w_p
            hybrid_scores = w_mi * mi_norm + w_p * p_norm
            df_hybrid = pd.DataFrame({"Feature": X.columns, "MI_Pearson_Hybrid": hybrid_scores})

            # 4. Random Forest Importance
            rf = RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X, y)
            df_rf = pd.DataFrame({"Feature": X.columns, "RF_Importance": rf.feature_importances_})

            # Merge all importance metrics
            merged = pd.merge(df_mi, df_pearson, on="Feature")
            merged = pd.merge(merged, df_hybrid, on="Feature")
            merged = pd.merge(merged, df_rf, on="Feature")
            sort_col = self.ranking_metric if self.ranking_metric in merged.columns else "MI_Score"
            self.importance_df = merged.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
        else:
            # Fast-path for CV folds: compute only the requested ranking_metric
            if self.ranking_metric == "Pearson":
                p_scores = self._compute_pearson(X, y)
                self.importance_df = pd.DataFrame({"Feature": X.columns, "Pearson": p_scores}).sort_values(
                    by="Pearson", ascending=False
                ).reset_index(drop=True)
            elif self.ranking_metric == "MI_Score":
                mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
                self.importance_df = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores}).sort_values(
                    by="MI_Score", ascending=False
                ).reset_index(drop=True)
            elif self.ranking_metric == "MI_Pearson_Hybrid":
                mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
                p_scores = self._compute_pearson(X, y)
                mi_arr, p_arr = np.array(mi_scores), np.array(p_scores)
                mi_norm = (mi_arr - mi_arr.min()) / (mi_arr.max() - mi_arr.min() + 1e-8)
                p_norm = (p_arr - p_arr.min()) / (p_arr.max() - p_arr.min() + 1e-8)
                hybrid_scores = (1.0 - self.hybrid_pearson_weight) * mi_norm + self.hybrid_pearson_weight * p_norm
                self.importance_df = pd.DataFrame({"Feature": X.columns, "MI_Pearson_Hybrid": hybrid_scores}).sort_values(
                    by="MI_Pearson_Hybrid", ascending=False
                ).reset_index(drop=True)
            elif self.ranking_metric == "RF_Importance":
                rf = RandomForestClassifier(
                    n_estimators=self.rf_n_estimators,
                    max_depth=self.rf_max_depth,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                rf.fit(X, y)
                self.importance_df = pd.DataFrame({"Feature": X.columns, "RF_Importance": rf.feature_importances_}).sort_values(
                    by="RF_Importance", ascending=False
                ).reset_index(drop=True)
            else:
                p_scores = self._compute_pearson(X, y)
                self.importance_df = pd.DataFrame({"Feature": X.columns, "Pearson": p_scores}).sort_values(
                    by="Pearson", ascending=False
                ).reset_index(drop=True)

        # Determine selected features
        if self.enabled and self.top_k is not None and self.top_k < len(X.columns):
            self.selected_features_ = self.importance_df["Feature"].head(self.top_k).tolist()
            if self.verbose:
                print(
                    f"[INFO] FeatureSelector ({self.ranking_metric}): Filtered {len(X.columns)} -> "
                    f"{len(self.selected_features_)} features (Top-{self.top_k} selected, "
                    f"dropped {len(X.columns) - len(self.selected_features_)} lowest-scoring features)"
                )
        else:
            self.selected_features_ = list(X.columns)
            if self.verbose and self.enabled:
                print(f"[INFO] FeatureSelector ({self.ranking_metric}): Retained all {len(self.selected_features_)} features (top_k={self.top_k})")

        self.created_features = self.selected_features_
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled or not self.selected_features_:
            return X.copy()

        available_cols = [c for c in self.selected_features_ if c in X.columns]
        return X[available_cols].copy()

    def get_importance_ranking(self) -> Optional[pd.DataFrame]:
        """Returns the full importance ranking DataFrame."""
        return self.importance_df

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "top_k": self.top_k,
            "ranking_metric": self.ranking_metric,
            "hybrid_pearson_weight": self.hybrid_pearson_weight,
            "rf_n_estimators": self.rf_n_estimators,
            "rf_max_depth": self.rf_max_depth,
            "random_state": self.random_state,
        })
        return params
