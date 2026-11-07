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

from .base import BaseFeatureTransformer


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
    ):
        super().__init__(name="FeatureSelector", enabled=enabled)
        self.top_k = top_k
        self.ranking_metric = ranking_metric
        self.hybrid_pearson_weight = float(np.clip(hybrid_pearson_weight, 0.0, 1.0))
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.random_state = random_state

        self.importance_df: Optional[pd.DataFrame] = None
        self.selected_features_: List[str] = []

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

        # 1. Mutual Information Score (Non-linear relationship)
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        df_mi = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores})

        # 2. Pearson Correlation Score (Linear relationship with binary target / Point-Biserial)
        y_vals = np.asarray(y, dtype=np.float64)
        pearson_scores = []
        for col in X.columns:
            x_col = np.asarray(X[col], dtype=np.float64)
            std_x = np.std(x_col)
            std_y = np.std(y_vals)
            if std_x > 1e-8 and std_y > 1e-8:
                corr = np.corrcoef(x_col, y_vals)[0, 1]
                pearson_scores.append(abs(corr) if not np.isnan(corr) else 0.0)
            else:
                pearson_scores.append(0.0)

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

        # Determine active sorting metric
        sort_col = self.ranking_metric if self.ranking_metric in merged.columns else "MI_Score"
        self.importance_df = merged.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

        # Determine selected features
        if self.enabled and self.top_k is not None and self.top_k < len(X.columns):
            self.selected_features_ = self.importance_df["Feature"].head(self.top_k).tolist()
        else:
            self.selected_features_ = list(X.columns)

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
