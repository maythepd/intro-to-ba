"""
Phase 6: Feature selection & predictive importance ranking.
Calculates Mutual Information (non-linear dependency) and Random Forest Gini Importance,
with support for Top-K feature filtering for ablation experiments.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from .base import BaseFeatureTransformer


class FeatureSelector(BaseFeatureTransformer):
    """
    Ranks and selects features using Mutual Information and Random Forest importance.
    
    Ablation options:
      - enabled: Toggle feature selection on/off.
      - top_k: Select top K features (if None, all features are retained).
      - ranking_metric: 'RF_Importance' or 'MI_Score'.
    """

    def __init__(
        self,
        enabled: bool = False,
        top_k: Optional[int] = None,
        ranking_metric: str = "MI_Score",
        rf_n_estimators: int = 100,
        rf_max_depth: int = 8,
        random_state: int = 42,
    ):
        super().__init__(name="FeatureSelector", enabled=enabled)
        self.top_k = top_k
        self.ranking_metric = ranking_metric
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

        # 1. Mutual Information Score
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        df_mi = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores})

        # 2. Random Forest Importance
        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        df_rf = pd.DataFrame({"Feature": X.columns, "RF_Importance": rf.feature_importances_})

        # Merge importance metrics
        self.importance_df = pd.merge(df_mi, df_rf, on="Feature").sort_values(
            by=self.ranking_metric, ascending=False
        ).reset_index(drop=True)

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
            "rf_n_estimators": self.rf_n_estimators,
            "rf_max_depth": self.rf_max_depth,
            "random_state": self.random_state,
        })
        return params
