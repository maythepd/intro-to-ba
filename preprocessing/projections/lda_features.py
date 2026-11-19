"""
Linear Discriminant Analysis (LDA) Supervised Manifold Feature Extractor.
Extracts Fisher canonical discriminant projection, posterior probabilities,
log-odds, and class centroid proximity strictly without target leakage.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


class LDAFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Supervised Linear Discriminant Analysis feature extractor for customer churn.
    Projects multi-dimensional customer features onto the 1D Fisher discriminant axis,
    Bayesian posterior log-odds, and class centroid distances.
    """

    def __init__(
        self,
        enabled: bool = False,
        solver: str = "lsqr",
        shrinkage: Optional[Union[str, float]] = "auto",
        enable_probabilities: bool = True,
        enable_log_odds: bool = True,
        enable_centroid_distances: bool = True,
        verbose: bool = False,
    ):
        self.enabled = enabled
        self.solver = solver
        self.shrinkage = shrinkage
        self.enable_probabilities = enable_probabilities
        self.enable_log_odds = enable_log_odds
        self.enable_centroid_distances = enable_centroid_distances
        self.verbose = verbose

        # Fitted state
        self.is_fitted_ = False
        self.scaler_ = None
        self.lda_ = None
        self.classes_: List[Any] = []
        self.means_: Optional[np.ndarray] = None
        self.fit_columns_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "LDAFeatureExtractor":
        """
        Fits StandardScaler and supervised LDA strictly on training partition with target y.
        Never fit on validation or test partitions.
        """
        if not self.enabled:
            return self

        if y is None:
            if self.verbose:
                print("[WARN] LDAFeatureExtractor requires target y for fitting; skipping.")
            return self

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.fit_columns_ = [c for c in numeric_cols if c not in ("customer_id", "churn", "is_churn")]

        if not self.fit_columns_:
            self.is_fitted_ = True
            return self

        y_arr = np.asarray(y)
        unique_classes = np.unique(y_arr)
        if len(unique_classes) < 2:
            if self.verbose:
                print(f"[WARN] LDAFeatureExtractor encountered single class {unique_classes}; skipping.")
            return self

        X_num = X[self.fit_columns_].fillna(0).to_numpy()

        # Step 1: Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_num)

        # Step 2: Fit Supervised LDA with optional Ledoit-Wolf shrinkage
        resolved_solver = self.solver
        resolved_shrinkage = self.shrinkage
        if resolved_shrinkage is not None and resolved_solver == "svd":
            resolved_solver = "lsqr"

        self.lda_ = LinearDiscriminantAnalysis(
            solver=resolved_solver,
            shrinkage=resolved_shrinkage,
        )
        self.lda_.fit(X_scaled, y_arr)

        self.classes_ = list(self.lda_.classes_)
        self.means_ = getattr(self.lda_, "means_", None)
        self.is_fitted_ = True

        if self.verbose:
            print(f"[OK] LDAFeatureExtractor fitted on {len(self.fit_columns_)} features (Solver: {resolved_solver}, Shrinkage: {resolved_shrinkage})")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Projects X onto Fisher discriminant axis and Bayesian posterior probabilities.
        """
        if not self.enabled or not self.is_fitted_ or self.lda_ is None or not self.fit_columns_:
            return X

        N = len(X)
        if N == 0:
            return X

        # Align columns
        missing_cols = [c for c in self.fit_columns_ if c not in X.columns]
        if missing_cols:
            X_work = X.copy()
            for mc in missing_cols:
                X_work[mc] = 0.0
            X_num = X_work[self.fit_columns_].fillna(0).to_numpy()
        else:
            X_num = X[self.fit_columns_].fillna(0).to_numpy()

        X_scaled = self.scaler_.transform(X_num)
        feat_dict: Dict[str, np.ndarray] = {}

        # 1. Fisher Canonical 1D Projection
        try:
            lda_proj = self.lda_.transform(X_scaled).ravel()
        except NotImplementedError:
            # Fallback for solvers like 'lsqr' that do not implement transform():
            # In binary classification, decision_function computes X.dot(coef_.T) + intercept_,
            # which is the exact Fisher discriminant 1D canonical projection axis.
            lda_proj = self.lda_.decision_function(X_scaled).ravel()
        feat_dict["lda_projection_churn"] = lda_proj

        # 2. Bayesian Posterior Probability & Log-Odds
        probs = self.lda_.predict_proba(X_scaled)
        churn_idx = self.classes_.index(1) if 1 in self.classes_ else (len(self.classes_) - 1)
        loyal_idx = self.classes_.index(0) if 0 in self.classes_ else 0

        p_churn = probs[:, churn_idx]
        p_loyal = probs[:, loyal_idx]

        if self.enable_probabilities:
            feat_dict["lda_prob_churn"] = p_churn

        if self.enable_log_odds:
            p_c = np.clip(p_churn, 1e-6, 1.0 - 1e-6)
            p_l = np.clip(p_loyal, 1e-6, 1.0 - 1e-6)
            feat_dict["lda_log_odds"] = np.log(p_c / p_l)

        # 3. Class centroid distances in standardized feature space
        if self.enable_centroid_distances and self.means_ is not None and len(self.means_) >= 2:
            mean_loyal = self.means_[loyal_idx]
            mean_churn = self.means_[churn_idx]
            dist_loyal = np.linalg.norm(X_scaled - mean_loyal, axis=1)
            dist_churn = np.linalg.norm(X_scaled - mean_churn, axis=1)
            
            feat_dict["lda_dist_to_loyal_centroid"] = dist_loyal
            feat_dict["lda_dist_to_churn_centroid"] = dist_churn
            feat_dict["lda_relative_churn_proximity"] = dist_loyal / (dist_loyal + dist_churn + 1e-6)

        df_lda = pd.DataFrame(feat_dict, index=X.index)
        self.feature_names_out_ = list(df_lda.columns)

        overlap = set(X.columns).intersection(set(df_lda.columns))
        if overlap:
            df_lda = df_lda.drop(columns=list(overlap))

        return pd.concat([X, df_lda], axis=1)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Returns the list of generated LDA feature names."""
        return list(self.feature_names_out_)
