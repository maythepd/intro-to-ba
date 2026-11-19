"""
Principal Component Analysis (PCA) Manifold Feature Extractor.
Extracts orthogonal variance axes, residual anomaly scores,
and manifold Mahalanobis distance strictly without target leakage.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts unsupervised orthogonal principal components, reconstruction errors,
    and manifold Mahalanobis distance from numeric customer features.
    """

    def __init__(
        self,
        enabled: bool = False,
        n_components: int = 5,
        whiten: bool = False,
        enable_reconstruction_error: bool = True,
        enable_mahalanobis: bool = True,
        verbose: bool = False,
    ):
        self.enabled = enabled
        self.n_components = n_components
        self.whiten = whiten
        self.enable_reconstruction_error = enable_reconstruction_error
        self.enable_mahalanobis = enable_mahalanobis
        self.verbose = verbose

        # Fitted state
        self.is_fitted_ = False
        self.scaler_ = None
        self.pca_ = None
        self.fit_columns_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PCAFeatureExtractor":
        """
        Fits StandardScaler and PCA on numeric columns of training partition only.
        Unsupervised: strictly ignores y.
        """
        if not self.enabled:
            return self

        # Extract numeric columns (strictly excluding IDs and target labels)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.fit_columns_ = [c for c in numeric_cols if c not in ("customer_id", "churn", "is_churn")]

        if not self.fit_columns_:
            self.is_fitted_ = True
            return self

        X_num = X[self.fit_columns_].fillna(0).to_numpy()

        # Step 1: Standardize features to mean=0, variance=1
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_num)

        # Step 2: Fit PCA
        k = min(int(self.n_components), X_scaled.shape[1], X_scaled.shape[0])
        self.pca_ = PCA(n_components=k, whiten=self.whiten, random_state=42)
        self.pca_.fit(X_scaled)

        self.is_fitted_ = True
        if self.verbose:
            var_ratio = np.sum(self.pca_.explained_variance_ratio_) * 100.0
            print(f"[OK] PCAFeatureExtractor fitted {k} components (Explained Variance: {var_ratio:.2f}%)")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Projects X onto fitted PCA manifold and appends extracted features.
        """
        if not self.enabled or not self.is_fitted_ or self.pca_ is None or not self.fit_columns_:
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
        pca_coords = self.pca_.transform(X_scaled)  # (N, k)
        k = pca_coords.shape[1]

        feat_dict: Dict[str, np.ndarray] = {}

        # 1. Principal orthogonal coordinates
        for i in range(k):
            feat_dict[f"pca_component_{i+1}"] = pca_coords[:, i]

        # 2. Residual Reconstruction Anomaly Score
        if self.enable_reconstruction_error:
            X_recon = self.pca_.inverse_transform(pca_coords)
            recon_diff = X_scaled - X_recon
            recon_err = np.linalg.norm(recon_diff, axis=1)
            feat_dict["pca_reconstruction_error"] = recon_err
            
            # Relative reconstruction ratio: ||x - \hat{x}|| / (||x|| + eps)
            norm_orig = np.linalg.norm(X_scaled, axis=1) + 1e-6
            feat_dict["pca_relative_recon_error"] = recon_err / norm_orig

        # 3. Manifold Mahalanobis distance
        if self.enable_mahalanobis:
            ev = np.maximum(self.pca_.explained_variance_, 1e-6)
            maha_sq = np.sum((pca_coords ** 2) / ev, axis=1)
            feat_dict["pca_mahalanobis_dist"] = np.sqrt(np.maximum(maha_sq, 0.0))

        df_pca = pd.DataFrame(feat_dict, index=X.index)
        self.feature_names_out_ = list(df_pca.columns)

        overlap = set(X.columns).intersection(set(df_pca.columns))
        if overlap:
            df_pca = df_pca.drop(columns=list(overlap))

        return pd.concat([X, df_pca], axis=1)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Returns the list of generated PCA feature names."""
        return list(self.feature_names_out_)
