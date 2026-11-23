"""
Modular Cluster-Based Feature Extractor for Digital Bank Churn Prediction.

Extracts rich, interpretable, anti-leakage customer segmentation features:
  1. Centroid Distances (Euclidean & Cosine distances to each cluster center)
  2. Soft Membership Probabilities & Assignment Entropy (Uncertainty on cluster boundary)
  3. Atypicality, Inertia & Boundary Margin (Distance to closest, 2nd closest, silhouette margin)
  4. Risk Affinity & Churn Persona Contrast (Distance ratio between safe vs risky cluster prototypes)
  5. Peer-Relative Deviation Ratios (Customer metrics relative to cluster baseline means)
  6. Discrete Cluster IDs & One-Hot Encoding
  7. Hierarchical Linkage Features (Distances to Ward hierarchical medoids & agreement flag)

All feature sub-groups are independently toggleable via config.yaml and CLI flags.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Support cross-platform unpickling if needed
try:
    import pathlib
    if not hasattr(pathlib, "PosixPath"):
        pathlib.PosixPath = pathlib.WindowsPath
except Exception:
    pass

# Ensure BalancedFeatures is available for unpickling cluster_bundle.joblib
try:
    from api.segmentation_service import (
        BalancedFeatures,
        FEATURES as SEG_FEATURES,
        ORIGINAL_NUMERIC,
        BEHAVIORAL_NUMERIC,
        ORIGINAL_CATEGORICAL,
    )
except ImportError:
    BalancedFeatures = None
    SEG_FEATURES = [
        "credit_score", "age", "tenure", "balance", "products_number", "credit_card",
        "active_member", "estimated_salary", "country", "gender", "recency_days",
        "monthly_transactions", "trans_amount_3m", "trans_trend_ratio",
        "app_logins_monthly", "customer_service_complaints", "has_unresolved_complaint",
    ]


@dataclass
class ClusterFeatureConfig:
    """Configuration container for cluster feature extraction."""

    enabled: bool = False
    algorithm: str = "kmeans"  # "kmeans" | "hierarchical" | "both"
    n_clusters: int = 4
    use_saved_bundle: bool = True
    bundle_path: str = "artifacts/segmentation/cluster_bundle.joblib"
    
    # Sub-feature group toggles
    enable_centroid_distances: bool = True
    enable_soft_probabilities: bool = True
    enable_inertia_boundary: bool = True
    enable_risk_affinity: bool = True
    enable_relative_ratios: bool = True
    enable_cluster_one_hot: bool = True
    enable_hierarchical_features: bool = False

    # Temperature for soft membership softmax
    softmax_temperature: float = 1.0


class ClusterFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that enriches tabular customer data
    with unsupervised cluster geometry, soft memberships, boundary margins,
    and peer-relative deviation ratios.
    """

    def __init__(
        self,
        enabled: bool = False,
        algorithm: str = "kmeans",
        n_clusters: int = 4,
        use_saved_bundle: bool = True,
        bundle_path: Union[str, Path] = "artifacts/segmentation/cluster_bundle.joblib",
        enable_centroid_distances: bool = True,
        enable_soft_probabilities: bool = True,
        enable_inertia_boundary: bool = True,
        enable_risk_affinity: bool = True,
        enable_relative_ratios: bool = True,
        enable_cluster_one_hot: bool = True,
        enable_hierarchical_features: bool = False,
        softmax_temperature: float = 1.0,
        verbose: bool = False,
    ):
        self.enabled = enabled
        self.algorithm = algorithm.lower() if algorithm else "kmeans"
        self.n_clusters = n_clusters
        self.use_saved_bundle = use_saved_bundle
        self.bundle_path = str(bundle_path)
        self.enable_centroid_distances = enable_centroid_distances
        self.enable_soft_probabilities = enable_soft_probabilities
        self.enable_inertia_boundary = enable_inertia_boundary
        self.enable_risk_affinity = enable_risk_affinity
        self.enable_relative_ratios = enable_relative_ratios
        self.enable_cluster_one_hot = enable_cluster_one_hot
        self.enable_hierarchical_features = enable_hierarchical_features or (self.algorithm in ("hierarchical", "both"))
        self.softmax_temperature = softmax_temperature
        self.verbose = verbose

        # Fitted state
        self.is_fitted_ = False
        self.preprocessor_ = None
        self.kmeans_ = None
        self.cluster_centers_ = None  # Shape (K, D)
        self.hierarchical_centers_ = None  # Shape (K, D) if available
        self.cluster_churn_rates_ = {}  # {cluster_id: churn_rate}
        self.safe_cluster_id_ = 0
        self.risky_cluster_id_ = 1
        self.cluster_mean_distances_ = {}  # {cluster_id: mean_dist}
        self.cluster_peer_stats_ = {}  # {col: {cluster_id: mean_val}}
        self.feature_names_out_: List[str] = []

    def _resolve_bundle_path(self) -> Path:
        """Finds cluster bundle file across working directory and project root."""
        p = Path(self.bundle_path)
        if p.is_absolute() and p.exists():
            return p
        candidates = [
            Path.cwd() / p,
            Path(__file__).resolve().parent.parent.parent / p,
            Path(__file__).resolve().parent.parent / p,
        ]
        for cand in candidates:
            if cand.exists():
                return cand.resolve()
        return p

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ClusterFeatureExtractor":
        """
        Fits or loads clustering models and computes peer baseline statistics.
        Unsupervised for geometry; strictly anti-leakage.
        """
        if not self.enabled:
            return self

        bundle_loaded = False
        resolved_bundle = self._resolve_bundle_path()

        if self.use_saved_bundle and resolved_bundle.exists():
            try:
                import joblib
                bundle = joblib.load(resolved_bundle)
                if isinstance(bundle, dict) and "kmeans" in bundle and "preprocessor" in bundle:
                    self.kmeans_ = bundle["kmeans"]
                    self.preprocessor_ = bundle["preprocessor"]
                    self.cluster_centers_ = self.kmeans_.cluster_centers_
                    self.n_clusters = self.cluster_centers_.shape[0]
                    self.hierarchical_centers_ = bundle.get("hierarchical_centers")
                    bundle_loaded = True
                    if self.verbose:
                        print(f"[OK] ClusterFeatureExtractor loaded bundle from {resolved_bundle} (K={self.n_clusters})")
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Could not load cluster bundle ({e}), fitting dynamically.")

        if not bundle_loaded:
            # Fit dynamic preprocessor and KMeans on numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude identifiers and targets
            cols_to_fit = [c for c in numeric_cols if c not in ("customer_id", "churn", "is_churn")]
            self.preprocessor_ = StandardScaler()
            scaled_X = self.preprocessor_.fit_transform(X[cols_to_fit].fillna(0))
            self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.kmeans_.fit(scaled_X)
            self.cluster_centers_ = self.kmeans_.cluster_centers_
            self._fit_columns = cols_to_fit

        # Predict cluster memberships on fit data for peer statistics
        prep_X = self._preprocess(X)
        dists = self._compute_euclidean_dists(prep_X, self.cluster_centers_)
        cluster_ids = np.argmin(dists, axis=1)

        # 1. Cluster mean distance for inertia ratio
        for k in range(self.n_clusters):
            mask = (cluster_ids == k)
            if np.any(mask):
                self.cluster_mean_distances_[k] = float(np.mean(dists[mask, k]))
            else:
                self.cluster_mean_distances_[k] = 1.0

        # 2. Risk affinity cluster resolution
        # Known banking personas: Cluster 0 is Active Digital (14.5% churn), Cluster 1 is Dormant (25.8% churn)
        # If training target y is provided, compute empirical cluster churn rate on train set
        if y is not None:
            y_arr = np.asarray(y)
            for k in range(self.n_clusters):
                mask = (cluster_ids == k)
                if np.any(mask):
                    self.cluster_churn_rates_[k] = float(np.mean(y_arr[mask]))
                else:
                    self.cluster_churn_rates_[k] = 0.20
            # Identify safe (lowest churn) and risky (highest churn) clusters
            sorted_clusters = sorted(self.cluster_churn_rates_.items(), key=lambda kv: kv[1])
            self.safe_cluster_id_ = sorted_clusters[0][0]
            self.risky_cluster_id_ = sorted_clusters[-1][0]
        else:
            # Fallback to known cluster bundle profile
            self.cluster_churn_rates_ = {0: 0.1445, 1: 0.2582, 2: 0.2308, 3: 0.1808}
            self.safe_cluster_id_ = 0
            self.risky_cluster_id_ = 1

        # 3. Peer-relative statistics (means per cluster for key behavioral and financial columns)
        key_peer_cols = [
            "balance", "monthly_transactions", "app_logins_monthly",
            "trans_amount_3m", "trans_trend_ratio", "customer_service_complaints",
            "estimated_salary", "credit_score", "recency_days",
        ]
        self.cluster_peer_stats_ = {}
        for col in key_peer_cols:
            if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                col_vals = pd.to_numeric(X[col], errors="coerce").fillna(0).to_numpy()
                self.cluster_peer_stats_[col] = {}
                for k in range(self.n_clusters):
                    mask = (cluster_ids == k)
                    if np.any(mask):
                        self.cluster_peer_stats_[col][k] = float(np.mean(col_vals[mask]))
                    else:
                        self.cluster_peer_stats_[col][k] = float(np.mean(col_vals))

        self.is_fitted_ = True
        return self

    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms input DataFrame to representation space expected by cluster centers."""
        if hasattr(self.preprocessor_, "transform"):
            if isinstance(self.preprocessor_, BalancedFeatures) or hasattr(self.preprocessor_, "original_"):
                # BalancedFeatures expects list of SEG_FEATURES
                seg_cols = list(SEG_FEATURES)
                missing = [c for c in seg_cols if c not in X.columns]
                if missing:
                    # Provide defaults for missing columns to prevent pipeline crash
                    X_work = X.copy()
                    for m in missing:
                        X_work[m] = 0.0 if m not in ("country", "gender") else "Unknown"
                    return self.preprocessor_.transform(X_work[seg_cols])
                return self.preprocessor_.transform(X[seg_cols])
            elif hasattr(self, "_fit_columns"):
                return self.preprocessor_.transform(X[self._fit_columns].fillna(0))
            else:
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                cols = [c for c in numeric_cols if c not in ("customer_id", "churn", "is_churn")]
                return self.preprocessor_.transform(X[cols].fillna(0))
        return np.asarray(X.select_dtypes(include=[np.number]).fillna(0), dtype=float)

    @staticmethod
    def _compute_euclidean_dists(X_prep: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Vectorized Euclidean distance calculation (N x K)."""
        # diff shape: (N, K, D)
        diff = X_prep[:, np.newaxis, :] - centers[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=2)

    @staticmethod
    def _compute_cosine_dists(X_prep: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Vectorized Cosine distance calculation (N x K)."""
        norm_x = np.linalg.norm(X_prep, axis=1, keepdims=True) + 1e-9
        norm_c = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9
        cosine_sim = (X_prep / norm_x) @ (centers / norm_c).T
        return 1.0 - np.clip(cosine_sim, -1.0, 1.0)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates enriched cluster features and concatenates them with X.
        """
        if not self.enabled or not self.is_fitted_ or self.cluster_centers_ is None:
            return X

        N = len(X)
        if N == 0:
            return X

        X_prep = self._preprocess(X)
        K = self.n_clusters
        centers = self.cluster_centers_

        # 1. Distances to KMeans centers
        dists = self._compute_euclidean_dists(X_prep, centers)  # (N, K)
        feature_dict: Dict[str, np.ndarray] = {}

        # Assigned cluster index (closest centroid)
        assigned_cluster = np.argmin(dists, axis=1)  # (N,)

        # Group 1: Centroid Distances
        if self.enable_centroid_distances:
            sum_dists = np.sum(dists, axis=1, keepdims=True) + 1e-9
            for k in range(K):
                feature_dict[f"cluster_dist_{k}"] = dists[:, k]
                feature_dict[f"cluster_norm_dist_{k}"] = dists[:, k] / sum_dists.ravel()
            
            # Cosine distances
            cos_dists = self._compute_cosine_dists(X_prep, centers)
            for k in range(K):
                feature_dict[f"cluster_cos_dist_{k}"] = cos_dists[:, k]

        # Group 2: Soft Membership Probabilities & Assignment Entropy
        scaled_dists = -dists / max(float(self.softmax_temperature), 1e-3)
        max_scaled = np.max(scaled_dists, axis=1, keepdims=True)
        exp_d = np.exp(scaled_dists - max_scaled)
        probs = exp_d / (np.sum(exp_d, axis=1, keepdims=True) + 1e-9)  # (N, K)

        if self.enable_soft_probabilities:
            for k in range(K):
                feature_dict[f"cluster_prob_{k}"] = probs[:, k]
            feature_dict["cluster_prob_max"] = np.max(probs, axis=1)
            # Shannon entropy of cluster membership distribution
            entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
            feature_dict["cluster_entropy"] = entropy

        # Group 3: Atypicality, Inertia & Boundary Margin
        sorted_dists = np.sort(dists, axis=1)  # Shape (N, K)
        min_dist = sorted_dists[:, 0]
        second_min_dist = sorted_dists[:, 1] if K > 1 else min_dist
        max_dist = sorted_dists[:, -1]

        if self.enable_inertia_boundary:
            feature_dict["cluster_min_dist"] = min_dist
            feature_dict["cluster_second_min_dist"] = second_min_dist
            feature_dict["cluster_max_dist"] = max_dist
            
            # Silhouette-like boundary margin: (d_second - d_min)
            silhouette_margin = second_min_dist - min_dist
            feature_dict["cluster_silhouette_margin"] = silhouette_margin
            
            # Normalized relative margin: (d_second - d_min) / (d_second + d_min)
            feature_dict["cluster_relative_margin"] = silhouette_margin / (second_min_dist + min_dist + 1e-6)
            
            # Inertia ratio: min_dist / mean_dist_of_assigned_cluster
            cluster_means = np.array([self.cluster_mean_distances_.get(k, 1.0) for k in range(K)])
            assigned_means = cluster_means[assigned_cluster]
            feature_dict["cluster_inertia_ratio"] = min_dist / (assigned_means + 1e-6)

        # Group 4: Risk Affinity & Churn Persona Contrast
        if self.enable_risk_affinity:
            safe_k = self.safe_cluster_id_
            risky_k = self.risky_cluster_id_
            dist_to_safe = dists[:, safe_k]
            dist_to_risky = dists[:, risky_k]

            feature_dict["cluster_dist_to_safe"] = dist_to_safe
            feature_dict["cluster_dist_to_risky"] = dist_to_risky
            
            # Affinity ratio: closer to risky means higher ratio (0 = completely safe, 1 = extremely risky)
            affinity_ratio = dist_to_safe / (dist_to_safe + dist_to_risky + 1e-6)
            feature_dict["cluster_risk_affinity_ratio"] = affinity_ratio
            feature_dict["cluster_risk_dist_diff"] = dist_to_safe - dist_to_risky
            feature_dict["cluster_prob_risky"] = probs[:, risky_k]
            feature_dict["cluster_prob_safe"] = probs[:, safe_k]

        # Group 5: Peer-Relative Deviation Ratios
        if self.enable_relative_ratios and self.cluster_peer_stats_:
            for col, stats in self.cluster_peer_stats_.items():
                if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                    col_vals = pd.to_numeric(X[col], errors="coerce").fillna(0).to_numpy()
                    means_arr = np.array([stats.get(k, 1.0) for k in range(K)])
                    assigned_col_mean = means_arr[assigned_cluster]
                    
                    # Ratio relative to peer mean
                    feature_dict[f"cluster_rel_{col}"] = col_vals / (assigned_col_mean + 1e-6)
                    # Difference from peer mean
                    feature_dict[f"cluster_diff_{col}"] = col_vals - assigned_col_mean

        # Group 6: Discrete Cluster IDs & One-Hot Encoding
        if self.enable_cluster_one_hot:
            feature_dict["cluster_id"] = assigned_cluster.astype(np.int32)
            for k in range(K):
                feature_dict[f"is_cluster_{k}"] = (assigned_cluster == k).astype(np.float32)

        # Group 7: Hierarchical Clustering Linkage Features
        if self.enable_hierarchical_features and self.hierarchical_centers_ is not None:
            hier_centers = np.asarray(self.hierarchical_centers_, dtype=float)
            hier_dists = self._compute_euclidean_dists(X_prep, hier_centers)
            hier_k = hier_centers.shape[0]
            hier_assigned = np.argmin(hier_dists, axis=1)

            for hk in range(hier_k):
                feature_dict[f"hier_dist_{hk}"] = hier_dists[:, hk]
            feature_dict["hier_min_dist"] = np.min(hier_dists, axis=1)
            feature_dict["hier_cluster_id"] = hier_assigned.astype(np.int32)
            # Agreement between KMeans persona and Hierarchical persona
            feature_dict["hier_kmeans_agreement"] = (assigned_cluster == hier_assigned).astype(np.float32)

        # Assemble new DataFrame and concatenate
        cluster_df = pd.DataFrame(feature_dict, index=X.index)
        self.feature_names_out_ = list(cluster_df.columns)
        
        # Avoid column name collision
        overlap = set(X.columns).intersection(set(cluster_df.columns))
        if overlap:
            cluster_df = cluster_df.drop(columns=list(overlap))

        return pd.concat([X, cluster_df], axis=1)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Returns the list of generated cluster feature column names."""
        return list(self.feature_names_out_)
