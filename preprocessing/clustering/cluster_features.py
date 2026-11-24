"""
Independent Behavioral & Psychological Cluster Feature Extractor.
Completely decoupled from Module 5 (Segmentation Service).

Enriches customer representations with unsupervised cluster geometry, soft memberships,
boundary margins, and peer-relative deviation ratios computed over a joint
Demographic + Behavioral + Psychological feature space:
  1. Centroid Distances (Euclidean & Cosine distances to each cluster prototype)
  2. Soft Membership Probabilities & Assignment Entropy (Softmax over distance)
  3. Atypicality, Inertia & Boundary Margin (Silhouette-like distance margins)
  4. Risk Affinity & Churn Archetype Contrast (Distance ratio to safe vs risky prototype)
  5. Peer-Relative Deviation Ratios (Customer metrics relative to cluster archetype means)
  6. Discrete Cluster IDs & One-Hot Encoding
  7. Hierarchical Linkage Features (Ward agglomerative medoids & agreement flag)
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Curated demographic & financial wealth attributes
CORE_DEMOGRAPHIC_FEATURES: List[str] = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "credit_card",
    "active_member",
    "estimated_salary",
]

# Curated behavioral engagement & operational velocity attributes
CORE_BEHAVIORAL_FEATURES: List[str] = [
    "recency_days",
    "monthly_transactions",
    "trans_amount_3m",
    "trans_trend_ratio",
    "app_logins_monthly",
    "customer_service_complaints",
]

# Core consumer & behavioral psychology dimensions
CORE_PSYCHOLOGY_DIMENSIONS: List[str] = [
    "psych_savings_commitment_ratio",       # Mental accounting: commitment in savings
    "psych_early_break_propensity",         # Mental accounting: early penalty breaking
    "psych_renewal_inertia_status",         # Status quo bias: auto vs manual broken inertia
    "psych_phantom_login_ratio",            # Anxiety: surveillance without financial action
    "psych_anxious_checking_index",         # Anxiety: high logins with falling spend
    "psych_fee_retaliation_score",          # Pain of paying: spending curtailment post fee
    "psych_cash_detachment_ratio",          # Decoupling: physical cash extraction
    "psych_exponential_recency_complaint",  # Peak-End: time-decayed dispute severity
    "psych_terminal_failure_shock",         # Catastrophe: unresolved complaint + fail shock
    "psych_financial_runway_months",        # Scarcity: survival buffer in months
    "psych_hand_to_mouth_burn_rate",        # Scarcity: outflow / inflow burn speed
]


@dataclass
class ClusterFeatureConfig:
    """Configuration container for independent cluster feature extraction."""

    enabled: bool = False
    algorithm: str = "kmeans"  # "kmeans" | "hierarchical" | "both"
    n_clusters: int = 4
    use_saved_bundle: bool = False
    bundle_path: Optional[str] = None
    
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
    Independent Scikit-learn transformer that clusters customers over a joint
    Demographic + Behavioral + Psychological representation space.
    Completely decoupled from external segmentation services.
    """

    def __init__(
        self,
        enabled: bool = False,
        algorithm: str = "kmeans",
        n_clusters: int = 4,
        use_saved_bundle: bool = False,
        bundle_path: Optional[Union[str, Path]] = None,
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
        self.bundle_path = str(bundle_path) if bundle_path else None
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
        self.selected_features_: List[str] = []
        self.preprocessor_ = StandardScaler()
        self.kmeans_ = None
        self.cluster_centers_ = None             # Shape (K, D)
        self.hierarchical_centers_ = None         # Shape (K, D)
        self.cluster_churn_rates_: Dict[int, float] = {}
        self.safe_cluster_id_: int = 0
        self.risky_cluster_id_: int = 1
        self.cluster_mean_distances_: Dict[int, float] = {}
        self.cluster_peer_stats_: Dict[str, Dict[int, float]] = {}
        self.feature_names_out_: List[str] = []

    def _select_clustering_features(self, X: pd.DataFrame) -> List[str]:
        """
        Dynamically selects the richest possible feature subspace:
        prioritizes Demographic + Behavioral + newly generated Psychology features.
        """
        target_features: List[str] = []

        # 1. Add demographic features present
        for f in CORE_DEMOGRAPHIC_FEATURES:
            if f in X.columns and pd.api.types.is_numeric_dtype(X[f]):
                target_features.append(f)

        # 2. Add behavioral features present
        for f in CORE_BEHAVIORAL_FEATURES:
            if f in X.columns and pd.api.types.is_numeric_dtype(X[f]):
                target_features.append(f)

        # 3. Add psychology dimensions present
        for f in CORE_PSYCHOLOGY_DIMENSIONS:
            if f in X.columns and pd.api.types.is_numeric_dtype(X[f]):
                target_features.append(f)

        # Also capture any other psych_* features present in X
        for col in X.columns:
            if col.startswith("psych_") and col not in target_features:
                if pd.api.types.is_numeric_dtype(X[col]):
                    target_features.append(col)

        # If none of the curated features matched, fallback to all numeric features excluding ID/target
        if not target_features:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            target_features = [c for c in numeric_cols if c not in ("customer_id", "churn", "is_churn")]

        return target_features

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ClusterFeatureExtractor":
        """
        Fits independent KMeans and Ward Agglomerative models on training partition.
        Learns cluster centers, risk affinity archetypes, and peer baseline statistics.
        """
        if not self.enabled:
            return self

        # 1. Feature Subspace Selection (Demographic + Behavioral + Psychology)
        self.selected_features_ = self._select_clustering_features(X)
        if self.verbose:
            psych_count = sum(1 for f in self.selected_features_ if f.startswith("psych_"))
            print(f"[OK] Independent ClusterFeatureExtractor fitting on {len(self.selected_features_)} features ({psych_count} psychology dimensions)")

        # 2. Leakage-free Standard Scaling
        X_sub = X[self.selected_features_].fillna(0.0).to_numpy(dtype=float)
        scaled_X = self.preprocessor_.fit_transform(X_sub)

        # 3. Fit Independent KMeans Archetypes
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans_.fit(scaled_X)
        self.cluster_centers_ = self.kmeans_.cluster_centers_

        # Predict cluster memberships on fit data for peer & boundary statistics
        dists = self._compute_euclidean_dists(scaled_X, self.cluster_centers_)
        cluster_ids = np.argmin(dists, axis=1)

        # 4. Fit Independent Hierarchical Linkage (Ward Medoids) if enabled
        if self.enable_hierarchical_features:
            # Use up to 3000 points to keep hierarchical fitting responsive
            n_samples = len(scaled_X)
            if n_samples > 3000:
                np.random.seed(42)
                sample_idx = np.random.choice(n_samples, size=3000, replace=False)
                hier_sub = scaled_X[sample_idx]
            else:
                hier_sub = scaled_X

            agg = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="ward")
            agg_labels = agg.fit_predict(hier_sub)

            # Compute Ward cluster medoid centers as centroids of hierarchical partitions
            hier_centers = []
            for k in range(self.n_clusters):
                mask_k = (agg_labels == k)
                if np.any(mask_k):
                    hier_centers.append(np.mean(hier_sub[mask_k], axis=0))
                else:
                    hier_centers.append(self.cluster_centers_[k])
            self.hierarchical_centers_ = np.array(hier_centers)

        # 5. Cluster Mean Distance (Inertia baseline)
        for k in range(self.n_clusters):
            mask = (cluster_ids == k)
            if np.any(mask):
                self.cluster_mean_distances_[k] = float(np.mean(dists[mask, k]))
            else:
                self.cluster_mean_distances_[k] = 1.0

        # 6. Risk Affinity Resolution (Identifies Safe vs Risky Archetypes)
        if y is not None:
            y_arr = np.asarray(y)
            for k in range(self.n_clusters):
                mask = (cluster_ids == k)
                if np.any(mask):
                    self.cluster_churn_rates_[k] = float(np.mean(y_arr[mask]))
                else:
                    self.cluster_churn_rates_[k] = 0.20
            sorted_clusters = sorted(self.cluster_churn_rates_.items(), key=lambda kv: kv[1])
            self.safe_cluster_id_ = sorted_clusters[0][0]
            self.risky_cluster_id_ = sorted_clusters[-1][0]
        else:
            self.cluster_churn_rates_ = {k: 0.20 for k in range(self.n_clusters)}
            self.safe_cluster_id_ = 0
            self.risky_cluster_id_ = max(1, self.n_clusters - 1)

        # 7. Peer-Relative Statistics (means per cluster for key financial & psychological dimensions)
        key_peer_cols = [
            "balance", "monthly_transactions", "app_logins_monthly",
            "trans_amount_3m", "trans_trend_ratio", "customer_service_complaints",
            "estimated_salary", "credit_score", "recency_days",
            "psych_financial_runway_months", "psych_phantom_login_ratio",
            "psych_savings_commitment_ratio", "psych_fee_retaliation_score",
        ]
        self.cluster_peer_stats_ = {}
        for col in key_peer_cols:
            if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                col_vals = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy()
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
        """Transforms input DataFrame using strictly fitted StandardScaler."""
        if not self.selected_features_:
            return np.zeros((len(X), 1), dtype=float)

        # Fill missing columns defensively if unseen
        cols = self.selected_features_
        missing = [c for c in cols if c not in X.columns]
        if missing:
            X_work = X.copy()
            for m in missing:
                X_work[m] = 0.0
            X_sub = X_work[cols].fillna(0.0).to_numpy(dtype=float)
        else:
            X_sub = X[cols].fillna(0.0).to_numpy(dtype=float)

        return self.preprocessor_.transform(X_sub)

    @staticmethod
    def _compute_euclidean_dists(X_prep: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Vectorized Euclidean distance calculation (N x K)."""
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

        # Group 4: Risk Affinity & Churn Archetype Contrast
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
                    col_vals = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy()
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
            # Agreement between KMeans prototype and Hierarchical prototype
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
