"""
Master Feature Engineering Pipeline for Digital Bank Churn Prediction.
Orchestrates skewness transformation, scaling, domain feature engineering,
categorical encoding, SMOTE class balancing, feature selection, and sample weighting.

Configuration is decoupled into the config/ module with YAML support.
"""

import copy
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Ensure config module is discoverable
try:
    from config import PipelineConfig
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import PipelineConfig

from .base import BaseFeatureTransformer
from .ingestion.etl import BankDataETL, load_or_build_dataset
from .ingestion.splitting import DataSplitter, DatasetSplit, CVFold, CVResult
from .transforms.skewness_transform import SkewnessTransformer
from .transforms.scaling import FeatureScaler
from .transforms.categorical_encoding import CategoricalEncoder
from .domain.domain_features import DomainFeatureEngineer
from .psychology.orchestrator import PsychologyFeatureEngineer
from .selection.balancing import ClassBalancer
from .selection.feature_selector import FeatureSelector
from .selection.sample_weighting import SampleWeightConfig, SampleWeightGenerator
from .temporal.manager import TemporalEmbeddingManager
from .gnn.manager import GNNEmbeddingManager
from .quant_signals.extractor import QuantSignalFeatureExtractor
from .clustering.cluster_features import ClusterFeatureExtractor
from .projections.pca_features import PCAFeatureExtractor
from .projections.lda_features import LDAFeatureExtractor


def load_dataset(
    features_cfg: Optional[PipelineConfig] = None,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Unified dataset ingestion & loading function.
    Resolves data_path, checks cache, and builds from raw multi-table files via BankDataETL if needed.
    """
    cfg = features_cfg or PipelineConfig()
    target_data_str = cfg.data_path
    rebuild = force_rebuild or getattr(cfg, "rebuild_full_dataset", False)

    target_path = Path(target_data_str)
    if not target_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        resolved = (repo_root / target_path).resolve()
        if not resolved.exists() and not rebuild:
            cwd_cand = Path.cwd() / target_path
            resolved = cwd_cand if cwd_cand.exists() else resolved
    else:
        resolved = target_path

    if rebuild or not resolved.exists() or resolved.is_dir():
        raw_dir = getattr(cfg, "raw_full_dataset_dir", "data/full_dataset")
        out_path = str(resolved) if not resolved.is_dir() else "data/full_bank_data.csv"
        return load_or_build_dataset(
            raw_data_dir=raw_dir,
            output_path=out_path,
            force_rebuild=rebuild,
            verbose=verbose,
        )

    if verbose:
        print(f"[OK] Ingesting dataset from: {resolved}")
    return pd.read_csv(resolved)


# Backward-compatible alias
load_or_build_full_dataset = load_or_build_dataset


@dataclass
class PipelineResult:
    """Output container for prepared train/test feature sets."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_resampled: pd.DataFrame
    y_train_resampled: pd.Series
    model_features: List[str]
    X_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.Series] = None
    sample_weight: Optional[np.ndarray] = None
    sample_weight_resampled: Optional[np.ndarray] = None
    feature_importance: Optional[pd.DataFrame] = None
    full_transformed_df: Optional[pd.DataFrame] = None
    train_df: Optional[pd.DataFrame] = None
    val_df: Optional[pd.DataFrame] = None
    train_val_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None
    auxiliary_temporal_metrics: Optional[Dict[str, Any]] = None
    auxiliary_nonlinear_metrics: Optional[Dict[str, Any]] = None


class FeatureEngineeringPipeline:
    """
    Final end-to-end Feature Engineering & Transformation Pipeline.
    
    Coordinates:
      1. SkewnessTransformer: Log1p, zero-balance hurdle, Yeo-Johnson
      2. FeatureScaler: RobustScaler and StandardScaler
      3. DomainFeatureEngineer: Velocity, friction, bundling, interaction terms
      4. CategoricalEncoder: One-Hot and smoothed Target Encoding
      5. ClassBalancer: SMOTE minority oversampling (train only)
      6. FeatureSelector: MI + RF importance ranking and Top-K selection
      7. SampleWeightGenerator: Exponential recency decay + financial value weighting
    """

    def __init__(
        self,
        config: Optional[Union[PipelineConfig, Dict[str, Any], str, Path]] = None,
        fold_id: Optional[Union[int, str]] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        if config is None:
            self.config = PipelineConfig.from_yaml()
        elif isinstance(config, (str, Path)):
            self.config = PipelineConfig.from_yaml(config)
        elif isinstance(config, dict):
            self.config = PipelineConfig.from_dict(config)
        elif isinstance(config, PipelineConfig):
            self.config = config
        elif hasattr(config, "features") and isinstance(config.features, PipelineConfig):
            self.config = config.features
        else:
            raise TypeError(f"Invalid config type: {type(config)}. Expected PipelineConfig, UnifiedConfig, dict, str, or Path.")

        self.fold_id = fold_id or getattr(self.config, "fold_id", None)
        self._init_modules()

    @classmethod
    def from_yaml(cls, path: Optional[Union[str, Path]] = None) -> "FeatureEngineeringPipeline":
        """Instantiate pipeline directly from a YAML config file."""
        return cls(config=PipelineConfig.from_yaml(path))

    def _init_modules(self) -> None:
        """Instantiate sub-modules based on current configuration."""
        cfg = self.config

        # 1. Skewness
        self.skewness_transformer = SkewnessTransformer(
            enabled=cfg.enable_skewness,
            transform_trans_amount_log=cfg.transform_trans_amount_log,
            transform_recency_log=cfg.transform_recency_log,
            zero_balance_hurdle=cfg.zero_balance_hurdle,
            transform_yeo_johnson=cfg.transform_yeo_johnson,
        )

        # 2. Scaling
        self.scaler = FeatureScaler(
            enabled=cfg.enable_scaling,
            apply_robust_scaling=cfg.apply_robust_scaling,
            apply_standard_scaling=cfg.apply_standard_scaling,
            robust_features=cfg.robust_features,
            standard_features=cfg.standard_features,
        )

        # 3. Domain Features (Coordinated modular transformers)
        self.domain_engineer = DomainFeatureEngineer(
            enabled=cfg.enable_domain_features,
            enable_behavioral_momentum=cfg.enable_behavioral_momentum,
            enable_customer_friction=cfg.enable_customer_friction,
            enable_product_bundling=cfg.enable_product_bundling,
            enable_risk_interactions=cfg.enable_risk_interactions,
            spending_collapse_threshold=getattr(cfg, "spending_collapse_threshold", 0.50),
            login_collapse_threshold=getattr(cfg, "login_collapse_threshold", 0.50),
            velocity_smoothing=getattr(cfg, "velocity_smoothing", 1.0),
            unresolved_weight=getattr(cfg, "unresolved_weight", 3.0),
            complaint_count_weight=getattr(cfg, "complaint_count_weight", 1.5),
            tx_failure_weight=getattr(cfg, "tx_failure_weight", 2.0),
            high_friction_complaints_thresh=getattr(cfg, "high_friction_complaints_thresh", 2),
            sweetspot_min_products=getattr(cfg, "sweetspot_min_products", 2),
            danger_zone_min_products=getattr(cfg, "danger_zone_min_products", 3),
            vulnerable_age_min=getattr(cfg, "vulnerable_age_min", 45),
            vulnerable_age_max=getattr(cfg, "vulnerable_age_max", 60),
            high_risk_country=getattr(cfg, "high_risk_country", "Germany"),
        )
        self.momentum_engineer = self.domain_engineer.momentum_engineer
        self.friction_engineer = self.domain_engineer.friction_engineer
        self.portfolio_engineer = self.domain_engineer.portfolio_engineer
        self.risk_engineer = self.domain_engineer.risk_engineer

        # 3b. Psychology Features (Behavioral Economics & Consumer Psychology)
        self.psychology_engineer = PsychologyFeatureEngineer(
            enabled=getattr(cfg, "enable_psychology_features", True),
            enable_mental_accounting=getattr(cfg, "enable_mental_accounting", True),
            enable_anxiety_monitoring=getattr(cfg, "enable_anxiety_monitoring", True),
            enable_pain_of_paying=getattr(cfg, "enable_pain_of_paying", True),
            enable_peak_end_friction=getattr(cfg, "enable_peak_end_friction", True),
            enable_scarcity_mindset=getattr(cfg, "enable_scarcity_mindset", True),
            recency_half_life_days=getattr(cfg, "psych_recency_half_life_days", 30.0),
            login_recency_scale=getattr(cfg, "psych_login_recency_scale", 15.0),
            outlier_clip_quantile=getattr(cfg, "psych_outlier_clip_quantile", 0.995),
            prune_sparse=getattr(cfg, "psych_prune_sparse", True),
            max_sparsity_threshold=getattr(cfg, "psych_max_sparsity_threshold", 0.98),
            min_variance_threshold=getattr(cfg, "psych_min_variance_threshold", 1e-6),
        )

        # 4. Categorical Encoding
        self.categorical_encoder = CategoricalEncoder(
            enabled=(cfg.enable_one_hot_encoding or cfg.enable_target_encoding),
            apply_ohe=cfg.enable_one_hot_encoding,
            apply_target_enc=cfg.enable_target_encoding,
            categorical_cols=cfg.categorical_cols,
            m_weight=cfg.target_encoding_m_weight,
            ohe_drop_first=cfg.ohe_drop_first,
        )

        # 5. Balancing
        self.balancer = ClassBalancer(
            enabled=cfg.enable_smote,
            method="smote",
            random_state=cfg.random_state,
            sampling_strategy=cfg.smote_sampling_strategy,
            k_neighbors=cfg.smote_k_neighbors,
        )

        # 6. Feature Selection (MI, Pearson, MI_Pearson_Hybrid, RF)
        self.feature_selector = FeatureSelector(
            enabled=cfg.enable_feature_selection,
            top_k=cfg.top_k_features,
            ranking_metric=getattr(cfg, "feature_selection_metric", "MI_Score"),
            hybrid_pearson_weight=getattr(cfg, "hybrid_pearson_weight", 0.50),
            random_state=cfg.random_state,
            verbose=getattr(self, "verbose", True),
        )

        # 7. Sample Weighting
        sw_cfg = getattr(cfg, "sample_weighting", None)
        if sw_cfg is None:
            self.sample_weighter = SampleWeightGenerator(
                enabled=getattr(cfg, "enable_sample_weighting", False),
                decay_mode=getattr(cfg, "weight_decay_mode", "exponential"),
                recency_col=getattr(cfg, "weight_recency_col", "recency_days"),
                decay_half_life_days=getattr(cfg, "weight_decay_half_life_days", 60.0),
                enable_value_weighting=getattr(cfg, "weight_enable_value_weighting", True),
                balance_col=getattr(cfg, "weight_balance_col", "balance"),
                value_weight_alpha=getattr(cfg, "weight_value_weight_alpha", 0.5),
                churn_weight_multiplier=getattr(cfg, "weight_churn_multiplier", 1.5),
                normalize=getattr(cfg, "weight_normalize", True),
            )
        else:
            self.sample_weighter = SampleWeightGenerator(sw_cfg)

        # 8. Temporal Multi-Snapshot Sequence Embedding Manager
        strat_kwargs = dict(getattr(cfg, "temporal_strategy_kwargs", {}) or {})
        strat_kwargs.setdefault("curvature", getattr(cfg, "temporal_curvature", 1.0))
        strat_kwargs.setdefault("lambda_poincare", getattr(cfg, "temporal_lambda_poincare", 0.25))
        strat_kwargs.setdefault("lambda_drift", getattr(cfg, "temporal_lambda_drift", 0.20))
        strat_kwargs.setdefault("alpha_fpr", getattr(cfg, "temporal_alpha_fpr", 0.10))
        strat_kwargs.setdefault("lambda_np", getattr(cfg, "temporal_lambda_np", 0.30))
        strat_kwargs.setdefault("tau", getattr(cfg, "temporal_tau", 0.50))
        strat_kwargs.setdefault("use_feature_attention", getattr(cfg, "temporal_use_feature_attention", True))
        strat_kwargs.setdefault("use_temporal_attention", getattr(cfg, "temporal_use_temporal_attention", True))
        strat_kwargs.setdefault("use_multiscale_conv", getattr(cfg, "temporal_use_multiscale_conv", True))
        strat_kwargs.setdefault("conv_kernels", getattr(cfg, "temporal_conv_kernels", [2, 3]))
        strat_kwargs.setdefault("noise_std", getattr(cfg, "temporal_noise_std", 0.02))
        strat_kwargs.setdefault("noise_decay", getattr(cfg, "temporal_noise_decay", 0.95))
        strat_kwargs.setdefault("lambda_direction", getattr(cfg, "temporal_lambda_direction", 0.10))
        strat_kwargs.setdefault("lambda_correlation", getattr(cfg, "temporal_lambda_correlation", 0.05))
        strat_kwargs.setdefault("delta", getattr(cfg, "temporal_huber_delta", 1.0))

        self.temporal_manager = TemporalEmbeddingManager(
            enabled=getattr(cfg, "enable_temporal_embeddings", False),
            raw_data_dir=getattr(cfg, "raw_full_dataset_dir", "data/full_dataset"),
            temporal_input_mode=getattr(cfg, "temporal_input_mode", "extractors"),
            raw_sequence_length=getattr(cfg, "raw_sequence_length", 50),
            latent_dim=getattr(cfg, "temporal_latent_dim", 16),
            hidden_dim=getattr(cfg, "temporal_hidden_dim", 64),
            num_layers=getattr(cfg, "temporal_num_layers", 2),
            epochs=getattr(cfg, "temporal_epochs", 25),
            batch_size=getattr(cfg, "temporal_batch_size", 256),
            lr=getattr(cfg, "temporal_learning_rate", 1e-3),
            dropout=getattr(cfg, "temporal_dropout", 0.20),
            checkpoint_dir=getattr(cfg, "temporal_checkpoint_dir", "models/saved/temporal"),
            cache_path=getattr(cfg, "temporal_cache_path", "data/processed/temporal_embeddings.csv"),
            random_state=cfg.random_state,
            verbose=self.verbose,
            fold_id=self.fold_id,
            force_retrain=getattr(cfg, "temporal_force_retrain", False),
            manifold_strategy=getattr(cfg, "temporal_manifold_strategy", "unsupervised"),
            strategy_kwargs=strat_kwargs,
            lambda_angular=getattr(cfg, "temporal_lambda_angular", 0.30),
            margin_churn=getattr(cfg, "temporal_margin_churn", 0.20),
            margin_loyal=getattr(cfg, "temporal_margin_loyal", 0.00),
            angular_scale=getattr(cfg, "temporal_angular_scale", 4.0),
        )

        self.gnn_manager = GNNEmbeddingManager(
            enabled=getattr(cfg, "enable_gnn_embeddings", False),
            k_neighbors=getattr(cfg, "gnn_k_neighbors", 5),
            latent_dim=getattr(cfg, "gnn_latent_dim", 16),
            hidden_dim=getattr(cfg, "gnn_hidden_dim", 32),
            heads=getattr(cfg, "gnn_heads", 2),
            epochs=getattr(cfg, "gnn_epochs", 20),
            lr=getattr(cfg, "gnn_learning_rate", 0.005),
            dropout=getattr(cfg, "gnn_dropout", 0.20),
            checkpoint_dir=getattr(cfg, "gnn_checkpoint_dir", "models/saved/gnn"),
            cache_path=getattr(cfg, "gnn_cache_path", "data/processed/gnn/gnn_embeddings.csv"),
            random_state=cfg.random_state,
            verbose=self.verbose,
            fold_id=self.fold_id,
            force_retrain=getattr(cfg, "gnn_force_retrain", False),
        )

        # 8. Classical Quantitative & Signal Processing Feature Extractor
        from .quant_signals import QuantSignalFeatureExtractor
        self.quant_engineer = QuantSignalFeatureExtractor(
            enabled=getattr(cfg, "enable_quant_signal_features", False),
            enable_momentum=getattr(cfg, "quant_enable_momentum", True),
            enable_macd=getattr(cfg, "quant_enable_macd", True),
            enable_volatility=getattr(cfg, "quant_enable_volatility", True),
            enable_fourier=getattr(cfg, "quant_enable_fourier", True),
            enable_exogenous=getattr(cfg, "quant_enable_exogenous", True),
            enable_lags=getattr(cfg, "quant_enable_lags", True),
            enable_rolling=getattr(cfg, "quant_enable_rolling", True),
            enable_cyclic_time=getattr(cfg, "quant_enable_cyclic_time", True),
            enable_interactions=getattr(cfg, "quant_enable_interactions", True),
            enable_multiscale_rolling=getattr(cfg, "quant_enable_multiscale_rolling", True),
            prune_sparse=getattr(cfg, "quant_prune_sparse", True),
            max_sparsity_threshold=getattr(cfg, "quant_max_sparsity_threshold", 0.98),
            min_variance_threshold=getattr(cfg, "quant_min_variance_threshold", 1e-6),
            outlier_clip_quantile=getattr(cfg, "quant_outlier_clip_quantile", 0.995),
            cache_path=getattr(cfg, "quant_cache_path", "data/processed/quant_signal_features.csv"),
            force_recompute=getattr(cfg, "quant_force_recompute", False),
            verbose=self.verbose,
        )

        # 9. Clustering & Customer Segmentation Feature Extractor
        self.cluster_extractor = ClusterFeatureExtractor(
            enabled=getattr(cfg, "enable_clustering_features", False),
            algorithm=getattr(cfg, "clustering_algorithm", "kmeans"),
            n_clusters=getattr(cfg, "cluster_count", 4),
            use_saved_bundle=getattr(cfg, "cluster_use_saved_bundle", True),
            bundle_path=getattr(cfg, "cluster_bundle_path", "artifacts/segmentation/cluster_bundle.joblib"),
            enable_centroid_distances=getattr(cfg, "cluster_enable_centroid_distances", True),
            enable_soft_probabilities=getattr(cfg, "cluster_enable_soft_probabilities", True),
            enable_inertia_boundary=getattr(cfg, "cluster_enable_inertia_boundary", True),
            enable_risk_affinity=getattr(cfg, "cluster_enable_risk_affinity", True),
            enable_relative_ratios=getattr(cfg, "cluster_enable_relative_ratios", True),
            enable_cluster_one_hot=getattr(cfg, "cluster_enable_cluster_one_hot", True),
            enable_hierarchical_features=getattr(cfg, "cluster_enable_hierarchical_features", False),
            softmax_temperature=getattr(cfg, "cluster_softmax_temperature", 1.0),
            verbose=self.verbose,
        )

        # 10. PCA Manifold Feature Extractor
        self.pca_extractor = PCAFeatureExtractor(
            enabled=getattr(cfg, "enable_pca_features", False),
            n_components=getattr(cfg, "pca_n_components", 5),
            whiten=getattr(cfg, "pca_whiten", False),
            enable_reconstruction_error=getattr(cfg, "pca_enable_reconstruction_error", True),
            enable_mahalanobis=getattr(cfg, "pca_enable_mahalanobis", True),
            verbose=self.verbose,
        )

        # 11. LDA Supervised Fisher Discriminant Feature Extractor
        self.lda_extractor = LDAFeatureExtractor(
            enabled=getattr(cfg, "enable_lda_features", False),
            solver=getattr(cfg, "lda_solver", "lsqr"),
            shrinkage=getattr(cfg, "lda_shrinkage", "auto"),
            enable_probabilities=getattr(cfg, "lda_enable_probabilities", True),
            enable_log_odds=getattr(cfg, "lda_enable_log_odds", True),
            enable_centroid_distances=getattr(cfg, "lda_enable_centroid_distances", True),
            verbose=self.verbose,
        )

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> "FeatureEngineeringPipeline":
        """
        Fit all stateful transformers sequentially on training dataframe.
        """
        target_name = target_col or self.config.target_col
        y = df[target_name] if target_name in df.columns else None

        current = df.copy()

        # Step 1: Skewness
        self.skewness_transformer.fit(current, y)
        current = self.skewness_transformer.transform(current)

        # Step 2: Domain Features (Run before scaling so domain features can be scaled)
        self.domain_engineer.fit(current, y)
        current = self.domain_engineer.transform(current)

        # Step 2b: Psychology Features (Behavioral Economics & Consumer Psychology)
        self.psychology_engineer.fit(current, y)
        current = self.psychology_engineer.transform(current)

        # Step 3: Scaling (Properly scales raw and domain features like monthly_transactions, trans_trend_ratio, app_logins_monthly)
        self.scaler.fit(current, y)
        current = self.scaler.transform(current)

        # Step 4: Categorical
        self.categorical_encoder.fit(current, y)
        current = self.categorical_encoder.transform(current)

        # Step 5: Numeric Imputer (handles genuine NaNs like avg_resolution_delay_days cleanly without leakage)
        feature_cols = self.extract_model_features(current)
        self.imputer = SimpleImputer(strategy="median")
        current[feature_cols] = self.imputer.fit_transform(current[feature_cols])

        # Step 6: Temporal Behavioral Sequence Embeddings (Self-Supervised Bi-GRU + Manifold Metric Learning)
        if getattr(self.config, "enable_temporal_embeddings", False):
            self.temporal_manager.fit(df, y=y)
            current = self.temporal_manager.transform(current)

        # Step 7: GNN Topological Network Embeddings (Self-Supervised k-NN Graph Attention Network)
        if getattr(self.config, "enable_gnn_embeddings", False):
            self.gnn_manager.fit(current, y=None)
            current = self.gnn_manager.transform(current)
            if self.verbose and hasattr(self.gnn_manager, "embeddings_df_") and self.gnn_manager.embeddings_df_ is not None:
                print(f"[OK] GNNEmbeddingManager fitted {self.gnn_manager.latent_dim} latent dimensions (k={self.gnn_manager.k_neighbors})")

        # Step 8: Classical Quantitative & Signal Processing Features (Pure Classical ML)
        if getattr(self.config, "enable_quant_signal_features", False):
            self.quant_engineer.fit(current, y=y)
            current = self.quant_engineer.transform(current)

        # Step 9: Clustering & Customer Segmentation Features (Geometry, Boundary, Risk Affinity & Peer Ratios)
        if getattr(self.config, "enable_clustering_features", False):
            self.cluster_extractor.fit(current, y=y)
            current = self.cluster_extractor.transform(current)

        # Step 10: PCA Manifold Features (Orthogonal variance axes & residual anomaly score)
        if getattr(self.config, "enable_pca_features", False):
            self.pca_extractor.fit(current, y=None)
            current = self.pca_extractor.transform(current)

        # Step 11: Supervised Fisher LDA Features (Linear discriminant axis & Bayes log-odds)
        if getattr(self.config, "enable_lda_features", False):
            self.lda_extractor.fit(current, y=y)
            current = self.lda_extractor.transform(current)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe using all fitted transformers sequentially.
        """
        out = df.copy()

        # Step 1: Skewness
        out = self.skewness_transformer.transform(out)

        # Step 2: Domain Features
        out = self.domain_engineer.transform(out)

        # Step 2b: Psychology Features
        if hasattr(self, "psychology_engineer") and self.psychology_engineer is not None:
            out = self.psychology_engineer.transform(out)

        # Step 3: Scaling
        out = self.scaler.transform(out)

        # Step 4: Categorical
        out = self.categorical_encoder.transform(out)

        # Step 5: Numeric Imputer
        if hasattr(self, "imputer"):
            feature_cols = self.extract_model_features(out)
            out[feature_cols] = self.imputer.transform(out[feature_cols])

        # Step 6: Temporal Behavioral Sequence Embeddings (Augment with dense z embeddings)
        if getattr(self.config, "enable_temporal_embeddings", False):
            out = self.temporal_manager.transform(out)

        # Step 7: GNN Topological Network Embeddings
        if getattr(self.config, "enable_gnn_embeddings", False):
            out = self.gnn_manager.transform(out)

        # Step 8: Classical Quantitative & Signal Processing Features (Pure Classical ML)
        if getattr(self.config, "enable_quant_signal_features", False):
            out = self.quant_engineer.transform(out)

        # Step 9: Clustering & Customer Segmentation Features
        if getattr(self.config, "enable_clustering_features", False):
            out = self.cluster_extractor.transform(out)

        # Step 10: PCA Manifold Features
        if getattr(self.config, "enable_pca_features", False):
            out = self.pca_extractor.transform(out)

        # Step 11: Supervised Fisher LDA Features
        if getattr(self.config, "enable_lda_features", False):
            out = self.lda_extractor.transform(out)

        return out

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Fit on df and return transformed dataframe."""
        return self.fit(df, target_col).transform(df)

    def extract_model_features(
        self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select all numeric feature columns, filtering out specified exclusion columns.
        """
        excludes = exclude_cols or self.config.exclude_cols
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in excludes]

    def prepare_train_test(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        compute_feature_importance: Optional[bool] = None,
    ) -> PipelineResult:
        """
        End-to-end execution with leak-free Train/Test partitioning:
          1. Stratified split (80/20 by default).
          2. Fit transformers exclusively on train partition.
          3. Transform train and test partitions.
          4. Extract numerical model features.
          5. Apply SMOTE balancing on train partition.
          6. Compute sample weights on train partition (exponential recency decay + value weighting).
          7. Compute feature importances & optionally select top-K features.
        """
        target = target_col or self.config.target_col
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found in dataframe.")

        # If baseline mode is requested, restrict dataframe strictly to original raw Kaggle features
        if getattr(self.config, "use_raw_kaggle_features_only", False):
            raw_cols = [
                "credit_score", "country", "gender", "age", "tenure",
                "balance", "products_number", "credit_card", "active_member",
                "estimated_salary", target
            ]
            available_raw = [c for c in raw_cols if c in df.columns]
            df = df[available_raw].copy()
            self.config.enable_domain_features = False
            self.domain_engineer.enabled = False

        # 1. Stratified Partitioning via DataSplitter
        splitter = DataSplitter(
            test_size=self.config.test_size,
            val_size=getattr(self.config, "val_size", 0.0),
            stratify=self.config.stratify,
            random_state=self.config.random_state,
            group_col=getattr(self.config, "group_col", "customer_id"),
        )
        split = splitter.train_val_test_split(df, target_col=target)

        train_df = split.train_df
        test_df = split.test_df
        val_df = split.val_df

        # 2. Fit transformers exclusively on train_df
        self.fit(train_df, target_col=target)

        # 3. Transform partitions
        train_tr = self.transform(train_df)
        test_tr = self.transform(test_df)
        val_tr = self.transform(val_df) if split.has_val else None

        # 4. Extract numeric feature matrices
        feature_cols = self.extract_model_features(train_tr)
        initial_features_count = len(feature_cols)
        X_train = train_tr[feature_cols].copy()
        y_train = train_tr[target].copy()
        X_test = test_tr[feature_cols].copy()
        y_test = test_tr[target].copy()

        if val_tr is not None:
            X_val = val_tr[feature_cols].copy()
            y_val = val_tr[target].copy()
        else:
            X_val = None
            y_val = None

        # 5. Class Imbalance Remediation (SMOTE) strictly on training set
        X_train_res, y_train_res = self.balancer.fit_resample(X_train, y_train)

        # 5a. Sample Weighting strictly computed on training partition (zero leakage)
        self.sample_weighter.fit(train_df, y_train)
        sample_weight_train = self.sample_weighter.compute_weights(train_df, y_train)
        if self.config.enable_smote and len(y_train_res) > len(y_train):
            sample_weight_res = SampleWeightGenerator.expand_for_smote(
                weights=sample_weight_train,
                y_train_original=y_train,
                y_train_resampled=y_train_res,
            )
        else:
            sample_weight_res = sample_weight_train

        # 6. Feature Selection / Importance
        should_rank = (
            self.config.enable_feature_selection
            or (compute_feature_importance if compute_feature_importance is not None else self.config.compute_feature_importance)
        )
        self.feature_selector.fit(X_train, y_train, compute_ranking=should_rank)
        df_importance = self.feature_selector.get_importance_ranking()

        if self.config.enable_feature_selection and self.config.top_k_features:
            n_before = len(feature_cols)
            X_train = self.feature_selector.transform(X_train)
            X_test = self.feature_selector.transform(X_test)
            X_train_res = self.feature_selector.transform(X_train_res)
            if X_val is not None:
                X_val = self.feature_selector.transform(X_val)
            feature_cols = list(X_train.columns)
            if self.verbose:
                print(
                    f"[INFO] Pipeline Feature Selection: Filtered {n_before} -> {len(feature_cols)} features "
                    f"(Top-{self.config.top_k_features} via {self.config.feature_selection_metric})"
                )

        if self.verbose:
            print("\n" + "=" * 65)
            print("PIPELINE DATA MATRICES SUMMARY:")
            print(f"  • Extracted Candidate Features: {initial_features_count} columns")
            print(f"  • Initial Train Partition:     ({X_train.shape[0]} samples x {initial_features_count} features)")
            if X_val is not None:
                print(f"  • Initial Val Partition:       ({X_val.shape[0]} samples x {initial_features_count} features)")
            print(f"  • Initial Test Holdout:        ({X_test.shape[0]} samples x {initial_features_count} features)")
            if self.config.enable_smote:
                print(f"  • SMOTE Balanced Train:        ({X_train_res.shape[0]} samples x {initial_features_count} features) [Ratio 1:1]")
            if self.config.enable_feature_selection and self.config.top_k_features:
                print(f"  • Feature Selection Applied:   Filtered {initial_features_count} -> {len(feature_cols)} features (Top-{self.config.top_k_features} via {self.config.feature_selection_metric})")
            print(f"  -> FINAL MATRIX PASSED TO MODEL: ({X_train_res.shape[0]} samples x {len(feature_cols)} features)")
            print("=" * 65 + "\n")

        # Full transformed df (fit_transform on whole dataset for exploratory export)
        full_transformed = self.transform(df)

        # Full training development pool (train + val, strictly excluding held-out test partition)
        train_val_pool = pd.concat([train_df, val_df], ignore_index=True) if split.has_val else train_df

        return PipelineResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_train_resampled=X_train_res,
            y_train_resampled=y_train_res,
            model_features=feature_cols,
            X_val=X_val,
            y_val=y_val,
            sample_weight=sample_weight_train,
            sample_weight_resampled=sample_weight_res,
            feature_importance=df_importance,
            full_transformed_df=full_transformed,
            train_df=train_df,
            val_df=val_df,
            train_val_df=train_val_pool,
            test_df=test_df,
            auxiliary_temporal_metrics=self.temporal_manager.get_auxiliary_metrics(),
            auxiliary_nonlinear_metrics=self.temporal_manager.get_auxiliary_metrics(),
        )

    def cross_validate(
        self,
        df: pd.DataFrame,
        model_builder: Callable[[Any, pd.Series], Any],
        target_col: Optional[str] = None,
        n_splits: Optional[int] = None,
        verbose: bool = False,
    ) -> Any:
        """
        Executes leak-free Cross-Validation using DataSplitter.
        """
        target = target_col or self.config.target_col
        if getattr(self.config, "use_raw_kaggle_features_only", False):
            raw_cols = [
                "credit_score", "country", "gender", "age", "tenure",
                "balance", "products_number", "credit_card", "active_member",
                "estimated_salary", target
            ]
            available_raw = [c for c in raw_cols if c in df.columns]
            df = df[available_raw].copy()
            self.config.enable_domain_features = False

        k = n_splits or getattr(self.config, "cv_folds", 5)
        splitter = DataSplitter(
            n_splits=k,
            stratify=self.config.stratify,
            random_state=self.config.random_state,
            group_col=getattr(self.config, "group_col", "customer_id"),
        )
        return splitter.evaluate_cv(
            df=df,
            pipeline_builder=lambda: FeatureEngineeringPipeline(copy.deepcopy(self.config)),
            model_builder=model_builder,
            target_col=target,
            n_splits=k,
            verbose=verbose,
        )

    def export_processed_data(
        self,
        result: PipelineResult,
        output_dir: str = "data/processed",
        verbose: bool = True,
    ) -> Dict[str, str]:
        """
        Export all transformed feature matrices and splits into data/processed/.
        """
        os.makedirs(output_dir, exist_ok=True)
        exported: Dict[str, str] = {}

        # 1. Full transformed dataset
        if result.full_transformed_df is not None:
            p = os.path.join(output_dir, "full_transformed.csv")
            result.full_transformed_df.to_csv(p, index=False)
            exported["full_transformed"] = p

        # 2. Complete split datasets with target
        p_train = os.path.join(output_dir, "train.csv")
        df_train = result.X_train.copy()
        df_train["churn"] = result.y_train.values
        df_train.to_csv(p_train, index=False)
        exported["train"] = p_train

        p_test = os.path.join(output_dir, "test.csv")
        df_test = result.X_test.copy()
        df_test["churn"] = result.y_test.values
        df_test.to_csv(p_test, index=False)
        exported["test"] = p_test

        if result.X_val is not None and result.y_val is not None:
            p_val = os.path.join(output_dir, "val.csv")
            df_val = result.X_val.copy()
            df_val["churn"] = result.y_val.values
            df_val.to_csv(p_val, index=False)
            exported["val"] = p_val

        # 3. SMOTE balanced sets
        p_smote = os.path.join(output_dir, "train_smote.csv")
        df_smote = result.X_train_resampled.copy()
        df_smote["churn"] = result.y_train_resampled.values
        df_smote.to_csv(p_smote, index=False)
        exported["train_smote"] = p_smote

        # 4. Feature importance ranking
        if result.feature_importance is not None:
            p_imp = os.path.join(output_dir, "feature_importance_ranking.csv")
            result.feature_importance.to_csv(p_imp, index=False)
            exported["feature_importance"] = p_imp

        # 5. Metadata JSON (list of feature names)
        import json
        p_meta = os.path.join(output_dir, "feature_metadata.json")
        with open(p_meta, "w", encoding="utf-8") as f:
            json.dump({
                "num_features": len(result.model_features),
                "model_features": result.model_features,
            }, f, indent=2)
        exported["feature_metadata"] = p_meta

        if verbose:
            print(f"[OK] Saved processed datasets to: {output_dir}")

        return exported

    # Backward-compatible alias
    export_artifacts = export_processed_data
