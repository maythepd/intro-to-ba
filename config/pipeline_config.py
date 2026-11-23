"""
Unified configuration system for Digital Bank Churn Prediction.
Encapsulates Feature Engineering, Dataset, Model (XGBoost, LR, RF), and Experiment settings.
All configurations are defined in config/config.yaml.
"""

from dataclasses import asdict, dataclass, field, fields
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "config.yaml")
)
DEFAULT_UNIFIED_CONFIG_PATH = DEFAULT_CONFIG_PATH


@dataclass
class ModelConfig:
    """Configuration for Machine Learning models (XGBoost, Logistic Regression, Random Forest)."""

    name: str = "xgboost"  # "xgboost" | "lr" | "rf"
    # Unified model-specific hyperparameter dictionaries
    xgboost_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "min_child_weight": 1,
            "scale_pos_weight": 1.0,
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        }
    )
    early_stopping_rounds: Optional[int] = 30
    lr_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "C": 0.05,
            "solver": "saga",
            "penalty": "l1",
            "max_iter": 3000,
            "tol": 1e-3,
            "class_weight": "balanced",
            "random_state": 42,
        }
    )
    rf_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": 12,
            "min_samples_split": 15,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "oob_score": True,
            "random_state": 42,
            "n_jobs": -1,
        }
    )
    svm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "C": 0.1,
            "penalty": "l2",
            "loss": "squared_hinge",
            "dual": "auto",
            "tol": 1e-4,
            "max_iter": 5000,
            "class_weight": "balanced",
            "random_state": 42,
        }
    )
    # Generic params dict for backward compatibility or direct overrides
    params: Dict[str, Any] = field(default_factory=dict)

    # Focal Loss configuration
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # Decision threshold for binary classification
    decision_threshold: float = 0.50

    def get_params_for_model(self, model_type: str) -> Dict[str, Any]:
        """
        Returns the resolved hyperparameters for the requested model type:
        1. Model-specific dictionary (xgboost_params, lr_params, rf_params)
        2. Fallback to generic `params` if matching `name` or model-specific dict is empty.
        """
        mtype = model_type.lower()
        if mtype == "xgboost":
            p = dict(self.xgboost_params) if self.xgboost_params else dict(self.params)
            if self.name.lower() == "xgboost" and self.params:
                p.update(self.params)
        elif mtype in ("lr", "logistic_regression"):
            p = dict(self.lr_params) if self.lr_params else dict(self.params)
            if self.name.lower() in ("lr", "logistic_regression") and self.params:
                p.update(self.params)
        elif mtype in ("rf", "random_forest"):
            p = dict(self.rf_params) if self.rf_params else dict(self.params)
            if self.name.lower() in ("rf", "random_forest") and self.params:
                p.update(self.params)
        elif mtype in ("svm", "linear_svm"):
            p = dict(self.svm_params) if self.svm_params else dict(self.params)
            if self.name.lower() in ("svm", "linear_svm") and self.params:
                p.update(self.params)
        else:
            p = dict(self.params)

        # Attach focal loss settings
        if self.use_focal_loss:
            p["use_focal_loss"] = True
            p["focal_gamma"] = self.focal_gamma
            p["focal_alpha"] = self.focal_alpha

        return p

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        if "focal_loss" in data and isinstance(data["focal_loss"], dict):
            fl_data = data["focal_loss"]
            filtered["use_focal_loss"] = fl_data.get("enabled", False)
            filtered["focal_gamma"] = fl_data.get("gamma", 2.0)
            filtered["focal_alpha"] = fl_data.get("alpha", 0.25)
        if "decision_threshold" in data:
            filtered["decision_threshold"] = float(data["decision_threshold"])
        return cls(**filtered)


@dataclass
class ExperimentConfig:
    """Configuration for Experiment tracking, model serialization, and reporting."""

    output_dir: str = "results"
    save_model: bool = True
    model_save_path: str = "models/saved/{model}_churn.{ext}"
    report_filename: str = "{mode}_{model}_report.md"

    def get_output_dir(self, mode: str, model_type: Optional[str] = None) -> str:
        """
        Dynamically returns the dedicated subfolder inside output_dir:
          - mode == 'compare' -> {output_dir}/comparison
          - model_type == 'lr' -> {output_dir}/lr
          - model_type == 'rf' -> {output_dir}/rf
          - model_type == 'xgboost' -> {output_dir}/xgboost
        """
        base = Path(self.output_dir)
        mode_clean = mode.lower()
        if mode_clean in ("compare", "comparison", "comparision"):
            sub = "comparison"
        else:
            sub = (model_type or "common").lower()
        target = base / sub
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    def get_model_save_path(self, model_type: str) -> str:
        """
        Dynamically formats and returns the model artifact filepath for any model.
        Maps model types to appropriate extensions:
          - xgboost -> .json
          - lr / rf -> .pkl
        """
        m = model_type.lower()
        ext_map = {"xgboost": "json", "lr": "pkl", "rf": "pkl", "svm": "pkl", "linear_svm": "pkl"}
        ext = ext_map.get(m, "pkl")

        raw = self.model_save_path
        if "{model}" in raw or "{model_type}" in raw or "{ext}" in raw:
            return raw.format(model=m, model_type=m, ext=ext)

        p = Path(raw)
        stem = p.stem
        if "xgboost" in stem:
            stem = stem.replace("xgboost", m)
        elif not stem.startswith(m):
            stem = f"{m}_{stem}"
        return str(p.parent / f"{stem}.{ext}")

    def get_report_filename(self, mode: str, model_type: Optional[str] = None) -> str:
        """
        Dynamically formats and returns report filename for any mode (compare | ablation | train)
        and model type (xgboost | lr | rf).
        """
        m = (model_type or "all").lower()
        mode_clean = mode.lower()
        if mode_clean == "compare":
            return "model_comparison_report.md"

        raw = self.report_filename
        if "{mode}" in raw or "{model}" in raw:
            return raw.format(mode=mode_clean, model=m)

        if mode_clean == "ablation":
            return f"{m}_ablation_report.md"
        if mode_clean == "train":
            return f"{m}_train_report.md"
        return f"{m}_{mode_clean}_report.md"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class PipelineConfig:
    """Configuration class for the FeatureEngineeringPipeline."""

    # Dataset partition & exclusion defaults
    data_path: str = "data/full_bank_data.csv"
    raw_full_dataset_dir: str = "data/full_dataset"
    rebuild_full_dataset: bool = False
    target_col: str = "churn"
    group_col: Optional[str] = "customer_id"
    test_size: float = 0.20
    val_size: float = 0.10
    enable_cv: bool = True
    cv_folds: int = 5
    random_state: int = 42
    stratify: bool = True
    exclude_cols: List[str] = field(
        default_factory=lambda: [
            "customer_id",
            "churn",
            "country",
            "gender",
            "preferred_channel",
            "primary_device",
            "yj_trans_amount_3m",
        ]
    )

    # High-level module toggles (Ablations)
    use_raw_kaggle_features_only: bool = False  # Ablation baseline: only original 10 Kaggle features
    enable_skewness: bool = True
    enable_scaling: bool = True
    enable_domain_features: bool = True
    enable_one_hot_encoding: bool = True
    enable_target_encoding: bool = True
    enable_smote: bool = True
    enable_feature_selection: bool = False
    enable_temporal_embeddings: bool = False
    enable_gnn_embeddings: bool = False
    enable_quant_signal_features: bool = False

    # Skewness sub-toggles
    transform_trans_amount_log: bool = True
    transform_recency_log: bool = True
    zero_balance_hurdle: bool = True
    transform_yeo_johnson: bool = True

    # Scaling sub-toggles & feature lists
    apply_robust_scaling: bool = True
    apply_standard_scaling: bool = True
    robust_features: Optional[List[str]] = None
    standard_features: Optional[List[str]] = None

    # Domain features sub-toggles
    enable_behavioral_momentum: bool = True
    enable_customer_friction: bool = True
    enable_product_bundling: bool = True
    enable_risk_interactions: bool = True

    # Domain features parameters (Avoiding hardcoding)
    spending_collapse_threshold: float = 0.50
    login_collapse_threshold: float = 0.50
    velocity_smoothing: float = 1.0
    unresolved_weight: float = 3.0
    complaint_count_weight: float = 1.5
    tx_failure_weight: float = 2.0
    high_friction_complaints_thresh: int = 2
    sweetspot_min_products: int = 2
    danger_zone_min_products: int = 3
    vulnerable_age_min: int = 45
    vulnerable_age_max: int = 60
    high_risk_country: str = "Germany" 

    # Categorical encoding configuration
    categorical_cols: Optional[List[str]] = None
    target_encoding_m_weight: float = 10.0
    ohe_drop_first: bool = False

    # SMOTE balancing configuration
    smote_sampling_strategy: str = "auto"
    smote_k_neighbors: int = 5

    # Feature selection configuration
    top_k_features: Optional[int] = None
    feature_selection_metric: str = "MI_Score"  # "MI_Score" | "Pearson" | "MI_Pearson_Hybrid" | "RF_Importance"
    hybrid_pearson_weight: float = 0.50
    compute_feature_importance: bool = False

    # Sample Weighting configuration (Exponential recency decay + financial value weighting)
    enable_sample_weighting: bool = False
    weight_decay_mode: str = "exponential"
    weight_recency_col: str = "recency_days"
    weight_decay_half_life_days: float = 60.0
    weight_enable_value_weighting: bool = True
    weight_balance_col: str = "balance"
    weight_value_weight_alpha: float = 0.5
    weight_churn_multiplier: float = 1.5
    weight_normalize: bool = True

    # Temporal Behavioral Sequence Representation configuration (Bi-GRU Event Autoencoder)
    enable_temporal_embeddings: bool = False
    temporal_input_mode: str = "extractors"  # "extractors" | "raw_events"
    raw_sequence_length: int = 50
    decision_threshold: float = 0.50
    temporal_manifold_strategy: str = "unsupervised"
    temporal_lambda_angular: float = 0.30
    temporal_margin_churn: float = 0.20
    temporal_margin_loyal: float = 0.00
    temporal_angular_scale: float = 4.0
    temporal_curvature: float = 1.0
    temporal_lambda_poincare: float = 0.25
    temporal_lambda_drift: float = 0.20
    temporal_alpha_fpr: float = 0.10
    temporal_lambda_np: float = 0.30
    temporal_tau: float = 0.50
    temporal_strategy_kwargs: Dict[str, Any] = field(default_factory=dict)
    temporal_latent_dim: int = 16
    temporal_hidden_dim: int = 64
    temporal_num_layers: int = 2
    temporal_epochs: int = 25
    temporal_batch_size: int = 256
    temporal_learning_rate: float = 0.001
    temporal_dropout: float = 0.20
    temporal_checkpoint_dir: str = "models/saved/temporal"
    temporal_cache_path: str = "data/processed/temporal_embeddings.csv"
    temporal_force_retrain: bool = False

    # Deep Learning Architecture Operations (Pillars 1-4 from BAREFOOTS)
    temporal_use_feature_attention: bool = True
    temporal_use_temporal_attention: bool = True
    temporal_use_multiscale_conv: bool = True
    temporal_conv_kernels: List[int] = field(default_factory=lambda: [2, 3])
    temporal_noise_std: float = 0.02
    temporal_noise_decay: float = 0.95
    temporal_lambda_direction: float = 0.10
    temporal_lambda_correlation: float = 0.05
    temporal_huber_delta: float = 1.0

    # Graph Neural Network (GNN) Representation configuration (Self-Supervised GAT Autoencoder)
    gnn_k_neighbors: int = 5
    gnn_latent_dim: int = 16
    gnn_hidden_dim: int = 32
    gnn_heads: int = 2
    gnn_epochs: int = 20
    gnn_learning_rate: float = 0.005
    gnn_dropout: float = 0.20
    gnn_checkpoint_dir: str = "models/saved/gnn"
    gnn_cache_path: str = "data/processed/gnn/gnn_embeddings.csv"
    gnn_force_retrain: bool = False

    # Classical Quantitative & Signal Processing Features (Momentum, MACD, Volatility, Fourier, Exogenous, Lags, Rolling, Cyclic, Interactions)
    quant_enable_momentum: bool = True
    quant_enable_macd: bool = True
    quant_enable_volatility: bool = True
    quant_enable_fourier: bool = True
    quant_enable_exogenous: bool = True
    quant_enable_lags: bool = True
    quant_enable_rolling: bool = True
    quant_enable_cyclic_time: bool = True
    quant_enable_interactions: bool = True
    quant_enable_multiscale_rolling: bool = True
    quant_prune_sparse: bool = True
    quant_max_sparsity_threshold: float = 0.98
    quant_min_variance_threshold: float = 1e-6
    quant_outlier_clip_quantile: float = 0.995
    quant_cache_path: str = "data/processed/quant_signal_features.csv"
    quant_force_recompute: bool = False

    # Clustering / Customer Segmentation Features (Geometry, Soft Membership, Boundary & Peer Ratios)
    enable_clustering_features: bool = False
    clustering_algorithm: str = "kmeans"  # "kmeans" | "hierarchical" | "both"
    cluster_count: int = 4
    cluster_use_saved_bundle: bool = True
    cluster_bundle_path: str = "artifacts/segmentation/cluster_bundle.joblib"
    cluster_softmax_temperature: float = 1.0
    cluster_enable_centroid_distances: bool = True
    cluster_enable_soft_probabilities: bool = True
    cluster_enable_inertia_boundary: bool = True
    cluster_enable_risk_affinity: bool = True
    cluster_enable_relative_ratios: bool = True
    cluster_enable_cluster_one_hot: bool = True
    cluster_enable_hierarchical_features: bool = False

    # PCA Manifold Features configuration
    enable_pca_features: bool = False
    pca_n_components: int = 5
    pca_whiten: bool = False
    pca_enable_reconstruction_error: bool = True
    pca_enable_mahalanobis: bool = True

    # LDA Fisher Discriminant Features configuration
    enable_lda_features: bool = False
    lda_solver: str = "lsqr"
    lda_shrinkage: Optional[Union[str, float]] = "auto"
    lda_enable_probabilities: bool = True
    lda_enable_log_odds: bool = True
    lda_enable_centroid_distances: bool = True

    # Legacy compatibility aliases (deprecated)
    allow_nonlinear_representations: bool = False
    nonlinear_method: str = "supervised"
    nonlinear_mode: str = "augment"
    nonlinear_latent_dim: int = 16
    nonlinear_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    nonlinear_epochs: int = 30
    nonlinear_batch_size: int = 256
    nonlinear_learning_rate: float = 0.001
    nonlinear_dropout: float = 0.20
    nonlinear_noise_level: float = 0.15
    nonlinear_temperature: float = 0.10
    nonlinear_kernel_components: int = 32
    nonlinear_checkpoint_dir: str = "models/saved/nonlinear"
    nonlinear_fold_idx: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """
        Creates a PipelineConfig from dictionary, extracting from 'features' and 'dataset'
        sections if present, or flattening nested sections.
        """
        flat_data: Dict[str, Any] = {}

        # If data is a unified dict with 'dataset' and 'features'
        if "dataset" in data and isinstance(data["dataset"], dict):
            flat_data.update(data["dataset"])

        feature_dict = data.get("features", data)
        if isinstance(feature_dict, dict):
            section_keys = [
                "modules",
                "skewness_transform",
                "skewness",
                "scaling",
                "domain_features",
                "categorical_encoding",
                "balancing",
                "feature_selection",
                "sample_weighting",
                "nonlinear_representations",
                "nonlinear",
                "dataset",
                "clustering_features",
                "clustering",
                "pca_features",
                "pca",
                "lda_features",
                "lda",
            ]
            for sec in section_keys:
                if sec in feature_dict and isinstance(feature_dict[sec], dict):
                    flat_data.update(feature_dict[sec])

            for k, v in feature_dict.items():
                if k not in section_keys:
                    flat_data[k] = v

        # Top level overrides
        for k, v in data.items():
            if k not in ["dataset", "features", "model", "experiment"]:
                flat_data[k] = v

        # Handle PCA mappings
        pca_source = data.get("pca_features") or data.get("pca") or (feature_dict.get("pca_features") if isinstance(feature_dict, dict) else None) or (feature_dict.get("pca") if isinstance(feature_dict, dict) else None)
        if isinstance(pca_source, dict):
            if "enabled" in pca_source:
                flat_data["enable_pca_features"] = pca_source["enabled"]
            if "n_components" in pca_source:
                flat_data["pca_n_components"] = pca_source["n_components"]
            if "whiten" in pca_source:
                flat_data["pca_whiten"] = pca_source["whiten"]
            if "enable_reconstruction_error" in pca_source:
                flat_data["pca_enable_reconstruction_error"] = pca_source["enable_reconstruction_error"]
            if "enable_mahalanobis" in pca_source:
                flat_data["pca_enable_mahalanobis"] = pca_source["enable_mahalanobis"]

        # Handle LDA mappings
        lda_source = data.get("lda_features") or data.get("lda") or (feature_dict.get("lda_features") if isinstance(feature_dict, dict) else None) or (feature_dict.get("lda") if isinstance(feature_dict, dict) else None)
        if isinstance(lda_source, dict):
            if "enabled" in lda_source:
                flat_data["enable_lda_features"] = lda_source["enabled"]
            if "solver" in lda_source:
                flat_data["lda_solver"] = lda_source["solver"]
            if "shrinkage" in lda_source:
                flat_data["lda_shrinkage"] = lda_source["shrinkage"]
            if "enable_probabilities" in lda_source:
                flat_data["lda_enable_probabilities"] = lda_source["enable_probabilities"]
            if "enable_log_odds" in lda_source:
                flat_data["lda_enable_log_odds"] = lda_source["enable_log_odds"]
            if "enable_centroid_distances" in lda_source:
                flat_data["lda_enable_centroid_distances"] = lda_source["enable_centroid_distances"]

        # Handle clustering_features mappings
        clust_source = data.get("clustering_features") or (feature_dict.get("clustering_features") if isinstance(feature_dict, dict) else None)
        if isinstance(clust_source, dict):
            if "enabled" in clust_source:
                flat_data["enable_clustering_features"] = clust_source["enabled"]
            if "algorithm" in clust_source:
                flat_data["clustering_algorithm"] = clust_source["algorithm"]
            if "n_clusters" in clust_source:
                flat_data["cluster_count"] = clust_source["n_clusters"]
            if "use_saved_bundle" in clust_source:
                flat_data["cluster_use_saved_bundle"] = clust_source["use_saved_bundle"]
            if "bundle_path" in clust_source:
                flat_data["cluster_bundle_path"] = clust_source["bundle_path"]
            if "softmax_temperature" in clust_source:
                flat_data["cluster_softmax_temperature"] = clust_source["softmax_temperature"]
            
            sub_feats = clust_source.get("features", {})
            if isinstance(sub_feats, dict):
                if "enable_centroid_distances" in sub_feats:
                    flat_data["cluster_enable_centroid_distances"] = sub_feats["enable_centroid_distances"]
                if "enable_soft_probabilities" in sub_feats:
                    flat_data["cluster_enable_soft_probabilities"] = sub_feats["enable_soft_probabilities"]
                if "enable_cluster_inertia_boundary" in sub_feats:
                    flat_data["cluster_enable_inertia_boundary"] = sub_feats["enable_cluster_inertia_boundary"]
                if "enable_risk_affinity_ratio" in sub_feats:
                    flat_data["cluster_enable_risk_affinity"] = sub_feats["enable_risk_affinity_ratio"]
                if "enable_cluster_relative_ratios" in sub_feats:
                    flat_data["cluster_enable_relative_ratios"] = sub_feats["enable_cluster_relative_ratios"]
                if "enable_cluster_one_hot" in sub_feats:
                    flat_data["cluster_enable_cluster_one_hot"] = sub_feats["enable_cluster_one_hot"]
                if "enable_hierarchical_features" in sub_feats:
                    flat_data["cluster_enable_hierarchical_features"] = sub_feats["enable_hierarchical_features"]

        # Handle sample_weighting mappings
        sw_source = data.get("sample_weighting") or (feature_dict.get("sample_weighting") if isinstance(feature_dict, dict) else None)
        if isinstance(sw_source, dict):
            if "enabled" in sw_source:
                flat_data["enable_sample_weighting"] = sw_source["enabled"]
            if "decay_mode" in sw_source:
                flat_data["weight_decay_mode"] = sw_source["decay_mode"]
            if "recency_col" in sw_source:
                flat_data["weight_recency_col"] = sw_source["recency_col"]
            if "decay_half_life_days" in sw_source:
                flat_data["weight_decay_half_life_days"] = sw_source["decay_half_life_days"]
            if "enable_value_weighting" in sw_source:
                flat_data["weight_enable_value_weighting"] = sw_source["enable_value_weighting"]
            if "balance_col" in sw_source:
                flat_data["weight_balance_col"] = sw_source["balance_col"]
            if "value_weight_alpha" in sw_source:
                flat_data["weight_value_weight_alpha"] = sw_source["value_weight_alpha"]
            if "churn_weight_multiplier" in sw_source:
                flat_data["weight_churn_multiplier"] = sw_source["churn_weight_multiplier"]
            if "normalize" in sw_source:
                flat_data["weight_normalize"] = sw_source["normalize"]

        # Handle feature_selection mappings
        fs_source = data.get("feature_selection") or (feature_dict.get("feature_selection") if isinstance(feature_dict, dict) else None)
        if isinstance(fs_source, dict):
            if "top_k_features" in fs_source:
                flat_data["top_k_features"] = fs_source["top_k_features"]
            if "feature_selection_metric" in fs_source:
                flat_data["feature_selection_metric"] = fs_source["feature_selection_metric"]
            if "hybrid_pearson_weight" in fs_source:
                flat_data["hybrid_pearson_weight"] = fs_source["hybrid_pearson_weight"]
            if "compute_feature_importance" in fs_source:
                flat_data["compute_feature_importance"] = fs_source["compute_feature_importance"]

        # Handle temporal_embeddings mappings
        temp_source = data.get("temporal_embeddings") or (feature_dict.get("temporal_embeddings") if isinstance(feature_dict, dict) else None)
        if isinstance(temp_source, dict):
            if "enabled" in temp_source:
                flat_data["enable_temporal_embeddings"] = temp_source["enabled"]
            if "temporal_input_mode" in temp_source:
                flat_data["temporal_input_mode"] = temp_source["temporal_input_mode"]
            elif "input_mode" in temp_source:
                flat_data["temporal_input_mode"] = temp_source["input_mode"]
            elif "use_raw_event_logs" in temp_source:
                flat_data["temporal_input_mode"] = "raw_events" if temp_source["use_raw_event_logs"] else "extractors"
            if "raw_sequence_length" in temp_source:
                flat_data["raw_sequence_length"] = temp_source["raw_sequence_length"]
            elif "raw_seq_len" in temp_source:
                flat_data["raw_sequence_length"] = temp_source["raw_seq_len"]
            if "manifold_strategy" in temp_source:
                flat_data["temporal_manifold_strategy"] = temp_source["manifold_strategy"]
            if "lambda_angular" in temp_source:
                flat_data["temporal_lambda_angular"] = temp_source["lambda_angular"]
            if "margin_churn" in temp_source:
                flat_data["temporal_margin_churn"] = temp_source["margin_churn"]
            if "margin_loyal" in temp_source:
                flat_data["temporal_margin_loyal"] = temp_source["margin_loyal"]
            if "angular_scale" in temp_source:
                flat_data["temporal_angular_scale"] = temp_source["angular_scale"]
            if "curvature" in temp_source:
                flat_data["temporal_curvature"] = temp_source["curvature"]
            if "lambda_poincare" in temp_source:
                flat_data["temporal_lambda_poincare"] = temp_source["lambda_poincare"]
            if "lambda_drift" in temp_source:
                flat_data["temporal_lambda_drift"] = temp_source["lambda_drift"]
            if "alpha_fpr" in temp_source:
                flat_data["temporal_alpha_fpr"] = temp_source["alpha_fpr"]
            if "lambda_np" in temp_source:
                flat_data["temporal_lambda_np"] = temp_source["lambda_np"]
            if "tau" in temp_source:
                flat_data["temporal_tau"] = temp_source["tau"]
            if "strategy_kwargs" in temp_source and isinstance(temp_source["strategy_kwargs"], dict):
                flat_data["temporal_strategy_kwargs"] = temp_source["strategy_kwargs"]
            if "latent_dim" in temp_source:
                flat_data["temporal_latent_dim"] = temp_source["latent_dim"]
            if "hidden_dim" in temp_source:
                flat_data["temporal_hidden_dim"] = temp_source["hidden_dim"]
            if "num_layers" in temp_source:
                flat_data["temporal_num_layers"] = temp_source["num_layers"]
            if "epochs" in temp_source:
                flat_data["temporal_epochs"] = temp_source["epochs"]
            if "batch_size" in temp_source:
                flat_data["temporal_batch_size"] = temp_source["batch_size"]
            if "learning_rate" in temp_source:
                flat_data["temporal_learning_rate"] = temp_source["learning_rate"]
            if "dropout" in temp_source:
                flat_data["temporal_dropout"] = temp_source["dropout"]
            if "checkpoint_dir" in temp_source:
                flat_data["temporal_checkpoint_dir"] = temp_source["checkpoint_dir"]
            if "cache_path" in temp_source:
                flat_data["temporal_cache_path"] = temp_source["cache_path"]
            if "force_retrain" in temp_source:
                flat_data["temporal_force_retrain"] = temp_source["force_retrain"]
            if "use_feature_attention" in temp_source:
                flat_data["temporal_use_feature_attention"] = temp_source["use_feature_attention"]
            if "use_temporal_attention" in temp_source:
                flat_data["temporal_use_temporal_attention"] = temp_source["use_temporal_attention"]
            if "use_multiscale_conv" in temp_source:
                flat_data["temporal_use_multiscale_conv"] = temp_source["use_multiscale_conv"]
            if "conv_kernels" in temp_source:
                flat_data["temporal_conv_kernels"] = temp_source["conv_kernels"]
            if "noise_std" in temp_source:
                flat_data["temporal_noise_std"] = temp_source["noise_std"]
            if "noise_decay" in temp_source:
                flat_data["temporal_noise_decay"] = temp_source["noise_decay"]
            if "lambda_direction" in temp_source:
                flat_data["temporal_lambda_direction"] = temp_source["lambda_direction"]
            if "lambda_correlation" in temp_source:
                flat_data["temporal_lambda_correlation"] = temp_source["lambda_correlation"]
            if "huber_delta" in temp_source:
                flat_data["temporal_huber_delta"] = temp_source["huber_delta"]

        # Handle gnn_embeddings mappings
        gnn_source = data.get("gnn_embeddings") or (feature_dict.get("gnn_embeddings") if isinstance(feature_dict, dict) else None)
        if isinstance(gnn_source, dict):
            if "enabled" in gnn_source:
                flat_data["enable_gnn_embeddings"] = gnn_source["enabled"]
            if "k_neighbors" in gnn_source:
                flat_data["gnn_k_neighbors"] = gnn_source["k_neighbors"]
            if "latent_dim" in gnn_source:
                flat_data["gnn_latent_dim"] = gnn_source["latent_dim"]
            if "hidden_dim" in gnn_source:
                flat_data["gnn_hidden_dim"] = gnn_source["hidden_dim"]
            if "heads" in gnn_source:
                flat_data["gnn_heads"] = gnn_source["heads"]
            if "epochs" in gnn_source:
                flat_data["gnn_epochs"] = gnn_source["epochs"]
            if "learning_rate" in gnn_source:
                flat_data["gnn_learning_rate"] = gnn_source["learning_rate"]
            if "dropout" in gnn_source:
                flat_data["gnn_dropout"] = gnn_source["dropout"]
            if "checkpoint_dir" in gnn_source:
                flat_data["gnn_checkpoint_dir"] = gnn_source["checkpoint_dir"]
            if "cache_path" in gnn_source:
                flat_data["gnn_cache_path"] = gnn_source["cache_path"]
            if "force_retrain" in gnn_source:
                flat_data["gnn_force_retrain"] = gnn_source["force_retrain"]

        # Parse Classical Quantitative & Signal Processing Features
        quant_source = data.get("quant_signals") or (feature_dict.get("quant_signals") if isinstance(feature_dict, dict) else None)
        if isinstance(quant_source, dict):
            if "enabled" in quant_source:
                flat_data["enable_quant_signal_features"] = quant_source["enabled"]
            if "enable_momentum" in quant_source:
                flat_data["quant_enable_momentum"] = quant_source["enable_momentum"]
            if "enable_macd" in quant_source:
                flat_data["quant_enable_macd"] = quant_source["enable_macd"]
            if "enable_volatility" in quant_source:
                flat_data["quant_enable_volatility"] = quant_source["enable_volatility"]
            if "enable_fourier" in quant_source:
                flat_data["quant_enable_fourier"] = quant_source["enable_fourier"]
            if "enable_exogenous" in quant_source:
                flat_data["quant_enable_exogenous"] = quant_source["enable_exogenous"]
            if "enable_lags" in quant_source:
                flat_data["quant_enable_lags"] = quant_source["enable_lags"]
            if "enable_rolling" in quant_source:
                flat_data["quant_enable_rolling"] = quant_source["enable_rolling"]
            if "enable_cyclic_time" in quant_source:
                flat_data["quant_enable_cyclic_time"] = quant_source["enable_cyclic_time"]
            if "enable_interactions" in quant_source:
                flat_data["quant_enable_interactions"] = quant_source["enable_interactions"]
            if "enable_multiscale_rolling" in quant_source:
                flat_data["quant_enable_multiscale_rolling"] = quant_source["enable_multiscale_rolling"]
            if "prune_sparse_features" in quant_source:
                flat_data["quant_prune_sparse"] = quant_source["prune_sparse_features"]
            if "max_sparsity_threshold" in quant_source:
                flat_data["quant_max_sparsity_threshold"] = quant_source["max_sparsity_threshold"]
            if "min_variance_threshold" in quant_source:
                flat_data["quant_min_variance_threshold"] = quant_source["min_variance_threshold"]
            if "outlier_clip_quantile" in quant_source:
                flat_data["quant_outlier_clip_quantile"] = quant_source["outlier_clip_quantile"]
            if "cache_path" in quant_source:
                flat_data["quant_cache_path"] = quant_source["cache_path"]
            if "force_recompute" in quant_source:
                flat_data["quant_force_recompute"] = quant_source["force_recompute"]

        # Handle legacy nonlinear_representations mappings (backward compatibility)
        nl_source = data.get("nonlinear_representations") or (feature_dict.get("nonlinear_representations") if isinstance(feature_dict, dict) else None)
        if isinstance(nl_source, dict):
            if "enabled" in nl_source:
                flat_data["allow_nonlinear_representations"] = nl_source["enabled"]
            if "method" in nl_source:
                flat_data["nonlinear_method"] = nl_source["method"]
            if "mode" in nl_source:
                flat_data["nonlinear_mode"] = nl_source["mode"]
            if "latent_dim" in nl_source:
                flat_data["nonlinear_latent_dim"] = nl_source["latent_dim"]
            if "hidden_dims" in nl_source:
                flat_data["nonlinear_hidden_dims"] = nl_source["hidden_dims"]
            if "epochs" in nl_source:
                flat_data["nonlinear_epochs"] = nl_source["epochs"]
            if "batch_size" in nl_source:
                flat_data["nonlinear_batch_size"] = nl_source["batch_size"]
            if "learning_rate" in nl_source:
                flat_data["nonlinear_learning_rate"] = nl_source["learning_rate"]
            if "dropout" in nl_source:
                flat_data["nonlinear_dropout"] = nl_source["dropout"]
            if "noise_level" in nl_source:
                flat_data["nonlinear_noise_level"] = nl_source["noise_level"]
            if "temperature" in nl_source:
                flat_data["nonlinear_temperature"] = nl_source["temperature"]
            if "kernel_components" in nl_source:
                flat_data["nonlinear_kernel_components"] = nl_source["kernel_components"]
            if "checkpoint_dir" in nl_source:
                flat_data["nonlinear_checkpoint_dir"] = nl_source["checkpoint_dir"]

        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in flat_data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: Optional[Union[str, Path]] = None) -> "PipelineConfig":
        """
        Load configuration from YAML file.
        Defaults to config/config.yaml.
        """
        config_path = str(path) if path is not None else DEFAULT_CONFIG_PATH

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration YAML not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        return cls.from_dict(raw_data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Export current configuration to a YAML file."""
        data = asdict(self)
        with open(str(path), "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)


@dataclass
class TuningConfig:
    """Configuration for Optuna Hyperparameter Optimization."""

    enabled: bool = False
    n_trials: int = 30
    timeout: Optional[int] = None
    metric: str = "PR_AUC"  # "PR_AUC" | "ROC_AUC" | "F1_Score" | "Recall" | "F2_Score"
    cv_folds: int = 5
    study_name: str = "churn_optuna_tuning"
    direction: str = "maximize"
    random_state: int = 42

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TuningConfig":
        if not data:
            return cls()
        fields_set = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields_set})


@dataclass
class UnifiedConfig:
    """Master Unified Configuration combining features, model, experiment, and tuning."""

    features: PipelineConfig = field(default_factory=PipelineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedConfig":
        features_cfg = PipelineConfig.from_dict(data)
        model_data = dict(data.get("model", {}))
        dt = data.get("decision_threshold") or model_data.get("decision_threshold")
        if dt is not None:
            model_data["decision_threshold"] = float(dt)
            features_cfg.decision_threshold = float(dt)
        model_cfg = ModelConfig.from_dict(model_data)
        experiment_cfg = ExperimentConfig.from_dict(data.get("experiment", {}))
        tuning_cfg = TuningConfig.from_dict(data.get("tuning", {}))
        return cls(features=features_cfg, model=model_cfg, experiment=experiment_cfg, tuning=tuning_cfg)

    @classmethod
    def from_yaml(cls, path: Optional[Union[str, Path]] = None) -> "UnifiedConfig":
        config_path = str(path) if path is not None else DEFAULT_CONFIG_PATH
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Unified config YAML not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        return cls.from_dict(raw_data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        raw_dict = {
            "dataset": {
                "data_path": self.features.data_path,
                "raw_full_dataset_dir": self.features.raw_full_dataset_dir,
                "rebuild_full_dataset": self.features.rebuild_full_dataset,
                "target_col": self.features.target_col,
                "group_col": self.features.group_col,
                "test_size": self.features.test_size,
                "val_size": self.features.val_size,
                "enable_cv": self.features.enable_cv,
                "cv_folds": self.features.cv_folds,
                "random_state": self.features.random_state,
                "stratify": self.features.stratify,
                "exclude_cols": self.features.exclude_cols,
            },
            "features": asdict(self.features),
            "model": asdict(self.model),
            "experiment": asdict(self.experiment),
            "tuning": asdict(self.tuning),
        }
        with open(str(path), "w", encoding="utf-8") as f:
            yaml.dump(raw_dict, f, sort_keys=False, allow_unicode=True)


def load_config(path: Optional[Union[str, Path]] = None) -> PipelineConfig:
    """Helper function to load PipelineConfig from YAML."""
    return PipelineConfig.from_yaml(path)


def load_unified_config(path: Optional[Union[str, Path]] = None) -> UnifiedConfig:
    """Helper function to load the full UnifiedConfig (features + model + experiment)."""
    return UnifiedConfig.from_yaml(path)
