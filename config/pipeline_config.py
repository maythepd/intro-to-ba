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
            "max_iter": 2000,
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
    # Generic params dict for backward compatibility or direct overrides
    params: Dict[str, Any] = field(default_factory=dict)

    # Focal Loss configuration
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

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
        ext_map = {"xgboost": "json", "lr": "pkl", "rf": "pkl"}
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
    feature_selection_metric: str = "MI_Score"  # Default: Mutual Information Classif
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
                "dataset",
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
        model_cfg = ModelConfig.from_dict(data.get("model", {}))
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
                "target_col": self.features.target_col,
                "test_size": self.features.test_size,
                "random_state": self.features.random_state,
                "stratify": self.features.stratify,
                "exclude_cols": self.features.exclude_cols,
            },
            "features": asdict(self.features),
            "model": asdict(self.model),
            "experiment": asdict(self.experiment),
        }
        with open(str(path), "w", encoding="utf-8") as f:
            yaml.dump(raw_dict, f, sort_keys=False, allow_unicode=True)


def load_config(path: Optional[Union[str, Path]] = None) -> PipelineConfig:
    """Helper function to load PipelineConfig from YAML."""
    return PipelineConfig.from_yaml(path)


def load_unified_config(path: Optional[Union[str, Path]] = None) -> UnifiedConfig:
    """Helper function to load the full UnifiedConfig (features + model + experiment)."""
    return UnifiedConfig.from_yaml(path)
