"""
Unified configuration system for Digital Bank Churn Prediction.
Encapsulates Feature Engineering, Dataset, Model (XGBoost), and Experiment settings.
All configurations are defined in config/config.yaml.
"""

from dataclasses import asdict, dataclass, field
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
    """Configuration for Machine Learning model."""

    name: str = "xgboost"
    params: Dict[str, Any] = field(
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class ExperimentConfig:
    """Configuration for Experiment tracking and artifacts."""

    output_dir: str = "results"
    save_model: bool = True
    model_save_path: str = "models/saved/xgboost_churn.json"
    report_filename: str = "xgboost_ablation_report.md"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class PipelineConfig:
    """Configuration class for the FeatureEngineeringPipeline."""

    # Dataset partition & exclusion defaults
    data_path: str = "data/cleaned_bank_data.csv"
    target_col: str = "churn"
    test_size: float = 0.20
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

    # Categorical encoding configuration
    categorical_cols: Optional[List[str]] = None
    target_encoding_m_weight: float = 10.0
    ohe_drop_first: bool = False

    # SMOTE balancing configuration
    smote_sampling_strategy: str = "auto"
    smote_k_neighbors: int = 5

    # Feature selection configuration
    top_k_features: Optional[int] = None
    feature_selection_metric: str = "RF_Importance"
    compute_feature_importance: bool = False

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
            # Known sub-sections
            section_keys = [
                "modules",
                "skewness_transform",
                "skewness",
                "scaling",
                "domain_features",
                "categorical_encoding",
                "balancing",
                "feature_selection",
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
class UnifiedConfig:
    """Master Unified Configuration combining features, model, and experiment."""

    features: PipelineConfig = field(default_factory=PipelineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedConfig":
        features_cfg = PipelineConfig.from_dict(data)
        model_cfg = ModelConfig.from_dict(data.get("model", {}))
        experiment_cfg = ExperimentConfig.from_dict(data.get("experiment", {}))
        return cls(features=features_cfg, model=model_cfg, experiment=experiment_cfg)

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
