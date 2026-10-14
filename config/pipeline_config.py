"""
Configuration dataclass and YAML loader for Feature Engineering Pipeline.
"""

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "feature_engineering.yaml")
)


@dataclass
class PipelineConfig:
    """Configuration class for the FeatureEngineeringPipeline."""

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

    # Dataset partition & exclusion defaults
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """
        Creates a PipelineConfig from a dictionary, flattening nested sections
        (modules, skewness_transform, scaling, domain_features, categorical_encoding,
         balancing, feature_selection, dataset) if present.
        """
        flat_data: Dict[str, Any] = {}

        # Known nested sections in YAML
        section_keys = [
            "modules",
            "skewness_transform",
            "scaling",
            "domain_features",
            "categorical_encoding",
            "balancing",
            "feature_selection",
            "dataset",
        ]

        for sec in section_keys:
            if sec in data and isinstance(data[sec], dict):
                flat_data.update(data[sec])

        # Flat top-level keys take precedence or fill remaining fields
        for k, v in data.items():
            if k not in section_keys:
                flat_data[k] = v

        # Filter only valid dataclass field names
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in flat_data.items() if k in valid_keys}

        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: Optional[Union[str, Path]] = None) -> "PipelineConfig":
        """
        Load configuration from a YAML file.
        Defaults to config/feature_engineering.yaml if path is not specified.
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


def load_config(path: Optional[Union[str, Path]] = None) -> PipelineConfig:
    """Helper function to load PipelineConfig from YAML."""
    return PipelineConfig.from_yaml(path)
