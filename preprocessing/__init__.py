"""
Preprocessing and Feature Engineering Package for Digital Bank Churn Prediction.
Modular architecture extracted from 04_Data_Transformation.
"""

from .base import BaseFeatureTransformer
from .skewness_transform import SkewnessTransformer
from .scaling import FeatureScaler
from .domain_features import DomainFeatureEngineer
from .categorical_encoding import CategoricalEncoder
from .balancing import ClassBalancer
from .feature_selector import FeatureSelector
from .pipeline import FeatureEngineeringPipeline, PipelineConfig, PipelineResult

__all__ = [
    "BaseFeatureTransformer",
    "SkewnessTransformer",
    "FeatureScaler",
    "DomainFeatureEngineer",
    "CategoricalEncoder",
    "ClassBalancer",
    "FeatureSelector",
    "FeatureEngineeringPipeline",
    "PipelineConfig",
    "PipelineResult",
]
