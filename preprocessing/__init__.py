"""
Preprocessing and Feature Engineering Package for Digital Bank Churn Prediction.
All modular components (ETL, Skewness, Scaling, Domain, Categorical, Balancing,
Feature Selection, Splitting) are orchestrated via pipeline.py.
"""

from . import pipeline
from .base import BaseFeatureTransformer
from .etl import BankDataETL, load_or_build_dataset, FullDatasetProcessor, load_or_build_full_dataset
from .skewness_transform import SkewnessTransformer
from .scaling import FeatureScaler
from .domain_features import DomainFeatureEngineer
from .categorical_encoding import CategoricalEncoder
from .balancing import ClassBalancer
from .feature_selector import FeatureSelector
from .splitting import DataSplitter, DatasetSplit, CVFold, CVResult
from .pipeline import (
    FeatureEngineeringPipeline,
    PipelineConfig,
    PipelineResult,
    load_dataset,
)

__all__ = [
    "pipeline",
    "BaseFeatureTransformer",
    "BankDataETL",
    "load_or_build_dataset",
    "FullDatasetProcessor",
    "load_or_build_full_dataset",
    "SkewnessTransformer",
    "FeatureScaler",
    "DomainFeatureEngineer",
    "CategoricalEncoder",
    "ClassBalancer",
    "FeatureSelector",
    "FeatureEngineeringPipeline",
    "PipelineConfig",
    "PipelineResult",
    "load_dataset",
    "DataSplitter",
    "DatasetSplit",
    "CVFold",
    "CVResult",
]
