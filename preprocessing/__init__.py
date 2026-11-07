"""
Preprocessing and Feature Engineering Package for Digital Bank Churn Prediction.
All modular components (ETL, Skewness, Scaling, Domain, Categorical, Balancing,
Feature Selection, Sample Weighting, Splitting) are orchestrated via pipeline.py.
"""

from . import pipeline
from .base import BaseFeatureTransformer
from .etl import BankDataETL, load_or_build_dataset, FullDatasetProcessor, load_or_build_full_dataset
from .skewness_transform import SkewnessTransformer
from .scaling import FeatureScaler
from .behavioral_momentum import BehavioralMomentumEngineer
from .customer_friction import CustomerFrictionEngineer
from .product_portfolio import ProductPortfolioEngineer
from .risk_interactions import RiskInteractionsEngineer
from .domain_features import DomainFeatureEngineer
from .categorical_encoding import CategoricalEncoder
from .balancing import ClassBalancer
from .feature_selector import FeatureSelector
from .sample_weighting import SampleWeightConfig, SampleWeightGenerator
from .splitting import DataSplitter, DatasetSplit, CVFold, CVResult
from .pipeline import (
    FeatureEngineeringPipeline,
    PipelineConfig,
    PipelineResult,
    load_dataset,
)

from .temporal import (
    TemporalSnapshotExtractor,
    BiGRUTemporalAutoencoder,
    TemporalAutoencoderTrainer,
    TemporalEmbeddingManager,
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
    "BehavioralMomentumEngineer",
    "CustomerFrictionEngineer",
    "ProductPortfolioEngineer",
    "RiskInteractionsEngineer",
    "DomainFeatureEngineer",
    "CategoricalEncoder",
    "ClassBalancer",
    "FeatureSelector",
    "SampleWeightConfig",
    "SampleWeightGenerator",
    "FeatureEngineeringPipeline",
    "PipelineConfig",
    "PipelineResult",
    "load_dataset",
    "DataSplitter",
    "DatasetSplit",
    "CVFold",
    "CVResult",
    "TemporalSnapshotExtractor",
    "BiGRUTemporalAutoencoder",
    "TemporalAutoencoderTrainer",
    "TemporalEmbeddingManager",
]
