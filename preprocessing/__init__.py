"""
Preprocessing and Feature Engineering Package for Digital Bank Churn Prediction.
All modular components (ETL, Skewness, Scaling, Domain, Categorical, Balancing,
Feature Selection, Sample Weighting, Splitting) are orchestrated via pipeline.py.
"""

from . import pipeline
from .base import BaseFeatureTransformer
from .ingestion import BankDataETL, load_or_build_dataset, FullDatasetProcessor, load_or_build_full_dataset, DataSplitter, DatasetSplit, CVFold, CVResult
from .transforms import SkewnessTransformer, FeatureScaler, CategoricalEncoder
from .domain import (
    BehavioralMomentumEngineer,
    CustomerFrictionEngineer,
    ProductPortfolioEngineer,
    RiskInteractionsEngineer,
    DomainFeatureEngineer,
)
from .selection import FeatureSelector, ClassBalancer, SampleWeightConfig, SampleWeightGenerator
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
from .gnn import (
    CustomerGraphBuilder,
    GraphAttentionLayer,
    SelfSupervisedGNNAutoencoder,
    GNNTrainer,
    GNNEmbeddingManager,
)
from .quant_signals import QuantSignalFeatureExtractor

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
