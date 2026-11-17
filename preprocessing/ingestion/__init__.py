"""
Data Ingestion, Multi-Table ETL, and Stratified Partitioning Package.
"""

from .etl import (
    BankDataETL as BankDataETL,
    FullDatasetProcessor as FullDatasetProcessor,
    load_or_build_dataset as load_or_build_dataset,
    load_or_build_full_dataset as load_or_build_full_dataset,
)
from .splitting import (
    DataSplitter as DataSplitter,
    DatasetSplit as DatasetSplit,
    CVFold as CVFold,
    CVResult as CVResult,
)

__all__ = [
    "BankDataETL",
    "FullDatasetProcessor",
    "load_or_build_dataset",
    "load_or_build_full_dataset",
    "DataSplitter",
    "DatasetSplit",
    "CVFold",
    "CVResult",
]
