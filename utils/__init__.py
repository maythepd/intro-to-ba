"""
Utility package for Digital Bank Churn Prediction.
Modularized components for path resolution, model experimentation, and executive reporting.
"""

from .common import resolve_data_path
from .experiments import (
    MODEL_REGISTRY,
    AnyChurnModel,
    build_model,
    _build_model,
    run_single_experiment,
    run_model_comparison,
    run_ablation_suite,
)
from .reporting import (
    generate_report,
    generate_ablation_report,
    generate_comparison_report,
    generate_train_report,
    generate_markdown_report,
)

__all__ = [
    "resolve_data_path",
    "MODEL_REGISTRY",
    "AnyChurnModel",
    "build_model",
    "_build_model",
    "run_single_experiment",
    "run_model_comparison",
    "run_ablation_suite",
    "generate_report",
    "generate_ablation_report",
    "generate_comparison_report",
    "generate_train_report",
    "generate_markdown_report",
]

