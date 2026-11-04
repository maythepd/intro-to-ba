"""
Models package for Digital Bank Churn Prediction.
Exports XGBoostChurnModel, LogisticRegressionChurnModel, RandomForestChurnModel,
OptunaTuner, and Recall/Lift optimization utilities.
"""

from .logistic_regression import LogisticRegressionChurnModel
from .random_forest import RandomForestChurnModel
from .xgboost import XGBoostChurnModel
from .tuner import OptunaTuner, TuningResult
from .recall_eval import (
    run_recall_optimization,
    compute_decile_lift_table,
    optimize_thresholds,
    evaluate_with_baselines,
)

__all__ = [
    "XGBoostChurnModel",
    "LogisticRegressionChurnModel",
    "RandomForestChurnModel",
    "OptunaTuner",
    "TuningResult",
    "run_recall_optimization",
    "compute_decile_lift_table",
    "optimize_thresholds",
    "evaluate_with_baselines",
]
