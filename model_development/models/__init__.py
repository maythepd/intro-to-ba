"""
Models package for Digital Bank Churn Prediction.
Exports XGBoostChurnModel, LogisticRegressionChurnModel, RandomForestChurnModel,
OptunaTuner, and Recall/Lift optimization utilities.
"""

from .logistic_regression import LogisticRegressionChurnModel
from .random_forest import RandomForestChurnModel
from .xgboost import XGBoostChurnModel
from .focal_loss import make_xgboost_focal_loss, FocalLossLogisticRegression
# Research helpers have optional dependencies and legacy CLI imports. Loading a
# trusted inference checkpoint must not initialize Optuna, plotting or splitters.
def __getattr__(name):
    if name in {"OptunaTuner", "TuningResult"}:
        from . import tuner
        return getattr(tuner, name)
    if name in {"run_recall_optimization", "compute_decile_lift_table", "optimize_thresholds", "evaluate_with_baselines"}:
        from . import recall_eval
        return getattr(recall_eval, name)
    raise AttributeError(name)

__all__ = [
    "XGBoostChurnModel",
    "LogisticRegressionChurnModel",
    "RandomForestChurnModel",
    "make_xgboost_focal_loss",
    "FocalLossLogisticRegression",
    "OptunaTuner",
    "TuningResult",
    "run_recall_optimization",
    "compute_decile_lift_table",
    "optimize_thresholds",
    "evaluate_with_baselines",
]
