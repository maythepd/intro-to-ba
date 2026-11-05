"""
Visualization package for Digital Bank Churn Prediction.
"""

from .plots import (
    plot_feature_importance,
    plot_model_comparison,
    plot_optuna_tuning_history,
    plot_lift_and_gains_curve,
)
from .separability import (
    run_all_separability_analyses,
    plot_pca_projection,
    plot_tsne_projection,
    plot_categorical_churn_bars,
    plot_continuous_boxplots,
    plot_continuous_kde_distributions,
    compute_separability_metrics,
)

__all__ = [
    "plot_feature_importance",
    "plot_model_comparison",
    "plot_optuna_tuning_history",
    "plot_lift_and_gains_curve",
    "run_all_separability_analyses",
    "plot_pca_projection",
    "plot_tsne_projection",
    "plot_categorical_churn_bars",
    "plot_continuous_boxplots",
    "plot_continuous_kde_distributions",
    "compute_separability_metrics",
]
