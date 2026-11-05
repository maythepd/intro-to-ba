"""
Experiment runner utilities and model factory for Digital Bank Churn Prediction.
Orchestrates single runs, K-Fold cross-validation, multi-model comparisons, and ablation suites.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from config import (
    ModelConfig,
    PipelineConfig,
    UnifiedConfig,
)
from models import (
    LogisticRegressionChurnModel,
    RandomForestChurnModel,
    XGBoostChurnModel,
)
from preprocessing import pipeline

# Supported model registry
MODEL_REGISTRY = {
    "xgboost": XGBoostChurnModel,
    "lr": LogisticRegressionChurnModel,
    "rf": RandomForestChurnModel,
}

# Type alias for any supported churn model instance
AnyChurnModel = Union[XGBoostChurnModel, LogisticRegressionChurnModel, RandomForestChurnModel]


def build_model(
    model_type: str,
    model_cfg: Union[Dict[str, Any], ModelConfig],
    features_cfg: PipelineConfig,
    y_train: pd.Series,
) -> AnyChurnModel:
    """
    Factory: instantiates the correct model class from model_type string.
    Dynamically extracts hyperparameter overrides from ModelConfig or dict.
    For XGBoost, also handles scale_pos_weight adjustment when SMOTE is disabled.
    """
    model_type = model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    if isinstance(model_cfg, ModelConfig):
        params = copy.deepcopy(model_cfg.get_params_for_model(model_type))
        early_stopping = model_cfg.early_stopping_rounds
        use_focal = model_cfg.use_focal_loss
        focal_gamma = model_cfg.focal_gamma
        focal_alpha = model_cfg.focal_alpha
    else:
        # Dictionary fallback
        if model_type == "xgboost":
            params = copy.deepcopy(model_cfg.get("xgboost_params") or model_cfg.get("params") or {})
        elif model_type in ("lr", "logistic_regression"):
            params = copy.deepcopy(
                model_cfg.get("lr_params")
                or (model_cfg.get("params") if model_cfg.get("name") in ("lr", "logistic_regression") else {})
            )
        else:  # rf
            params = copy.deepcopy(
                model_cfg.get("rf_params")
                or (model_cfg.get("params") if model_cfg.get("name") in ("rf", "random_forest") else {})
            )
        early_stopping = model_cfg.get("early_stopping_rounds", 30)
        use_focal = bool(model_cfg.get("use_focal_loss", False) or model_cfg.get("focal_loss", {}).get("enabled", False))
        focal_gamma = float(model_cfg.get("focal_gamma") or model_cfg.get("focal_loss", {}).get("gamma", 2.0))
        focal_alpha = float(model_cfg.get("focal_alpha") or model_cfg.get("focal_loss", {}).get("alpha", 0.25))

    if use_focal and model_type in ("xgboost", "lr", "logistic_regression"):
        params["use_focal_loss"] = True
        params["focal_gamma"] = focal_gamma
        params["focal_alpha"] = focal_alpha

    if model_type == "xgboost":
        if not features_cfg.enable_smote and "scale_pos_weight" in params:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            params["scale_pos_weight"] = float(round(neg_count / max(1, pos_count), 2))
        return XGBoostChurnModel(params=params, early_stopping_rounds=early_stopping)

    elif model_type in ("lr", "logistic_regression"):
        return LogisticRegressionChurnModel(params=params if params else None)

    else:  # rf
        return RandomForestChurnModel(params=params if params else None)


# Backward-compatible alias
_build_model = build_model


def run_single_experiment(
    df: pd.DataFrame,
    features_cfg: PipelineConfig,
    model_cfg: Union[Dict[str, Any], ModelConfig],
    experiment_name: str,
    model_type: str = "xgboost",
    run_cv: bool = False,
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], AnyChurnModel, pd.DataFrame]:
    """
    Executes a single experiment:
      1. Prepares train/val/test sets using FeatureEngineeringPipeline and DataSplitter.
      2. Instantiates and trains the requested model (xgboost | lr | rf).
      3. Evaluates metrics on held-out test partition at specified threshold.
      4. Optionally evaluates K-Fold Cross-Validation.
      5. Extracts feature importances.
    """
    print(
        f"\n>>> Running Experiment: [{experiment_name}] | Model: {model_type.upper()}"
        + (f" | Threshold: {threshold:.2f}" if threshold != 0.5 else "")
    )
    pipe = pipeline.FeatureEngineeringPipeline(features_cfg)
    res = pipe.prepare_train_test(df)

    model = build_model(model_type, model_cfg, features_cfg, res.y_train_resampled)

    # Strictly isolate validation set; never use test set for early stopping / validation
    X_val = res.X_val
    y_val = res.y_val

    fit_kwargs: Dict[str, Any] = {
        "X_train": res.X_train_resampled,
        "y_train": res.y_train_resampled,
        "X_val": X_val,
        "y_val": y_val,
        "verbose": False,
    }
    if getattr(features_cfg, "enable_sample_weighting", False) and res.sample_weight_resampled is not None:
        fit_kwargs["sample_weight"] = res.sample_weight_resampled

    model.fit(**fit_kwargs)

    metrics = model.evaluate(res.X_test, res.y_test, threshold=threshold)
    metrics["Experiment"] = experiment_name
    metrics["Model"] = model_type.upper()
    metrics["Threshold"] = round(threshold, 2)
    metrics["Num_Features"] = len(res.model_features)
    metrics["X_train_Shape"] = f"{res.X_train_resampled.shape[0]}x{res.X_train_resampled.shape[1]}"
    if res.X_val is not None:
        metrics["X_val_Shape"] = f"{res.X_val.shape[0]}x{res.X_val.shape[1]}"
    metrics["X_test_Shape"] = f"{res.X_test.shape[0]}x{res.X_test.shape[1]}"
    metrics["SMOTE_Applied"] = features_cfg.enable_smote

    # Optional K-Fold Cross-Validation evaluation strictly on training partition (zero test leakage)
    if run_cv and getattr(features_cfg, "enable_cv", False) and not features_cfg.enable_feature_selection:
        k_folds = getattr(features_cfg, "cv_folds", 5)
        print(f"    Evaluating {k_folds}-Fold Stratified Cross-Validation on training set...")
        try:
            cv_source_df = res.train_val_df if res.train_val_df is not None else (
                res.train_df if res.train_df is not None else df
            )
            cv_res = pipe.cross_validate(
                df=cv_source_df,
                model_builder=lambda p_cfg, y_tr: build_model(model_type, model_cfg, p_cfg, y_tr),
                n_splits=k_folds,
                verbose=False,
            )
            metrics["CV_ROC_AUC"] = round(cv_res.mean_metrics.get("ROC_AUC", 0.0), 4)
            metrics["CV_ROC_AUC_Std"] = round(cv_res.std_metrics.get("ROC_AUC", 0.0), 4)
            metrics["CV_PR_AUC"] = round(cv_res.mean_metrics.get("PR_AUC", 0.0), 4)
            metrics["CV_F1"] = round(cv_res.mean_metrics.get("F1_Score", 0.0), 4)
            print(
                f"    {k_folds}-Fold CV Scores  | "
                f"ROC-AUC: {metrics['CV_ROC_AUC']} ± {metrics['CV_ROC_AUC_Std']} | "
                f"PR-AUC: {metrics['CV_PR_AUC']} | "
                f"F1: {metrics['CV_F1']}"
            )
        except Exception as e:
            print(f"    [WARN] CV execution skipped: {e}")

    # Use default importance_type per model family
    imp_type = "gain" if model_type == "xgboost" else "gini" if model_type == "rf" else "coefficient"
    df_imp = model.get_feature_importances(importance_type=imp_type)

    # Attach evaluation predictions to model instance for universal lift export
    # Test set is strictly held-out; validation predictions (or train if no val) are used for threshold tuning.
    p_te = model.predict_proba(res.X_test)
    model.last_test_y = res.y_test.values
    model.last_test_prob = p_te[:, 1] if getattr(p_te, "ndim", 1) > 1 else p_te
    if res.X_val is not None:
        p_va = model.predict_proba(res.X_val)
        model.last_val_y = res.y_val.values
        model.last_val_prob = p_va[:, 1] if getattr(p_va, "ndim", 1) > 1 else p_va
    else:
        # Fallback strictly to train predictions; never touch test set for threshold calibration
        p_tr = model.predict_proba(res.X_train)
        model.last_val_y = res.y_train.values
        model.last_val_prob = p_tr[:, 1] if getattr(p_tr, "ndim", 1) > 1 else p_tr

    print(
        f"    Test Set Scores   | "
        f"Features: {metrics['Num_Features']} | "
        f"ROC-AUC: {metrics['ROC_AUC']:.4f} | "
        f"PR-AUC: {metrics['PR_AUC']:.4f} | "
        f"F1-Score: {metrics['F1_Score']:.4f} | "
        f"Recall: {metrics['Recall']:.4f}"
    )

    return metrics, model, df_imp


def run_model_comparison(
    df: pd.DataFrame,
    base_config: UnifiedConfig,
) -> Tuple[pd.DataFrame, Dict[str, AnyChurnModel], Dict[str, pd.DataFrame]]:
    """
    Trains LR, RF, and XGBoost on the same default feature pipeline and compares metrics side-by-side.
    This is the canonical model-selection benchmark for Module 5 of the project.
    """
    model_types = [
        ("Logistic Regression", "lr"),
        ("Random Forest", "rf"),
        ("XGBoost", "xgboost"),
    ]

    results: List[Dict[str, Any]] = []
    models_dict: Dict[str, AnyChurnModel] = {}
    importances_dict: Dict[str, pd.DataFrame] = {}

    feat_cfg = copy.deepcopy(base_config.features)
    model_dict = copy.deepcopy(base_config.model.__dict__)
    run_cv = getattr(feat_cfg, "enable_cv", True)

    for name, mtype in model_types:
        metrics, trained_model, df_imp = run_single_experiment(
            df=df,
            features_cfg=feat_cfg,
            model_cfg=model_dict,
            experiment_name=name,
            model_type=mtype,
            run_cv=run_cv,
        )
        results.append(metrics)
        models_dict[name] = trained_model
        importances_dict[name] = df_imp

    df_results = pd.DataFrame(results)
    return df_results, models_dict, importances_dict


def run_ablation_suite(
    df: pd.DataFrame,
    base_config: UnifiedConfig,
    model_type: str = "xgboost",
) -> Tuple[pd.DataFrame, Dict[str, AnyChurnModel], Dict[str, pd.DataFrame]]:
    """
    Runs a comprehensive set of ablation experiments by modifying feature configs.
    By default uses XGBoost; pass model_type='lr' or 'rf' to ablate a different base model.
    """
    print(f"\n[INFO] Ablation suite using model: {model_type.upper()}") 
    experiments: List[Tuple[str, Dict[str, Any]]] = [
        (
            "0. Raw Kaggle Baseline (Original 10 Features Only)",
            {
                "use_raw_kaggle_features_only": True,
                "enable_domain_features": False,
            },
        ),
        (
            "1. Full Pipeline (Default - All Multi-Table & Domain)",
            {},
        ),
        (
            "2. No SMOTE (Native scale_pos_weight)",
            {"enable_smote": False},
        ),
        (
            "3. No Domain Features (All 4 Groups Disabled)",
            {"enable_domain_features": False},
        ),
        (
            "3a. Ablation: No Behavioral Momentum",
            {"enable_behavioral_momentum": False},
        ),
        (
            "3b. Ablation: No Customer Friction",
            {"enable_customer_friction": False},
        ),
        (
            "3c. Ablation: No Product Portfolio",
            {"enable_product_bundling": False},
        ),
        (
            "3d. Ablation: No Risk Interactions",
            {"enable_risk_interactions": False},
        ),
        (
            "4. No Target Encoding",
            {"enable_target_encoding": False},
        ),
        (
            "5. No One-Hot Encoding",
            {"enable_one_hot_encoding": False},
        ),
        (
            "6. No Scaling & No Skewness Transform",
            {"enable_scaling": False, "enable_skewness": False},
        ),
        (
            "7. Top 15 Features Only (Feature Selection - MI Classif)",
            {
                "enable_feature_selection": True,
                "top_k_features": 15,
                "feature_selection_metric": "MI_Score",
            },
        ),
        (
            "8. With Sample Weighting (Recency Decay + Financial Value)",
            {"enable_sample_weighting": True},
        ),
    ]

    if model_type in ("xgboost", "lr"):
        experiments.append((
            "9. With Focal Loss Objective (Focus on Hard Samples)",
            {"model_override": {"use_focal_loss": True, "focal_gamma": 2.0, "focal_alpha": 0.25}},
        ))

    results: List[Dict[str, Any]] = []
    models_dict: Dict[str, AnyChurnModel] = {}
    importances_dict: Dict[str, pd.DataFrame] = {}

    for name, overrides in experiments:
        # Create a deep copy of base features config and apply overrides
        feat_dict = copy.deepcopy(base_config.features.__dict__)
        model_dict = copy.deepcopy(base_config.model.__dict__)

        m_overrides = overrides.get("model_override", {})
        f_overrides = {k: v for k, v in overrides.items() if k != "model_override"}

        feat_dict.update(f_overrides)
        model_dict.update(m_overrides)
        features_cfg = PipelineConfig(**feat_dict)

        metrics, trained_model, df_imp = run_single_experiment(
            df=df,
            features_cfg=features_cfg,
            model_cfg=model_dict,
            experiment_name=name,
            model_type=model_type,
        )

        results.append(metrics)
        models_dict[name] = trained_model
        importances_dict[name] = df_imp

    df_results = pd.DataFrame(results)
    return df_results, models_dict, importances_dict

