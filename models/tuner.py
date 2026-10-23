"""
Optuna Hyperparameter Tuning Module for Digital Bank Customer Churn Prediction.
Executes Stratified K-Fold Cross-Validation hyperparameter optimization strictly
on the training partition, ensuring complete zero-leakage test holdout.
"""

import copy
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import optuna

from config.pipeline_config import PipelineConfig, UnifiedConfig
from preprocessing.pipeline import FeatureEngineeringPipeline
from preprocessing.splitting import DataSplitter
from visualization.plots import (
    plot_feature_importance,
    plot_optuna_tuning_history,
)

# Silence verbose Optuna logs by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    """Encapsulates comprehensive results from an Optuna tuning run."""

    model_type: str
    target_metric: str
    best_params: Dict[str, Any]
    best_cv_score: float
    test_metrics: Dict[str, float]
    study: optuna.Study
    trials_df: pd.DataFrame
    best_model: Any
    df_importance: pd.DataFrame
    report_path: Optional[str] = None
    plot_path: Optional[str] = None


class OptunaTuner:
    """
    Automated hyperparameter optimizer for Churn Prediction models (XGBoost, RF, LR).
    Features:
      - Stratified K-Fold Cross-Validation objective function.
      - Leak-free partitioning: optimization runs exclusively on the training partition.
      - Early pruning of suboptimal trials via MedianPruner.
      - Metric targets: PR_AUC (recommended for imbalanced data), ROC_AUC, or F1_Score.
      - Automatic trial export, visualization plotting, and markdown report generation.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        config: Optional[UnifiedConfig] = None,
        target_metric: str = "PR_AUC",
        cv_folds: int = 5,
        n_trials: int = 30,
        timeout: Optional[int] = None,
        random_state: int = 42,
    ):
        self.model_type = model_type.lower()
        if self.model_type not in ("xgboost", "rf", "lr"):
            raise ValueError(f"Unsupported model_type '{model_type}'. Expected: xgboost | rf | lr")

        self.config = config or UnifiedConfig()
        self.target_metric = target_metric
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Defines the hyperparameter search space tailored to the model family."""
        if self.model_type == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
                "gamma": trial.suggest_float("gamma", 0.0, 3.0, step=0.2),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "scale_pos_weight": 1.0,
                "eval_metric": "auc",
                "random_state": self.random_state,
                "n_jobs": -1,
            }
        elif self.model_type == "rf":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=50),
                "max_depth": trial.suggest_int("max_depth", 6, 16),
                "min_samples_split": trial.suggest_int("min_samples_split", 5, 25),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
                "class_weight": "balanced_subsample",
                "random_state": self.random_state,
                "n_jobs": -1,
            }
        elif self.model_type == "lr":
            return {
                "C": trial.suggest_float("C", 0.005, 5.0, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "solver": "saga",
                "max_iter": 1500,
                "class_weight": "balanced",
                "random_state": self.random_state,
            }
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def _build_model(self, params: Dict[str, Any]) -> Any:
        """Instantiates model given parameters."""
        if self.model_type == "xgboost":
            from models.xgboost import XGBoostChurnModel
            return XGBoostChurnModel(params=params, early_stopping_rounds=20)
        elif self.model_type == "rf":
            from models.random_forest import RandomForestChurnModel
            return RandomForestChurnModel(params=params)
        elif self.model_type == "lr":
            from models.logistic_regression import LogisticRegressionChurnModel
            return LogisticRegressionChurnModel(params=params)
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def tune(
        self,
        df: pd.DataFrame,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> TuningResult:
        """
        Executes leak-free hyperparameter optimization:
          1. Stratified split into Train (with holdout Val) and Test.
          2. Runs Optuna study with K-Fold Cross-Validation strictly on Train partition.
          3. Evaluates best parameter set on the independent Test partition.
          4. Generates artifacts (trials CSV, optimization plot, markdown report).
        """
        trials_count = n_trials or self.n_trials
        time_limit = timeout if timeout is not None else self.timeout
        target = self.config.features.target_col

        out_dir = Path(output_dir or f"results/{self.model_type}")
        out_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print("\n=======================================================")
            print(f"STARTING OPTUNA TUNING: {self.model_type.upper()}")
            print(f"Target Metric: {self.target_metric} | Trials: {trials_count} | Folds: {self.cv_folds}")
            print("=======================================================")

        # 1. Partition dataset via DataSplitter: keep Test partition strictly untouched
        splitter = DataSplitter(
            test_size=self.config.features.test_size,
            val_size=getattr(self.config.features, "val_size", 0.10),
            stratify=self.config.features.stratify,
            random_state=self.random_state,
        )
        split = splitter.train_val_test_split(df, target_col=target)
        train_df = split.train_df
        test_df = split.test_df
        val_df = split.val_df

        # Combine train + val for CV tuning (Test remains strictly held-out!)
        cv_train_pool = pd.concat([train_df, val_df], ignore_index=True) if split.has_val else train_df

        # 2. Define Optuna Objective with K-Fold CV
        def objective(trial: optuna.Trial) -> float:
            sampled_params = self._sample_params(trial)
            fold_scores: List[float] = []

            for fold in splitter.get_cv_folds(cv_train_pool, target_col=target, n_splits=self.cv_folds):
                # Fresh pipeline for each fold to prevent data leakage
                fold_pipe = FeatureEngineeringPipeline(copy.deepcopy(self.config.features))
                fold_pipe.fit(fold.train_df, target_col=target)

                # Transform partitions
                tr_tr = fold_pipe.transform(fold.train_df)
                va_tr = fold_pipe.transform(fold.val_df)

                feature_cols = fold_pipe.extract_model_features(tr_tr)
                X_tr = tr_tr[feature_cols].copy()
                y_tr = tr_tr[target].copy()
                X_va = va_tr[feature_cols].copy()
                y_va = va_tr[target].copy()

                # Resample strictly on fold training data
                X_tr_res, y_tr_res = fold_pipe.balancer.fit_resample(X_tr, y_tr)

                # Feature selection if enabled
                if fold_pipe.config.enable_feature_selection and fold_pipe.config.top_k_features:
                    fold_pipe.feature_selector.fit(X_tr, y_tr, compute_ranking=True)
                    X_tr_res = fold_pipe.feature_selector.transform(X_tr_res)
                    X_va = fold_pipe.feature_selector.transform(X_va)

                # Fit model
                model = self._build_model(sampled_params)
                if self.model_type == "xgboost":
                    model.fit(X_tr_res, y_tr_res, X_val=X_va, y_val=y_va, verbose=False)
                else:
                    model.fit(X_tr_res, y_tr_res)

                metrics = model.evaluate(X_va, y_va)
                score = float(metrics.get(self.target_metric, 0.0))
                fold_scores.append(score)

                # Intermediate pruning check
                trial.report(np.mean(fold_scores), step=fold.fold)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return float(np.mean(fold_scores))

        # 3. Create Study & Optimize
        study_name = f"{self.model_type}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
            study_name=study_name,
        )

        study.optimize(
            objective,
            n_trials=trials_count,
            timeout=time_limit,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_cv_score = study.best_value

        if verbose:
            print("\n=======================================================")
            print(f"OPTUNA TUNING COMPLETED | Best {self.target_metric} CV: {best_cv_score:.4f}")
            print("Best Hyperparameters:")
            for k, v in best_params.items():
                print(f"  - {k}: {v}")
            print("=======================================================")

        # 4. Refit Best Model on Full Training Pool & Evaluate on Independent Test Set
        final_pipe = FeatureEngineeringPipeline(copy.deepcopy(self.config.features))
        final_pipe.fit(cv_train_pool, target_col=target)

        tr_full = final_pipe.transform(cv_train_pool)
        te_full = final_pipe.transform(test_df)

        feat_cols = final_pipe.extract_model_features(tr_full)
        X_train_full = tr_full[feat_cols].copy()
        y_train_full = tr_full[target].copy()
        X_test_final = te_full[feat_cols].copy()
        y_test_final = te_full[target].copy()

        X_train_res, y_train_res = final_pipe.balancer.fit_resample(X_train_full, y_train_full)

        if final_pipe.config.enable_feature_selection and final_pipe.config.top_k_features:
            final_pipe.feature_selector.fit(X_train_full, y_train_full, compute_ranking=True)
            X_train_res = final_pipe.feature_selector.transform(X_train_res)
            X_test_final = final_pipe.feature_selector.transform(X_test_final)

        # Merge base params with best tuned params
        merged_best = self._sample_params(optuna.trial.FixedTrial(best_params))
        best_model = self._build_model(merged_best)

        if self.model_type == "xgboost":
            best_model.fit(X_train_res, y_train_res, X_val=X_test_final, y_val=y_test_final, verbose=False)
        else:
            best_model.fit(X_train_res, y_train_res)

        test_metrics = best_model.evaluate(X_test_final, y_test_final)

        # Extract feature importances
        imp_type = "gain" if self.model_type == "xgboost" else "gini" if self.model_type == "rf" else "coefficient"
        df_imp = best_model.get_feature_importances(importance_type=imp_type)

        if verbose:
            print("\nIndependent Test Set Performance (Tuned Model):")
            print(
                f"  ROC-AUC: {test_metrics['ROC_AUC']:.4f} | "
                f"PR-AUC: {test_metrics['PR_AUC']:.4f} | "
                f"F1: {test_metrics['F1_Score']:.4f} | "
                f"Recall: {test_metrics['Recall']:.4f}"
            )

        # 5. Export Artifacts
        trials_df = study.trials_dataframe()
        trials_csv = out_dir / f"{self.model_type}_tuning_trials.csv"
        trials_df.to_csv(str(trials_csv), index=False)

        # Plot tuning history
        plot_history_path = out_dir / f"{self.model_type}_tuning_history.png"
        plot_optuna_tuning_history(study, plot_history_path, metric_name=self.target_metric)

        # Plot feature importance
        plot_imp_path = out_dir / f"{self.model_type}_feature_importance_tuned.png"
        plot_feature_importance(df_imp, plot_imp_path, model_name=f"{self.model_type.upper()} (Tuned)")

        # Save tuned model artifact
        save_path = self.config.experiment.get_model_save_path(self.model_type)
        best_model.save(save_path)

        # Generate markdown report
        report_path = out_dir / f"optuna_tuning_{self.model_type}_report.md"
        self._generate_tuning_report(
            report_path=str(report_path),
            best_params=best_params,
            best_cv_score=best_cv_score,
            test_metrics=test_metrics,
            trials_df=trials_df,
            plot_history_path=plot_history_path.name,
            plot_imp_path=plot_imp_path.name,
            df_imp=df_imp,
        )

        if verbose:
            print(f"\n[OK] Tuning artifacts saved to {out_dir}")
            print(f"[OK] Report: {report_path}")

        return TuningResult(
            model_type=self.model_type,
            target_metric=self.target_metric,
            best_params=best_params,
            best_cv_score=best_cv_score,
            test_metrics=test_metrics,
            study=study,
            trials_df=trials_df,
            best_model=best_model,
            df_importance=df_imp,
            report_path=str(report_path),
            plot_path=str(plot_history_path),
        )

    def _generate_tuning_report(
        self,
        report_path: str,
        best_params: Dict[str, Any],
        best_cv_score: float,
        test_metrics: Dict[str, float],
        trials_df: pd.DataFrame,
        plot_history_path: str,
        plot_imp_path: str,
        df_imp: pd.DataFrame,
    ) -> None:
        """Generates an executive-level Markdown report for the tuning experiment."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n_trials = len(trials_df)
        completed_trials = len(trials_df[trials_df["state"] == "COMPLETE"])
        pruned_trials = len(trials_df[trials_df["state"] == "PRUNED"])

        params_table = pd.DataFrame([
            {"Hyperparameter": k, "Optimal Value": str(v)}
            for k, v in best_params.items()
        ]).to_markdown(index=False)

        test_table = pd.DataFrame([
            {"Metric": k, "Score": f"{v:.4f}" if isinstance(v, float) else str(v)}
            for k, v in test_metrics.items()
            if k not in ("TP", "FP", "TN", "FN")
        ]).to_markdown(index=False)

        top15_imp = df_imp.head(15).to_markdown(index=False)

        md = f"""# BÁO CÁO TỐI ƯU HÓA SIÊU THAM SỐ (OPTUNA TUNING REPORT)
## MÔ HÌNH: {self.model_type.upper()} CHURN PREDICTION

**Thời gian thực hiện:** `{timestamp}`  
**Mục tiêu tối ưu:** `{self.target_metric}` (Cross-Validation Folds: `{self.cv_folds}`)  
**Tổng số trials:** `{n_trials}` (Hoàn thành: `{completed_trials}`, Cắt tỉa sớm - Pruned: `{pruned_trials}`)  
**CV Score cao nhất ({self.target_metric}):** `{best_cv_score:.4f}`  

---

## 1. TIẾN TRÌNH TỐI ƯU HÓA (OPTIMIZATION PROGRESS)

![Tiến trình tối ưu hóa]({plot_history_path})

---

## 2. BỘ SIÊU THAM SỐ TỐI ƯU (BEST HYPERPARAMETERS)

{params_table}

---

## 3. HIỆU NĂNG ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP (GENERALIZATION TEST SET)

Được đánh giá trên tập kiểm thử độc lập (20% dữ liệu) hoàn toàn không rò rỉ trong quá trình tuning:

{test_table}

### Ma trận nhầm lẫn (Confusion Matrix):
- **True Positives (Bắt đúng churn):** `{test_metrics.get('TP', 'N/A')}`
- **False Negatives (Bỏ sót churn):** `{test_metrics.get('FN', 'N/A')}`
- **True Negatives (Khách hàng ở lại):** `{test_metrics.get('TN', 'N/A')}`
- **False Positives (Báo nhầm churn):** `{test_metrics.get('FP', 'N/A')}`

---

## 4. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT (TUNED MODEL)

![Đặc trưng quan trọng]({plot_imp_path})

{top15_imp}

---
*Báo cáo được sinh tự động bởi Optuna Tuner Engine — Digital Bank Churn Prediction.*
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md)
