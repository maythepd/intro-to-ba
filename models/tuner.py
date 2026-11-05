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
from models.recall_eval import export_lift_artifacts
from visualization.plots import (
    plot_feature_importance,
    plot_optuna_tuning_history,
)

# Configure Optuna logger with the familiar green INFO logs on terminal
os.system('')  # Enable ANSI VT100 color support on Windows console
try:
    import colorlog
    optuna.logging.enable_default_handler()
    for _h in optuna.logging._get_library_root_logger().handlers:
        _h.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        )
except ImportError:
    optuna.logging.enable_default_handler()

optuna.logging.set_verbosity(optuna.logging.INFO)


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
            is_recall_mode = self.target_metric.lower() in ("recall", "f2_score", "f2")
            scale_weight = (
                trial.suggest_float("scale_pos_weight", 1.5, 4.5, step=0.5)
                if is_recall_mode
                else 1.0
            )
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
                "scale_pos_weight": scale_weight,
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
                "max_iter": 3000,
                "tol": 1e-3,
                "class_weight": "balanced",
                "random_state": self.random_state,
            }
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def _build_model(self, params: Dict[str, Any]) -> Any:
        """Instantiates model given parameters."""
        p = dict(params)
        use_focal = getattr(self.config.model, "use_focal_loss", False)
        if use_focal and self.model_type in ("xgboost", "lr"):
            p["use_focal_loss"] = True
            p["focal_gamma"] = getattr(self.config.model, "focal_gamma", 2.0)
            p["focal_alpha"] = getattr(self.config.model, "focal_alpha", 0.25)

        if self.model_type == "xgboost":
            from models.xgboost import XGBoostChurnModel
            return XGBoostChurnModel(params=p, early_stopping_rounds=20)
        elif self.model_type == "rf":
            from models.random_forest import RandomForestChurnModel
            return RandomForestChurnModel(params=p)
        elif self.model_type == "lr":
            from models.logistic_regression import LogisticRegressionChurnModel
            return LogisticRegressionChurnModel(params=p)
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def tune(
        self,
        df: pd.DataFrame,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        output_dir: Optional[str] = None,
        manual_threshold: Optional[float] = None,
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
        eval_th = float(manual_threshold) if manual_threshold is not None else 0.5

        out_dir = Path(output_dir or f"results/{self.model_type}").resolve()
        os.makedirs(out_dir, exist_ok=True)

        if verbose:
            print("\n=======================================================")
            print(f"STARTING OPTUNA TUNING: {self.model_type.upper()}")
            print(f"Target Metric: {self.target_metric} | Trials: {trials_count} | Folds: {self.cv_folds}")
            if manual_threshold is not None:
                print(f"Decision Threshold: {eval_th:.2f}")
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

                # Sample weighting strictly computed on fold training data
                sw_fold = None
                if hasattr(fold_pipe, "sample_weighter") and getattr(fold_pipe.sample_weighter.config, "enabled", False):
                    fold_pipe.sample_weighter.fit(fold.train_df, y_tr)
                    sw_raw = fold_pipe.sample_weighter.compute_weights(fold.train_df, y_tr)
                    if fold_pipe.config.enable_smote and len(y_tr_res) > len(y_tr):
                        sw_fold = fold_pipe.sample_weighter.expand_for_smote(sw_raw, y_tr, y_tr_res)
                    else:
                        sw_fold = sw_raw

                # Fit model
                model = self._build_model(sampled_params)
                fit_kwargs = {"sample_weight": sw_fold} if sw_fold is not None else {}
                if self.model_type == "xgboost":
                    model.fit(X_tr_res, y_tr_res, X_val=X_va, y_val=y_va, verbose=False, **fit_kwargs)
                else:
                    model.fit(X_tr_res, y_tr_res, **fit_kwargs)

                metrics = model.evaluate(X_va, y_va, threshold=eval_th)
                if self.target_metric.lower() in ("recall", "rec"):
                    score = float(metrics.get("Recall", 0.0))
                elif self.target_metric.lower() in ("f2_score", "f2"):
                    p = metrics.get("Precision", 0.0)
                    r = metrics.get("Recall", 0.0)
                    score = float((5.0 * p * r) / (4.0 * p + r + 1e-5))
                elif self.target_metric.lower() in ("precision", "prec"):
                    score = float(metrics.get("Precision", 0.0))
                else:
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

        # Final refit sample weights
        sw_final = None
        if hasattr(final_pipe, "sample_weighter") and getattr(final_pipe.sample_weighter.config, "enabled", False):
            final_pipe.sample_weighter.fit(cv_train_pool, y_train_full)
            sw_raw = final_pipe.sample_weighter.compute_weights(cv_train_pool, y_train_full)
            if final_pipe.config.enable_smote and len(y_train_res) > len(y_train_full):
                sw_final = final_pipe.sample_weighter.expand_for_smote(sw_raw, y_train_full, y_train_res)
            else:
                sw_final = sw_raw

        # Transform validation partition if available (strictly isolated for threshold calibration & validation)
        X_val_final = None
        y_val_final = None
        if split.has_val and val_df is not None:
            te_val = final_pipe.transform(val_df)
            X_val_final = te_val[feat_cols].copy()
            y_val_final = te_val[target].copy()
            if final_pipe.config.enable_feature_selection and final_pipe.config.top_k_features:
                X_val_final = final_pipe.feature_selector.transform(X_val_final)

        # Merge base params with best tuned params
        merged_best = self._sample_params(optuna.trial.FixedTrial(best_params))
        best_model = self._build_model(merged_best)

        fit_kwargs = {"sample_weight": sw_final} if sw_final is not None else {}
        if self.model_type == "xgboost":
            if X_val_final is not None and y_val_final is not None:
                best_model.fit(X_train_res, y_train_res, X_val=X_val_final, y_val=y_val_final, verbose=False, **fit_kwargs)
            else:
                # No validation split: fit on cv_train_pool with tuned n_estimators without early stopping.
                # Strictly prevents data leakage by keeping test set completely unseen during fitting.
                best_model.fit(X_train_res, y_train_res, X_val=None, y_val=None, verbose=False, **fit_kwargs)
        else:
            best_model.fit(X_train_res, y_train_res, **fit_kwargs)

        test_metrics = best_model.evaluate(X_test_final, y_test_final, threshold=eval_th)
        ref_05 = best_model.evaluate(X_test_final, y_test_final, threshold=0.5) if manual_threshold is not None else None

        # Extract feature importances
        imp_type = "gain" if self.model_type == "xgboost" else "gini" if self.model_type == "rf" else "coefficient"
        df_imp = best_model.get_feature_importances(importance_type=imp_type)

        if verbose:
            print(f"\nIndependent Test Set Performance (Tuned Model @ Threshold {eval_th:.2f}):")
            print(
                f"  ROC-AUC: {test_metrics['ROC_AUC']:.4f} | "
                f"PR-AUC: {test_metrics['PR_AUC']:.4f} | "
                f"Precision: {test_metrics['Precision']:.4f} | "
                f"Recall: {test_metrics['Recall']:.4f} | "
                f"F1: {test_metrics['F1_Score']:.4f}"
            )
            if ref_05 is not None:
                print(
                    f"  [Reference @ Threshold 0.50] Precision: {ref_05['Precision']:.4f} | "
                    f"Recall: {ref_05['Recall']:.4f} | F1: {ref_05['F1_Score']:.4f}"
                )

        # 5. Export Artifacts (ensuring directory exists on all platforms)
        os.makedirs(out_dir, exist_ok=True)
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

        # Export Lift & Gains artifacts universally across flags
        p_final = best_model.predict_proba(X_test_final)
        test_prob_final = p_final[:, 1] if getattr(p_final, "ndim", 1) > 1 else p_final

        # Learn operating thresholds strictly on validation predictions (or train if no val); never on test set
        if X_val_final is not None and y_val_final is not None:
            p_val = best_model.predict_proba(X_val_final)
            val_prob_final = p_val[:, 1] if getattr(p_val, "ndim", 1) > 1 else p_val
            y_val_for_lift = y_val_final
        else:
            p_train = best_model.predict_proba(X_train_full)
            val_prob_final = p_train[:, 1] if getattr(p_train, "ndim", 1) > 1 else p_train
            y_val_for_lift = y_train_full

        lift_res = export_lift_artifacts(
            y_test=y_test_final,
            test_prob=test_prob_final,
            output_dir=out_dir,
            model_type=self.model_type,
            y_val=y_val_for_lift,
            val_prob=val_prob_final,
            manual_threshold=manual_threshold,
            quiet=not verbose,
        )

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
            lift_res=lift_res,
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
        lift_res: Optional[Dict[str, Any]] = None,
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

        lift_section = ""
        if lift_res:
            df_dec = lift_res.get("df_deciles")
            df_comp = lift_res.get("df_comparison")
            top10_lift = df_dec.iloc[0]["Decile_Lift"] if df_dec is not None and len(df_dec) > 0 else "N/A"
            top20_lift = df_dec.iloc[1]["Cum_Lift"] if df_dec is not None and len(df_dec) > 1 else "N/A"
            lift_section = f"""

---

## 5. ĐÁNH GIÁ HỆ SỐ LIFT & 10 PHÂN VỊ RỦI RO (DECILE LIFT & GAINS)

![Lift and Cumulative Gains Curve](lift_and_gains_curve.png)

### 5.1. Bảng 10 phân vị rủi ro (Decile Lift Analysis):
* **Top 10% rủi ro cao nhất (Decile D1):** Đạt hệ số **Lift {top10_lift}** so với chọn ngẫu nhiên.
* **Top 20% rủi ro cao nhất (Decile D1 + D2):** Đạt hệ số **Cumulative Lift {top20_lift}**.

{df_dec.to_markdown(index=False) if df_dec is not None else ""}

### 5.2. So sánh với các Baselines & Điểm vận hành tối ưu:
{df_comp.to_markdown(index=False) if df_comp is not None else ""}
"""

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

{top15_imp}{lift_section}

---
*Báo cáo được sinh tự động bởi Optuna Tuner Engine - Digital Bank Churn Prediction.*
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md)
