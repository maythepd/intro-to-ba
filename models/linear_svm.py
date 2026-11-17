"""
Linear Support Vector Machine (Linear SVM) Classifier for Bank Customer Churn Prediction.
Follows the unified model contract:
  fit(), predict(), predict_proba(), evaluate(), get_feature_importances(), save(), load().

Features:
  - Primal/Dual linear hyperplane margin maximization via LinearSVC (LIBLINEAR)
  - Automatic feature standardization (StandardScaler) critical for distance-based SVM
  - Well-calibrated class probability output via Platt scaling (CalibratedClassifierCV)
  - Geometric separability diagnostics (margin width, slack violations, hyperplane distance)
  - Native class imbalance handling (class_weight='balanced')
  - Sample weighting support (recency decay + financial value weighting)
  - Absolute weight vector |w_j| feature importance extraction
"""

from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class LinearSVMChurnModel:
    """
    Linear Support Vector Classifier with Platt Probability Calibration
    and Geometric Separability Diagnostics.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "C": 0.1,
        "penalty": "l2",
        "loss": "squared_hinge",
        "dual": "auto",
        "tol": 1e-4,
        "max_iter": 5000,
        "class_weight": "balanced",
        "random_state": 42,
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        scale_features: bool = True,
        calibrate_probabilities: bool = True,
        calibration_cv: int = 3,
        decision_threshold: float = 0.35,
        **kwargs: Any,
    ):
        self.params = dict(self.DEFAULT_PARAMS)
        if params is not None:
            self.params.update(params)
        if kwargs:
            self.params.update(kwargs)

        self.scale_features = bool(self.params.pop("scale_features", scale_features))
        self.calibrate_probabilities = bool(
            self.params.pop("calibrate_probabilities", calibrate_probabilities)
        )
        self.calibration_cv = int(self.params.pop("calibration_cv", calibration_cv))
        self.decision_threshold = float(
            self.params.pop("decision_threshold", decision_threshold)
        )

        self.scaler_: Optional[StandardScaler] = (
            StandardScaler() if self.scale_features else None
        )
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

        # Raw estimator for hyperplane geometry analysis
        self.raw_svc_ = LinearSVC(**self.params)
        # Calibrated estimator for reliable probability scoring
        self.calibrated_model_: Optional[CalibratedClassifierCV] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        verbose: Union[bool, int] = False,
    ) -> "LinearSVMChurnModel":
        """
        Trains the Linear SVM classifier and calibrates probability outputs via Platt scaling.
        Feature matrices are standardized to guarantee optimal hyperplane convergence.
        """
        self.feature_names_ = list(X_train.columns)

        if self.scaler_ is not None:
            X_fit = self.scaler_.fit_transform(X_train)
        else:
            X_fit = X_train.values if isinstance(X_train, pd.DataFrame) else X_train

        y_fit = y_train.values if isinstance(y_train, pd.Series) else y_train

        # 1. Fit raw LinearSVC for hyperplane parameters (w, b)
        if sample_weight is not None:
            self.raw_svc_.fit(X_fit, y_fit, sample_weight=sample_weight)
        else:
            self.raw_svc_.fit(X_fit, y_fit)

        # 2. Fit calibrated classifier for calibrated probabilities
        if self.calibrate_probabilities:
            # Calibrate on training fold using cross-validation (or prefit)
            base = LinearSVC(**self.params)
            self.calibrated_model_ = CalibratedClassifierCV(
                estimator=base,
                method="sigmoid",
                cv=self.calibration_cv,
            )
            if sample_weight is not None:
                self.calibrated_model_.fit(X_fit, y_fit, sample_weight=sample_weight)
            else:
                self.calibrated_model_.fit(X_fit, y_fit)

        self.is_fitted = True
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns raw signed geometric distance to the separating hyperplane:
            f(x) = w^T x + b
        Positive indicates class 1 (churn), negative indicates class 0 (loyal).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        if self.scaler_ is not None:
            X_eval = self.scaler_.transform(X)
        else:
            X_eval = X.values if isinstance(X, pd.DataFrame) else X

        return self.raw_svc_.decision_function(X_eval)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts calibrated probability of churn (P(y=1|x)).
        Falls back to sigmoid-transformed decision function if calibration was disabled.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        if self.scaler_ is not None:
            X_eval = self.scaler_.transform(X)
        else:
            X_eval = X.values if isinstance(X, pd.DataFrame) else X

        if self.calibrate_probabilities and self.calibrated_model_ is not None:
            probs = self.calibrated_model_.predict_proba(X_eval)
            return probs[:, 1] if probs.ndim > 1 and probs.shape[1] > 1 else probs

        # Fallback: standard logistic sigmoid on raw margin distance
        dist = self.raw_svc_.decision_function(X_eval)
        return 1.0 / (1.0 + np.exp(-dist))

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Predicts binary churn labels using decision threshold."""
        th = self.decision_threshold if threshold is None else float(threshold)
        probs = self.predict_proba(X)
        return (probs >= th).astype(int)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Computes comprehensive classification metrics on test data."""
        th = self.decision_threshold if threshold is None else float(threshold)
        y_prob = self.predict_proba(X_test)
        y_pred = (y_prob >= th).astype(int)

        y_true = y_test.values if isinstance(y_test, pd.Series) else y_test
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "ROC_AUC": float(round(roc_auc_score(y_true, y_prob), 4)),
            "PR_AUC": float(round(average_precision_score(y_true, y_prob), 4)),
            "F1_Score": float(round(f1_score(y_true, y_pred, zero_division=0), 4)),
            "Precision": float(round(precision_score(y_true, y_pred, zero_division=0), 4)),
            "Recall": float(round(recall_score(y_true, y_pred, zero_division=0), 4)),
            "Accuracy": float(round(accuracy_score(y_true, y_pred), 4)),
            "Balanced_Accuracy": float(round(balanced_accuracy_score(y_true, y_pred), 4)),
            "Log_Loss": float(round(log_loss(y_true, y_prob), 4)),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        }
        return metrics

    def compute_separability_metrics(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Diagnoses whether the feature space is linearly separable.

        Metrics computed:
          - geometric_margin: 1 / ||w||_2
          - weight_norm: ||w||_2
          - train_misclassification_rate: fraction of samples with y_i * f(x_i) < 0
          - margin_violations_rate: fraction of samples violating margin (slack xi_i > 0)
          - severe_margin_violations: fraction of samples misclassified past margin (xi_i > 1)
          - mean_slack: average slack penalty sum(xi_i) / N
          - is_strictly_separable: True if zero misclassifications and zero margin violations
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        y_arr = y.values if isinstance(y, pd.Series) else np.array(y)
        # Convert {0, 1} to {-1, +1}
        y_pm = np.where(y_arr == 1, 1.0, -1.0)

        f_x = self.decision_function(X)
        functional_margin = y_pm * f_x
        slack = np.maximum(0.0, 1.0 - functional_margin)

        w = self.raw_svc_.coef_[0]
        w_norm = float(np.linalg.norm(w))
        geom_margin = float(1.0 / (w_norm + 1e-8))

        misclassified = float(np.mean(functional_margin < 0.0))
        margin_violations = float(np.mean(slack > 0.0))
        severe_violations = float(np.mean(slack > 1.0))
        mean_slack = float(np.mean(slack))

        return {
            "weight_norm": round(w_norm, 4),
            "geometric_margin": round(geom_margin, 4),
            "misclassification_rate": round(misclassified, 4),
            "margin_violation_rate": round(margin_violations, 4),
            "severe_violation_rate": round(severe_violations, 4),
            "mean_slack": round(mean_slack, 4),
            "is_strictly_separable": bool(misclassified == 0.0 and margin_violations == 0.0),
        }

    def get_feature_importances(
        self,
        importance_type: str = "weight",
    ) -> pd.DataFrame:
        """
        Extracts standardized weight-based feature importances from trained Linear SVM.
        Uses |w_j| as feature importance magnitude.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        weights = self.raw_svc_.coef_[0]
        df_imp = pd.DataFrame({
            "Feature": self.feature_names_,
            "Importance": np.abs(weights),
            "Weight": weights,
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        total = df_imp["Importance"].sum()
        df_imp["Relative_Pct"] = (
            (df_imp["Importance"] / total * 100).round(2) if total > 0 else 0.0
        )
        return df_imp

    def save(self, path: Union[str, Path]) -> str:
        """Saves model to a pickle file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(str(filepath), "wb") as f:
            pickle.dump(self, f)
        return str(filepath)

    def load(self, path: Union[str, Path]) -> "LinearSVMChurnModel":
        """Loads model from a pickle file."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")
        with open(str(filepath), "rb") as f:
            loaded = pickle.load(f)
        self.__dict__.update(loaded.__dict__)
        return self
