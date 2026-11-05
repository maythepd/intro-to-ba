"""
Logistic Regression Classifier module for Digital Bank Customer Churn Prediction.
Follows the same interface contract as XGBoostChurnModel:
  fit(), predict(), predict_proba(), evaluate(), get_feature_importances(), save(), load().
Supports:
  - Native sample weighting (exponential recency decay + financial value weighting)
  - Custom Focal Loss objective via L-BFGS-B (FocalLossLogisticRegression)
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

from .focal_loss import FocalLossLogisticRegression


class LogisticRegressionChurnModel:
    """
    Logistic Regression Classifier for Bank Customer Churn Prediction.

    Supports:
      - Flexible initialization via keyword arguments or params dict
      - Native class-imbalance handling via class_weight='balanced'
      - Sample weighting (exponential recency decay + financial value weighting)
      - Custom Focal Loss objective via L-BFGS-B (FocalLossLogisticRegression)
      - Comprehensive business evaluation metrics (ROC-AUC, PR-AUC, F1, Recall, LogLoss)
      - Coefficient-based feature importance extraction
      - Serialization via pickle
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "C": 0.1,
        "max_iter": 3000,
        "tol": 1e-3,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        **kwargs: Any,
    ):
        self.params = dict(self.DEFAULT_PARAMS)
        if params is not None:
            self.params.update(params)
        if kwargs:
            self.params.update(kwargs)

        self.use_focal_loss = bool(self.params.pop("use_focal_loss", use_focal_loss))
        self.focal_gamma = float(self.params.pop("focal_gamma", focal_gamma))
        self.focal_alpha = float(self.params.pop("focal_alpha", focal_alpha))

        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

        if self.use_focal_loss:
            c_val = float(self.params.get("C", 1.0))
            max_iter = int(self.params.get("max_iter", 500))
            rnd = int(self.params.get("random_state", 42))
            self.model = FocalLossLogisticRegression(
                C=c_val,
                gamma=self.focal_gamma,
                alpha=self.focal_alpha,
                max_iter=max_iter,
                random_state=rnd,
            )
        else:
            self.model = LogisticRegression(**self.params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        verbose: Union[bool, int] = False,
    ) -> "LogisticRegressionChurnModel":
        """
        Train the Logistic Regression classifier with optional sample weighting.
        X_val / y_val are accepted for API compatibility but not used (LR has no early stopping).
        """
        self.feature_names_ = list(X_train.columns)
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary churn labels using decision threshold."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of churn (class 1)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict_proba().")
        probs = self.model.predict_proba(X)
        if probs.ndim > 1 and probs.shape[1] > 1:
            return probs[:, 1]
        return probs

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Computes a comprehensive suite of classification metrics on test data."""
        y_prob = self.predict_proba(X_test)
        y_pred = (y_prob >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "ROC_AUC": float(round(roc_auc_score(y_test, y_prob), 4)),
            "PR_AUC": float(round(average_precision_score(y_test, y_prob), 4)),
            "F1_Score": float(round(f1_score(y_test, y_pred, zero_division=0), 4)),
            "Precision": float(round(precision_score(y_test, y_pred, zero_division=0), 4)),
            "Recall": float(round(recall_score(y_test, y_pred, zero_division=0), 4)),
            "Accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
            "Balanced_Accuracy": float(round(balanced_accuracy_score(y_test, y_pred), 4)),
            "Log_Loss": float(round(log_loss(y_test, y_prob), 4)),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        }
        return metrics

    def get_feature_importances(
        self,
        importance_type: str = "coefficient",
    ) -> pd.DataFrame:
        """
        Extract feature importances from trained Logistic Regression.
        Uses absolute coefficient values as importance proxy.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        coefs = self.model.coef_[0]
        df_imp = pd.DataFrame({
            "Feature": self.feature_names_,
            "Importance": np.abs(coefs),
            "Coefficient": coefs,
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        total = df_imp["Importance"].sum()
        df_imp["Relative_Pct"] = (
            (df_imp["Importance"] / total * 100).round(2) if total > 0 else 0.0
        )
        return df_imp

    def save(self, path: Union[str, Path]) -> str:
        """Save model to a pickle file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(str(filepath), "wb") as f:
            pickle.dump(self, f)
        return str(filepath)

    def load(self, path: Union[str, Path]) -> "LogisticRegressionChurnModel":
        """Load model from a pickle file."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")
        with open(str(filepath), "rb") as f:
            loaded = pickle.load(f)
        self.__dict__.update(loaded.__dict__)
        return self
