"""
XGBoost Classifier module for Digital Bank Customer Churn Prediction.
Encapsulates training, early stopping, probabilistic prediction, evaluation,
and feature importance extraction.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
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
import xgboost as xgb


class XGBoostChurnModel:
    """
    XGBoost Classifier for Bank Customer Churn Prediction.
    
    Supports:
      - Flexible initialization via ModelConfig or keyword arguments
      - Native handling of class imbalance (scale_pos_weight)
      - Early stopping with validation sets
      - Comprehensive business evaluation metrics (ROC-AUC, PR-AUC, F1, Recall, LogLoss)
      - Feature importance extraction (Gain, Weight, Cover)
      - Serialization to JSON or binary formats
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "min_child_weight": 1,
        "scale_pos_weight": 1.0,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: Optional[int] = 30,
        **kwargs: Any,
    ):
        self.params = dict(self.DEFAULT_PARAMS)
        if params is not None:
            self.params.update(params)
        if kwargs:
            self.params.update(kwargs)

        self.early_stopping_rounds = early_stopping_rounds
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

        # Initialize underlying XGBClassifier
        model_params = dict(self.params)
        if self.early_stopping_rounds is not None:
            model_params["early_stopping_rounds"] = self.early_stopping_rounds

        self.model = xgb.XGBClassifier(**model_params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: Union[bool, int] = False,
    ) -> "XGBoostChurnModel":
        """
        Train the XGBoost classifier.
        If validation set is provided, monitors validation metric for early stopping.
        """
        self.feature_names_ = list(X_train.columns)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose,
        )

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
        return probs[:, 1]

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Computes a comprehensive suite of classification metrics on test data.
        """
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
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """
        Extract feature importances from trained model.
        Args:
          importance_type: 'gain' (default), 'weight', 'cover', or 'total_gain'.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        booster = self.model.get_booster()
        score_dict = booster.get_score(importance_type=importance_type)

        df_imp = pd.DataFrame([
            {"Feature": f, "Importance": score_dict.get(f, 0.0)}
            for f in self.feature_names_
        ]).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        # Normalize relative importance
        total = df_imp["Importance"].sum()
        if total > 0:
            df_imp["Relative_Pct"] = (df_imp["Importance"] / total * 100).round(2)
        else:
            df_imp["Relative_Pct"] = 0.0

        return df_imp

    def save(self, path: Union[str, Path]) -> str:
        """Save model to a JSON file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(filepath))
        return str(filepath)

    def load(self, path: Union[str, Path]) -> "XGBoostChurnModel":
        """Load model from a JSON file."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")
        self.model.load_model(str(filepath))
        self.is_fitted = True
        return self
