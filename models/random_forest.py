"""
Random Forest Classifier module for Digital Bank Customer Churn Prediction.
Follows the same interface contract as XGBoostChurnModel:
  fit(), predict(), predict_proba(), evaluate(), get_feature_importances(), save(), load().
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


class RandomForestChurnModel:
    """
    Random Forest Classifier for Bank Customer Churn Prediction.

    Supports:
      - Flexible initialization via keyword arguments or params dict
      - Native class-imbalance handling via class_weight='balanced'
      - Comprehensive business evaluation metrics (ROC-AUC, PR-AUC, F1, Recall, LogLoss)
      - Gini/MDI feature importance extraction
      - Serialization via pickle
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        # 300 trees: cân bằng giữa phương sai thấp và runtime.
        "n_estimators": 300,
        # Cài max_depth=15 thay vì None: giới hạn chiều sâu cây để kiểm soát
        # overfitting trên tabular churn data (55 features, 10K rows).
        "max_depth": 15,
        # min_samples_split=10: tạo node mới chỉ khi đủ mẫu, giảm chi phí
        # chia trên các node nhỏ có ít thông tin.
        "min_samples_split": 10,
        # min_samples_leaf=4: lá cây phải có ít nhất 4 mẫu — regularization
        # quan trọng hơn khi thiếu balanced label.
        "min_samples_leaf": 4,
        # sqrt(n_features) là chuẩn cho classification: giảm tương quan giữa cây.
        "max_features": "sqrt",
        # balanced_subsample: kết hợp với bootstrap, cân bằng class weight
        # trên từng subsample thay vì toàn bộ dataset.
        "class_weight": "balanced_subsample",
        # oob_score=True: tận dụng out-of-bag samples như tập validation miễn phí.
        "oob_score": True,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.params = dict(self.DEFAULT_PARAMS)
        if params is not None:
            self.params.update(params)
        if kwargs:
            self.params.update(kwargs)

        self.feature_names_: List[str] = []
        self.is_fitted: bool = False
        self.model = RandomForestClassifier(**self.params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: Union[bool, int] = False,
    ) -> "RandomForestChurnModel":
        """
        Train the Random Forest classifier.
        X_val / y_val are accepted for API compatibility but not used.
        """
        self.feature_names_ = list(X_train.columns)
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
        return self.model.predict_proba(X)[:, 1]

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
        importance_type: str = "gini",
    ) -> pd.DataFrame:
        """
        Extract feature importances from trained Random Forest.
        Uses mean decrease in impurity (Gini importance) by default.
        importance_type is accepted for interface compatibility but 'gini' (MDI) is always used.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        importances = self.model.feature_importances_
        df_imp = pd.DataFrame({
            "Feature": self.feature_names_,
            "Importance": importances,
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

    def load(self, path: Union[str, Path]) -> "RandomForestChurnModel":
        """Load model from a pickle file."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")
        with open(str(filepath), "rb") as f:
            loaded = pickle.load(f)
        self.__dict__.update(loaded.__dict__)
        return self
