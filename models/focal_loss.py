"""
Focal Loss implementation for Gradient-Based Classifiers.
Provides:
  1. XGBoost Custom Objective & Evaluation Metric (make_xgboost_focal_loss).
  2. FocalLossLogisticRegression: scikit-learn compatible estimator trained via L-BFGS-B.

Theoretical Background:
  Standard Cross-Entropy Loss:
      CE(p_t) = -log(p_t)
  Focal Loss (Lin et al., ICCV 2017):
      FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
  where:
      p_t = p if y=1 else (1 - p)
      alpha_t = alpha if y=1 else (1 - alpha)
      gamma >= 0 is the focusing parameter.

By dampening the loss contribution of easy, well-classified examples ((1 - p_t)^gamma -> 0),
the model focuses gradient updates on hard-to-classify samples and class boundaries,
effectively pushing the Precision-Recall frontier outward and reducing False Negatives.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    z_clipped = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def make_xgboost_focal_loss(
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tuple[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]], Callable[[np.ndarray, np.ndarray], Tuple[str, float]]]:
    """
    Factory creating a custom objective and evaluation metric for XGBoost.

    Args:
        gamma: Focusing parameter (default: 2.0).
        alpha: Weight for positive class (churn=1), default: 0.25.

    Returns:
        focal_obj: Callable(labels, preds) -> (grad, hess)
        focal_eval: Callable(labels, preds) -> (metric_name, metric_value)
    """

    def focal_obj(labels: Any, preds: Any) -> Tuple[np.ndarray, np.ndarray]:
        # Support both numpy arrays and xgb.DMatrix
        if hasattr(labels, "get_label"):
            y = labels.get_label()
        else:
            y = np.asarray(labels)

        z = np.asarray(preds)
        p = _sigmoid(z)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)

        # 1. Gradient with respect to logit z
        # y=1: alpha * (1 - p)^gamma * [ gamma * p * log(p) + p - 1 ]
        # y=0: (1 - alpha) * p^gamma * [ -gamma * (1 - p) * log(1 - p) + p ]
        log_p = np.log(p)
        log_1_p = np.log(1.0 - p)

        g_pos = alpha * ((1.0 - p) ** gamma) * (gamma * p * log_p + p - 1.0)
        g_neg = (1.0 - alpha) * (p ** gamma) * (-gamma * (1.0 - p) * log_1_p + p)
        grad = np.where(y == 1, g_pos, g_neg)

        # 2. Exact Hessian with respect to logit z: h = (dg/dp) * (dp/dz)
        dp_dz = p * (1.0 - p)

        p_safe_pos = np.maximum(1.0 - p, 1e-6)
        term1_pos = -gamma * (p_safe_pos ** (gamma - 1.0)) * (gamma * p * log_p + p - 1.0)
        term2_pos = ((1.0 - p) ** gamma) * (gamma * (log_p + 1.0) + 1.0)
        dg_dp_pos = alpha * (term1_pos + term2_pos)

        p_safe_neg = np.maximum(p, 1e-6)
        term1_neg = gamma * (p_safe_neg ** (gamma - 1.0)) * (-gamma * (1.0 - p) * log_1_p + p)
        term2_neg = (p ** gamma) * (gamma * (log_1_p + 1.0) + 1.0)
        dg_dp_neg = (1.0 - alpha) * (term1_neg + term2_neg)

        hess = np.where(y == 1, dg_dp_pos, dg_dp_neg) * dp_dz
        # Ensure positive definite hessian for stable tree splitting
        hess = np.clip(hess, 1e-4, 100.0)

        return grad, hess

    def focal_eval(labels: Any, preds: Any) -> Tuple[str, float]:
        if hasattr(labels, "get_label"):
            y = labels.get_label()
        else:
            y = np.asarray(labels)

        z = np.asarray(preds)
        p = _sigmoid(z)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)

        loss_pos = -alpha * ((1.0 - p) ** gamma) * np.log(p)
        loss_neg = -(1.0 - alpha) * (p ** gamma) * np.log(1.0 - p)
        fl_val = float(np.mean(np.where(y == 1, loss_pos, loss_neg)))

        return "focal_loss", fl_val

    return focal_obj, focal_eval


class FocalLossLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression with Focal Loss objective optimized via L-BFGS-B.
    
    Compatible with scikit-learn API:
      - fit(X, y, sample_weight=None)
      - predict(X, threshold=0.5)
      - predict_proba(X)
      - coef_, intercept_ attributes for explainability.
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: float = 2.0,
        alpha: float = 0.25,
        max_iter: int = 500,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        self.C = C
        self.gamma = gamma
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.classes_: np.ndarray = np.array([0, 1])
        self.n_iter_: int = 0

    def fit(
        self,
        X: Union[np.ndarray, Any],
        y: Union[np.ndarray, Any],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "FocalLossLogisticRegression":
        X_mat = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n_samples, n_features = X_mat.shape

        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            if np.sum(sw) > 0:
                sw = sw * (n_samples / np.sum(sw))
        else:
            sw = np.ones(n_samples, dtype=float)

        w0 = np.zeros(n_features + 1, dtype=float)
        reg_c = max(float(self.C), 1e-6)

        def objective_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
            w = theta[:-1]
            b = theta[-1]

            z = X_mat @ w + b
            p = _sigmoid(z)
            p = np.clip(p, 1e-12, 1.0 - 1e-12)

            # Focal loss values
            loss_pos = -self.alpha * ((1.0 - p) ** self.gamma) * np.log(p)
            loss_neg = -(1.0 - self.alpha) * (p ** self.gamma) * np.log(1.0 - p)
            loss_samples = np.where(y_arr == 1, loss_pos, loss_neg) * sw
            loss_val = np.sum(loss_samples) + (0.5 / reg_c) * np.dot(w, w)

            # Sample gradients w.r.t z
            g_pos = self.alpha * ((1.0 - p) ** self.gamma) * (self.gamma * p * np.log(p) + p - 1.0)
            g_neg = (1.0 - self.alpha) * (p ** self.gamma) * (-self.gamma * (1.0 - p) * np.log(1.0 - p) + p)
            g_z = np.where(y_arr == 1, g_pos, g_neg) * sw

            # Gradients w.r.t parameters
            grad_w = X_mat.T @ g_z + (1.0 / reg_c) * w
            grad_b = np.sum(g_z)
            grad_all = np.append(grad_w, grad_b)

            return float(loss_val), grad_all

        opt_result = minimize(
            fun=objective_and_grad,
            x0=w0,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.max_iter,
                "ftol": self.tol,
            },
        )

        self.coef_ = np.array([opt_result.x[:-1]])
        self.intercept_ = np.array([opt_result.x[-1]])
        self.n_iter_ = int(opt_result.nit)

        return self

    def predict_proba(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X_mat = np.asarray(X, dtype=float)
        z = X_mat @ self.coef_[0] + self.intercept_[0]
        p = _sigmoid(z)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Union[np.ndarray, Any], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

