"""
Sample Weighting Module for Digital Bank Customer Churn Prediction.
Implements:
  1. Exponential recency decay (prioritizing recent customer activity signals).
  2. Financial value weighting (prioritizing high-balance / high-CLV customers).
  3. Class-asymmetric weighting (penalizing false negatives on churners).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class SampleWeightConfig:
    """Configuration container for sample weighting."""

    enabled: bool = False
    decay_mode: str = "exponential"  # "exponential" | "linear" | "none"
    recency_col: str = "recency_days"
    decay_half_life_days: float = 60.0
    enable_value_weighting: bool = True
    balance_col: str = "balance"
    value_weight_alpha: float = 0.5
    churn_weight_multiplier: float = 1.5
    normalize: bool = True


class SampleWeightGenerator:
    """
    Computes sample-level importance weights for model training.
    
    Composite Weight Formulation:
        w_i = w_recency(i) * w_value(i) * w_churn(i)
        w_normalized = w_i * (N / sum(w_i))
    """

    def __init__(
        self,
        config: Optional[Union[SampleWeightConfig, Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        if config is None:
            self.config = SampleWeightConfig(**kwargs)
        elif isinstance(config, dict):
            cfg_dict = dict(config)
            cfg_dict.update(kwargs)
            self.config = SampleWeightConfig(**{k: v for k, v in cfg_dict.items() if hasattr(SampleWeightConfig, k)})
        else:
            self.config = config

        self.median_balance_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> "SampleWeightGenerator":
        """Learn baseline reference values (e.g. median balance) from training partition."""
        bal_col = self.config.balance_col
        if bal_col in df.columns:
            positive_balances = df[df[bal_col] > 0][bal_col]
            self.median_balance_ = float(positive_balances.median()) if len(positive_balances) > 0 else 1000.0
        else:
            self.median_balance_ = 1000.0

        self.is_fitted = True
        return self

    def compute_weights(
        self,
        df: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Calculates sample weights for the given dataframe.
        """
        n = len(df)
        if not self.config.enabled or n == 0:
            return np.ones(n, dtype=float)

        if not self.is_fitted:
            self.fit(df, y)

        weights = np.ones(n, dtype=float)

        # 1. Exponential Recency Decay
        rec_col = self.config.recency_col
        if self.config.decay_mode == "exponential" and rec_col in df.columns:
            recency = df[rec_col].fillna(self.config.decay_half_life_days).values.astype(float)
            recency = np.maximum(recency, 0.0)
            t_half = max(float(self.config.decay_half_life_days), 1.0)
            lam = np.log(2.0) / t_half
            w_rec = np.exp(-lam * recency)
            # Clip between 0.1 and 1.0 to prevent sample starvation
            w_rec = np.clip(w_rec, 0.10, 1.0)
            weights *= w_rec

        # 2. Financial Value Weighting (Balance)
        bal_col = self.config.balance_col
        if self.config.enable_value_weighting and bal_col in df.columns:
            balance = df[bal_col].fillna(0.0).values.astype(float)
            balance = np.maximum(balance, 0.0)
            denom = max(self.median_balance_, 1.0)
            alpha = float(self.config.value_weight_alpha)
            w_val = 1.0 + alpha * np.log1p(balance / denom)
            weights *= w_val

        # 3. Class-asymmetric Weighting for Churners (reduces FN)
        if y is not None and self.config.churn_weight_multiplier > 1.0:
            y_arr = np.asarray(y)
            churn_mult = float(self.config.churn_weight_multiplier)
            w_churn = np.where(y_arr == 1, churn_mult, 1.0)
            weights *= w_churn

        # 4. Normalization (preserves gradient scale such that mean(w) == 1.0)
        if self.config.normalize and np.sum(weights) > 0:
            weights = weights * (n / np.sum(weights))

        return weights

    @staticmethod
    def expand_for_smote(
        weights: np.ndarray,
        y_train_original: Union[pd.Series, np.ndarray],
        y_train_resampled: Union[pd.Series, np.ndarray],
    ) -> np.ndarray:
        """
        Expands sample weights when synthetic minority samples are created by SMOTE.
        Synthetic samples receive the mean weight of the real minority class samples.
        """
        n_orig = len(y_train_original)
        n_res = len(y_train_resampled)
        if n_res <= n_orig:
            return weights

        y_orig = np.asarray(y_train_original)
        pos_weights = weights[y_orig == 1]
        mean_pos_weight = float(np.mean(pos_weights)) if len(pos_weights) > 0 else 1.0

        n_synthetic = n_res - n_orig
        synthetic_weights = np.full(n_synthetic, fill_value=mean_pos_weight, dtype=float)
        expanded = np.concatenate([weights, synthetic_weights])

        # Renormalize to maintain mean == 1.0
        return expanded * (n_res / np.sum(expanded))

