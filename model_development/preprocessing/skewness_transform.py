"""
Phase 1: Skewness diagnosis & power transformations.
Handles right-skewed and heavy-tailed distributions via Log1p, zero-inflation hurdle,
and Yeo-Johnson power transformations.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from .base import BaseFeatureTransformer


class SkewnessTransformer(BaseFeatureTransformer):
    """
    Transforms skewed numerical distributions.
    Features engineered:
      - log_trans_amount_3m: Log1p transformation on 3-month transaction volume
      - log_recency_days: Log1p transformation on days since last transaction
      - is_zero_balance: Binary hurdle indicator (1 if balance == 0, else 0)
      - log_positive_balance: Log1p of positive balance portion
      - yj_trans_amount_3m: Yeo-Johnson power transformation on trans_amount_3m (optional)
    """

    def __init__(
        self,
        enabled: bool = True,
        transform_trans_amount_log: bool = True,
        transform_recency_log: bool = True,
        zero_balance_hurdle: bool = True,
        transform_yeo_johnson: bool = True,
    ):
        super().__init__(name="SkewnessTransformer", enabled=enabled)
        self.transform_trans_amount_log = transform_trans_amount_log
        self.transform_recency_log = transform_recency_log
        self.zero_balance_hurdle = zero_balance_hurdle
        self.transform_yeo_johnson = transform_yeo_johnson

        self._pt: Optional[PowerTransformer] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "SkewnessTransformer":
        if not self.enabled:
            self.is_fitted = True
            return self

        if self.transform_yeo_johnson and "trans_amount_3m" in X.columns:
            self._pt = PowerTransformer(method="yeo-johnson")
            self._pt.fit(X[["trans_amount_3m"]])

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()

        out = X.copy()
        created: List[str] = []

        # 1. Log1p trans_amount_3m
        if self.transform_trans_amount_log and "trans_amount_3m" in out.columns:
            out["log_trans_amount_3m"] = np.log1p(out["trans_amount_3m"].clip(lower=0))
            created.append("log_trans_amount_3m")

        # 2. Log1p recency_days
        if self.transform_recency_log and "recency_days" in out.columns:
            out["log_recency_days"] = np.log1p(out["recency_days"].clip(lower=0))
            created.append("log_recency_days")

        # 3. Balance Zero-inflation Hurdle Decomposition
        if self.zero_balance_hurdle and "balance" in out.columns:
            out["is_zero_balance"] = (out["balance"] == 0).astype(int)
            out["log_positive_balance"] = np.log1p(np.maximum(0, out["balance"]))
            created.extend(["is_zero_balance", "log_positive_balance"])

        # 4. Yeo-Johnson Power Transformation on trans_amount_3m
        if self.transform_yeo_johnson and "trans_amount_3m" in out.columns and self._pt is not None:
            yj_vals = self._pt.transform(out[["trans_amount_3m"]])
            out["yj_trans_amount_3m"] = yj_vals.ravel()
            created.append("yj_trans_amount_3m")

        self.created_features = created
        return out

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "transform_trans_amount_log": self.transform_trans_amount_log,
            "transform_recency_log": self.transform_recency_log,
            "zero_balance_hurdle": self.zero_balance_hurdle,
            "transform_yeo_johnson": self.transform_yeo_johnson,
        })
        return params
