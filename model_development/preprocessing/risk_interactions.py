"""
Risk Interactions & Financial Demographics Feature Engineering Module.
Engineers non-linear interaction terms between account balances, financial velocity,
geographic exposure, and customer demographics without hardcoded bounds.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .base import BaseFeatureTransformer


class RiskInteractionsEngineer(BaseFeatureTransformer):
    """
    Engineers domain risk interactions across financial wealth, churn demographics, and activity.

    Features created:
      - balance_to_salary_ratio: Committed balance relative to customer annual income
      - wealth_deceleration_risk: Severe risk term when large balance holders experience sudden spending decline
      - germany_high_balance: Regional risk interaction capturing elevated German market vulnerability
      - is_middle_aged_vulnerable: Demographic vulnerability indicator for life-stage wealth transitions
      - inactive_high_balance: Urgent capital flight risk (substantial balance sitting dormant)
      - credit_to_age_ratio: Credit maturity ratio comparing financial discipline against age
    """

    def __init__(
        self,
        enabled: bool = True,
        vulnerable_age_min: int = 45,
        vulnerable_age_max: int = 60,
        high_risk_country: str = "Germany",
    ):
        super().__init__(name="RiskInteractionsEngineer", enabled=enabled)
        self.vulnerable_age_min = vulnerable_age_min
        self.vulnerable_age_max = vulnerable_age_max
        self.high_risk_country = high_risk_country

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "RiskInteractionsEngineer":
        """Stateless transformer; records fitting status."""
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()

        out = X.copy()
        created: List[str] = []

        # 1. Wealth Deceleration Risk (Capital outflow / spending freezing)
        trend_col = "trans_trend_ratio" if "trans_trend_ratio" in out.columns else None
        if "balance" in out.columns and trend_col:
            out["wealth_deceleration_risk"] = np.log1p(np.maximum(0, out["balance"])) * np.maximum(
                0.0, 1.0 - out[trend_col]
            )
            created.append("wealth_deceleration_risk")

        # 2. Regional High Balance Exposure (German market risk interaction)
        if "country" in out.columns and "balance" in out.columns:
            is_high_risk = (out["country"] == self.high_risk_country).astype(int)
            out["germany_high_balance"] = is_high_risk * np.log1p(np.maximum(0, out["balance"]))
            created.append("germany_high_balance")

        # 3. Demographic Vulnerability Band (Parameter-driven age interval)
        if "age" in out.columns:
            out["is_middle_aged_vulnerable"] = (
                (out["age"] >= self.vulnerable_age_min) & (out["age"] <= self.vulnerable_age_max)
            ).astype(int)
            created.append("is_middle_aged_vulnerable")

        # 4. Committed Balance to Income Ratio
        if "balance" in out.columns and "estimated_salary" in out.columns:
            out["balance_to_salary_ratio"] = np.maximum(0, out["balance"]) / (out["estimated_salary"] + 1.0)
            created.append("balance_to_salary_ratio")

        # 5. Inactive High Balance (Dormant Flight Risk)
        if "active_member" in out.columns and "balance" in out.columns:
            out["inactive_high_balance"] = ((out["active_member"] == 0) & (out["balance"] > 0)).astype(int)
            created.append("inactive_high_balance")

        # 6. Credit Maturity Ratio
        if "credit_score" in out.columns and "age" in out.columns:
            out["credit_to_age_ratio"] = out["credit_score"] / np.maximum(out["age"], 1.0)
            created.append("credit_to_age_ratio")

        self.created_features = created
        return out

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "vulnerable_age_min": self.vulnerable_age_min,
            "vulnerable_age_max": self.vulnerable_age_max,
            "high_risk_country": self.high_risk_country,
        })
        return params

