"""
Phase 3: Advanced behavioral, friction, bundling, and interaction feature engineering.
Engineers domain-driven features derived from banking behavioral mechanics.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .base import BaseFeatureTransformer


class DomainFeatureEngineer(BaseFeatureTransformer):
    """
    Engineers banking domain-specific behavioral and risk features.
    
    Sub-feature groups for ablation studies:
      1. behavioral_momentum:
         - interaction_velocity: 1.0 / np.log1p(recency_days)
         - is_spending_collapsing: trans_trend_ratio < 0.50
         - avg_ticket_size: trans_amount_3m / (3 * (monthly_transactions + 1))
         - app_to_trans_ratio: app_logins_monthly / (monthly_transactions + 1)
      2. customer_friction:
         - friction_index: has_unresolved_complaint * 3.0 + customer_service_complaints * 1.5
         - is_high_friction: has_unresolved_complaint == 1 or customer_service_complaints >= 2
      3. product_bundling:
         - is_product_sweetspot: products_number == 2 (low-churn zone)
         - is_product_danger_zone: products_number >= 3 (high-churn zone)
      4. risk_interactions:
         - wealth_deceleration_risk: log1p(balance) * max(0, 1 - trans_trend_ratio)
         - germany_high_balance: (country == 'Germany') * log1p(balance)
         - is_middle_aged_vulnerable: 45 <= age <= 60
    """

    def __init__(
        self,
        enabled: bool = True,
        enable_behavioral_momentum: bool = True,
        enable_customer_friction: bool = True,
        enable_product_bundling: bool = True,
        enable_risk_interactions: bool = True,
    ):
        super().__init__(name="DomainFeatureEngineer", enabled=enabled)
        self.enable_behavioral_momentum = enable_behavioral_momentum
        self.enable_customer_friction = enable_customer_friction
        self.enable_product_bundling = enable_product_bundling
        self.enable_risk_interactions = enable_risk_interactions

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DomainFeatureEngineer":
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()

        out = X.copy()
        created: List[str] = []

        # 1. Behavioral Momentum & Velocity Features
        if self.enable_behavioral_momentum:
            if "recency_days" in out.columns:
                out["interaction_velocity"] = 1.0 / np.log1p(out["recency_days"].clip(lower=0))
                created.append("interaction_velocity")

            if "trans_trend_ratio" in out.columns:
                out["is_spending_collapsing"] = (out["trans_trend_ratio"] < 0.50).astype(int)
                created.append("is_spending_collapsing")

            if "trans_amount_3m" in out.columns and "monthly_transactions" in out.columns:
                out["avg_ticket_size"] = out["trans_amount_3m"] / (3.0 * (out["monthly_transactions"] + 1.0))
                created.append("avg_ticket_size")

            if "app_logins_monthly" in out.columns and "monthly_transactions" in out.columns:
                out["app_to_trans_ratio"] = out["app_logins_monthly"] / (out["monthly_transactions"] + 1.0)
                created.append("app_to_trans_ratio")

        # 2. Customer Service Friction Features
        if self.enable_customer_friction:
            has_unres = "has_unresolved_complaint" in out.columns
            has_complaints = "customer_service_complaints" in out.columns

            if has_unres and has_complaints:
                out["friction_index"] = (
                    out["has_unresolved_complaint"] * 3.0 + out["customer_service_complaints"] * 1.5
                )
                out["is_high_friction"] = (
                    (out["has_unresolved_complaint"] == 1) | (out["customer_service_complaints"] >= 2)
                ).astype(int)
                created.extend(["friction_index", "is_high_friction"])
            elif has_unres:
                out["friction_index"] = out["has_unresolved_complaint"] * 3.0
                out["is_high_friction"] = (out["has_unresolved_complaint"] == 1).astype(int)
                created.extend(["friction_index", "is_high_friction"])
            elif has_complaints:
                out["friction_index"] = out["customer_service_complaints"] * 1.5
                out["is_high_friction"] = (out["customer_service_complaints"] >= 2).astype(int)
                created.extend(["friction_index", "is_high_friction"])

        # 3. Product Bundling Non-linear Features
        if self.enable_product_bundling and "products_number" in out.columns:
            out["is_product_sweetspot"] = (out["products_number"] == 2).astype(int)
            out["is_product_danger_zone"] = (out["products_number"] >= 3).astype(int)
            created.extend(["is_product_sweetspot", "is_product_danger_zone"])

        # 4. Wealth & Regional Risk Interaction Terms
        if self.enable_risk_interactions:
            if "balance" in out.columns and "trans_trend_ratio" in out.columns:
                out["wealth_deceleration_risk"] = np.log1p(
                    np.maximum(0, out["balance"])
                ) * np.maximum(0, 1.0 - out["trans_trend_ratio"])
                created.append("wealth_deceleration_risk")

            if "country" in out.columns and "balance" in out.columns:
                is_germany = (out["country"] == "Germany").astype(int)
                out["germany_high_balance"] = is_germany * np.log1p(np.maximum(0, out["balance"]))
                created.append("germany_high_balance")

            if "age" in out.columns:
                out["is_middle_aged_vulnerable"] = ((out["age"] >= 45) & (out["age"] <= 60)).astype(int)
                created.append("is_middle_aged_vulnerable")

        self.created_features = created
        return out

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "enable_behavioral_momentum": self.enable_behavioral_momentum,
            "enable_customer_friction": self.enable_customer_friction,
            "enable_product_bundling": self.enable_product_bundling,
            "enable_risk_interactions": self.enable_risk_interactions,
        })
        return params
