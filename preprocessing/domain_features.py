"""
Phase 3: Banking Domain & Dynamic Behavioral Feature Engineering.
Engineers domain-driven features derived from banking behavioral mechanics,
momentum trends, service friction, portfolio bundling, and risk interactions.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .base import BaseFeatureTransformer


class DomainFeatureEngineer(BaseFeatureTransformer):
    """
    Engineers banking domain-specific behavioral momentum, friction, and risk features.
    
    Sub-feature groups for ablation studies:
      1. behavioral_momentum:
         - monthly_transactions, monthly_logins, app_logins_monthly
         - trans_trend_ratio, trans_count_trend_ratio, login_trend_ratio
         - interaction_velocity, login_velocity
         - is_spending_collapsing, is_login_collapsing
         - avg_ticket_size, app_to_trans_ratio
         - net_cash_flow, inflow_outflow_ratio
         - trans_failure_rate, login_failure_rate, desktop_login_ratio, mobile_login_ratio
      2. customer_friction:
         - has_unresolved_complaint, friction_index, is_high_friction
         - complaint_velocity, has_fee_penalty
      3. product_bundling:
         - is_product_sweetspot, is_product_danger_zone
         - has_savings_account, has_closed_account, closed_account_ratio
         - savings_share, multi_account_holder
      4. risk_interactions:
         - wealth_deceleration_risk, germany_high_balance, is_middle_aged_vulnerable
         - salary_to_balance_ratio, inactive_high_balance
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

        # Safe helper for exposure months
        exp_m = out["exposure_months"] if "exposure_months" in out.columns else 6.0

        # ----------------------------------------------------------------------
        # 1. Behavioral Momentum & Velocity Features
        # ----------------------------------------------------------------------
        if self.enable_behavioral_momentum:
            # Monthly rates adjusted by customer exposure
            if "total_transactions" in out.columns:
                out["monthly_transactions"] = out["total_transactions"] / exp_m
                created.append("monthly_transactions")

            if "total_logins" in out.columns:
                out["monthly_logins"] = out["total_logins"] / exp_m
                created.append("monthly_logins")

            if "app_logins_total" in out.columns:
                out["app_logins_monthly"] = out["app_logins_total"] / exp_m
                created.append("app_logins_monthly")

            # Momentum trend ratios (Q3 vs Q2)
            if "trans_amount_3m" in out.columns and "trans_amount_prior_3m" in out.columns:
                out["trans_trend_ratio"] = out["trans_amount_3m"] / (out["trans_amount_prior_3m"] + 1.0)
                out["is_spending_collapsing"] = (out["trans_trend_ratio"] < 0.50).astype(int)
                created.extend(["trans_trend_ratio", "is_spending_collapsing"])

            if "trans_count_3m" in out.columns and "trans_count_prior_3m" in out.columns:
                out["trans_count_trend_ratio"] = out["trans_count_3m"] / (out["trans_count_prior_3m"] + 1.0)
                created.append("trans_count_trend_ratio")

            if "login_count_q3" in out.columns and "login_count_q2" in out.columns:
                out["login_trend_ratio"] = out["login_count_q3"] / (out["login_count_q2"] + 1.0)
                out["is_login_collapsing"] = (out["login_trend_ratio"] < 0.50).astype(int)
                created.extend(["login_trend_ratio", "is_login_collapsing"])

            # Interaction recency velocities
            if "recency_days" in out.columns:
                out["interaction_velocity"] = 1.0 / np.log1p(out["recency_days"].clip(lower=0))
                created.append("interaction_velocity")

            if "login_recency_days" in out.columns:
                out["login_velocity"] = 1.0 / np.log1p(out["login_recency_days"].clip(lower=0))
                created.append("login_velocity")

            # Ticket size and cross-channel engagement
            has_amt = "trans_amount_3m" in out.columns
            has_m_tx = "monthly_transactions" in out.columns
            if has_amt and has_m_tx:
                out["avg_ticket_size"] = out["trans_amount_3m"] / (3.0 * (out["monthly_transactions"] + 1.0))
                created.append("avg_ticket_size")

            has_app = "app_logins_monthly" in out.columns
            if has_app and has_m_tx:
                out["app_to_trans_ratio"] = out["app_logins_monthly"] / (out["monthly_transactions"] + 1.0)
                created.append("app_to_trans_ratio")

            # Financial flow balances
            if "total_inflow" in out.columns and "total_outflow" in out.columns:
                out["net_cash_flow"] = out["total_inflow"] - out["total_outflow"]
                out["inflow_outflow_ratio"] = out["total_inflow"] / (out["total_outflow"] + 1.0)
                created.extend(["net_cash_flow", "inflow_outflow_ratio"])

            # Failure and friction rates in channels
            if "failed_transactions" in out.columns and "total_tx_attempts" in out.columns:
                out["trans_failure_rate"] = out["failed_transactions"] / (out["total_tx_attempts"] + 1e-5)
                created.append("trans_failure_rate")

            if "failed_logins" in out.columns and "total_logins" in out.columns:
                out["login_failure_rate"] = out["failed_logins"] / (out["total_logins"] + 1e-5)
                created.append("login_failure_rate")

            if "desktop_logins_total" in out.columns and "total_logins" in out.columns:
                out["desktop_login_ratio"] = out["desktop_logins_total"] / (out["total_logins"] + 1e-5)
                out["mobile_login_ratio"] = out["mobile_logins_total"] / (out["total_logins"] + 1e-5)
                created.extend(["desktop_login_ratio", "mobile_login_ratio"])

        # ----------------------------------------------------------------------
        # 2. Customer Service Friction Features
        # ----------------------------------------------------------------------
        if self.enable_customer_friction:
            if "unresolved_complaints_count" in out.columns:
                out["has_unresolved_complaint"] = (out["unresolved_complaints_count"] > 0).astype(int)
                created.append("has_unresolved_complaint")
            elif "has_unresolved_complaint" not in out.columns:
                out["has_unresolved_complaint"] = 0

            has_unres = "has_unresolved_complaint" in out.columns
            has_cmps = "customer_service_complaints" in out.columns
            has_fail_tx = "trans_failure_rate" in out.columns

            fail_term = out["trans_failure_rate"] * 2.0 if has_fail_tx else 0.0

            if has_unres and has_cmps:
                out["friction_index"] = (
                    out["has_unresolved_complaint"] * 3.0
                    + out["customer_service_complaints"] * 1.5
                    + fail_term
                )
                out["is_high_friction"] = (
                    (out["has_unresolved_complaint"] == 1) | (out["customer_service_complaints"] >= 2)
                ).astype(int)
                created.extend(["friction_index", "is_high_friction"])
            elif has_unres:
                out["friction_index"] = out["has_unresolved_complaint"] * 3.0 + fail_term
                out["is_high_friction"] = (out["has_unresolved_complaint"] == 1).astype(int)
                created.extend(["friction_index", "is_high_friction"])
            elif has_cmps:
                out["friction_index"] = out["customer_service_complaints"] * 1.5 + fail_term
                out["is_high_friction"] = (out["customer_service_complaints"] >= 2).astype(int)
                created.extend(["friction_index", "is_high_friction"])

            if "complaint_recency_days" in out.columns:
                out["complaint_velocity"] = 1.0 / np.log1p(out["complaint_recency_days"].clip(lower=0))
                created.append("complaint_velocity")

            if "bank_fee_count" in out.columns:
                out["has_fee_penalty"] = (out["bank_fee_count"] > 0).astype(int)
                created.append("has_fee_penalty")

        # ----------------------------------------------------------------------
        # 3. Product Bundling & Account Portfolio
        # ----------------------------------------------------------------------
        if self.enable_product_bundling:
            if "products_number" in out.columns:
                out["is_product_sweetspot"] = (out["products_number"] == 2).astype(int)
                out["is_product_danger_zone"] = (out["products_number"] >= 3).astype(int)
                created.extend(["is_product_sweetspot", "is_product_danger_zone"])

            if "num_savings_accounts" in out.columns:
                out["has_savings_account"] = (out["num_savings_accounts"] > 0).astype(int)
                created.append("has_savings_account")

            if "num_closed_accounts" in out.columns:
                out["has_closed_account"] = (out["num_closed_accounts"] > 0).astype(int)
                created.append("has_closed_account")
                if "total_accounts" in out.columns:
                    out["closed_account_ratio"] = out["num_closed_accounts"] / np.maximum(out["total_accounts"], 1)
                    created.append("closed_account_ratio")

            if "savings_balance" in out.columns and "balance" in out.columns:
                out["savings_share"] = out["savings_balance"] / (np.maximum(0, out["balance"]) + 1.0)
                created.append("savings_share")

            if "total_accounts" in out.columns:
                out["multi_account_holder"] = (out["total_accounts"] > 1).astype(int)
                created.append("multi_account_holder")

        # ----------------------------------------------------------------------
        # 4. Wealth & Regional Risk Interaction Terms
        # ----------------------------------------------------------------------
        if self.enable_risk_interactions:
            trend_col = "trans_trend_ratio" if "trans_trend_ratio" in out.columns else None
            if "balance" in out.columns and trend_col:
                out["wealth_deceleration_risk"] = np.log1p(
                    np.maximum(0, out["balance"])
                ) * np.maximum(0, 1.0 - out[trend_col])
                created.append("wealth_deceleration_risk")

            if "country" in out.columns and "balance" in out.columns:
                is_germany = (out["country"] == "Germany").astype(int)
                out["germany_high_balance"] = is_germany * np.log1p(np.maximum(0, out["balance"]))
                created.append("germany_high_balance")

            if "age" in out.columns:
                out["is_middle_aged_vulnerable"] = ((out["age"] >= 45) & (out["age"] <= 60)).astype(int)
                created.append("is_middle_aged_vulnerable")

            if "balance" in out.columns and "estimated_salary" in out.columns:
                out["salary_to_balance_ratio"] = np.maximum(0, out["balance"]) / (out["estimated_salary"] + 1.0)
                created.append("salary_to_balance_ratio")

            if "active_member" in out.columns and "balance" in out.columns:
                out["inactive_high_balance"] = ((out["active_member"] == 0) & (out["balance"] > 0)).astype(int)
                created.append("inactive_high_balance")

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
