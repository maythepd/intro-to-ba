"""
Behavioral Momentum & Velocity Feature Engineering Module.
Engineers dynamic transaction and login momentum, velocity, cash flow,
and operational reliability metrics while preventing penalization of newly joined customers.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .base import BaseFeatureTransformer


class BehavioralMomentumEngineer(BaseFeatureTransformer):
    """
    Engineers behavioral momentum, velocity, cash flow, and channel engagement metrics.

    Features created:
      - is_new_customer: Binary indicator for customers who joined in Q3 (exposure <= 3m)
      - monthly_transactions: Total transaction frequency adjusted by customer exposure
      - monthly_logins: Total digital login frequency adjusted by customer exposure
      - app_logins_monthly: Mobile app login frequency adjusted by customer exposure
      - trans_trend_ratio: Ratio of Q3 spending vs Q2 spending (1.0 neutral baseline for new customers)
      - trans_count_trend_ratio: Ratio of Q3 transaction count vs Q2 transaction count (1.0 for new customers)
      - login_trend_ratio: Ratio of Q3 logins vs Q2 logins (1.0 for new customers)
      - is_spending_collapsing: Binary alert flag if spending dropped by > 50% (0 for new customers)
      - is_login_collapsing: Binary alert flag if digital engagement dropped by > 50% (0 for new customers)
      - interaction_velocity: Continuous velocity of financial activity (zero-division safe)
      - login_velocity: Continuous velocity of digital login engagement (zero-division safe)
      - avg_ticket_size: Average transaction volume per customer-initiated transaction in Q3
      - app_to_trans_ratio: Intensity of digital mobile engagement per financial transaction
      - net_cash_flow: Net external cash flow (total inflows minus total external outflows)
      - inflow_outflow_ratio: Ratio of money received vs money spent externally
      - trans_failure_rate: Operational failure rate across financial transactions
      - login_failure_rate: Operational failure rate across login attempts
      - desktop_login_ratio: Proportion of logins via desktop
      - mobile_login_ratio: Proportion of logins via mobile (iOS + Android)
    """

    def __init__(
        self,
        enabled: bool = True,
        spending_collapse_threshold: float = 0.50,
        login_collapse_threshold: float = 0.50,
        velocity_smoothing: float = 1.0,
        new_customer_exposure_months: float = 3.0,
        max_trend_ratio: float = 50.0,
        max_cash_flow_ratio: float = 50.0,
    ):
        super().__init__(name="BehavioralMomentumEngineer", enabled=enabled)
        self.spending_collapse_threshold = spending_collapse_threshold
        self.login_collapse_threshold = login_collapse_threshold
        self.velocity_smoothing = velocity_smoothing
        self.new_customer_exposure_months = new_customer_exposure_months
        self.max_trend_ratio = max_trend_ratio
        self.max_cash_flow_ratio = max_cash_flow_ratio

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BehavioralMomentumEngineer":
        """Stateless transformer; records fitting status."""
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()

        out = X.copy()
        created: List[str] = []

        # Safe helper for exposure months (defaulting to 6.0 months if not present)
        exp_m = out["exposure_months"] if "exposure_months" in out.columns else 6.0

        # 1. Identify newly joined customers (exposure <= 3 months, i.e. joined in Q3)
        # Per synth_data_dict.md: History before joining is unobserved, not zero activity.
        if "is_joined_q3" in out.columns:
            is_new_cust = out["is_joined_q3"] == 1
        else:
            is_new_cust = (exp_m <= self.new_customer_exposure_months)

        if "is_new_customer" not in out.columns:
            out["is_new_customer"] = is_new_cust.astype(int)
            created.append("is_new_customer")

        # 2. Exposure-adjusted monthly frequencies (engagement strictly measures successful usage)
        if "total_transactions" in out.columns:
            out["monthly_transactions"] = out["total_transactions"] / exp_m
            created.append("monthly_transactions")

        if "successful_logins" in out.columns:
            out["monthly_logins"] = out["successful_logins"] / exp_m
            created.append("monthly_logins")
        elif "total_logins" in out.columns:
            out["monthly_logins"] = out["total_logins"] / exp_m
            created.append("monthly_logins")

        if "app_logins_total" in out.columns:
            out["app_logins_monthly"] = out["app_logins_total"] / exp_m
            created.append("app_logins_monthly")

        # 3. Momentum trend ratios (Q3 vs Q2)
        # Normalize by exact days observed in each quarter (preventing observation length bias for mid-quarter joiners).
        # For new customers who joined in Q3, Q2 is unobserved. Set neutral ratio (1.0) and collapse flags to 0.
        has_obs_days = "days_observed_q2" in out.columns and "days_observed_q3" in out.columns
        d_q3 = np.maximum(out["days_observed_q3"], 1.0) if has_obs_days else 92.0
        d_q2 = np.maximum(out["days_observed_q2"], 1.0) if has_obs_days else 91.0
        is_unobserved_q2 = (out["days_observed_q2"] <= 0) if has_obs_days else is_new_cust

        if "trans_amount_3m" in out.columns and "trans_amount_prior_3m" in out.columns:
            daily_spend_q3 = out["trans_amount_3m"] / d_q3
            daily_spend_q2 = out["trans_amount_prior_3m"] / d_q2
            raw_trend = np.clip(daily_spend_q3 / (daily_spend_q2 + 0.01), 0.0, self.max_trend_ratio)
            out["trans_trend_ratio"] = np.where(is_new_cust | is_unobserved_q2, 1.0, raw_trend)
            out["is_spending_collapsing"] = np.where(
                is_new_cust | is_unobserved_q2, 0, (out["trans_trend_ratio"] < self.spending_collapse_threshold).astype(int)
            )
            created.extend(["trans_trend_ratio", "is_spending_collapsing"])

        if "trans_count_3m" in out.columns and "trans_count_prior_3m" in out.columns:
            daily_cnt_q3 = out["trans_count_3m"] / d_q3
            daily_cnt_q2 = out["trans_count_prior_3m"] / d_q2
            raw_count_trend = np.clip(daily_cnt_q3 / (daily_cnt_q2 + 0.01), 0.0, self.max_trend_ratio)
            out["trans_count_trend_ratio"] = np.where(is_new_cust | is_unobserved_q2, 1.0, raw_count_trend)
            created.append("trans_count_trend_ratio")

        if "login_count_q3" in out.columns and "login_count_q2" in out.columns:
            daily_log_q3 = out["login_count_q3"] / d_q3
            daily_log_q2 = out["login_count_q2"] / d_q2
            raw_login_trend = np.clip(daily_log_q3 / (daily_log_q2 + 0.01), 0.0, self.max_trend_ratio)
            out["login_trend_ratio"] = np.where(is_new_cust | is_unobserved_q2, 1.0, raw_login_trend)
            out["is_login_collapsing"] = np.where(
                is_new_cust | is_unobserved_q2, 0, (out["login_trend_ratio"] < self.login_collapse_threshold).astype(int)
            )
            created.extend(["login_trend_ratio", "is_login_collapsing"])

        # 4. Interaction recency velocities (Zero-division safe & smooth)
        if "recency_days" in out.columns:
            out["interaction_velocity"] = 1.0 / (out["recency_days"].clip(lower=0) + self.velocity_smoothing)
            created.append("interaction_velocity")

        if "login_recency_days" in out.columns:
            out["login_velocity"] = 1.0 / (out["login_recency_days"].clip(lower=0) + self.velocity_smoothing)
            created.append("login_velocity")

        # 5. Ticket size and cross-channel engagement
        has_amt = "trans_amount_3m" in out.columns
        has_cnt_3m = "trans_count_3m" in out.columns
        has_m_tx = "monthly_transactions" in out.columns

        if has_amt and has_cnt_3m:
            out["avg_ticket_size"] = np.where(
                out["trans_count_3m"] > 0,
                out["trans_amount_3m"] / out["trans_count_3m"],
                0.0,
            )
            created.append("avg_ticket_size")
        elif has_amt and has_m_tx:
            denom = 3.0 * out["monthly_transactions"]
            out["avg_ticket_size"] = np.where(
                denom > 0,
                out["trans_amount_3m"] / denom,
                0.0,
            )
            created.append("avg_ticket_size")

        has_app = "app_logins_monthly" in out.columns
        if has_app and has_m_tx:
            raw_app_trans = np.where(
                out["monthly_transactions"] > 0,
                out["app_logins_monthly"] / out["monthly_transactions"],
                out["app_logins_monthly"],
            )
            out["app_to_trans_ratio"] = np.clip(raw_app_trans, 0.0, self.max_trend_ratio)
            created.append("app_to_trans_ratio")

        # 6. Financial cash flows
        if "total_inflow" in out.columns and "total_outflow" in out.columns:
            out["net_cash_flow"] = out["total_inflow"] - out["total_outflow"]
            raw_in_out = np.where(
                out["total_outflow"] > 0,
                out["total_inflow"] / out["total_outflow"],
                np.where(out["total_inflow"] > 0, self.max_cash_flow_ratio, 1.0),
            )
            out["inflow_outflow_ratio"] = np.clip(raw_in_out, 0.0, self.max_cash_flow_ratio)
            created.extend(["net_cash_flow", "inflow_outflow_ratio"])

        # 7. Channel failure and device usage ratios
        if "failed_transactions" in out.columns and "total_tx_attempts" in out.columns:
            out["trans_failure_rate"] = np.where(
                out["total_tx_attempts"] > 0,
                np.clip(out["failed_transactions"] / out["total_tx_attempts"], 0.0, 1.0),
                0.0,
            )
            created.append("trans_failure_rate")

        if "failed_logins" in out.columns and "total_logins" in out.columns:
            out["login_failure_rate"] = np.where(
                out["total_logins"] > 0,
                np.clip(out["failed_logins"] / out["total_logins"], 0.0, 1.0),
                0.0,
            )
            created.append("login_failure_rate")

        # Device usage ratios: evaluated relative to successful logins (active usage sessions)
        dev_denom = out["successful_logins"] if "successful_logins" in out.columns else (
            out["total_logins"] if "total_logins" in out.columns else None
        )
        if "desktop_logins_total" in out.columns and dev_denom is not None:
            out["desktop_login_ratio"] = np.where(
                dev_denom > 0,
                np.clip(out["desktop_logins_total"] / dev_denom, 0.0, 1.0),
                0.0,
            )
            out["mobile_login_ratio"] = np.where(
                dev_denom > 0,
                np.clip(out["mobile_logins_total"] / dev_denom, 0.0, 1.0),
                0.0,
            )
            created.extend(["desktop_login_ratio", "mobile_login_ratio"])

        self.created_features = created
        return out

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "spending_collapse_threshold": self.spending_collapse_threshold,
            "login_collapse_threshold": self.login_collapse_threshold,
            "velocity_smoothing": self.velocity_smoothing,
            "new_customer_exposure_months": self.new_customer_exposure_months,
            "max_trend_ratio": self.max_trend_ratio,
            "max_cash_flow_ratio": self.max_cash_flow_ratio,
        })
        return params

