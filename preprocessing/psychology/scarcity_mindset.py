"""
Scarcity Mindset & Financial Survival Pressure Feature Engineering Module.
Theories: Mullainathan & Shafir (Scarcity: Bandwidth Tax & Cognitive Depletion, 2013);
Kaplan, Violante & Weidner (Hand-to-Mouth Consumers, 2014).
Engineers metrics measuring the customer's financial runway (survival cushion in months),
hand-to-mouth burn rate, and extreme sensitivity to bank errors when living paycheck-to-paycheck.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ..base import BaseFeatureTransformer


class ScarcityMindsetEngineer(BaseFeatureTransformer):
    """
    Features created:
      - psych_financial_runway_months: Number of months current balance can sustain outflows (capped [0, 24])
      - psych_is_runway_vulnerable: Binary alert for acute financial scarcity (< 1 month runway)
      - psych_hand_to_mouth_burn_rate: Outflow to inflow ratio (burn rate)
      - psych_paycheck_absorption_rate: Proportion of estimated earnings consumed by outflows
      - psych_scarcity_friction_interaction: Compound volatility (financial vulnerability x customer friction)
    """

    def __init__(
        self,
        enabled: bool = True,
        runway_cap_months: float = 24.0,
        max_burn_rate: float = 10.0,
    ):
        super().__init__(name="ScarcityMindsetEngineer", enabled=enabled)
        self.runway_cap_months = runway_cap_months
        self.max_burn_rate = max_burn_rate

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ScarcityMindsetEngineer":
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()

        out = X.copy()
        created: List[str] = []

        bal = np.maximum(0.0, out["balance"]) if "balance" in out.columns else pd.Series(0.0, index=out.index)
        tot_out = np.maximum(0.0, out["total_outflow"]) if "total_outflow" in out.columns else pd.Series(0.0, index=out.index)
        tot_in = np.maximum(0.0, out["total_inflow"]) if "total_inflow" in out.columns else pd.Series(0.0, index=out.index)
        exp_m = np.maximum(0.5, out["exposure_months"]) if "exposure_months" in out.columns else pd.Series(6.0, index=out.index)
        monthly_outflow = tot_out / exp_m

        # 1. Financial Runway Months (How long before funds run out at current spending rate)
        # Low runway (< 1 month) induces severe cognitive bandwidth tax and zero patience for bank mistakes
        runway = np.clip(bal / (monthly_outflow + 1.0), 0.0, self.runway_cap_months)
        out["psych_financial_runway_months"] = runway
        created.append("psych_financial_runway_months")

        # 2. Runway Vulnerability Flag (< 1 month buffer)
        out["psych_is_runway_vulnerable"] = (runway < 1.0).astype(float)
        created.append("psych_is_runway_vulnerable")

        # 3. Hand-to-Mouth Burn Rate (Spending outpacing incoming funds)
        burn_rate = np.clip(tot_out / (tot_in + 1.0), 0.0, self.max_burn_rate)
        out["psych_hand_to_mouth_burn_rate"] = burn_rate
        created.append("psych_hand_to_mouth_burn_rate")

        # 4. Paycheck Absorption Rate (Outflows relative to estimated earnings across exposure)
        if "estimated_salary" in out.columns:
            sal = np.maximum(1.0, out["estimated_salary"])
            expected_earnings = (sal / 12.0) * exp_m
            absorption = np.clip(tot_out / (expected_earnings + 1.0), 0.0, 5.0)
            out["psych_paycheck_absorption_rate"] = absorption
            created.append("psych_paycheck_absorption_rate")
        else:
            out["psych_paycheck_absorption_rate"] = 0.0
            created.append("psych_paycheck_absorption_rate")

        # 5. Scarcity Friction Interaction: When broke customers experience complaints or fees
        # They react far more aggressively than wealthy, insulated customers
        cmps = np.maximum(0.0, out["customer_service_complaints"]) if "customer_service_complaints" in out.columns else 0.0
        fee_cnt = (out["bank_fee_count"] > 0).astype(float) if "bank_fee_count" in out.columns else 0.0
        out["psych_scarcity_friction_interaction"] = (runway < 1.0).astype(float) * (cmps + fee_cnt)
        created.append("psych_scarcity_friction_interaction")

        self.created_features = created
        return out
