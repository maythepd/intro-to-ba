"""
Peak-End Rule & Catastrophic Friction Feature Engineering Module.
Theories: Daniel Kahneman (Peak-End Rule & Duration Neglect, 1993); Richard Oliver (Expectation Disconfirmation, 1980).
Engineers non-linear metrics where acute recent friction (End) and severe operational failures (Peak)
disproportionately dominate overall memory and churn intent.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ..base import BaseFeatureTransformer


class PeakEndFrictionEngineer(BaseFeatureTransformer):
    """
    Features created:
      - psych_exponential_recency_complaint: Time-decayed complaint intensity (recent complaints weigh 10x more)
      - psych_unresolved_expectation_gap: Unresolved dispute multiplied by excessive resolution delay
      - psych_terminal_failure_shock: Compound catastrophe score (unresolved dispute + operational failure)
      - psych_peak_friction_intensity: Severity of operational failures relative to transaction attempts
      - psych_complaint_recency_velocity: Inverse recency of customer disputes (hyperbolic recency)
    """

    def __init__(
        self,
        enabled: bool = True,
        recency_half_life_days: float = 30.0,
        max_expectation_gap: float = 50.0,
    ):
        super().__init__(name="PeakEndFrictionEngineer", enabled=enabled)
        self.recency_half_life_days = max(1.0, float(recency_half_life_days))
        self.max_expectation_gap = max_expectation_gap

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PeakEndFrictionEngineer":
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()

        out = X.copy()
        created: List[str] = []

        cmps = np.maximum(0.0, out["customer_service_complaints"]) if "customer_service_complaints" in out.columns else pd.Series(0.0, index=out.index)
        unres = np.maximum(0.0, out["unresolved_complaints_count"]) if "unresolved_complaints_count" in out.columns else pd.Series(0.0, index=out.index)
        fail_tx = np.maximum(0.0, out["failed_transactions"]) if "failed_transactions" in out.columns else pd.Series(0.0, index=out.index)
        fail_log = np.maximum(0.0, out["failed_logins"]) if "failed_logins" in out.columns else pd.Series(0.0, index=out.index)
        tx_att = np.maximum(1.0, out["total_tx_attempts"]) if "total_tx_attempts" in out.columns else (
            np.maximum(1.0, out["total_transactions"]) if "total_transactions" in out.columns else pd.Series(1.0, index=out.index)
        )
        exp_m = np.maximum(1.0, out["exposure_months"]) if "exposure_months" in out.columns else pd.Series(6.0, index=out.index)

        # 1. Exponential Recency Complaint Impact (Peak-End Rule: Recent dispute dominates memory)
        if "complaint_recency_days" in out.columns:
            rec_days = np.clip(out["complaint_recency_days"], 0.0, 365.0)
            rec_decay = np.exp(-rec_days / self.recency_half_life_days)
            out["psych_exponential_recency_complaint"] = cmps * rec_decay
            created.append("psych_exponential_recency_complaint")

            # Hyperbolic recency velocity (1 / (recency + 1)) strictly when complaints exist
            out["psych_complaint_recency_velocity"] = np.where(cmps > 0, 1.0 / (rec_days + 1.0), 0.0)
            created.append("psych_complaint_recency_velocity")
        else:
            out["psych_exponential_recency_complaint"] = cmps
            out["psych_complaint_recency_velocity"] = 0.0
            created.extend(["psych_exponential_recency_complaint", "psych_complaint_recency_velocity"])

        # 2. Unresolved Expectation Gap: Resolution delay outlasting patience
        if "avg_resolution_delay_days" in out.columns:
            delay = np.maximum(0.0, out["avg_resolution_delay_days"])
            raw_gap = unres * (delay / exp_m)
            out["psych_unresolved_expectation_gap"] = np.clip(raw_gap, 0.0, self.max_expectation_gap)
            created.append("psych_unresolved_expectation_gap")
        else:
            out["psych_unresolved_expectation_gap"] = unres
            created.append("psych_unresolved_expectation_gap")

        # 3. Terminal Failure Shock: Compound catastrophe ("The straw that breaks the camel's back")
        # Combines an active unresolved complaint with operational friction
        has_unres = (unres > 0).astype(float)
        compound_fails = fail_tx + fail_log
        out["psych_terminal_failure_shock"] = has_unres * np.log1p(compound_fails)
        created.append("psych_terminal_failure_shock")

        # 4. Peak Friction Intensity (Rate of friction incidents across attempts)
        out["psych_peak_friction_intensity"] = np.clip((cmps * 2.0 + fail_tx) / tx_att, 0.0, 10.0)
        created.append("psych_peak_friction_intensity")

        self.created_features = created
        return out
