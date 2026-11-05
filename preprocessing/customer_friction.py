"""
Customer Service Friction Feature Engineering Module.
Engineers indicators of operational and service friction (complaints, resolution status,
operational banking fees, and dispute velocity) without arbitrary hardcoding or biased imputation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .base import BaseFeatureTransformer


class CustomerFrictionEngineer(BaseFeatureTransformer):
    """
    Engineers customer service friction, complaint urgency, and fee impact metrics.

    Features created:
      - has_unresolved_complaint: Binary flag indicating active dispute at snapshot
      - has_resolved_complaint: Binary indicator of past resolved dispute
      - unresolved_complaint_ratio: Proportion of customer complaints left unresolved
      - complaint_velocity: Recency velocity of complaints (0.0 for customers without complaints)
      - friction_index: Composite friction index with parameterized component weights
      - is_high_friction: Alert indicator for severe customer dissatisfaction
      - has_fee_penalty: Binary flag indicating customer was charged bank fees
      - fee_burden_ratio: Ratio of bank fees to total external financial outflows
    """

    def __init__(
        self,
        enabled: bool = True,
        unresolved_weight: float = 3.0,
        complaint_count_weight: float = 1.5,
        tx_failure_weight: float = 2.0,
        high_friction_complaints_thresh: int = 2,
        velocity_smoothing: float = 1.0,
    ):
        super().__init__(name="CustomerFrictionEngineer", enabled=enabled)
        self.unresolved_weight = unresolved_weight
        self.complaint_count_weight = complaint_count_weight
        self.tx_failure_weight = tx_failure_weight
        self.high_friction_complaints_thresh = high_friction_complaints_thresh
        self.velocity_smoothing = velocity_smoothing

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CustomerFrictionEngineer":
        """Stateless transformer; records fitting status."""
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()

        out = X.copy()
        created: List[str] = []

        # 1. Unresolved and Resolved complaint indicators
        if "unresolved_complaints_count" in out.columns:
            out["has_unresolved_complaint"] = (out["unresolved_complaints_count"] > 0).astype(int)
            created.append("has_unresolved_complaint")
        elif "has_unresolved_complaint" not in out.columns:
            out["has_unresolved_complaint"] = 0

        if "resolved_complaints_count" in out.columns:
            out["has_resolved_complaint"] = (out["resolved_complaints_count"] > 0).astype(int)
            created.append("has_resolved_complaint")

        # 2. Unresolved complaint ratio (strictly bounded in [0.0, 1.0])
        if "unresolved_complaints_count" in out.columns and "customer_service_complaints" in out.columns:
            out["unresolved_complaint_ratio"] = np.where(
                out["customer_service_complaints"] > 0,
                np.clip(out["unresolved_complaints_count"] / out["customer_service_complaints"], 0.0, 1.0),
                0.0,
            )
            created.append("unresolved_complaint_ratio")

        # 3. Parameterized Friction Index & High Friction Flag
        has_unres = "has_unresolved_complaint" in out.columns
        has_cmps = "customer_service_complaints" in out.columns
        has_fail_tx = "trans_failure_rate" in out.columns

        fail_term = out["trans_failure_rate"] * self.tx_failure_weight if has_fail_tx else 0.0

        if has_unres and has_cmps:
            out["friction_index"] = (
                out["has_unresolved_complaint"] * self.unresolved_weight
                + out["customer_service_complaints"] * self.complaint_count_weight
                + fail_term
            )
            out["is_high_friction"] = (
                (out["has_unresolved_complaint"] == 1)
                | (out["customer_service_complaints"] >= self.high_friction_complaints_thresh)
            ).astype(int)
            created.extend(["friction_index", "is_high_friction"])
        elif has_unres:
            out["friction_index"] = out["has_unresolved_complaint"] * self.unresolved_weight + fail_term
            out["is_high_friction"] = (out["has_unresolved_complaint"] == 1).astype(int)
            created.extend(["friction_index", "is_high_friction"])
        elif has_cmps:
            out["friction_index"] = out["customer_service_complaints"] * self.complaint_count_weight + fail_term
            out["is_high_friction"] = (
                out["customer_service_complaints"] >= self.high_friction_complaints_thresh
            ).astype(int)
            created.extend(["friction_index", "is_high_friction"])

        # 4. Complaint Velocity (Clean 0.0 for customers who never filed a complaint)
        if "complaint_recency_days" in out.columns:
            has_cmp = (out["customer_service_complaints"] > 0) if "customer_service_complaints" in out.columns else False
            raw_cmp_vel = 1.0 / (out["complaint_recency_days"].clip(lower=0) + self.velocity_smoothing)
            out["complaint_velocity"] = np.where(has_cmp, raw_cmp_vel, 0.0)
            created.append("complaint_velocity")

        # 5. Fee penalties and fee burden
        if "bank_fee_count" in out.columns:
            out["has_fee_penalty"] = (out["bank_fee_count"] > 0).astype(int)
            created.append("has_fee_penalty")

        if "bank_fee_amount" in out.columns and "total_outflow" in out.columns:
            out["fee_burden_ratio"] = np.where(
                out["total_outflow"] > 0,
                np.clip(out["bank_fee_amount"] / out["total_outflow"], 0.0, 1.0),
                0.0,
            )
            created.append("fee_burden_ratio")

        self.created_features = created
        return out

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "unresolved_weight": self.unresolved_weight,
            "complaint_count_weight": self.complaint_count_weight,
            "tx_failure_weight": self.tx_failure_weight,
            "high_friction_complaints_thresh": self.high_friction_complaints_thresh,
            "velocity_smoothing": self.velocity_smoothing,
        })
        return params

