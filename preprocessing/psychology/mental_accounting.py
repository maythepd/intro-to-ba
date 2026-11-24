"""
Mental Accounting & Commitment Violation Feature Engineering Module.
Theories: Richard Thaler (Mental Accounting, 1985, 1999); Samuelson & Zeckhauser (Status Quo Bias, 1988).
Engineers metrics capturing non-fungible categorization of funds between Everyday spending and Term Savings,
as well as psychological commitment breaches (early withdrawals) and renewal inertia.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ..base import BaseFeatureTransformer


class MentalAccountingEngineer(BaseFeatureTransformer):
    """
    Features created:
      - psych_savings_commitment_ratio: Share of committed savings vs total balance [0, 1]
      - psych_early_break_propensity: Propensity to break term savings commitment early, taking penalty
      - psych_renewal_inertia_status: Status quo bias indicator (1: auto sticky, -1: manual broken, 0: neutral)
      - psych_everyday_drain_ratio: Proportion of balance sitting in everyday liquid account vs savings [0, 1]
      - psych_savings_to_salary_ratio: Accumulated savings relative to annual salary (safety cushion)
    """

    def __init__(
        self,
        enabled: bool = True,
        max_salary_ratio: float = 10.0,
    ):
        super().__init__(name="MentalAccountingEngineer", enabled=enabled)
        self.max_salary_ratio = max_salary_ratio

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MentalAccountingEngineer":
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()

        out = X.copy()
        created: List[str] = []

        bal = np.maximum(0.0, out["balance"]) if "balance" in out.columns else pd.Series(0.0, index=out.index)
        sav_bal = np.maximum(0.0, out["savings_balance"]) if "savings_balance" in out.columns else pd.Series(0.0, index=out.index)
        ev_bal = np.maximum(0.0, out["everyday_balance"]) if "everyday_balance" in out.columns else pd.Series(0.0, index=out.index)

        # 1. Savings commitment ratio (Share of total balance locked in mental 'savings bucket')
        # Bounded strictly in [0.0, 1.0]
        out["psych_savings_commitment_ratio"] = np.clip(sav_bal / (bal + 1.0), 0.0, 1.0)
        created.append("psych_savings_commitment_ratio")

        # 2. Early withdrawal break propensity (Breaking psychological term contract prematurely)
        if "avg_early_withdrawal_rate" in out.columns and "num_closed_accounts" in out.columns:
            tot_ever = out["total_accounts_ever"] if "total_accounts_ever" in out.columns else (out["total_accounts"] if "total_accounts" in out.columns else 1.0)
            closed_ratio = np.clip(out["num_closed_accounts"] / np.maximum(1.0, tot_ever), 0.0, 1.0)
            has_early_rate = (out["avg_early_withdrawal_rate"] > 0).astype(float)
            out["psych_early_break_propensity"] = has_early_rate * closed_ratio
            created.append("psych_early_break_propensity")
        else:
            out["psych_early_break_propensity"] = 0.0
            created.append("psych_early_break_propensity")

        # 3. Renewal inertia status (Status Quo Bias vs Deliberate Liquidation)
        # 1.0: Auto renewal (relies on default inertia)
        # -1.0: Manual renewal + closed accounts (actively refused renewal / broke relationship)
        # 0.0: Neutral / No term savings
        has_auto = (out["has_auto_renewal"] == 1) if "has_auto_renewal" in out.columns else pd.Series(False, index=out.index)
        has_man = (out["has_manual_renewal"] == 1) if "has_manual_renewal" in out.columns else pd.Series(False, index=out.index)
        has_closed = (out["num_closed_accounts"] > 0) if "num_closed_accounts" in out.columns else pd.Series(False, index=out.index)

        status = np.zeros(len(out), dtype=float)
        status[has_auto] = 1.0
        status[has_man & has_closed] = -1.0
        out["psych_renewal_inertia_status"] = status
        created.append("psych_renewal_inertia_status")

        # 4. Everyday drain ratio (Mental bucket allocation towards immediate liquid spending)
        out["psych_everyday_drain_ratio"] = np.clip(ev_bal / (bal + 1.0), 0.0, 1.0)
        created.append("psych_everyday_drain_ratio")

        # 5. Committed savings to salary ratio (Accumulated wealth security buffer)
        if "estimated_salary" in out.columns:
            sal = np.maximum(1.0, out["estimated_salary"])
            out["psych_savings_to_salary_ratio"] = np.clip(sav_bal / sal, 0.0, self.max_salary_ratio)
            created.append("psych_savings_to_salary_ratio")
        else:
            out["psych_savings_to_salary_ratio"] = 0.0
            created.append("psych_savings_to_salary_ratio")

        self.created_features = created
        return out
