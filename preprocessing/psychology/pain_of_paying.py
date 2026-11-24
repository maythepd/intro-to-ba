"""
Pain of Paying & Loss Sensitivity Feature Engineering Module.
Theories: Prelec & Loewenstein (Pain of Paying, 1998); Kahneman & Tversky (Loss Aversion & Prospect Theory, 1979).
Engineers metrics capturing behavioral reactions to penalty fees (punishment retaliation),
cash decoupling (withdrawing physical cash to detach from the bank), and disproportionate loss burden.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ..base import BaseFeatureTransformer


class PainOfPayingEngineer(BaseFeatureTransformer):
    """
    Features created:
      - psych_fee_retaliation_score: Spending curtailment directly following bank penalty fees
      - psych_fee_burden_to_balance: Psychological pain of fees relative to current wealth [%]
      - psych_cash_detachment_ratio: Preference for cash withdrawals over electronic payments [0, 1]
      - psych_loss_aversion_drain_index: Net capital drain relative to income (loss salience)
      - psych_fee_frequency_shock: Ratio of fee occurrences to total financial transactions
    """

    def __init__(
        self,
        enabled: bool = True,
        max_fee_burden_pct: float = 100.0,
        max_loss_drain_ratio: float = 5.0,
    ):
        super().__init__(name="PainOfPayingEngineer", enabled=enabled)
        self.max_fee_burden_pct = max_fee_burden_pct
        self.max_loss_drain_ratio = max_loss_drain_ratio

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PainOfPayingEngineer":
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()

        out = X.copy()
        created: List[str] = []

        bal = np.maximum(0.0, out["balance"]) if "balance" in out.columns else pd.Series(0.0, index=out.index)
        fee_amt = np.maximum(0.0, out["bank_fee_amount"]) if "bank_fee_amount" in out.columns else pd.Series(0.0, index=out.index)
        fee_cnt = np.maximum(0.0, out["bank_fee_count"]) if "bank_fee_count" in out.columns else pd.Series(0.0, index=out.index)
        tot_out = np.maximum(0.0, out["total_outflow"]) if "total_outflow" in out.columns else pd.Series(0.0, index=out.index)
        tot_in = np.maximum(0.0, out["total_inflow"]) if "total_inflow" in out.columns else pd.Series(0.0, index=out.index)

        # 1. Fee Retaliation Score (Customer cuts spending in protest after being hit by bank fees)
        t3m = np.maximum(0.0, out["trans_amount_3m"]) if "trans_amount_3m" in out.columns else 0.0
        tprior = np.maximum(0.0, out["trans_amount_prior_3m"]) if "trans_amount_prior_3m" in out.columns else 0.0
        spending_drop = np.maximum(0.0, 1.0 - (t3m / (tprior + 1.0)))
        has_fee = (fee_cnt > 0).astype(float)
        out["psych_fee_retaliation_score"] = has_fee * spending_drop
        created.append("psych_fee_retaliation_score")

        # 2. Fee Burden to Balance (Emotional pain of fees on low-balance accounts)
        out["psych_fee_burden_to_balance"] = np.clip((fee_amt / (bal + 1.0)) * 100.0, 0.0, self.max_fee_burden_pct)
        created.append("psych_fee_burden_to_balance")

        # 3. Cash Detachment Ratio (Withdrawing cash removes digital money from the bank)
        cash_amt = np.maximum(0.0, out["cash_withdrawal_amount"]) if "cash_withdrawal_amount" in out.columns else 0.0
        out["psych_cash_detachment_ratio"] = np.clip(cash_amt / (tot_out + 1.0), 0.0, 1.0)
        created.append("psych_cash_detachment_ratio")

        # 4. Loss Aversion Drain Index: Net negative cash flow scaled by estimated salary
        sal = np.maximum(1.0, out["estimated_salary"]) if "estimated_salary" in out.columns else pd.Series(100000.0, index=out.index)
        net_drain = np.maximum(0.0, tot_out - tot_in)
        out["psych_loss_aversion_drain_index"] = np.clip(net_drain / sal, 0.0, self.max_loss_drain_ratio)
        created.append("psych_loss_aversion_drain_index")

        # 5. Fee Frequency Shock: Ratio of penalties to legitimate transactions
        tot_tx = np.maximum(1.0, out["total_transactions"]) if "total_transactions" in out.columns else pd.Series(1.0, index=out.index)
        out["psych_fee_frequency_shock"] = np.clip(fee_cnt / tot_tx, 0.0, 1.0)
        created.append("psych_fee_frequency_shock")

        self.created_features = created
        return out
