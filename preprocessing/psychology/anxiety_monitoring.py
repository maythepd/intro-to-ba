"""
Anxiety Monitoring & Phantom Engagement Feature Engineering Module.
Theories: Wendy Wood & David Neal (Habit Disruption, 2009); Anxious Surveillance & Vigilance Behavior.
Engineers metrics capturing disproportionate account surveillance without financial activity (phantom logins),
anxious balance checking while freezing spending, and channel panic migrations.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ..base import BaseFeatureTransformer


class AnxietyMonitoringEngineer(BaseFeatureTransformer):
    """
    Features created:
      - psych_phantom_login_ratio: Frequency of logins per financial action (surveillance intensity)
      - psych_login_to_tx_intensity: Ratio of digital logins to transactions
      - psych_anxious_checking_index: Interaction of high login volume with collapsing spend (hyper-vigilance)
      - psych_panic_channel_migration: Proportion of desktop logins when digital engagement shifts
      - psych_login_recency_decay: Exponential recency of digital app engagement
    """

    def __init__(
        self,
        enabled: bool = True,
        login_recency_scale: float = 15.0,
        max_phantom_ratio: float = 50.0,
    ):
        super().__init__(name="AnxietyMonitoringEngineer", enabled=enabled)
        self.login_recency_scale = max(1.0, float(login_recency_scale))
        self.max_phantom_ratio = max_phantom_ratio

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AnxietyMonitoringEngineer":
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()

        out = X.copy()
        created: List[str] = []

        logins = np.maximum(0.0, out["successful_logins"]) if "successful_logins" in out.columns else (
            np.maximum(0.0, out["total_logins"]) if "total_logins" in out.columns else pd.Series(0.0, index=out.index)
        )
        tx_cust = np.maximum(0.0, out["customer_initiated_tx_count"]) if "customer_initiated_tx_count" in out.columns else (
            np.maximum(0.0, out["total_transactions"]) if "total_transactions" in out.columns else pd.Series(0.0, index=out.index)
        )

        # 1. Phantom Login Ratio (Logins without initiating spending - Anxious monitoring)
        out["psych_phantom_login_ratio"] = np.clip(logins / (tx_cust + 1.0), 0.0, self.max_phantom_ratio)
        created.append("psych_phantom_login_ratio")

        # 2. Login to Transaction Intensity (Monthly or total scale)
        if "monthly_logins" in out.columns and "monthly_transactions" in out.columns:
            m_log = np.maximum(0.0, out["monthly_logins"])
            m_tx = np.maximum(0.0, out["monthly_transactions"])
            out["psych_login_to_tx_intensity"] = np.clip(m_log / (m_tx + 1.0), 0.0, self.max_phantom_ratio)
            created.append("psych_login_to_tx_intensity")
        else:
            out["psych_login_to_tx_intensity"] = out["psych_phantom_login_ratio"]
            created.append("psych_login_to_tx_intensity")

        # 3. Anxious Checking Index: High logins accompanied by sudden spending collapse
        # Customer logs in repeatedly to watch dwindling funds or audit charges
        t3m = np.maximum(0.0, out["trans_amount_3m"]) if "trans_amount_3m" in out.columns else 0.0
        tprior = np.maximum(0.0, out["trans_amount_prior_3m"]) if "trans_amount_prior_3m" in out.columns else 0.0
        
        spending_drop = np.maximum(0.0, 1.0 - (t3m / (tprior + 1.0)))
        out["psych_anxious_checking_index"] = np.log1p(logins) * spending_drop
        created.append("psych_anxious_checking_index")

        # 4. Panic Channel Migration (Proportion of Desktop logins)
        # In a mobile-first banking environment, sudden desktop usage often signals administrative actions or dispute escalation
        if "desktop_logins_total" in out.columns and "total_logins" in out.columns:
            desk = np.maximum(0.0, out["desktop_logins_total"])
            tot_log = np.maximum(1.0, out["total_logins"])
            out["psych_panic_channel_migration"] = np.clip(desk / tot_log, 0.0, 1.0)
            created.append("psych_panic_channel_migration")
        else:
            out["psych_panic_channel_migration"] = 0.0
            created.append("psych_panic_channel_migration")

        # 5. Login Recency Decay: Exponential recency of digital touchpoints
        if "login_recency_days" in out.columns:
            rec_days = np.clip(out["login_recency_days"], 0.0, 365.0)
            out["psych_login_recency_decay"] = np.exp(-rec_days / self.login_recency_scale)
            created.append("psych_login_recency_decay")
        else:
            out["psych_login_recency_decay"] = 1.0
            created.append("psych_login_recency_decay")

        self.created_features = created
        return out
