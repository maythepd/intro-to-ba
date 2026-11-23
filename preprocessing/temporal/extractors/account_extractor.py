"""
Account Structural Dynamics Extractor.
Extracts 4 longitudinal channels capturing active account holding stock,
premature savings liquidation, everyday account closure, and manual renewal friction.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

from .base import BaseSubExtractor


class AccountSnapshotExtractor(BaseSubExtractor):
    """
    Extracts multi-snapshot account portfolio and closure dynamics.
    """

    FEATURE_NAMES: List[str] = [
        "acc_active_count",
        "acc_savings_closed_in_month",
        "acc_everyday_closed_in_month",
        "acc_manual_renewal_ratio",
    ]

    MONTH_DAYS: Dict[int, int] = {4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30}

    def extract(
        self,
        customer_ids: np.ndarray,
        months: List[int],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        idx = customer_ids
        if context is None:
            context = {}

        acc_path = self._resolve_file("accounts.csv")
        acc_df = pd.read_csv(
            acc_path,
            usecols=[
                "account_id",
                "customer_id",
                "account_type",
                "created_at",
                "closed_at",
                "renewal_type",
            ],
        )

        acc_df["created_dt"] = pd.to_datetime(acc_df["created_at"], errors="coerce")
        acc_df["closed_dt"] = pd.to_datetime(acc_df["closed_at"], errors="coerce")

        # Also store acct_map into context if not present for other extractors
        if "acct_map" not in context:
            context["acct_map"] = dict(zip(acc_df["account_id"], acc_df["customer_id"]))

        active_counts_m = []
        savings_closed_m = []
        everyday_closed_m = []
        manual_ratio_m = []

        for m in months:
            days = self.MONTH_DAYS.get(m, 30)
            month_end = pd.Timestamp(2026, m, days, 23, 59, 59)

            # 1. Active accounts at month snapshot boundary
            is_active = (acc_df["created_dt"] <= month_end) & (
                acc_df["closed_dt"].isna() | (acc_df["closed_dt"] > month_end)
            )
            act_m = acc_df[is_active]
            c_act = (
                act_m.groupby("customer_id")
                .size()
                .reindex(idx, fill_value=0.0)
                .values.astype(np.float32)
            )
            active_counts_m.append(c_act)

            # 2 & 3. Closures in month m
            is_closed_m = (
                (acc_df["closed_dt"].dt.year == 2026)
                & (acc_df["closed_dt"].dt.month == m)
            )

            # Savings closures (capital flight indicator)
            sav_closed = acc_df[is_closed_m & (acc_df["account_type"] == "savings")]
            c_sav = (
                sav_closed.groupby("customer_id")
                .size()
                .reindex(idx, fill_value=0.0)
                .values.astype(np.float32)
            )
            savings_closed_m.append(c_sav)

            # Everyday closures
            eve_closed = acc_df[is_closed_m & (acc_df["account_type"] == "everyday")]
            c_eve = (
                eve_closed.groupby("customer_id")
                .size()
                .reindex(idx, fill_value=0.0)
                .values.astype(np.float32)
            )
            everyday_closed_m.append(c_eve)

            # 4. Manual renewal ratio among active savings accounts
            sav_act = act_m[act_m["account_type"] == "savings"]
            tot_sav = sav_act.groupby("customer_id").size()
            man_sav = (
                sav_act[sav_act["renewal_type"] == "manual"]
                .groupby("customer_id")
                .size()
            )
            ratio = (man_sav / tot_sav).reindex(idx, fill_value=0.0).fillna(0.0).values.astype(np.float32)
            manual_ratio_m.append(ratio)

        c1 = np.column_stack(active_counts_m).astype(np.float32)
        c2 = np.column_stack(savings_closed_m).astype(np.float32)
        c3 = np.column_stack(everyday_closed_m).astype(np.float32)
        c4 = np.column_stack(manual_ratio_m).astype(np.float32)

        # Store in context if needed
        context["acc_active_cnt"] = c1
        context["acc_savings_closed"] = c2

        # Numerical bounds
        c1 = self.safe_clip(c1, 0.0, 10.0)
        c2 = self.safe_clip(c2, 0.0, 10.0)
        c3 = self.safe_clip(c3, 0.0, 10.0)
        c4 = self.safe_clip(c4, 0.0, 1.0)

        channels = [c1, c2, c3, c4]
        tensor_slice = np.stack(channels, axis=-1).astype(np.float32)

        return tensor_slice, self.FEATURE_NAMES
