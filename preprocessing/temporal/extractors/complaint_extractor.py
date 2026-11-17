"""
Customer Complaints Extractor.
Extracts 5 longitudinal channels capturing complaint volume, resolution delay,
service SLA breaches, unresolved complaint stock, and monthly escalation velocity.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

from .base import BaseSubExtractor


class ComplaintSnapshotExtractor(BaseSubExtractor):
    """
    Extracts multi-snapshot customer dispute and service friction metrics.
    """

    FEATURE_NAMES: List[str] = [
        "complaint_count_log",
        "complaint_avg_delay_days",
        "complaint_sla_breach_count",
        "complaint_unresolved_stock",
        "complaint_escalation_velocity",
    ]

    LEGACY_FEATURE_NAMES: List[str] = [
        "complaints_count",
    ]

    def extract_legacy(
        self,
        customer_ids: np.ndarray,
        months: List[int],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extracts the 1 original legacy complaint count channel."""
        if context is None:
            context = {}
        self.extract(customer_ids, months, context)
        cmp_cnt = context["cmp_cnt_raw"].astype(np.float32)
        legacy_tensor = cmp_cnt[:, :, np.newaxis]
        return legacy_tensor, self.LEGACY_FEATURE_NAMES

    def extract(
        self,
        customer_ids: np.ndarray,
        months: List[int],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        idx = customer_ids
        if context is None:
            context = {}

        cmp_path = self._resolve_file("complaints.csv")
        cmp_df = pd.read_csv(
            cmp_path,
            usecols=["customer_id", "complaint_datetime", "resolved_datetime"],
        )

        cmp_df["month"] = (
            pd.to_numeric(cmp_df["complaint_datetime"].str.slice(5, 7), errors="coerce")
            .fillna(0)
            .astype(int)
        )
        cmp_df["is_unresolved"] = cmp_df["resolved_datetime"].isna().astype(int)

        t_comp = pd.to_datetime(cmp_df["complaint_datetime"], errors="coerce")
        t_res = pd.to_datetime(cmp_df["resolved_datetime"], errors="coerce")
        cmp_df["delay_days"] = (t_res - t_comp).dt.total_seconds() / (24 * 3600)
        cmp_df["delay_days"] = cmp_df["delay_days"].fillna(30.0)  # Unresolved defaulted to 30 days penalty

        cmp_df["is_sla_breach"] = (
            (cmp_df["delay_days"] > 7.0) | (cmp_df["is_unresolved"] == 1)
        ).astype(int)

        # 1. Total monthly complaints count
        cmp_cnt_df = (
            cmp_df.groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        cmp_cnt = cmp_cnt_df.values.astype(np.float32)
        context["cmp_cnt_raw"] = cmp_cnt

        # 2. Average resolution delay in days (safe log1p)
        delay_df = (
            cmp_df.groupby(["customer_id", "month"])["delay_days"]
            .mean()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        avg_delay = delay_df.values.astype(np.float32)

        # 3. SLA breaches count (>7 days or unresolved)
        sla_df = (
            cmp_df[cmp_df["is_sla_breach"] == 1]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        sla_breach_cnt = sla_df.values.astype(np.float32)

        # 4. Unresolved complaints stock
        unres_df = (
            cmp_df[cmp_df["is_unresolved"] == 1]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        # Cumulative unresolved complaints stock across months
        unres_stock = np.cumsum(unres_df.values.astype(np.float32), axis=1)

        # 5. Escalation velocity: delta complaints vs prior month (first month delta is 0)
        padded = np.column_stack([cmp_cnt[:, :1], cmp_cnt])
        escalation_velocity = (padded[:, 1:] - padded[:, :-1]).astype(np.float32)

        # Transform and clip
        c1 = self.safe_log1p(cmp_cnt)
        c2 = self.safe_log1p(avg_delay)
        c3 = self.safe_log1p(sla_breach_cnt)
        c4 = self.safe_log1p(unres_stock)
        c5 = self.safe_clip(escalation_velocity, -10.0, 10.0)

        channels = [c1, c2, c3, c4, c5]
        tensor_slice = np.stack(channels, axis=-1).astype(np.float32)

        return tensor_slice, self.FEATURE_NAMES
