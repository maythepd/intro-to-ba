"""
Transaction Behavioral Extractor.
Extracts 11 longitudinal channels capturing cash flow, transaction volume,
external capital flight, friction fees, and physical branch escalations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

from .base import BaseSubExtractor


class TransactionSnapshotExtractor(BaseSubExtractor):
    """
    Extracts multi-snapshot financial and transactional behavioral channels.
    """

    FEATURE_NAMES: List[str] = [
        "tx_in_vol_log",
        "tx_out_vol_log",
        "tx_net_flow_log",
        "tx_count_log",
        "tx_cancel_rate",
        "tx_avg_ticket_log",
        "tx_ext_transfer_vol_log",
        "tx_ext_transfer_ratio",
        "tx_cash_withdraw_vol_log",
        "tx_fee_count_log",
        "tx_branch_visit_log",
    ]

    LEGACY_FEATURE_NAMES: List[str] = [
        "tx_in_vol_log",
        "tx_out_vol_log",
        "tx_net_flow_log",
        "tx_count_log",
        "tx_cancel_rate",
        "tx_avg_ticket_log",
    ]

    def extract_legacy(
        self,
        customer_ids: np.ndarray,
        months: List[int],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extracts the 6 original legacy transaction channels."""
        tensor, _ = self.extract(customer_ids, months, context)
        return tensor[:, :, :6], self.LEGACY_FEATURE_NAMES

    def extract(
        self,
        customer_ids: np.ndarray,
        months: List[int],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        idx = customer_ids
        if context is None:
            context = {}

        # 1. Accounts mapping (account_id -> customer_id)
        if "acct_map" in context:
            acct_map = context["acct_map"]
        else:
            acc_path = self._resolve_file("accounts.csv")
            acc_df = pd.read_csv(acc_path, usecols=["account_id", "customer_id"])
            acct_map = dict(zip(acc_df["account_id"], acc_df["customer_id"]))
            context["acct_map"] = acct_map

        # 2. Ingest transactions
        tx_path = self._resolve_file("transactions.csv")
        tx_df = pd.read_csv(
            tx_path,
            usecols=[
                "account_id",
                "transaction_datetime",
                "amount",
                "direction",
                "transaction_type",
                "transaction_status",
                "channel",
            ],
        )
        tx_df["customer_id"] = tx_df["account_id"].map(acct_map)
        tx_df["month"] = (
            pd.to_numeric(tx_df["transaction_datetime"].str.slice(5, 7), errors="coerce")
            .fillna(0)
            .astype(int)
        )

        # Inward deposit volume
        tx_in = (
            tx_df[tx_df["direction"] == "in"]
            .groupby(["customer_id", "month"])["amount"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
            .values
        )

        # Outward payment volume
        tx_out = (
            tx_df[tx_df["direction"] == "out"]
            .groupby(["customer_id", "month"])["amount"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
            .values
        )

        # Total transaction count
        tx_cnt = (
            tx_df.groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
            .values
        )

        # Failed or cancelled transactions
        tx_cancel = (
            tx_df[tx_df["transaction_status"] != "success"]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
            .values
        )

        # External transfer volume (capital flight)
        tx_ext_vol = (
            tx_df[tx_df["transaction_type"] == "external_transfer"]
            .groupby(["customer_id", "month"])["amount"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
            .values
        )

        # External transfer count
        tx_ext_cnt = (
            tx_df[tx_df["transaction_type"] == "external_transfer"]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
            .values
        )

        # Cash withdrawal volume (ATM drainage)
        tx_cash_vol = (
            tx_df[tx_df["transaction_type"] == "cash_withdrawal"]
            .groupby(["customer_id", "month"])["amount"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
            .values
        )

        # Bank fee count (friction events)
        tx_fee_cnt = (
            tx_df[tx_df["transaction_type"] == "bank_fee"]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
            .values
        )

        # Branch interactions (escalations)
        tx_branch_cnt = (
            tx_df[tx_df["channel"] == "branch"]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
            .values
        )

        # Store raw counts in context for cross-channel extractor
        context["tx_cnt_raw"] = tx_cnt

        # Engineered channels
        c1 = self.safe_log1p(tx_in)
        c2 = self.safe_log1p(tx_out)

        net_flow = tx_in - tx_out
        c3 = (np.sign(net_flow) * np.log1p(np.abs(net_flow))).astype(np.float32)

        c4 = self.safe_log1p(tx_cnt)
        c5 = self.safe_clip(self.safe_div(tx_cancel, tx_cnt), 0.0, 1.0)
        c6 = self.safe_log1p(self.safe_div(tx_in + tx_out, tx_cnt))

        c7 = self.safe_log1p(tx_ext_vol)
        c8 = self.safe_clip(self.safe_div(tx_ext_cnt, tx_cnt), 0.0, 1.0)
        c9 = self.safe_log1p(tx_cash_vol)
        c10 = self.safe_log1p(tx_fee_cnt)
        c11 = self.safe_log1p(tx_branch_cnt)

        channels = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]
        tensor_slice = np.stack(channels, axis=-1).astype(np.float32)

        return tensor_slice, self.FEATURE_NAMES
