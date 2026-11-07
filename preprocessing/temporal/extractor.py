"""
Temporal Snapshot Extractor for Digital Bank Event Logs.
Ingests raw transactional, login, and complaint event streams from data/full_dataset/
and constructs a 3D temporal sequence tensor (N_customers, T_months, D_metrics)
capturing longitudinal customer dynamics across monthly snapshots.
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class TemporalSnapshotExtractor:
    """
    Constructs multi-period monthly behavioral snapshots per customer from raw event logs.
    
    Snapshot Months: April (4), May (5), June (6), July (7), August (8), September (9) 2026.
    
    Engineered Monthly Behavioral Channels (D=10):
      1. tx_in_vol_log:       log1p(total inward deposit/transfer volume in month)
      2. tx_out_vol_log:      log1p(total outward payment/withdrawal volume in month)
      3. tx_net_flow_log:     signed log1p of net flow (in - out)
      4. tx_count_log:        log1p(number of transactions in month)
      5. tx_cancel_rate:      cancellation/failure rate in month
      6. tx_avg_ticket_log:   log1p(average transaction ticket size)
      7. login_count_log:     log1p(digital banking logins in month)
      8. login_fail_rate:     fraction of failed login attempts
      9. login_mobile_share:  fraction of logins performed via mobile (iOS/Android)
     10. complaints_count:    customer complaints filed in month
    """

    DEFAULT_MONTHS: List[int] = [4, 5, 6, 7, 8, 9]

    FEATURE_NAMES: List[str] = [
        "tx_in_vol_log",
        "tx_out_vol_log",
        "tx_net_flow_log",
        "tx_count_log",
        "tx_cancel_rate",
        "tx_avg_ticket_log",
        "login_count_log",
        "login_fail_rate",
        "login_mobile_share",
        "complaints_count",
    ]

    def __init__(
        self,
        raw_data_dir: Union[str, Path] = "data/full_dataset",
        months: Optional[List[int]] = None,
        tensor_cache_path: Union[str, Path] = "data/processed/temporal/raw_snapshot_tensor.npz",
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.months = months or self.DEFAULT_MONTHS
        self.tensor_cache_path = Path(tensor_cache_path)

    def _resolve_file(self, filename: str) -> Path:
        cand = self.raw_data_dir / filename
        if cand.exists():
            return cand
        cand_alt = Path(__file__).resolve().parent.parent.parent / "data" / "full_dataset" / filename
        if cand_alt.exists():
            return cand_alt
        raise FileNotFoundError(f"Raw table '{filename}' not found in '{self.raw_data_dir}'.")

    def extract(
        self,
        customer_ids: Union[List[int], np.ndarray, pd.Series],
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts temporal sequence tensor for target customer IDs.
        
        Returns:
          tensor: np.ndarray of shape (N_customers, len(months), 10), standardized per channel.
          cust_ids_arr: np.ndarray of shape (N_customers,).
        """
        t0 = time.time()
        cust_ids_arr = np.asarray(customer_ids, dtype=np.int64)

        # 0. Check pre-computed tensor cache
        if self.tensor_cache_path.exists():
            try:
                cached_data = np.load(self.tensor_cache_path)
                cached_ids = cached_data["customer_ids"]
                cached_tensor = cached_data["tensor"]

                cached_id_map = {cid: idx for idx, cid in enumerate(cached_ids)}
                if all(cid in cached_id_map for cid in cust_ids_arr):
                    indices = [cached_id_map[cid] for cid in cust_ids_arr]
                    sliced_tensor = cached_tensor[indices]
                    if verbose:
                        print(f"[INFO] Loaded pre-cached raw snapshot tensor for {len(cust_ids_arr)} customers in {time.time() - t0:.3f}s")
                    return sliced_tensor, cust_ids_arr
            except Exception:
                pass

        if verbose:
            print(f"[INFO] Extracting temporal event snapshots across {len(self.months)} months for {len(cust_ids_arr)} customers...")

        # 1. Accounts mapping (account_id -> customer_id)
        acc_path = self._resolve_file("accounts.csv")
        acc_df = pd.read_csv(acc_path, usecols=["account_id", "customer_id"])
        acct_map = dict(zip(acc_df["account_id"], acc_df["customer_id"]))

        # 2. Transactions
        tx_path = self._resolve_file("transactions.csv")
        tx_df = pd.read_csv(
            tx_path,
            usecols=["account_id", "transaction_datetime", "amount", "direction", "transaction_status"],
        )
        tx_df["customer_id"] = tx_df["account_id"].map(acct_map)
        try:
            tx_df["month"] = pd.to_numeric(tx_df["transaction_datetime"].str.slice(5, 7), errors="coerce").fillna(0).astype(int)
        except Exception:
            tx_df["month"] = pd.to_datetime(tx_df["transaction_datetime"], format="ISO8601").dt.month

        # 3. Logins
        log_path = self._resolve_file("login.csv")
        log_df = pd.read_csv(
            log_path,
            usecols=["customer_id", "sign_in_datetime", "login_status", "device_type"],
        )
        try:
            log_df["month"] = pd.to_numeric(log_df["sign_in_datetime"].str.slice(5, 7), errors="coerce").fillna(0).astype(int)
        except Exception:
            log_df["month"] = pd.to_datetime(log_df["sign_in_datetime"], format="ISO8601").dt.month

        # 4. Complaints
        cmp_path = self._resolve_file("complaints.csv")
        cmp_df = pd.read_csv(cmp_path, usecols=["customer_id", "complaint_datetime"])
        try:
            cmp_df["month"] = pd.to_numeric(cmp_df["complaint_datetime"].str.slice(5, 7), errors="coerce").fillna(0).astype(int)
        except Exception:
            cmp_df["month"] = pd.to_datetime(cmp_df["complaint_datetime"], format="ISO8601").dt.month

        # Grouped pivots aligned to cust_ids_arr and months
        months = self.months
        idx = cust_ids_arr

        # Transaction metrics
        tx_in = (
            tx_df[tx_df["direction"] == "in"]
            .groupby(["customer_id", "month"])["amount"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        tx_out = (
            tx_df[tx_df["direction"] == "out"]
            .groupby(["customer_id", "month"])["amount"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        tx_cnt = (
            tx_df.groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        tx_cancel = (
            tx_df[tx_df["transaction_status"] != "success"]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )

        # Login metrics
        log_cnt = (
            log_df.groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        log_fail = (
            log_df[log_df["login_status"] != "success"]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        log_mob = (
            log_df[log_df["device_type"].isin(["ios", "android"])]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )

        # Complaints metrics
        cmp_cnt = (
            cmp_df.groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )

        # Derived logarithmic and ratio transformations
        tx_in_log = np.log1p(np.maximum(tx_in.values, 0.0))
        tx_out_log = np.log1p(np.maximum(tx_out.values, 0.0))
        net_flow = tx_in.values - tx_out.values
        tx_net_flow_log = np.sign(net_flow) * np.log1p(np.abs(net_flow))

        tx_cnt_v = tx_cnt.values
        tx_cnt_log = np.log1p(tx_cnt_v)
        tx_cancel_rate = tx_cancel.values / np.maximum(tx_cnt_v, 1.0)
        tx_avg_ticket_log = np.log1p((tx_in.values + tx_out.values) / np.maximum(tx_cnt_v, 1.0))

        log_cnt_v = log_cnt.values
        log_cnt_log = np.log1p(log_cnt_v)
        log_fail_rate = log_fail.values / np.maximum(log_cnt_v, 1.0)
        log_mobile_share = log_mob.values / np.maximum(log_cnt_v, 1.0)
        complaints_cnt = cmp_cnt.values.astype(np.float32)

        # Stack into 3D array: (N, T, D)
        channels = [
            tx_in_log,
            tx_out_log,
            tx_net_flow_log,
            tx_cnt_log,
            tx_cancel_rate,
            tx_avg_ticket_log,
            log_cnt_log,
            log_fail_rate,
            log_mobile_share,
            complaints_cnt,
        ]
        raw_tensor = np.stack(channels, axis=-1).astype(np.float32)

        # Normalize across samples per channel for numerical stability
        mean = np.mean(raw_tensor, axis=(0, 1), keepdims=True)
        std = np.std(raw_tensor, axis=(0, 1), keepdims=True) + 1e-6
        norm_tensor = (raw_tensor - mean) / std

        # Save to tensor cache if large batch and larger or equal to existing cache
        if len(cust_ids_arr) >= 1000:
            should_save = True
            if self.tensor_cache_path.exists():
                try:
                    existing = np.load(self.tensor_cache_path, allow_pickle=True)
                    if len(existing.get("customer_ids", [])) > len(cust_ids_arr):
                        should_save = False
                except Exception:
                    pass
            if should_save:
                try:
                    self.tensor_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        self.tensor_cache_path,
                        tensor=norm_tensor,
                        customer_ids=cust_ids_arr,
                    )
                except Exception:
                    pass

        if verbose:
            t1 = time.time()
            print(f"[OK] Temporal sequence tensor constructed: shape={norm_tensor.shape} in {t1 - t0:.2f}s")

        return norm_tensor, cust_ids_arr
