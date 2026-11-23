"""
Raw Event Sequence Extractor for End-to-End Deep Learning.
Converts multi-table raw bank event streams (transactions, logins, complaints)
directly into a 3D chronological token sequence tensor (N_customers, L_events, D_raw_features)
without handcrafted monthly statistical aggregations.
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class RawEventSequenceExtractor:
    """
    Extracts chronological raw event sequences per customer for End-to-End
    Temporal Representation Learning (Deep Learning on Raw Event Streams).

    Each event is represented as a D-dimensional vector:
      - Continuous features (9 channels):
          1. signed_log_amount: sign(amount) * log1p(|amount|)
          2. is_inward: 1.0 for deposit/inflow
          3. is_outward: 1.0 for withdrawal/payment/outflow
          4. inter_event_hours_log: log1p(delta_t in hours since previous event)
          5. recency_days_log: log1p(days until observation cutoff 2026-10-01)
          6. hour_sin: sin(2 * pi * hour / 24)
          7. hour_cos: cos(2 * pi * hour / 24)
          8. day_of_week_sin: sin(2 * pi * dow / 7)
          9. day_of_week_cos: cos(2 * pi * dow / 7)
      - Event Type One-Hot (9 channels):
          10. type_interest_credit
          11. type_internal_transfer
          12. type_payment
          13. type_external_transfer
          14. type_bank_fee
          15. type_cash_withdrawal
          16. type_cash_deposit
          17. type_login
          18. type_complaint
      - Status One-Hot (3 channels):
          19. status_success
          20. status_failed
          21. status_cancelled
      - Channel / Device One-Hot (5 channels):
          22. channel_mobile_app
          23. channel_desktop
          24. channel_atm
          25. channel_branch
          26. channel_system

    Total: D = 26 raw channels per event token.
    Sequence shape: (N_customers, max_seq_len, 26), left-padded with 0s.
    """

    FEATURE_NAMES: List[str] = [
        "signed_log_amount",
        "is_inward",
        "is_outward",
        "inter_event_hours_log",
        "recency_days_log",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "type_interest_credit",
        "type_internal_transfer",
        "type_payment",
        "type_external_transfer",
        "type_bank_fee",
        "type_cash_withdrawal",
        "type_cash_deposit",
        "type_login",
        "type_complaint",
        "status_success",
        "status_failed",
        "status_cancelled",
        "channel_mobile_app",
        "channel_desktop",
        "channel_atm",
        "channel_branch",
        "channel_system",
    ]

    CUTOFF_DATETIME_STR = "2026-10-01 00:00:00"

    def __init__(
        self,
        raw_data_dir: Union[str, Path] = "data/full_dataset",
        max_seq_len: int = 50,
        tensor_cache_path: Optional[Union[str, Path]] = None,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.max_seq_len = int(max_seq_len)
        if tensor_cache_path is not None:
            self.tensor_cache_path = Path(tensor_cache_path)
        else:
            self.tensor_cache_path = Path(
                f"data/processed/temporal/raw_event_sequence_tensor_{self.max_seq_len}l_26d.npz"
            )

    def _resolve_file(self, filename: str) -> Path:
        p = self.raw_data_dir / filename
        if not p.exists():
            raise FileNotFoundError(f"Required raw event file not found: {p}")
        return p

    def load_all_raw_events(self) -> pd.DataFrame:
        """
        Loads and harmonizes raw events across transactions.csv, login.csv, complaints.csv.
        Returns a single unified DataFrame sorted chronologically by (customer_id, timestamp).
        """
        # 1. Accounts mapping (account_id -> customer_id)
        acc_path = self._resolve_file("accounts.csv")
        acc_df = pd.read_csv(acc_path, usecols=["account_id", "customer_id"])
        acct_map = dict(zip(acc_df["account_id"], acc_df["customer_id"]))

        # 2. Transactions
        tx_path = self._resolve_file("transactions.csv")
        tx_cols = [
            "account_id",
            "transaction_datetime",
            "amount",
            "direction",
            "transaction_type",
            "transaction_status",
            "channel",
        ]
        tx_df = pd.read_csv(tx_path, usecols=tx_cols)
        tx_df["customer_id"] = tx_df["account_id"].map(acct_map)
        tx_df = tx_df.dropna(subset=["customer_id"])

        ev_tx = pd.DataFrame({
            "customer_id": tx_df["customer_id"].astype(np.int64),
            "timestamp_str": tx_df["transaction_datetime"].str.slice(0, 19),
            "event_type": tx_df["transaction_type"].astype(str),
            "amount": np.where(tx_df["direction"] == "in", tx_df["amount"], -tx_df["amount"]).astype(np.float32),
            "status": tx_df["transaction_status"].astype(str),
            "channel": tx_df["channel"].astype(str),
        })

        # 3. Logins
        log_path = self._resolve_file("login.csv")
        log_cols = ["customer_id", "sign_in_datetime", "login_status", "device_type"]
        log_df = pd.read_csv(log_path, usecols=log_cols)
        log_df = log_df.dropna(subset=["customer_id"])

        device_map = {"ios": "mobile_app", "android": "mobile_app", "desktop": "desktop", "web": "desktop"}
        mapped_channel = log_df["device_type"].map(lambda x: device_map.get(str(x).lower(), "mobile_app"))

        ev_log = pd.DataFrame({
            "customer_id": log_df["customer_id"].astype(np.int64),
            "timestamp_str": log_df["sign_in_datetime"].str.slice(0, 19),
            "event_type": "login",
            "amount": np.float32(0.0),
            "status": log_df["login_status"].astype(str),
            "channel": mapped_channel.astype(str),
        })

        # 4. Complaints
        cmp_path = self._resolve_file("complaints.csv")
        cmp_cols = ["customer_id", "complaint_datetime"]
        cmp_df = pd.read_csv(cmp_path, usecols=cmp_cols)
        cmp_df = cmp_df.dropna(subset=["customer_id"])

        ev_cmp = pd.DataFrame({
            "customer_id": cmp_df["customer_id"].astype(np.int64),
            "timestamp_str": cmp_df["complaint_datetime"].str.slice(0, 19),
            "event_type": "complaint",
            "amount": np.float32(0.0),
            "status": "success",
            "channel": "system",
        })

        # 5. Concatenate and sort
        events = pd.concat([ev_tx, ev_log, ev_cmp], ignore_index=True)
        events = events.sort_values(["customer_id", "timestamp_str"]).reset_index(drop=True)
        return events

    def _vectorize_events(self, events: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes 26 raw features and returns (feature_matrix, customer_ids, pos_from_end).
        """
        # Parse timestamps into datetime
        dt = pd.to_datetime(events["timestamp_str"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        cutoff = pd.to_datetime(self.CUTOFF_DATETIME_STR)

        # 1. Continuous features
        amt = events["amount"].values.astype(np.float32)
        signed_log_amount = np.sign(amt) * np.log1p(np.abs(amt))
        is_inward = (amt > 0).astype(np.float32)
        is_outward = (amt < 0).astype(np.float32)

        # Inter-event duration (seconds since previous event of same customer)
        # Shift within group
        dt_series = pd.Series(dt)
        cust_series = events["customer_id"]
        prev_dt = dt_series.groupby(cust_series).shift(1)
        delta_sec = (dt_series - prev_dt).dt.total_seconds().fillna(0.0).clip(lower=0.0).values
        inter_event_hours_log = np.log1p(delta_sec / 3600.0).astype(np.float32)

        # Recency to cutoff in days
        recency_days = (cutoff - dt_series).dt.total_seconds().fillna(0.0).clip(lower=0.0).values / 86400.0
        recency_days_log = np.log1p(recency_days).astype(np.float32)

        # Cyclical time encodings
        hours = dt_series.dt.hour.fillna(12).values.astype(np.float32)
        dows = dt_series.dt.dayofweek.fillna(3).values.astype(np.float32)

        hour_sin = np.sin(2.0 * np.pi * hours / 24.0).astype(np.float32)
        hour_cos = np.cos(2.0 * np.pi * hours / 24.0).astype(np.float32)
        dow_sin = np.sin(2.0 * np.pi * dows / 7.0).astype(np.float32)
        dow_cos = np.cos(2.0 * np.pi * dows / 7.0).astype(np.float32)

        # 2. Event type one-hot
        ev_type = events["event_type"].values
        type_interest_credit = (ev_type == "interest_credit").astype(np.float32)
        type_internal_transfer = (ev_type == "internal_transfer").astype(np.float32)
        type_payment = (ev_type == "payment").astype(np.float32)
        type_external_transfer = (ev_type == "external_transfer").astype(np.float32)
        type_bank_fee = (ev_type == "bank_fee").astype(np.float32)
        type_cash_withdrawal = (ev_type == "cash_withdrawal").astype(np.float32)
        type_cash_deposit = (ev_type == "cash_deposit").astype(np.float32)
        type_login = (ev_type == "login").astype(np.float32)
        type_complaint = (ev_type == "complaint").astype(np.float32)

        # 3. Status one-hot
        st = events["status"].str.lower().values
        status_success = (st == "success").astype(np.float32)
        status_failed = (st == "failed").astype(np.float32)
        status_cancelled = ((st == "cancelled") | (st == "canceled")).astype(np.float32)

        # 4. Channel one-hot
        ch = events["channel"].str.lower().values
        channel_mobile_app = ((ch == "mobile_app") | (ch == "ios") | (ch == "android")).astype(np.float32)
        channel_desktop = ((ch == "desktop") | (ch == "web")).astype(np.float32)
        channel_atm = (ch == "atm").astype(np.float32)
        channel_branch = (ch == "branch").astype(np.float32)
        channel_system = (ch == "system").astype(np.float32)

        # Assemble full 26-dim feature matrix
        feature_matrix = np.column_stack([
            signed_log_amount,
            is_inward,
            is_outward,
            inter_event_hours_log,
            recency_days_log,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            type_interest_credit,
            type_internal_transfer,
            type_payment,
            type_external_transfer,
            type_bank_fee,
            type_cash_withdrawal,
            type_cash_deposit,
            type_login,
            type_complaint,
            status_success,
            status_failed,
            status_cancelled,
            channel_mobile_app,
            channel_desktop,
            channel_atm,
            channel_branch,
            channel_system,
        ]).astype(np.float32)

        # Compute position from end per customer (0 = latest event, 1 = second latest, ...)
        pos_from_end = events.groupby("customer_id").cumcount(ascending=False).values

        return feature_matrix, events["customer_id"].values, pos_from_end

    def extract(
        self,
        customer_ids: Union[List[int], np.ndarray, pd.Series],
        verbose: bool = True,
        save_cache: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts raw event sequence tensor (N_customers, max_seq_len, 26).
        Left-pads shorter sequences with 0s.
        """
        t0 = time.time()
        cust_ids_arr = np.asarray(customer_ids, dtype=np.int64)

        # 1. Check pre-computed tensor cache
        if self.tensor_cache_path.exists():
            try:
                cached_data = np.load(self.tensor_cache_path, allow_pickle=True)
                cached_ids = cached_data["customer_ids"]
                cached_tensor = cached_data["tensor"]

                if (
                    cached_tensor.shape[1] == self.max_seq_len
                    and cached_tensor.shape[2] == len(self.FEATURE_NAMES)
                ):
                    cached_id_map = {cid: idx for idx, cid in enumerate(cached_ids)}
                    if all(cid in cached_id_map for cid in cust_ids_arr):
                        indices = [cached_id_map[cid] for cid in cust_ids_arr]
                        subset_tensor = cached_tensor[indices]
                        if verbose:
                            print(
                                f"[INFO] Loaded pre-cached raw event sequence tensor ({subset_tensor.shape}) "
                                f"for {len(cust_ids_arr)} customers in {time.time() - t0:.3f}s"
                            )
                        return subset_tensor, cust_ids_arr
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to read raw tensor cache ({e}). Rebuilding...")

        if verbose:
            print(
                f"[INFO] Extracting raw event sequence tensor (L={self.max_seq_len}, D={len(self.FEATURE_NAMES)}) "
                f"across {len(cust_ids_arr)} customers..."
            )

        # 2. Ingest raw event streams
        events = self.load_all_raw_events()

        # 3. Filter to target customers
        target_set = set(cust_ids_arr)
        events = events[events["customer_id"].isin(target_set)].copy()

        # 4. Keep at most max_seq_len events per customer (most recent L events)
        events = events.groupby("customer_id", group_keys=False).tail(self.max_seq_len).reset_index(drop=True)

        # 5. Vectorize into features
        feat_matrix, ev_cust_ids, pos_from_end = self._vectorize_events(events)

        # 6. Standardize continuous channels (channels 0, 3, 4: amount, delta_t, recency)
        # Using bounded z-score standardization
        cont_cols = [0, 3, 4]
        for c in cont_cols:
            vals = feat_matrix[:, c]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            if std_val > 1e-6:
                feat_matrix[:, c] = np.clip((vals - mean_val) / std_val, -5.0, 5.0)

        # 7. Construct 3D Tensor (N, L, D) with left-padding
        N = len(cust_ids_arr)
        L = self.max_seq_len
        D = len(self.FEATURE_NAMES)
        tensor = np.zeros((N, L, D), dtype=np.float32)

        # Map customer_id -> index in cust_ids_arr
        cust_id_map = {cid: idx for idx, cid in enumerate(cust_ids_arr)}
        cust_indices = np.array([cust_id_map[cid] for cid in ev_cust_ids], dtype=np.int64)

        # Time indices: latest event placed at L - 1, earlier events at L - 1 - pos_from_end
        time_indices = L - 1 - pos_from_end
        valid_mask = (time_indices >= 0) & (time_indices < L)

        tensor[cust_indices[valid_mask], time_indices[valid_mask], :] = feat_matrix[valid_mask]

        if verbose:
            print(
                f"[OK] Extracted raw event sequence tensor {tensor.shape} "
                f"in {time.time() - t0:.2f}s."
            )

        # 8. Cache to disk
        should_save = save_cache if save_cache is not None else (len(cust_ids_arr) >= 9000)
        if should_save:
            try:
                self.tensor_cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    self.tensor_cache_path,
                    customer_ids=cust_ids_arr,
                    tensor=tensor,
                    feature_names=np.array(self.FEATURE_NAMES),
                )
                if verbose:
                    print(f"[OK] Cached raw event tensor to: {self.tensor_cache_path}")
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to cache raw event tensor: {e}")

        return tensor, cust_ids_arr
