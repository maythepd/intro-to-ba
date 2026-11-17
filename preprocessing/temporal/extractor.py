"""
Temporal Snapshot Extractor Coordinator for Digital Bank Event Logs.
Coordinates domain sub-extractors across transactional, login, complaint,
account, and cross-channel event streams to construct a 3D temporal sequence
tensor (N_customers, T_months, D_metrics) for longitudinal representation learning.
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

from .extractors import (
    TransactionSnapshotExtractor,
    LoginSnapshotExtractor,
    ComplaintSnapshotExtractor,
    AccountSnapshotExtractor,
    CrossChannelSnapshotExtractor,
    ALL_FEATURE_NAMES_37,
)


class TemporalSnapshotExtractor:
    """
    Coordinates multi-period monthly behavioral snapshots per customer.

    Snapshot Months: April (4), May (5), June (6), July (7), August (8), September (9) 2026.

    Modular Sub-Extractors:
      - TransactionSnapshotExtractor: 11 channels (T1 - T11)
      - LoginSnapshotExtractor: 15 channels (L1 - L15)
      - ComplaintSnapshotExtractor: 5 channels (C1 - C5)
      - AccountSnapshotExtractor: 4 channels (A1 - A4)
      - CrossChannelSnapshotExtractor: 2 channels (X1 - X2)
      Total: 37 dynamic behavioral channels (D=37).
    """

    DEFAULT_MONTHS: List[int] = [4, 5, 6, 7, 8, 9]

    LEGACY_FEATURE_NAMES_10: List[str] = (
        TransactionSnapshotExtractor.LEGACY_FEATURE_NAMES
        + LoginSnapshotExtractor.LEGACY_FEATURE_NAMES
        + ComplaintSnapshotExtractor.LEGACY_FEATURE_NAMES
    )

    FEATURE_NAMES: List[str] = ALL_FEATURE_NAMES_37

    def __init__(
        self,
        raw_data_dir: Union[str, Path] = "data/full_dataset",
        months: Optional[List[int]] = None,
        tensor_cache_path: Union[str, Path] = "data/processed/temporal/raw_snapshot_tensor_37d.npz",
        use_extended: bool = True,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.months = months or self.DEFAULT_MONTHS
        self.tensor_cache_path = Path(tensor_cache_path)
        self.use_extended = use_extended

        self.feature_names = (
            ALL_FEATURE_NAMES_37 if self.use_extended else self.LEGACY_FEATURE_NAMES_10
        )

        # Modular domain sub-extractors
        self.tx_extractor = TransactionSnapshotExtractor(self.raw_data_dir)
        self.login_extractor = LoginSnapshotExtractor(self.raw_data_dir)
        self.complaint_extractor = ComplaintSnapshotExtractor(self.raw_data_dir)
        self.account_extractor = AccountSnapshotExtractor(self.raw_data_dir)
        self.cross_extractor = CrossChannelSnapshotExtractor(self.raw_data_dir)

    def extract(
        self,
        customer_ids: Union[List[int], np.ndarray, pd.Series],
        verbose: bool = True,
        save_cache: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts standardized temporal sequence tensor for target customer IDs.

        Returns:
          tensor: np.ndarray of shape (N_customers, len(months), D_channels),
                  standardized per channel with bounded outlier clipping [-5.0, 5.0].
          cust_ids_arr: np.ndarray of shape (N_customers,).
        """
        t0 = time.time()
        cust_ids_arr = np.asarray(customer_ids, dtype=np.int64)

        # 0. Check pre-computed tensor cache
        if self.tensor_cache_path.exists():
            try:
                cached_data = np.load(self.tensor_cache_path, allow_pickle=True)
                cached_ids = cached_data["customer_ids"]
                cached_tensor = cached_data["tensor"]

                # Ensure cached dimension matches current configuration
                if cached_tensor.shape[-1] == len(self.feature_names):
                    cached_id_map = {cid: idx for idx, cid in enumerate(cached_ids)}
                    if all(cid in cached_id_map for cid in cust_ids_arr):
                        indices = [cached_id_map[cid] for cid in cust_ids_arr]
                        sliced_tensor = cached_tensor[indices]
                        if verbose:
                            print(
                                f"[INFO] Loaded pre-cached snapshot tensor "
                                f"({sliced_tensor.shape}) for {len(cust_ids_arr)} customers in {time.time() - t0:.3f}s"
                            )
                        return sliced_tensor, cust_ids_arr
            except Exception:
                pass

        if verbose:
            print(
                f"[INFO] Extracting {len(self.feature_names)}-dim temporal event snapshots "
                f"across {len(self.months)} months for {len(cust_ids_arr)} customers..."
            )

        if not self.use_extended:
            # Delegate to sub-extractors for legacy 10-dimensional extraction
            return self._extract_legacy(cust_ids_arr, verbose)

        # Shared execution context across sub-extractors to avoid redundant I/O
        context: Dict[str, Any] = {}

        # 1. Accounts & Portfolios (A1 - A4)
        acc_slice, _ = self.account_extractor.extract(cust_ids_arr, self.months, context)

        # 2. Financial Transactions (T1 - T11)
        tx_slice, _ = self.tx_extractor.extract(cust_ids_arr, self.months, context)

        # 3. Customer Service Complaints (C1 - C5)
        cmp_slice, _ = self.complaint_extractor.extract(cust_ids_arr, self.months, context)

        # 4. Digital Logins (L1 - L15)
        log_slice, _ = self.login_extractor.extract(cust_ids_arr, self.months, context)

        # 5. Cross-Channel Dynamics (X1 - X2)
        cross_slice, _ = self.cross_extractor.extract(cust_ids_arr, self.months, context)

        # Concatenate all slices along channel axis (axis=-1)
        raw_tensor = np.concatenate(
            [tx_slice, log_slice, cmp_slice, acc_slice, cross_slice],
            axis=-1,
        ).astype(np.float32)

        # Numerical safety: replace any unhandled NaNs/Infs before normalization
        raw_tensor = np.nan_to_num(raw_tensor, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Channel-wise Z-score normalization across samples and time snapshots
        mean = np.mean(raw_tensor, axis=(0, 1), keepdims=True)
        std = np.std(raw_tensor, axis=(0, 1), keepdims=True) + 1e-6
        norm_tensor = (raw_tensor - mean) / std

        # Clamping extreme outliers to [-5.0, 5.0] to safeguard BiGRU gates from exploding gradients
        norm_tensor = np.clip(norm_tensor, -5.0, 5.0).astype(np.float32)
        norm_tensor = np.nan_to_num(norm_tensor, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)

        # Cache tensor if requested or sufficiently large batch
        should_cache = save_cache if save_cache is not None else (len(cust_ids_arr) >= 1000)
        if should_cache:
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
            print(
                f"[OK] Multi-snapshot tensor constructed: shape={norm_tensor.shape} "
                f"(D={norm_tensor.shape[2]}) in {t1 - t0:.2f}s"
            )

        return norm_tensor, cust_ids_arr

    def _extract_legacy(
        self,
        cust_ids_arr: np.ndarray,
        verbose: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy 10-channel extraction path delegating cleanly to sub-extractors:
          - tx_extractor: 6 channels (tx_in, tx_out, net_flow, tx_count, cancel_rate, avg_ticket)
          - login_extractor: 3 channels (login_count, fail_rate, mobile_share)
          - complaint_extractor: 1 channel (complaints_count)
        """
        context: Dict[str, Any] = {}
        tx_slice, _ = self.tx_extractor.extract_legacy(cust_ids_arr, self.months, context)
        log_slice, _ = self.login_extractor.extract_legacy(cust_ids_arr, self.months, context)
        cmp_slice, _ = self.complaint_extractor.extract_legacy(cust_ids_arr, self.months, context)

        raw_tensor = np.concatenate([tx_slice, log_slice, cmp_slice], axis=-1).astype(np.float32)
        mean = np.mean(raw_tensor, axis=(0, 1), keepdims=True)
        std = np.std(raw_tensor, axis=(0, 1), keepdims=True) + 1e-6
        norm_tensor = (raw_tensor - mean) / std
        norm_tensor = np.clip(norm_tensor, -5.0, 5.0).astype(np.float32)
        norm_tensor = np.nan_to_num(norm_tensor, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)

        return norm_tensor, cust_ids_arr
