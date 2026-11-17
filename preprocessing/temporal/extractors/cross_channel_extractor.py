"""
Cross-Channel Interaction Extractor.
Extracts 2 longitudinal interaction channels modeling joint multi-table dynamics:
1. Login to Transaction Ratio: checks passive account anxiety / checking without spending.
2. Failed Login to Complaint Coupling: identifies compounding digital frustration.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

from .base import BaseSubExtractor


class CrossChannelSnapshotExtractor(BaseSubExtractor):
    """
    Extracts cross-domain behavioral ratios and interaction signals.
    """

    FEATURE_NAMES: List[str] = [
        "cross_login_to_tx_ratio",
        "cross_fail_login_complaint_coupling",
    ]

    def extract(
        self,
        customer_ids: np.ndarray,
        months: List[int],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        idx = customer_ids
        if context is None:
            context = {}

        # 1. Retrieve or compute login count and transaction count
        if "log_cnt_raw" in context and "tx_cnt_raw" in context:
            log_cnt = context["log_cnt_raw"]
            tx_cnt = context["tx_cnt_raw"]
        else:
            # Fallback: compute from raw files if called standalone
            from .login_extractor import LoginSnapshotExtractor
            from .transaction_extractor import TransactionSnapshotExtractor

            LoginSnapshotExtractor(self.raw_data_dir).extract(idx, months, context)
            TransactionSnapshotExtractor(self.raw_data_dir).extract(idx, months, context)
            log_cnt = context["log_cnt_raw"]
            tx_cnt = context["tx_cnt_raw"]

        # 2. Retrieve or compute failed login count and complaints count
        if "fail_cnt_raw" in context and "cmp_cnt_raw" in context:
            fail_cnt = context["fail_cnt_raw"]
            cmp_cnt = context["cmp_cnt_raw"]
        else:
            from .complaint_extractor import ComplaintSnapshotExtractor

            ComplaintSnapshotExtractor(self.raw_data_dir).extract(idx, months, context)
            fail_cnt = context.get("fail_cnt_raw", np.zeros_like(log_cnt))
            cmp_cnt = context["cmp_cnt_raw"]

        # Channel 1: Log-transformed login-to-transaction ratio
        # ratio = ln(1 + (logins + 1) / (tx + 1))
        # High value indicates obsessive app checking without monetary transactions
        ratio_raw = (log_cnt + 1.0) / (tx_cnt + 1.0)
        c1 = np.log1p(ratio_raw).astype(np.float32)

        # Channel 2: Multiplicative coupling of failed logins and complaints
        # Non-zero when BOTH tech difficulty (failed login) and service complaint occur
        c2 = np.log1p(fail_cnt * cmp_cnt).astype(np.float32)

        # Numerical safety clipping
        c1 = self.safe_clip(c1, 0.0, 10.0)
        c2 = self.safe_clip(c2, 0.0, 10.0)

        channels = [c1, c2]
        tensor_slice = np.stack(channels, axis=-1).astype(np.float32)

        return tensor_slice, self.FEATURE_NAMES
