"""
Domain Sub-Extractors for Multi-Snapshot Longitudinal Bank Dynamics.
Enriches the Bi-GRU sequence manifold with 37 dynamic behavioral channels
extracted from transactions, logins, complaints, accounts, and cross-channel dynamics.
"""

from typing import List

from .base import BaseSubExtractor
from .transaction_extractor import TransactionSnapshotExtractor
from .login_extractor import LoginSnapshotExtractor
from .complaint_extractor import ComplaintSnapshotExtractor
from .account_extractor import AccountSnapshotExtractor
from .cross_channel_extractor import CrossChannelSnapshotExtractor

ALL_FEATURE_NAMES_37: List[str] = (
    TransactionSnapshotExtractor.FEATURE_NAMES
    + LoginSnapshotExtractor.FEATURE_NAMES
    + ComplaintSnapshotExtractor.FEATURE_NAMES
    + AccountSnapshotExtractor.FEATURE_NAMES
    + CrossChannelSnapshotExtractor.FEATURE_NAMES
)

__all__ = [
    "BaseSubExtractor",
    "TransactionSnapshotExtractor",
    "LoginSnapshotExtractor",
    "ComplaintSnapshotExtractor",
    "AccountSnapshotExtractor",
    "CrossChannelSnapshotExtractor",
    "ALL_FEATURE_NAMES_37",
]

