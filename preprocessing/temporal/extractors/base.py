"""
Base Abstract Extractor for Domain-Specific Temporal Sequence Feature Engineering.
Provides contract and numerical safety helpers for vectorized multi-snapshot extraction.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class BaseSubExtractor(ABC):
    """
    Abstract contract for extracting multi-period monthly behavioral metrics
    from specific raw bank transaction, login, complaint, or account logs.
    """

    def __init__(self, raw_data_dir: Union[str, Path]):
        self.raw_data_dir = Path(raw_data_dir)

    def _resolve_file(self, filename: str) -> Path:
        """Finds raw table file in workspace or fallback directories."""
        cand = self.raw_data_dir / filename
        if cand.exists():
            return cand
        cand_alt = Path(__file__).resolve().parent.parent.parent.parent / "data" / "full_dataset" / filename
        if cand_alt.exists():
            return cand_alt
        raise FileNotFoundError(f"Raw table '{filename}' not found in '{self.raw_data_dir}'.")

    @abstractmethod
    def extract(
        self,
        customer_ids: np.ndarray,
        months: List[int],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extracts temporal channel tensor slice for target customer IDs and snapshot months.

        Args:
            customer_ids: 1D array of int64 customer IDs, shape (N,).
            months: Ordered list of month integers, e.g. [4, 5, 6, 7, 8, 9].
            context: Shared dictionary of pre-loaded tables (e.g. acct_to_cust mapping).

        Returns:
            tensor_slice: float32 array of shape (N, len(months), D_sub).
            channel_names: List of string names for each channel of length D_sub.
        """
        pass

    @staticmethod
    def safe_log1p(x: np.ndarray) -> np.ndarray:
        """Log1p transform bounded from below at 0 to eliminate negative values and heavy tails."""
        return np.log1p(np.maximum(x, 0.0)).astype(np.float32)

    @staticmethod
    def safe_div(num: np.ndarray, den: np.ndarray, fill_val: float = 0.0) -> np.ndarray:
        """Safe element-wise division preventing division by zero or NaN propagation."""
        mask = den > 0
        out = np.full_like(num, fill_val, dtype=np.float32)
        out[mask] = (num[mask] / np.maximum(den[mask], 1e-6)).astype(np.float32)
        return out

    @staticmethod
    def safe_clip(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Clips array to specified range, replacing NaNs with min_val."""
        cleaned = np.nan_to_num(x, nan=min_val, posinf=max_val, neginf=min_val)
        return np.clip(cleaned, min_val, max_val).astype(np.float32)

