"""
Lagged Dynamics & Shift Feature Extraction.
Pure Classical ML (Zero Deep Learning).
"""

from typing import Dict
import numpy as np


def extract_lag_features(
    ch_net_flow: np.ndarray,
    ch_tx_count: np.ndarray,
    ch_login_active: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extracts recent month-over-month shifts, velocity, and second-order acceleration."""
    features: Dict[str, np.ndarray] = {}

    features["quant_lag_net_flow_m6_m5"] = ch_net_flow[:, -1] - ch_net_flow[:, -2]
    features["quant_lag_net_flow_m5_m4"] = ch_net_flow[:, -2] - ch_net_flow[:, -3]
    features["quant_lag_net_flow_accel"] = (
        features["quant_lag_net_flow_m6_m5"] - features["quant_lag_net_flow_m5_m4"]
    )
    features["quant_lag_tx_count_m6_m5"] = ch_tx_count[:, -1] - ch_tx_count[:, -2]
    features["quant_lag_login_m6_m5"] = ch_login_active[:, -1] - ch_login_active[:, -2]

    return features
