"""
Volatility and Semi-variance Signal Feature Extraction.
Pure Classical ML (Zero Deep Learning).
"""

from typing import Dict
import numpy as np


def extract_volatility_features(
    ch_net_flow: np.ndarray,
    ch_tx_count: np.ndarray,
    ch_login_active: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extracts longitudinal volatility and downside semi-variance metrics."""
    features: Dict[str, np.ndarray] = {}

    # Standard Deviations
    nf_std = np.std(ch_net_flow, axis=1)
    features["quant_volatility_net_flow_std"] = nf_std
    features["quant_volatility_tx_count_std"] = np.std(ch_tx_count, axis=1)
    features["quant_volatility_login_std"] = np.std(ch_login_active, axis=1)

    # Coefficient of Variation (CV = std / (|mean| + eps))
    nf_mean = np.mean(ch_net_flow, axis=1)
    features["quant_volatility_net_flow_cv"] = nf_std / (np.abs(nf_mean) + 100.0)

    # Downside Semi-variance (Negative deviation from 0)
    neg_flows = np.where(ch_net_flow < 0, ch_net_flow, 0.0)
    features["quant_volatility_downside_semi_variance"] = np.sqrt(np.mean(neg_flows**2, axis=1))

    return features
