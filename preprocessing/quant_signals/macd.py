"""
Moving Average Convergence Divergence (MACD) Feature Extraction.
Pure Classical ML (Zero Deep Learning).
"""

from typing import Dict, Tuple
import numpy as np


def compute_macd_series(series_6: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes MACD line, Signal line, and Histogram on a 6-month sequence (N, 6).
    Fast EMA span=2 (alpha=2/3), Slow EMA span=4 (alpha=2/5), Signal span=2.
    """
    a_f = 2.0 / (2.0 + 1.0)  # 0.6667
    a_s = 2.0 / (4.0 + 1.0)  # 0.4000
    N, T = series_6.shape

    ema_f = np.zeros_like(series_6)
    ema_s = np.zeros_like(series_6)
    ema_f[:, 0] = series_6[:, 0]
    ema_s[:, 0] = series_6[:, 0]

    for t in range(1, T):
        ema_f[:, t] = a_f * series_6[:, t] + (1.0 - a_f) * ema_f[:, t - 1]
        ema_s[:, t] = a_s * series_6[:, t] + (1.0 - a_s) * ema_s[:, t - 1]

    macd_line = ema_f - ema_s
    signal = np.zeros_like(macd_line)
    signal[:, 0] = macd_line[:, 0]
    a_sig = 2.0 / (2.0 + 1.0)
    for t in range(1, T):
        signal[:, t] = a_sig * macd_line[:, t] + (1.0 - a_sig) * signal[:, t - 1]

    hist = macd_line - signal
    return macd_line[:, -1], signal[:, -1], hist[:, -1]


def extract_macd_features(
    ch_net_flow: np.ndarray,
    ch_tx_count: np.ndarray,
    ch_login_active: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extracts final-month MACD line, Signal line, and Histogram features."""
    features: Dict[str, np.ndarray] = {}

    m_line, m_sig, m_hist = compute_macd_series(ch_net_flow)
    features["quant_macd_net_flow_line"] = m_line
    features["quant_macd_net_flow_signal"] = m_sig
    features["quant_macd_net_flow_hist"] = m_hist

    tc_line, _, tc_hist = compute_macd_series(ch_tx_count)
    features["quant_macd_tx_count_line"] = tc_line
    features["quant_macd_tx_count_hist"] = tc_hist

    log_line, _, log_hist = compute_macd_series(ch_login_active)
    features["quant_macd_login_intensity_line"] = log_line
    features["quant_macd_login_intensity_hist"] = log_hist

    return features
