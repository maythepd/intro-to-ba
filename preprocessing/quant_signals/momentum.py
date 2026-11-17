"""
Momentum and Trend Rate of Change (ROC) Feature Extraction.
Pure Classical ML (Zero Deep Learning).
"""

from typing import Dict
import numpy as np


def extract_momentum_features(
    ch_net_flow: np.ndarray,
    ch_tx_count: np.ndarray,
    ch_login_active: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Computes momentum, velocity, acceleration, RSI cash flow, and negative streaks
    from 6-month longitudinal matrices of shape (N, 6).
    """
    features: Dict[str, np.ndarray] = {}

    # Net Flow Rate of Change (ROC)
    m6 = ch_net_flow[:, -1]
    m1 = ch_net_flow[:, 0]
    features["quant_mom_net_flow_roc_6m"] = (m6 - m1) / (np.abs(m1) + 100.0)

    # Net Flow Velocity (Recent 2-month slope)
    features["quant_mom_net_flow_velocity"] = (ch_net_flow[:, -1] - ch_net_flow[:, -3]) / 2.0

    # Net Flow Acceleration (Curvature of recent drop)
    vel_recent = ch_net_flow[:, -1] - ch_net_flow[:, -2]
    vel_prior = ch_net_flow[:, -2] - ch_net_flow[:, -3]
    features["quant_mom_net_flow_acceleration"] = vel_recent - vel_prior

    # Transaction Count Velocity & ROC
    tc6 = ch_tx_count[:, -1]
    tc1 = ch_tx_count[:, 0]
    features["quant_mom_tx_count_roc_6m"] = (tc6 - tc1) / (tc1 + 1.0)
    features["quant_mom_tx_count_velocity"] = (ch_tx_count[:, -1] - ch_tx_count[:, -3]) / 2.0

    # Login Active Days Velocity
    features["quant_mom_login_velocity"] = (ch_login_active[:, -1] - ch_login_active[:, -3]) / 2.0

    # RSI-like Relative Cash Flow Index: Up-months vs Down-months
    diffs = np.diff(ch_net_flow, axis=1)  # (N, 5)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = np.mean(gains, axis=1)
    avg_loss = np.mean(losses, axis=1)
    rs = avg_gain / (avg_loss + 1e-4)
    features["quant_mom_cash_flow_rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # Consecutive Negative Net Flow Streaks (Looking backwards from month 6)
    streaks = np.zeros(ch_net_flow.shape[0], dtype=float)
    for t in reversed(range(ch_net_flow.shape[1])):
        is_neg = ch_net_flow[:, t] < 0
        streaks = np.where(is_neg & (streaks == (5 - t)), streaks + 1.0, streaks)
    features["quant_mom_consecutive_negative_net_flow_streak"] = streaks

    return features
