"""
Fast Fourier Transform (FFT) Spectral Signal Feature Extraction.
Pure Classical ML (Zero Deep Learning).
"""

from typing import Dict
import numpy as np


def extract_fourier_features(
    ch_net_flow: np.ndarray,
    ch_tx_count: np.ndarray,
    ch_login_active: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extracts frequency-domain spectral energy, entropy, and harmonic power ratios."""
    features: Dict[str, np.ndarray] = {}

    for s_mat, pfx in [
        (ch_net_flow, "net_flow"),
        (ch_tx_count, "tx_count"),
        (ch_login_active, "login"),
    ]:
        s_det = s_mat - np.mean(s_mat, axis=1, keepdims=True)
        fft_vals = np.fft.rfft(s_det, axis=1)[:, 1:]  # k=1, 2, 3
        p = np.abs(fft_vals) ** 2
        e = np.sum(p, axis=1)
        pr = p / (e[:, None] + 1e-8)
        ent = -np.sum(pr * np.log(pr + 1e-8), axis=1)
        features[f"quant_fourier_{pfx}_spectral_energy"] = np.log1p(e)
        features[f"quant_fourier_{pfx}_spectral_entropy"] = ent
        if pfx == "net_flow":
            features[f"quant_fourier_{pfx}_low_high_ratio"] = np.log1p(
                p[:, 0] / (p[:, 1] + p[:, 2] + 1e-4)
            )

    return features
