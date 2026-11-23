"""
Classical Quantitative and Signal Processing Feature Engineering Package.
Pure Classical ML (Zero Deep Learning).
"""

from .extractor import QuantSignalFeatureExtractor as QuantSignalFeatureExtractor
from .momentum import extract_momentum_features as extract_momentum_features
from .macd import extract_macd_features as extract_macd_features
from .volatility import extract_volatility_features as extract_volatility_features
from .fourier import extract_fourier_features as extract_fourier_features
from .lags import extract_lag_features as extract_lag_features
from .cyclic import extract_cyclic_features as extract_cyclic_features
from .exogenous import extract_exogenous_features as extract_exogenous_features
from .interactions import extract_interaction_features as extract_interaction_features
from .multiscale_rolling import extract_multiscale_rolling_features as extract_multiscale_rolling_features

__all__ = [
    "QuantSignalFeatureExtractor",
    "extract_momentum_features",
    "extract_macd_features",
    "extract_volatility_features",
    "extract_fourier_features",
    "extract_lag_features",
    "extract_cyclic_features",
    "extract_exogenous_features",
    "extract_interaction_features",
    "extract_multiscale_rolling_features",
]
