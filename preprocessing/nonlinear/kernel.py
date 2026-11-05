"""
Non-Linear Kernel Representation Transformer using Nystroem RBF Approximation.
"""

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler


class NystroemKernelTransformer(BaseEstimator, TransformerMixin):
    """
    Maps input features into an explicit finite-dimensional Hilbert space approximation
    using Gaussian Radial Basis Function (RBF) kernel landmarks.
    """

    def __init__(
        self,
        n_components: int = 32,
        kernel: str = "rbf",
        gamma: float = 0.01,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.nystroem = Nystroem(
            kernel=self.kernel,
            gamma=self.gamma,
            n_components=self.n_components,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "NystroemKernelTransformer":
        X_s = self.scaler.fit_transform(X)
        self.nystroem.fit(X_s)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        return self.nystroem.transform(X_s).astype(np.float32)

