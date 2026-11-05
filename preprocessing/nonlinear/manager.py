"""
Nonlinear Representation Manager and Unified Orchestrator.
Coordinates non-linear and neural representation transformers, handles checkpoint caching,
and manages feature augmentation/replacement.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .kernel import NystroemKernelTransformer
from .autoencoder import TabularDenoisingAutoencoderTransformer
from .supervised_latent import SupervisedTabularEmbeddingTransformer
from .contrastive import ContrastiveTabularTransformer


class NonlinearRepresentationManager(BaseEstimator, TransformerMixin):
    """
    Unified manager for non-linear feature space transformations.
    
    Supports:
      - Methods: 'supervised' | 'contrastive' | 'autoencoder' | 'kernel'
      - Modes: 'augment' (concatenates raw features + latent z) | 'replace' (returns only z)
      - Checkpoint caching: if checkpoint exists in checkpoint_dir, loads directly from disk;
        otherwise trains on X_train and saves the checkpoint.
    """

    def __init__(
        self,
        enabled: bool = False,
        method: str = "supervised",
        mode: str = "augment",
        latent_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 1e-3,
        dropout: float = 0.20,
        noise_level: float = 0.15,
        temperature: float = 0.10,
        kernel_components: int = 32,
        checkpoint_dir: Union[str, Path] = "models/saved/nonlinear",
        random_state: int = 42,
    ):
        self.enabled = enabled
        self.method = method.lower()
        self.mode = mode.lower()  # "augment" or "replace"
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.noise_level = noise_level
        self.temperature = temperature
        self.kernel_components = kernel_components
        self.checkpoint_dir = Path(checkpoint_dir)
        self.random_state = random_state

        self.transformer: Optional[Any] = None
        self.feature_names_in_: List[str] = []
        self.latent_feature_names_: List[str] = []

    def _resolve_checkpoint_path(self) -> Path:
        """Returns the dedicated checkpoint file path for the active method."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir / f"{self.method}_encoder.pt"

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> "NonlinearRepresentationManager":
        if not self.enabled:
            return self

        # Extract numpy array & feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_arr = X.values.astype(np.float32)
        else:
            self.feature_names_in_ = [f"feat_{i}" for i in range(X.shape[1])]
            X_arr = np.asarray(X, dtype=np.float32)

        y_arr = np.asarray(y) if y is not None else None
        ckpt_path = self._resolve_checkpoint_path()

        if self.method == "supervised":
            if y_arr is None:
                raise ValueError("Method 'supervised' requires y target labels to fit.")
            self.transformer = SupervisedTabularEmbeddingTransformer(
                latent_dim=self.latent_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                random_state=self.random_state,
                checkpoint_path=ckpt_path,
            )
            self.transformer.fit(X_arr, y_arr)
            actual_dim = self.latent_dim

        elif self.method == "contrastive":
            if y_arr is None:
                raise ValueError("Method 'contrastive' requires y target labels to fit.")
            self.transformer = ContrastiveTabularTransformer(
                latent_dim=self.latent_dim,
                hidden_dims=self.hidden_dims,
                temperature=self.temperature,
                dropout=self.dropout,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                random_state=self.random_state,
                checkpoint_path=ckpt_path,
            )
            self.transformer.fit(X_arr, y_arr)
            actual_dim = self.latent_dim

        elif self.method in ("autoencoder", "dae"):
            self.transformer = TabularDenoisingAutoencoderTransformer(
                latent_dim=self.latent_dim,
                hidden_dims=self.hidden_dims,
                noise_level=self.noise_level,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                random_state=self.random_state,
                checkpoint_path=ckpt_path,
            )
            self.transformer.fit(X_arr)
            actual_dim = self.latent_dim

        elif self.method in ("kernel", "nystroem", "rbf"):
            self.transformer = NystroemKernelTransformer(
                n_components=self.kernel_components,
                kernel="rbf",
                gamma=0.01,
                random_state=self.random_state,
            )
            self.transformer.fit(X_arr)
            actual_dim = self.kernel_components

        else:
            raise ValueError(f"Unknown nonlinear representation method: '{self.method}'. Choose from: supervised, contrastive, autoencoder, kernel.")

        self.latent_feature_names_ = [f"nl_{self.method}_z{i}" for i in range(actual_dim)]
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if not self.enabled or self.transformer is None:
            return X

        is_df = isinstance(X, pd.DataFrame)
        cols = X.columns.tolist() if is_df else self.feature_names_in_
        X_arr = X.values.astype(np.float32) if is_df else np.asarray(X, dtype=np.float32)

        # Extract latent representations
        z = self.transformer.transform(X_arr)

        if self.mode == "replace":
            if is_df:
                return pd.DataFrame(z, columns=self.latent_feature_names_, index=X.index)
            return z

        # Default: "augment" (concatenates raw features + latent z)
        if is_df:
            z_df = pd.DataFrame(z, columns=self.latent_feature_names_, index=X.index)
            return pd.concat([X, z_df], axis=1)
        
        return np.hstack([X_arr, z])

