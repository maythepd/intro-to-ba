"""
GNN Embedding Manager for Customer Network Context.
Coordinates k-NN graph construction, self-supervised Graph Attention Network training,
inductive evaluation for unseen customers, artifact caching, and feature augmentation.
Zero target leakage (completely unsupervised).
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import torch

from .graph_builder import CustomerGraphBuilder
from .model import GNNTrainer


class GNNEmbeddingManager(BaseEstimator, TransformerMixin):
    """
    Coordinates extraction of topological customer neighborhood embeddings (z_gnn).
    Pipeline integration:
      - fit(df): Builds train k-NN similarity graph, trains GAT Autoencoder, caches train embeddings.
      - transform(df): Merges dense GNN embedding columns (gnn_z0..gnn_z{d-1}) via customer_id.
                       If unseen customers are present (Val/Test splits), runs inductive inference
                       by linking eval nodes strictly into reference train nodes (zero leakage)!
    """

    EXCLUDE_COLS_FOR_GRAPH = [
        "customer_id",
        "churn",
        "country",
        "gender",
        "preferred_channel",
        "primary_device",
    ]

    def __init__(
        self,
        enabled: bool = False,
        k_neighbors: int = 5,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        heads: int = 2,
        epochs: int = 20,
        lr: float = 0.005,
        dropout: float = 0.20,
        checkpoint_dir: Union[str, Path] = "models/saved/gnn",
        cache_path: Union[str, Path] = "data/processed/gnn/gnn_embeddings.csv",
        random_state: int = 42,
        verbose: bool = True,
        fold_id: Optional[Union[int, str]] = None,
        force_retrain: bool = False,
    ):
        self.enabled = enabled
        self.k_neighbors = k_neighbors
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cache_path = Path(cache_path)
        self.random_state = random_state
        self.verbose = verbose
        self.fold_id = fold_id
        self.force_retrain = force_retrain

        prefix = f"fold{self.fold_id}_" if self.fold_id is not None else ""
        if self.fold_id is not None:
            self.checkpoint_dir = self.checkpoint_dir / "folds"
            self.cache_path = Path("data/processed/gnn") / f"gnn_embeddings_fold{self.fold_id}.csv"
        else:
            self.cache_path = Path("data/processed/gnn") / f"gnn_embeddings_full.csv"

        self.builder = CustomerGraphBuilder(k_neighbors=self.k_neighbors)
        self.trainer: Optional[GNNTrainer] = None
        self.embeddings_df_: Optional[pd.DataFrame] = None
        self.train_features_: Optional[np.ndarray] = None
        self.train_cust_ids_: Optional[np.ndarray] = None
        self.feature_cols_: List[str] = []

        self.gnn_cols_: List[str] = [f"gnn_z{i}" for i in range(self.latent_dim)]

    def _resolve_checkpoint_path(self) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"fold{self.fold_id}_" if self.fold_id is not None else ""
        return self.checkpoint_dir / f"gnn_encoder_{prefix}full.pt"

    def _extract_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts and standardizes numeric feature matrix for graph construction."""
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in self.EXCLUDE_COLS_FOR_GRAPH
        ]
        if not self.feature_cols_:
            self.feature_cols_ = sorted(numeric_cols)

        # Ensure consistent column ordering
        cols_to_use = [c for c in self.feature_cols_ if c in df.columns]
        mat = df[cols_to_use].fillna(0.0).values.astype(np.float32)

        # Standardize features
        mean = np.mean(mat, axis=0, keepdims=True)
        std = np.std(mat, axis=0, keepdims=True) + 1e-6
        mat_norm = np.clip((mat - mean) / std, -5.0, 5.0)

        cust_ids = df["customer_id"].values.astype(np.int64)
        return mat_norm, cust_ids

    def fit(self, df: pd.DataFrame, y: Optional[Any] = None) -> "GNNEmbeddingManager":
        """
        Fits GNN Autoencoder on the training split graph (100% unsupervised).
        """
        if not self.enabled:
            return self

        if "customer_id" not in df.columns:
            raise KeyError("DataFrame must contain 'customer_id' column to construct GNN embeddings.")

        req_ids = set(df["customer_id"].values)

        # 1. Check pre-computed cache
        if not self.force_retrain and self.cache_path.exists():
            try:
                cached_df = pd.read_csv(self.cache_path)
                expected_cols = ["customer_id"] + self.gnn_cols_
                if all(c in cached_df.columns for c in expected_cols):
                    cached_ids = set(cached_df["customer_id"].values)
                    if req_ids.issubset(cached_ids):
                        self.embeddings_df_ = cached_df[expected_cols].copy()
                        if self.verbose:
                            fold_tag = f" [Fold {self.fold_id}]" if self.fold_id is not None else ""
                            print(
                                f"[INFO] Loaded pre-cached GNN embeddings{fold_tag} from: "
                                f"{self.cache_path} ({len(self.embeddings_df_)} customers)"
                            )
                        # Save train features for inductive eval
                        self.train_features_, self.train_cust_ids_ = self._extract_feature_matrix(df)
                        return self
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Failed to read GNN cache ({e}). Rebuilding...")

        # 2. Extract feature matrix for train nodes
        X_train, cust_ids = self._extract_feature_matrix(df)
        self.train_features_ = X_train
        self.train_cust_ids_ = cust_ids

        # 3. Build k-NN training graph
        edge_index, X_norm = self.builder.build_train_graph(X_train)

        # 4. Fit Self-Supervised GNN Autoencoder
        ckpt_path = self._resolve_checkpoint_path()
        self.trainer = GNNTrainer(
            in_features=X_train.shape[1],
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            heads=self.heads,
            lr=self.lr,
            dropout=self.dropout,
            epochs=self.epochs,
            checkpoint_path=ckpt_path,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.trainer.fit(X_norm, edge_index)

        # 5. Extract latent representations z_gnn for train nodes
        latents = self.trainer.transform(X_norm, edge_index)
        emb_data = {"customer_id": cust_ids}
        for i, col in enumerate(self.gnn_cols_):
            emb_data[col] = latents[:, i]
        self.embeddings_df_ = pd.DataFrame(emb_data)

        # 6. Save cache to disk
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.embeddings_df_.to_csv(self.cache_path, index=False)
            if self.verbose:
                print(f"[OK] Saved GNN embeddings to cache: {self.cache_path}")
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Could not cache GNN embeddings: {e}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments dataframe with GNN embeddings via customer_id.
        For unseen evaluation customers, performs inductive message passing
        connected strictly into reference train nodes (zero leakage).
        """
        if not self.enabled:
            return df.copy()

        df_out = df.copy()
        if "customer_id" not in df_out.columns:
            return df_out

        cust_ids = df_out["customer_id"].values
        known_ids = set(self.embeddings_df_["customer_id"].values) if self.embeddings_df_ is not None else set()
        missing_ids = [cid for cid in cust_ids if cid not in known_ids]

        if missing_ids and self.trainer is not None and self.train_features_ is not None:
            if self.verbose:
                print(f"[INFO] Inductive GNN inference for {len(missing_ids)} unseen customers...")

            df_missing = df_out[df_out["customer_id"].isin(set(missing_ids))].copy()
            X_eval, missing_cust_ids = self._extract_feature_matrix(df_missing)

            # Inductive bipartite graph: eval nodes connected to train nodes
            edge_index_eval, combined_X = self.builder.build_inductive_eval_graph(
                eval_features=X_eval,
                train_features=self.train_features_,
            )

            # Transform combined graph and slice the eval node representations
            combined_latents = self.trainer.transform(combined_X, edge_index_eval)
            N_train = self.train_features_.shape[0]
            eval_latents = combined_latents[N_train:]

            emb_missing = {"customer_id": missing_cust_ids}
            for i, col in enumerate(self.gnn_cols_):
                emb_missing[col] = eval_latents[:, i]
            df_new = pd.DataFrame(emb_missing)

            if self.embeddings_df_ is None:
                self.embeddings_df_ = df_new
            else:
                self.embeddings_df_ = pd.concat([self.embeddings_df_, df_new], ignore_index=True)

        if self.embeddings_df_ is not None:
            df_out = df_out.merge(self.embeddings_df_, on="customer_id", how="left")
            for col in self.gnn_cols_:
                if col in df_out.columns:
                    df_out[col] = df_out[col].fillna(0.0)

        return df_out
