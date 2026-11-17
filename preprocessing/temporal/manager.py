"""
Temporal Embedding Manager.
Coordinates raw multi-table event ingestion, multi-snapshot sequence tensor construction,
deep recurrent sequence autoencoding, artifact caching, and feature augmentation.
Supports pluggable representation learning strategies:
  - unsupervised (pure self-supervised baseline)
  - asymmetric_arcface (hyperspherical margin)
  - hyperbolic_poincare (Poincaré ball)
  - optimal_transport (drift velocity field)
  - neyman_pearson (bounded FPR risk control)
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .extractor import TemporalSnapshotExtractor
from .raw_extractor import RawEventSequenceExtractor
from .model import TemporalAutoencoderTrainer
from .strategies import get_temporal_strategy, BaseTemporalStrategy


class TemporalEmbeddingManager(BaseEstimator, TransformerMixin):
    """
    Coordinates extraction of longitudinal multi-period event snapshots and
    transformation into dense customer behavioral trajectory embeddings.
    
    Pipeline integration:
      - fit(df): Extracts raw event tensor, trains BiGRUTemporalAutoencoder strictly on df, caches train embeddings.
      - transform(df): Merges dense temporal embedding columns via customer_id.
                       If unseen customers are encountered (val/test sets), runs forward-pass inference in eval mode!
    """

    def __init__(
        self,
        enabled: bool = False,
        raw_data_dir: Union[str, Path] = "data/full_dataset",
        temporal_input_mode: str = "extractors",
        raw_sequence_length: int = 50,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        epochs: int = 25,
        batch_size: int = 256,
        lr: float = 1e-3,
        dropout: float = 0.20,
        checkpoint_dir: Union[str, Path] = "models/saved/temporal",
        cache_path: Union[str, Path] = "data/processed/temporal_embeddings.csv",
        random_state: int = 42,
        verbose: bool = True,
        fold_id: Optional[Union[int, str]] = None,
        force_retrain: bool = False,
        manifold_strategy: str = "unsupervised",
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        # Direct keyword args for backwards compatibility:
        lambda_angular: float = 0.30,
        margin_churn: float = 0.20,
        margin_loyal: float = 0.00,
        angular_scale: float = 4.0,
    ):
        self.enabled = enabled
        self.raw_data_dir = Path(raw_data_dir)
        self.temporal_input_mode = str(temporal_input_mode).lower()
        self.raw_sequence_length = int(raw_sequence_length)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cache_path = Path(cache_path)
        self.random_state = random_state
        self.verbose = verbose
        self.fold_id = fold_id
        self.force_retrain = force_retrain

        self.manifold_strategy = str(manifold_strategy).lower()
        self.strategy_kwargs = dict(strategy_kwargs or {})
        self.lambda_angular = lambda_angular
        self.margin_churn = margin_churn
        self.margin_loyal = margin_loyal
        self.angular_scale = angular_scale

        if self.manifold_strategy == "asymmetric_arcface":
            self.strategy_kwargs.setdefault("scale", self.angular_scale)
            self.strategy_kwargs.setdefault("margin_churn", self.margin_churn)
            self.strategy_kwargs.setdefault("margin_loyal", self.margin_loyal)
            self.strategy_kwargs.setdefault("lambda_angular", self.lambda_angular)

        self.strategy: BaseTemporalStrategy = get_temporal_strategy(self.manifold_strategy, **self.strategy_kwargs)

        # Route fold-specific paths if fold_id is provided
        mode_prefix = "raw_" if self.temporal_input_mode in ("raw_events", "raw", "raw_logs") else ""
        strat_prefix = f"{self.manifold_strategy}_" if self.manifold_strategy != "unsupervised" else ""
        prefix = f"{mode_prefix}{strat_prefix}"
        if self.fold_id is not None:
            self.checkpoint_dir = self.checkpoint_dir / "folds"
            self.cache_path = Path("data/processed/temporal") / f"temporal_embeddings_{prefix}fold{self.fold_id}.csv"
        else:
            self.cache_path = Path("data/processed/temporal") / f"temporal_embeddings_{prefix}full.csv"

        self.trainer: Optional[TemporalAutoencoderTrainer] = None
        self.embeddings_df_: Optional[pd.DataFrame] = None

        self.latent_cols_: List[str] = [f"temporal_z{i}" for i in range(self.latent_dim)]
        all_feature_cols = self.strategy.get_feature_names(self.latent_dim)
        self.geom_cols_: List[str] = [c for c in all_feature_cols if not c.startswith("temporal_z")]
        self.embedding_cols_: List[str] = self.latent_cols_ + self.geom_cols_

    def _get_extractor(self):
        if self.temporal_input_mode in ("raw_events", "raw", "raw_logs"):
            return RawEventSequenceExtractor(
                raw_data_dir=self.raw_data_dir,
                max_seq_len=self.raw_sequence_length,
            )
        return TemporalSnapshotExtractor(
            raw_data_dir=self.raw_data_dir,
        )

    def _resolve_checkpoint_path(self) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        mode_prefix = "raw_" if self.temporal_input_mode in ("raw_events", "raw", "raw_logs") else ""
        strat_prefix = f"{self.manifold_strategy}_" if self.manifold_strategy != "unsupervised" else ""
        prefix = f"{mode_prefix}{strat_prefix}"
        if self.fold_id is not None:
            return self.checkpoint_dir / f"temporal_encoder_{prefix}fold{self.fold_id}.pt"
        return self.checkpoint_dir / f"temporal_encoder_{prefix}full.pt"

    def fit(self, df: pd.DataFrame, y: Optional[Any] = None) -> "TemporalEmbeddingManager":
        """
        Fits temporal sequence autoencoder strictly on the training partition.
        Leverages training labels if the active strategy requires metric supervision.
        """
        if not self.enabled:
            return self

        if "customer_id" not in df.columns:
            raise KeyError("DataFrame must contain 'customer_id' column to extract temporal event sequences.")

        req_ids = set(df["customer_id"].values)

        # 1. Check if cached embeddings CSV already exists AND covers all customers in df (unless force_retrain)
        if not self.force_retrain and self.cache_path.exists():
            try:
                cached_df = pd.read_csv(self.cache_path)
                expected_cols = ["customer_id"] + self.embedding_cols_
                if all(c in cached_df.columns for c in expected_cols):
                    cached_ids = set(cached_df["customer_id"].values)
                    if req_ids.issubset(cached_ids):
                        self.embeddings_df_ = cached_df[expected_cols].copy()
                        if self.verbose:
                            fold_tag = f" [Fold {self.fold_id}]" if self.fold_id is not None else ""
                            print(
                                f"[INFO] Loaded pre-cached temporal embeddings{fold_tag} from: "
                                f"{self.cache_path} ({len(self.embeddings_df_)} customers)"
                            )
                        return self
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Failed to read cache ({e}). Re-extracting from raw event logs...")

        # 2. Extract sequence tensor strictly for training customers
        cust_ids = df["customer_id"].values
        extractor = self._get_extractor()
        tensor, extracted_ids = extractor.extract(cust_ids, verbose=self.verbose)

        # Extract training labels aligned by customer_id if available
        labels = None
        if y is not None:
            labels = np.asarray(y)
        elif "churn" in df.columns:
            labels = df["churn"].values.astype(int)

        # 3. Fit Temporal Sequence Representation strictly on this partition
        ckpt_path = self._resolve_checkpoint_path()
        self.trainer = TemporalAutoencoderTrainer(
            in_features=tensor.shape[2],
            seq_len=tensor.shape[1],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            latent_dim=self.latent_dim,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            dropout=self.dropout,
            checkpoint_path=ckpt_path,
            random_state=self.random_state,
            verbose=self.verbose,
            force_retrain=self.force_retrain,
            manifold_strategy=self.manifold_strategy,
            strategy_kwargs=self.strategy_kwargs,
            lambda_angular=self.lambda_angular,
            margin_churn=self.margin_churn,
            margin_loyal=self.margin_loyal,
            angular_scale=self.angular_scale,
        )
        self.trainer.fit(tensor, labels=labels)

        # 4. Generate Latent Embeddings z & Geometric features for Train customers
        latents, geom_dict = self.trainer.transform_with_geometry(tensor)
        emb_data = {"customer_id": extracted_ids}
        for i, col in enumerate(self.latent_cols_):
            emb_data[col] = latents[:, i]
        for col in self.geom_cols_:
            if col in geom_dict:
                emb_data[col] = geom_dict[col]
            else:
                emb_data[col] = np.zeros(len(extracted_ids), dtype=np.float32)

        self.embeddings_df_ = pd.DataFrame(emb_data)

        # 5. Save cached embeddings to disk
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.embeddings_df_.to_csv(self.cache_path, index=False)
            if self.verbose:
                print(f"[OK] Saved extracted temporal embeddings to cache: {self.cache_path}")
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Could not cache embeddings to disk: {e}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments dataframe by merging dense temporal embeddings via customer_id.
        If any customers in df are missing from embeddings_df_ (e.g. validation/test sets),
        extracts their temporal sequence tensor and infers latent embeddings z using
        the fitted BiGRUTemporalAutoencoder in eval mode (pure forward pass, zero target leakage)!
        """
        if not self.enabled:
            return df.copy()

        df_out = df.copy()
        if "customer_id" not in df_out.columns:
            return df_out

        cust_ids = df_out["customer_id"].values
        known_ids = set(self.embeddings_df_["customer_id"].values) if self.embeddings_df_ is not None else set()
        missing_ids = [cid for cid in cust_ids if cid not in known_ids]

        # If unseen customers are present (Val or Test set), run inference on them using the fitted trainer
        if missing_ids:
            if self.trainer is None:
                ckpt_path = self._resolve_checkpoint_path()
                if ckpt_path.exists():
                    in_feat = 26 if self.temporal_input_mode in ("raw_events", "raw", "raw_logs") else 37
                    seq_l = self.raw_sequence_length if self.temporal_input_mode in ("raw_events", "raw", "raw_logs") else 6
                    try:
                        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
                        in_feat = ckpt_dict.get("metadata", {}).get("in_features", in_feat)
                        seq_l = ckpt_dict.get("metadata", {}).get("seq_len", seq_l)
                    except Exception:
                        pass

                    self.trainer = TemporalAutoencoderTrainer(
                        in_features=in_feat,
                        seq_len=seq_l,
                        hidden_dim=self.hidden_dim,
                        num_layers=self.num_layers,
                        latent_dim=self.latent_dim,
                        checkpoint_path=ckpt_path,
                        verbose=False,
                        manifold_strategy=self.manifold_strategy,
                        strategy_kwargs=self.strategy_kwargs,
                        lambda_angular=self.lambda_angular,
                        margin_churn=self.margin_churn,
                        margin_loyal=self.margin_loyal,
                        angular_scale=self.angular_scale,
                    )
                    # Loads existing checkpoint
                    self.trainer.fit(np.zeros((1, seq_l, in_feat), dtype=np.float32))

            if self.trainer is not None:
                if self.verbose:
                    fold_tag = f" [Fold {self.fold_id}]" if self.fold_id is not None else ""
                    print(
                        f"[INFO] Inferring temporal embeddings{fold_tag} for {len(missing_ids)} "
                        f"unseen customers (eval mode)..."
                    )
                extractor = self._get_extractor()
                missing_tensor, actual_ids = extractor.extract(missing_ids, verbose=False)
                missing_latents, missing_geom = self.trainer.transform_with_geometry(missing_tensor)

                new_emb_data = {"customer_id": actual_ids}
                for i, col in enumerate(self.latent_cols_):
                    new_emb_data[col] = missing_latents[:, i]
                for col in self.geom_cols_:
                    if col in missing_geom:
                        new_emb_data[col] = missing_geom[col]
                    else:
                        new_emb_data[col] = np.zeros(len(actual_ids), dtype=np.float32)

                df_new = pd.DataFrame(new_emb_data)

                if self.embeddings_df_ is None:
                    self.embeddings_df_ = df_new
                else:
                    self.embeddings_df_ = pd.concat([self.embeddings_df_, df_new], ignore_index=True)

                # Update cache on disk with newly inferred customers
                try:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    self.embeddings_df_.to_csv(self.cache_path, index=False)
                except Exception:
                    pass

        if self.embeddings_df_ is not None:
            df_out = df_out.merge(self.embeddings_df_, on="customer_id", how="left")
            for col in self.embedding_cols_:
                if col in df_out.columns:
                    df_out[col] = df_out[col].fillna(0.0)

        return df_out

    def get_auxiliary_metrics(self) -> Dict[str, Any]:
        """Returns autoencoder reconstruction metrics."""
        if self.trainer is not None:
            return self.trainer.metrics_
        return {}

    def get_history(self) -> Dict[str, List[float]]:
        """Returns autoencoder training history."""
        if self.trainer is not None:
            return self.trainer.history_
        return {}
