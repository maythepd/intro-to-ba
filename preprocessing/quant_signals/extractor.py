"""
Classical Quantitative & Signal Processing Feature Engineering Orchestrator.
Pure Classical ML (Zero Deep Learning dependencies).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from ..base import BaseFeatureTransformer
from .momentum import extract_momentum_features
from .macd import extract_macd_features
from .volatility import extract_volatility_features
from .fourier import extract_fourier_features
from .lags import extract_lag_features
from .cyclic import extract_cyclic_features
from .exogenous import extract_exogenous_features
from .interactions import extract_interaction_features
from .multiscale_rolling import extract_multiscale_rolling_features


class QuantSignalFeatureExtractor(BaseFeatureTransformer):
    """
    Classical quantitative and signal processing feature extractor for customer transaction and login histories.
    """

    def __init__(
        self,
        enabled: bool = True,
        enable_momentum: bool = True,
        enable_macd: bool = True,
        enable_volatility: bool = True,
        enable_fourier: bool = True,
        enable_exogenous: bool = True,
        enable_lags: bool = True,
        enable_rolling: bool = True,
        enable_cyclic_time: bool = True,
        enable_interactions: bool = True,
        enable_multiscale_rolling: bool = True,
        prune_sparse: bool = True,
        max_sparsity_threshold: float = 0.98,
        min_variance_threshold: float = 1e-6,
        outlier_clip_quantile: float = 0.995,
        raw_data_dir: Union[str, Path] = "data/full_dataset",
        processed_dir: Union[str, Path] = "data/processed",
        cache_path: Optional[Union[str, Path]] = "data/processed/quant_signal_features.csv",
        force_recompute: bool = False,
        verbose: bool = True,
        **kwargs: Any,
    ):
        super().__init__(name="QuantSignalFeatureExtractor", enabled=enabled)
        self.enable_momentum = enable_momentum
        self.enable_macd = enable_macd
        self.enable_volatility = enable_volatility
        self.enable_fourier = enable_fourier
        self.enable_exogenous = enable_exogenous
        self.enable_lags = enable_lags
        self.enable_rolling = enable_rolling
        self.enable_cyclic_time = enable_cyclic_time
        self.enable_interactions = enable_interactions
        self.enable_multiscale_rolling = enable_multiscale_rolling
        self.prune_sparse = prune_sparse
        self.max_sparsity_threshold = max_sparsity_threshold
        self.min_variance_threshold = min_variance_threshold
        self.outlier_clip_quantile = outlier_clip_quantile
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.cache_path = Path(cache_path) if cache_path else self.processed_dir / "quant_signal_features.csv"
        self.force_recompute = force_recompute
        self.verbose = verbose

        self.cached_df_: Optional[pd.DataFrame] = None
        self.valid_features_: List[str] = []
        self.clip_bounds_: Dict[str, Tuple[float, float]] = {}

    def _extract_all_features(self) -> pd.DataFrame:
        """
        Extracts all signal families from raw temporal event streams and merges them.
        Saves result to cache for instant sub-second retrieval.
        """
        if self.verbose:
            print("[INFO] Computing Quant & Multi-Resolution Signal features from raw data...")

        # Load raw temporal matrix representations from tensor cache or CSV
        tensor_npz = self.processed_dir / "temporal" / "raw_snapshot_tensor_37d.npz"
        if not tensor_npz.exists():
            tensor_npz = Path("data/processed/temporal/raw_snapshot_tensor_37d.npz")

        snapshot_csv = self.processed_dir / "temporal" / "temporal_features_raw.csv"
        if not snapshot_csv.exists():
            snapshot_csv = Path("data/processed/temporal/temporal_features_raw.csv")

        ch_net_flow: Optional[np.ndarray] = None
        ch_tx_count: Optional[np.ndarray] = None
        ch_login_active: Optional[np.ndarray] = None
        customer_ids: List[int] = []

        if tensor_npz.exists():
            npz = np.load(tensor_npz, allow_pickle=True)
            customer_ids = npz["customer_ids"].tolist()
            tensor = npz["tensor"]  # (N, 6, 37)
            # Channel indices: tx_net_flow_log=2, tx_count_log=3, login_active_days=12
            ch_net_flow = tensor[:, :, 2]
            ch_tx_count = tensor[:, :, 3]
            ch_login_active = tensor[:, :, 12]
        elif snapshot_csv.exists():
            snap_df = pd.read_csv(snapshot_csv)
            customer_ids = snap_df["customer_id"].tolist()
            N = len(snap_df)

            def get_channel_matrix(col_prefix: str) -> np.ndarray:
                cols = [f"{col_prefix}_snap_{i}" for i in range(1, 7)]
                avail = [c for c in cols if c in snap_df.columns]
                if len(avail) == 6:
                    return snap_df[avail].values
                return np.zeros((N, 6))

            ch_net_flow = get_channel_matrix("net_flow")
            ch_tx_count = get_channel_matrix("tx_count")
            ch_login_active = get_channel_matrix("login_active_days")
        else:
            # Fallback to customer IDs from full bank data
            bank_csv = Path("data/full_bank_data.csv")
            if bank_csv.exists():
                bdf = pd.read_csv(bank_csv, usecols=["customer_id"])
                customer_ids = bdf["customer_id"].tolist()
            else:
                customer_ids = list(range(10000))

        features: Dict[str, np.ndarray] = {"customer_id": np.array(customer_ids)}

        # A. MOMENTUM
        if self.enable_momentum and ch_net_flow is not None:
            features.update(extract_momentum_features(ch_net_flow, ch_tx_count, ch_login_active))

        # B. MACD
        if self.enable_macd and ch_net_flow is not None:
            features.update(extract_macd_features(ch_net_flow, ch_tx_count, ch_login_active))

        # C. VOLATILITY
        if self.enable_volatility and ch_net_flow is not None:
            features.update(extract_volatility_features(ch_net_flow, ch_tx_count, ch_login_active))

        # D. FOURIER
        if self.enable_fourier and ch_net_flow is not None:
            features.update(extract_fourier_features(ch_net_flow, ch_tx_count, ch_login_active))

        # E. LAGGED DYNAMICS
        if self.enable_lags and ch_net_flow is not None:
            features.update(extract_lag_features(ch_net_flow, ch_tx_count, ch_login_active))

        # F. MULTI-SCALE & ROLLING DYNAMICS
        if self.enable_rolling or self.enable_multiscale_rolling:
            features.update(
                extract_multiscale_rolling_features(
                    raw_data_dir=self.raw_data_dir,
                    customer_ids=customer_ids,
                    ch_net_flow=ch_net_flow,
                    ch_tx_count=ch_tx_count,
                    ch_login_active=ch_login_active,
                )
            )

        df_base = pd.DataFrame(features)

        # G. EXOGENOUS SIGNALS
        bank_csv = Path("data/full_bank_data.csv")
        df_bank = pd.read_csv(bank_csv) if bank_csv.exists() else None

        tx_csv = self.raw_data_dir / "transactions.csv"
        acc_csv = self.raw_data_dir / "accounts.csv"
        tx_df: Optional[pd.DataFrame] = None
        if tx_csv.exists() and acc_csv.exists():
            acc = pd.read_csv(acc_csv, usecols=["account_id", "customer_id"])
            tx_raw = pd.read_csv(
                tx_csv,
                usecols=["account_id", "transaction_datetime", "amount", "direction", "transaction_status"],
            )
            tx_df = tx_raw[tx_raw["transaction_status"] == "success"].merge(acc, on="account_id", how="inner")

        if self.enable_exogenous and df_bank is not None:
            exo_feats = extract_exogenous_features(df_bank, tx_df, customer_ids)
            for col, arr in exo_feats.items():
                df_base[col] = arr

        # H. MULTIPLICATIVE INTERACTIONS
        if self.enable_interactions and df_bank is not None:
            q3_q2 = df_base["quant_exogenous_q3_to_q2_activity_ratio"].values if "quant_exogenous_q3_to_q2_activity_ratio" in df_base.columns else None
            inter_feats = extract_interaction_features(df_bank, customer_ids, q3_to_q2_ratio=q3_q2)
            for col, arr in inter_feats.items():
                df_base[col] = arr

        # I. SIN / COS CYCLICAL TIME EMBEDDINGS
        if self.enable_cyclic_time and tx_df is not None:
            cyclic_feats = extract_cyclic_features(tx_df, customer_ids)
            for col, arr in cyclic_feats.items():
                df_base[col] = arr

        # Clean infinities and NaNs
        df_base = df_base.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Save to cache
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_base.to_csv(self.cache_path, index=False)
        if self.verbose:
            print(f"[OK] Extracted and cached {df_base.shape[1] - 1} Quant & Multi-Resolution Signal features to {self.cache_path}")

        return df_base

    def _get_or_load_features(self) -> pd.DataFrame:
        """Loads cached features or extracts them from scratch."""
        if self.cached_df_ is not None and not self.force_recompute:
            return self.cached_df_

        if self.cache_path.exists() and not self.force_recompute:
            try:
                self.cached_df_ = pd.read_csv(self.cache_path)
                return self.cached_df_
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Failed to load cache {self.cache_path}: {e}. Recomputing...")

        self.cached_df_ = self._extract_all_features()
        return self.cached_df_

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "QuantSignalFeatureExtractor":
        """
        Fits variance thresholds and robust winsorization bounds strictly on X (training set).
        Prunes degenerate features (>98% zeros or zero variance).
        """
        if not self.enabled:
            return self

        full_df = self._get_or_load_features()
        if "customer_id" not in X.columns:
            if self.verbose:
                print("[WARN] 'customer_id' not found in X. QuantSignalFeatureExtractor skipped.")
            return self

        req_ids = set(X["customer_id"].values)
        train_slice = full_df[full_df["customer_id"].isin(req_ids)].copy()

        candidate_cols = [c for c in full_df.columns if c != "customer_id" and c.startswith("quant_")]
        valid_cols: List[str] = []
        clip_bounds: Dict[str, Tuple[float, float]] = {}

        for col in candidate_cols:
            vals = train_slice[col].dropna().values
            if len(vals) == 0:
                continue

            var_val = float(np.var(vals))
            zero_fraction = float(np.mean(vals == 0.0))

            # Pruning check
            if self.prune_sparse:
                if var_val < self.min_variance_threshold:
                    if self.verbose:
                        print(f"  [PRUNED] {col} (variance {var_val:.2e} < {self.min_variance_threshold})")
                    continue
                if zero_fraction > self.max_sparsity_threshold:
                    if self.verbose:
                        print(f"  [PRUNED] {col} (zero fraction {zero_fraction*100:.1f}% > {self.max_sparsity_threshold*100:.1f}%)")
                    continue

            # Robust outlier clip bounds computed strictly on training data
            q_low = float(np.quantile(vals, 1.0 - self.outlier_clip_quantile))
            q_high = float(np.quantile(vals, self.outlier_clip_quantile))
            clip_bounds[col] = (q_low, q_high)
            valid_cols.append(col)

        self.valid_features_ = valid_cols
        self.clip_bounds_ = clip_bounds
        self.created_features = list(valid_cols)
        self.is_fitted = True

        if self.verbose:
            print(f"[OK] QuantSignalFeatureExtractor fitted: Retained {len(self.valid_features_)} / {len(candidate_cols)} candidate features.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Merges validated and clipped quant features by customer_id."""
        if not self.enabled or not self.valid_features_:
            return X.copy()

        if "customer_id" not in X.columns:
            return X.copy()

        full_df = self._get_or_load_features()
        subset_cols = ["customer_id"] + self.valid_features_
        feat_df = full_df[subset_cols].copy()

        # Apply training clip bounds
        for col in self.valid_features_:
            q_low, q_high = self.clip_bounds_.get(col, (-np.inf, np.inf))
            feat_df[col] = feat_df[col].clip(lower=q_low, upper=q_high)

        # Merge with input DataFrame
        out = X.merge(feat_df, on="customer_id", how="left")

        # Fill any missing with 0.0
        for col in self.valid_features_:
            out[col] = out[col].fillna(0.0)

        return out
