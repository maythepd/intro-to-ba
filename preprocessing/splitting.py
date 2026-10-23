"""
Dataset Partitioning & Cross-Validation Module.
Provides leak-free Train/Validation/Test splitting and Stratified K-Fold Cross-Validation
with ratios and folds configured via config/config.yaml.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


@dataclass
class DatasetSplit:
    """Container for partitioned train, validation, and test dataframes."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    val_df: Optional[pd.DataFrame] = None

    @property
    def has_val(self) -> bool:
        return self.val_df is not None and not self.val_df.empty


@dataclass
class CVFold:
    """Container for a single cross-validation fold partition."""

    fold: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_df: pd.DataFrame
    val_df: pd.DataFrame


@dataclass
class CVResult:
    """Container for aggregated cross-validation evaluation results."""

    fold_metrics: List[Dict[str, float]]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    metrics_summary_df: pd.DataFrame


class DataSplitter:
    """
    Handles robust, leak-free dataset partitioning (Train / Validation / Test)
    and Stratified K-Fold Cross-Validation.
    """

    def __init__(
        self,
        test_size: float = 0.20,
        val_size: float = 0.10,
        n_splits: int = 5,
        stratify: bool = True,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.n_splits = n_splits
        self.stratify = stratify
        self.shuffle = shuffle
        self.random_state = random_state

    def train_val_test_split(
        self,
        df: pd.DataFrame,
        target_col: str = "churn",
    ) -> DatasetSplit:
        """
        Splits dataset into stratified Train, Validation, and Test sets.
        If val_size == 0.0, val_df will be None.
        
        Example:
          test_size=0.20, val_size=0.10 -> 70% Train, 10% Validation, 20% Test.
        """
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataframe.")

        stratify_series = df[target_col] if self.stratify else None

        # 1. Split off independent test set first
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_series,
            shuffle=self.shuffle,
        )

        # 2. If val_size is requested, split train_val into train and validation
        if self.val_size > 0.0:
            # Scale val_size relative to remaining train_val portion
            adjusted_val_ratio = self.val_size / (1.0 - self.test_size)
            adjusted_val_ratio = min(max(adjusted_val_ratio, 0.01), 0.50)

            val_stratify = train_val_df[target_col] if self.stratify else None
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=adjusted_val_ratio,
                random_state=self.random_state,
                stratify=val_stratify,
                shuffle=self.shuffle,
            )
            return DatasetSplit(
                train_df=train_df.copy(),
                val_df=val_df.copy(),
                test_df=test_df.copy(),
            )

        return DatasetSplit(
            train_df=train_val_df.copy(),
            test_df=test_df.copy(),
            val_df=None,
        )

    def get_cv_folds(
        self,
        df: pd.DataFrame,
        target_col: str = "churn",
        n_splits: Optional[int] = None,
    ) -> Generator[CVFold, None, None]:
        """
        Generates leak-free cross-validation folds.
        Yields CVFold containers with train_df and val_df slices.
        """
        k = n_splits or self.n_splits
        if self.stratify and target_col in df.columns:
            cv = StratifiedKFold(n_splits=k, shuffle=self.shuffle, random_state=self.random_state)
            splits = cv.split(df, df[target_col])
        else:
            cv = KFold(n_splits=k, shuffle=self.shuffle, random_state=self.random_state)
            splits = cv.split(df)

        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            yield CVFold(
                fold=fold_idx,
                train_indices=train_idx,
                val_indices=val_idx,
                train_df=df.iloc[train_idx].copy(),
                val_df=df.iloc[val_idx].copy(),
            )

    def evaluate_cv(
        self,
        df: pd.DataFrame,
        pipeline_builder: Callable[[], Any],
        model_builder: Callable[[Any, pd.Series], Any],
        target_col: str = "churn",
        n_splits: Optional[int] = None,
        verbose: bool = False,
    ) -> CVResult:
        """
        Executes leak-free Cross-Validation:
        For each fold:
          1. Fits pipeline strictly on fold train_df.
          2. Transforms fold train_df and val_df.
          3. Resamples fold train features (e.g. SMOTE).
          4. Fits model on train fold and evaluates on val fold.
        """
        k = n_splits or self.n_splits
        fold_metrics: List[Dict[str, float]] = []

        for fold in self.get_cv_folds(df, target_col=target_col, n_splits=k):
            # 1. Fresh pipeline for this fold to avoid any leakage
            pipe = pipeline_builder()
            pipe.fit(fold.train_df, target_col=target_col)

            # 2. Transform both folds
            train_tr = pipe.transform(fold.train_df)
            val_tr = pipe.transform(fold.val_df)

            # 3. Extract model numeric features
            features = pipe.extract_model_features(train_tr)
            X_tr = train_tr[features].copy()
            y_tr = train_tr[target_col].copy()
            X_v = val_tr[features].copy()
            y_v = val_tr[target_col].copy()

            # 4. Resample train partition (SMOTE)
            X_tr_res, y_tr_res = pipe.balancer.fit_resample(X_tr, y_tr)

            # 4b. Feature selection on fold train (strictly leak-free)
            if pipe.config.enable_feature_selection and pipe.config.top_k_features:
                pipe.feature_selector.fit(X_tr, y_tr, compute_ranking=True)
                X_tr_res = pipe.feature_selector.transform(X_tr_res)
                X_v = pipe.feature_selector.transform(X_v)

            # 5. Build and fit model
            model = model_builder(pipe.config, y_tr_res)
            model.fit(X_tr_res, y_tr_res, X_val=X_v, y_val=y_v, verbose=False)

            # 6. Evaluate
            metrics = model.evaluate(X_v, y_v)
            metrics["fold"] = fold.fold
            fold_metrics.append(metrics)

            if verbose:
                print(
                    f"Fold {fold.fold}/{k} | ROC-AUC: {metrics.get('ROC_AUC', 0):.4f} "
                    f"| PR-AUC: {metrics.get('PR_AUC', 0):.4f} | F1: {metrics.get('F1_Score', 0):.4f}"
                )

        df_metrics = pd.DataFrame(fold_metrics)
        metric_cols = [c for c in df_metrics.columns if c != "fold"]

        mean_metrics = df_metrics[metric_cols].mean().to_dict()
        std_metrics = df_metrics[metric_cols].std().to_dict()

        return CVResult(
            fold_metrics=fold_metrics,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            metrics_summary_df=df_metrics,
        )
