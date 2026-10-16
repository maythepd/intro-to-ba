"""
Master Feature Engineering Pipeline for Digital Bank Churn Prediction.
Orchestrates skewness transformation, scaling, domain feature engineering,
categorical encoding, SMOTE class balancing, and feature selection.

Configuration is decoupled into the config/ module with YAML support.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure config module is discoverable
try:
    from config import PipelineConfig
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import PipelineConfig

from .base import BaseFeatureTransformer
from .skewness_transform import SkewnessTransformer
from .scaling import FeatureScaler
from .domain_features import DomainFeatureEngineer
from .categorical_encoding import CategoricalEncoder
from .balancing import ClassBalancer
from .feature_selector import FeatureSelector


@dataclass
class PipelineResult:
    """Output container for prepared train/test feature sets."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_resampled: pd.DataFrame
    y_train_resampled: pd.Series
    model_features: List[str]
    feature_importance: Optional[pd.DataFrame] = None
    full_transformed_df: Optional[pd.DataFrame] = None


class FeatureEngineeringPipeline:
    """
    Final end-to-end Feature Engineering & Transformation Pipeline.
    
    Coordinates:
      1. SkewnessTransformer: Log1p, zero-balance hurdle, Yeo-Johnson
      2. FeatureScaler: RobustScaler and StandardScaler
      3. DomainFeatureEngineer: Velocity, friction, bundling, interaction terms
      4. CategoricalEncoder: One-Hot and smoothed Target Encoding
      5. ClassBalancer: SMOTE minority oversampling (train only)
      6. FeatureSelector: MI + RF importance ranking and Top-K selection
    """

    def __init__(
        self,
        config: Optional[Union[PipelineConfig, Dict[str, Any], str, Path]] = None,
    ):
        if config is None:
            self.config = PipelineConfig.from_yaml()
        elif isinstance(config, (str, Path)):
            self.config = PipelineConfig.from_yaml(config)
        elif isinstance(config, dict):
            self.config = PipelineConfig.from_dict(config)
        elif isinstance(config, PipelineConfig):
            self.config = config
        else:
            raise TypeError(f"Invalid config type: {type(config)}. Expected PipelineConfig, dict, str, or Path.")

        self._init_modules()

    @classmethod
    def from_yaml(cls, path: Optional[Union[str, Path]] = None) -> "FeatureEngineeringPipeline":
        """Instantiate pipeline directly from a YAML config file."""
        return cls(config=PipelineConfig.from_yaml(path))

    def _init_modules(self) -> None:
        """Instantiate sub-modules based on current configuration."""
        cfg = self.config

        # 1. Skewness
        self.skewness_transformer = SkewnessTransformer(
            enabled=cfg.enable_skewness,
            transform_trans_amount_log=cfg.transform_trans_amount_log,
            transform_recency_log=cfg.transform_recency_log,
            zero_balance_hurdle=cfg.zero_balance_hurdle,
            transform_yeo_johnson=cfg.transform_yeo_johnson,
        )

        # 2. Scaling
        self.scaler = FeatureScaler(
            enabled=cfg.enable_scaling,
            apply_robust_scaling=cfg.apply_robust_scaling,
            apply_standard_scaling=cfg.apply_standard_scaling,
            robust_features=cfg.robust_features,
            standard_features=cfg.standard_features,
        )

        # 3. Domain Features
        self.domain_engineer = DomainFeatureEngineer(
            enabled=cfg.enable_domain_features,
            enable_behavioral_momentum=cfg.enable_behavioral_momentum,
            enable_customer_friction=cfg.enable_customer_friction,
            enable_product_bundling=cfg.enable_product_bundling,
            enable_risk_interactions=cfg.enable_risk_interactions,
        )

        # 4. Categorical Encoding
        self.categorical_encoder = CategoricalEncoder(
            enabled=(cfg.enable_one_hot_encoding or cfg.enable_target_encoding),
            apply_ohe=cfg.enable_one_hot_encoding,
            apply_target_enc=cfg.enable_target_encoding,
            categorical_cols=cfg.categorical_cols,
            m_weight=cfg.target_encoding_m_weight,
            ohe_drop_first=cfg.ohe_drop_first,
        )

        # 5. Balancing
        self.balancer = ClassBalancer(
            enabled=cfg.enable_smote,
            method="smote",
            random_state=cfg.random_state,
            sampling_strategy=cfg.smote_sampling_strategy,
            k_neighbors=cfg.smote_k_neighbors,
        )

        # 6. Feature Selection
        self.feature_selector = FeatureSelector(
            enabled=cfg.enable_feature_selection,
            top_k=cfg.top_k_features,
            ranking_metric=cfg.feature_selection_metric,
            random_state=cfg.random_state,
        )

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> "FeatureEngineeringPipeline":
        """
        Fit all stateful transformers sequentially on training dataframe.
        """
        target_name = target_col or self.config.target_col
        y = df[target_name] if target_name in df.columns else None

        current = df.copy()

        # Step 1: Skewness
        self.skewness_transformer.fit(current, y)
        current = self.skewness_transformer.transform(current)

        # Step 2: Scaling
        self.scaler.fit(current, y)
        current = self.scaler.transform(current)

        # Step 3: Domain
        self.domain_engineer.fit(current, y)
        current = self.domain_engineer.transform(current)

        # Step 4: Categorical
        self.categorical_encoder.fit(current, y)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe using all fitted transformers sequentially.
        """
        out = df.copy()

        # Step 1: Skewness
        out = self.skewness_transformer.transform(out)

        # Step 2: Scaling
        out = self.scaler.transform(out)

        # Step 3: Domain
        out = self.domain_engineer.transform(out)

        # Step 4: Categorical
        out = self.categorical_encoder.transform(out)

        return out

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Fit on df and return transformed dataframe."""
        return self.fit(df, target_col).transform(df)

    def extract_model_features(
        self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select all numeric feature columns, filtering out specified exclusion columns.
        """
        excludes = exclude_cols or self.config.exclude_cols
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in excludes]

    def prepare_train_test(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        compute_feature_importance: Optional[bool] = None,
    ) -> PipelineResult:
        """
        End-to-end execution with leak-free Train/Test partitioning:
          1. Stratified split (80/20 by default).
          2. Fit transformers exclusively on train partition.
          3. Transform train and test partitions.
          4. Extract numerical model features.
          5. Apply SMOTE balancing on train partition.
          6. Compute feature importances & optionally select top-K features.
        """
        target = target_col or self.config.target_col
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found in dataframe.")

        # 1. Stratified Train/Test Split
        stratify_col = df[target] if self.config.stratify else None
        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify_col,
        )

        train_df = train_df.copy()
        test_df = test_df.copy()

        # 2. Fit transformers exclusively on train_df
        self.fit(train_df, target_col=target)

        # 3. Transform both train and test partitions
        train_tr = self.transform(train_df)
        test_tr = self.transform(test_df)

        # 4. Extract numeric feature matrices
        feature_cols = self.extract_model_features(train_tr)
        X_train = train_tr[feature_cols].copy()
        y_train = train_tr[target].copy()
        X_test = test_tr[feature_cols].copy()
        y_test = test_tr[target].copy()

        # 5. Class Imbalance Remediation (SMOTE) strictly on training set
        X_train_res, y_train_res = self.balancer.fit_resample(X_train, y_train)

        # 6. Feature Selection / Importance
        should_rank = (
            self.config.enable_feature_selection
            or (compute_feature_importance if compute_feature_importance is not None else self.config.compute_feature_importance)
        )
        self.feature_selector.fit(X_train, y_train, compute_ranking=should_rank)
        df_importance = self.feature_selector.get_importance_ranking()

        if self.config.enable_feature_selection and self.config.top_k_features:
            X_train = self.feature_selector.transform(X_train)
            X_test = self.feature_selector.transform(X_test)
            X_train_res = self.feature_selector.transform(X_train_res)
            feature_cols = list(X_train.columns)

        # Full transformed df (fit_transform on whole dataset for exploratory export)
        full_transformed = self.transform(df)

        return PipelineResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_train_resampled=X_train_res,
            y_train_resampled=y_train_res,
            model_features=feature_cols,
            feature_importance=df_importance,
            full_transformed_df=full_transformed,
        )

    def export_artifacts(
        self,
        result: PipelineResult,
        output_dir: str = "artifacts",
    ) -> Dict[str, str]:
        """
        Export all transformed feature matrices and reports matching notebook outputs.
        """
        os.makedirs(output_dir, exist_ok=True)
        exported: Dict[str, str] = {}

        # 1. Full transformed dataset
        if result.full_transformed_df is not None:
            p = os.path.join(output_dir, "transformed_features.csv")
            result.full_transformed_df.to_csv(p, index=False)
            exported["transformed_features"] = p

        # 2. X_train and X_test transformed
        p_xtrain = os.path.join(output_dir, "X_train_transformed.csv")
        result.X_train.to_csv(p_xtrain, index=False)
        exported["X_train"] = p_xtrain

        p_xtest = os.path.join(output_dir, "X_test_transformed.csv")
        result.X_test.to_csv(p_xtest, index=False)
        exported["X_test"] = p_xtest

        # 3. y_train and y_test
        p_ytrain = os.path.join(output_dir, "y_train.csv")
        result.y_train.to_csv(p_ytrain, index=False)
        exported["y_train"] = p_ytrain

        p_ytest = os.path.join(output_dir, "y_test.csv")
        result.y_test.to_csv(p_ytest, index=False)
        exported["y_test"] = p_ytest

        # 4. SMOTE balanced sets
        p_xsmote = os.path.join(output_dir, "X_train_smote.csv")
        result.X_train_resampled.to_csv(p_xsmote, index=False)
        exported["X_train_smote"] = p_xsmote

        p_ysmote = os.path.join(output_dir, "y_train_smote.csv")
        result.y_train_resampled.to_csv(p_ysmote, index=False)
        exported["y_train_smote"] = p_ysmote

        # 5. Feature importance ranking
        if result.feature_importance is not None:
            p_imp = os.path.join(output_dir, "feature_importance_ranking.csv")
            result.feature_importance.to_csv(p_imp, index=False)
            exported["feature_importance"] = p_imp

        return exported
