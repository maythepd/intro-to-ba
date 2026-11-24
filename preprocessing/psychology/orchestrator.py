"""
Composite Consumer & Behavioral Psychology Feature Engineering Orchestrator.
Coordinates specialized feature engineering modules inspired by behavioral economics and consumer psychology:
  1. MentalAccountingEngineer (Mental accounting buckets, commitment breaches, renewal inertia)
  2. AnxietyMonitoringEngineer (Phantom logins, anxious checking index, panic channel migrations)
  3. PainOfPayingEngineer (Penalty fee retaliation, cash decoupling, loss aversion drain)
  4. PeakEndFrictionEngineer (Peak-End rule, exponential recency complaints, terminal failure shocks)
  5. ScarcityMindsetEngineer (Financial runway months, hand-to-mouth burn, scarcity x friction)

Includes robust outlier containment (Winsorization on train) and degenerate sparse feature pruning.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..base import BaseFeatureTransformer
from .mental_accounting import MentalAccountingEngineer
from .anxiety_monitoring import AnxietyMonitoringEngineer
from .pain_of_paying import PainOfPayingEngineer
from .peak_end_friction import PeakEndFrictionEngineer
from .scarcity_mindset import ScarcityMindsetEngineer


class PsychologyFeatureEngineer(BaseFeatureTransformer):
    """
    Composite orchestrator for behavioral and consumer psychology feature engineering.
    Sequentially fits and transforms all 5 sub-modules without data leakage.
    """

    def __init__(
        self,
        enabled: bool = True,
        enable_mental_accounting: bool = True,
        enable_anxiety_monitoring: bool = True,
        enable_pain_of_paying: bool = True,
        enable_peak_end_friction: bool = True,
        enable_scarcity_mindset: bool = True,
        recency_half_life_days: float = 30.0,
        login_recency_scale: float = 15.0,
        outlier_clip_quantile: float = 0.995,
        prune_sparse: bool = True,
        max_sparsity_threshold: float = 0.98,
        min_variance_threshold: float = 1e-6,
        **kwargs: Any,
    ):
        super().__init__(name="PsychologyFeatureEngineer", enabled=enabled)
        self.enable_mental_accounting = enable_mental_accounting
        self.enable_anxiety_monitoring = enable_anxiety_monitoring
        self.enable_pain_of_paying = enable_pain_of_paying
        self.enable_peak_end_friction = enable_peak_end_friction
        self.enable_scarcity_mindset = enable_scarcity_mindset
        self.recency_half_life_days = recency_half_life_days
        self.login_recency_scale = login_recency_scale
        self.outlier_clip_quantile = outlier_clip_quantile
        self.prune_sparse = prune_sparse
        self.max_sparsity_threshold = max_sparsity_threshold
        self.min_variance_threshold = min_variance_threshold
        self.kwargs = kwargs

        # Initialize sub-modules
        self.mental_accounting_eng = MentalAccountingEngineer(
            enabled=(self.enabled and self.enable_mental_accounting)
        )
        self.anxiety_eng = AnxietyMonitoringEngineer(
            enabled=(self.enabled and self.enable_anxiety_monitoring),
            login_recency_scale=self.login_recency_scale,
        )
        self.pain_of_paying_eng = PainOfPayingEngineer(
            enabled=(self.enabled and self.enable_pain_of_paying)
        )
        self.peak_end_eng = PeakEndFrictionEngineer(
            enabled=(self.enabled and self.enable_peak_end_friction),
            recency_half_life_days=self.recency_half_life_days,
        )
        self.scarcity_eng = ScarcityMindsetEngineer(
            enabled=(self.enabled and self.enable_scarcity_mindset)
        )

        # Learned state on training partition (leakage-free)
        self.clip_bounds_: Dict[str, Tuple[float, float]] = {}
        self.valid_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PsychologyFeatureEngineer":
        """
        Fit all psychological modules and calculate outlier clipping bounds & sparsity filters.
        """
        if not self.enabled:
            self.is_fitted = True
            return self

        current = X.copy()
        current = self.mental_accounting_eng.fit_transform(current, y)
        current = self.anxiety_eng.fit_transform(current, y)
        current = self.pain_of_paying_eng.fit_transform(current, y)
        current = self.peak_end_eng.fit_transform(current, y)
        current = self.scarcity_eng.fit_transform(current, y)

        raw_created: List[str] = (
            self.mental_accounting_eng.created_features
            + self.anxiety_eng.created_features
            + self.pain_of_paying_eng.created_features
            + self.peak_end_eng.created_features
            + self.scarcity_eng.created_features
        )

        # Learn clipping bounds and evaluate sparsity/degeneracy on train only
        valid: List[str] = []
        bounds: Dict[str, Tuple[float, float]] = {}
        n_samples = len(current)

        for col in raw_created:
            if col not in current.columns:
                continue

            vals = current[col].values
            # Handle NaNs or infs defensively
            clean_vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

            # Sparsity / Variance check
            if self.prune_sparse and n_samples > 0:
                zero_ratio = float(np.sum(clean_vals == 0.0) / n_samples)
                var = float(np.var(clean_vals))
                if zero_ratio > self.max_sparsity_threshold or var < self.min_variance_threshold:
                    # Feature is degenerate or hyper-sparse (e.g. >98% zeros and zero variance)
                    continue

            # Learn Winsorization bounds on train
            if len(clean_vals) > 0 and self.outlier_clip_quantile < 1.0:
                low_q = float(np.percentile(clean_vals, (1.0 - self.outlier_clip_quantile) * 100))
                high_q = float(np.percentile(clean_vals, self.outlier_clip_quantile * 100))
                if low_q < high_q:
                    bounds[col] = (low_q, high_q)

            valid.append(col)

        self.valid_features_ = valid
        self.clip_bounds_ = bounds
        self.created_features = valid
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe through all psychology modules, applying learned clipping bounds.
        """
        if not self.enabled:
            return X.copy()

        out = X.copy()
        out = self.mental_accounting_eng.transform(out)
        out = self.anxiety_eng.transform(out)
        out = self.pain_of_paying_eng.transform(out)
        out = self.peak_end_eng.transform(out)
        out = self.scarcity_eng.transform(out)

        # Apply learned clipping bounds and prune rejected degenerate features
        all_candidate_cols = (
            self.mental_accounting_eng.created_features
            + self.anxiety_eng.created_features
            + self.pain_of_paying_eng.created_features
            + self.peak_end_eng.created_features
            + self.scarcity_eng.created_features
        )

        for col in all_candidate_cols:
            if col in out.columns:
                if self.is_fitted and col not in self.valid_features_:
                    # Drop pruned degenerate feature
                    out.drop(columns=[col], inplace=True)
                elif col in self.clip_bounds_:
                    low_b, high_b = self.clip_bounds_[col]
                    out[col] = np.clip(np.nan_to_num(out[col].values, nan=0.0), low_b, high_b)

        return out
