"""
Composite Domain Feature Engineering Orchestrator.
Coordinates specialized feature engineering modules:
  1. BehavioralMomentumEngineer (transaction/login velocity, momentum decay, cash flow)
  2. CustomerFrictionEngineer (dispute resolution status, complaint velocity, fee friction)
  3. ProductPortfolioEngineer (dynamic bundling regimes, portfolio utilization, savings stickiness)
  4. RiskInteractionsEngineer (wealth-loss interactions, geographic risk, life-stage vulnerability)
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from .base import BaseFeatureTransformer
from .behavioral_momentum import BehavioralMomentumEngineer
from .customer_friction import CustomerFrictionEngineer
from .product_portfolio import ProductPortfolioEngineer
from .risk_interactions import RiskInteractionsEngineer


class DomainFeatureEngineer(BaseFeatureTransformer):
    """
    Composite orchestrator for banking domain feature transformers.
    Delegates feature generation to dedicated sub-modules while preserving backward compatibility.
    """

    def __init__(
        self,
        enabled: bool = True,
        enable_behavioral_momentum: bool = True,
        enable_customer_friction: bool = True,
        enable_product_bundling: bool = True,
        enable_risk_interactions: bool = True,
        **kwargs: Any,
    ):
        super().__init__(name="DomainFeatureEngineer", enabled=enabled)
        self.enable_behavioral_momentum = enable_behavioral_momentum
        self.enable_customer_friction = enable_customer_friction
        self.enable_product_bundling = enable_product_bundling
        self.enable_risk_interactions = enable_risk_interactions
        self.kwargs = kwargs

        # 1. Behavioral Momentum Module
        self.momentum_engineer = BehavioralMomentumEngineer(
            enabled=(self.enabled and self.enable_behavioral_momentum),
            spending_collapse_threshold=kwargs.get("spending_collapse_threshold", 0.50),
            login_collapse_threshold=kwargs.get("login_collapse_threshold", 0.50),
            velocity_smoothing=kwargs.get("velocity_smoothing", 1.0),
            new_customer_exposure_months=kwargs.get("new_customer_exposure_months", 3.0),
        )

        # 2. Customer Friction Module
        self.friction_engineer = CustomerFrictionEngineer(
            enabled=(self.enabled and self.enable_customer_friction),
            unresolved_weight=kwargs.get("unresolved_weight", 3.0),
            complaint_count_weight=kwargs.get("complaint_count_weight", 1.5),
            tx_failure_weight=kwargs.get("tx_failure_weight", 2.0),
            high_friction_complaints_thresh=kwargs.get("high_friction_complaints_thresh", 2),
            velocity_smoothing=kwargs.get("velocity_smoothing", 1.0),
        )

        # 3. Product Portfolio Module
        self.portfolio_engineer = ProductPortfolioEngineer(
            enabled=(self.enabled and self.enable_product_bundling),
            sweetspot_min_products=kwargs.get("sweetspot_min_products", 2),
            danger_zone_min_products=kwargs.get("danger_zone_min_products", 3),
            active_weight=kwargs.get("active_weight", 0.35),
            trend_weight=kwargs.get("trend_weight", 0.25),
            util_weight=kwargs.get("util_weight", 0.20),
            friction_weight=kwargs.get("friction_weight", 0.20),
        )

        # 4. Risk Interactions Module
        self.risk_engineer = RiskInteractionsEngineer(
            enabled=(self.enabled and self.enable_risk_interactions),
            vulnerable_age_min=kwargs.get("vulnerable_age_min", 45),
            vulnerable_age_max=kwargs.get("vulnerable_age_max", 60),
            high_risk_country=kwargs.get("high_risk_country", "Germany"),
        )

    @property
    def median_utilization_(self) -> float:
        """Expose portfolio median utilization for inspection."""
        return getattr(self.portfolio_engineer, "median_utilization_", 1.0)

    @median_utilization_.setter
    def median_utilization_(self, val: float) -> None:
        self.portfolio_engineer.median_utilization_ = val

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DomainFeatureEngineer":
        """
        Fit all domain modules sequentially on training data (leak-free).
        """
        if not self.enabled:
            self.is_fitted = True
            return self

        current = X.copy()
        current = self.momentum_engineer.fit_transform(current, y)
        current = self.friction_engineer.fit_transform(current, y)
        current = self.portfolio_engineer.fit_transform(current, y)
        current = self.risk_engineer.fit_transform(current, y)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()

        out = X.copy()
        out = self.momentum_engineer.transform(out)
        out = self.friction_engineer.transform(out)
        out = self.portfolio_engineer.transform(out)
        out = self.risk_engineer.transform(out)

        self.created_features = (
            self.momentum_engineer.get_feature_names()
            + self.friction_engineer.get_feature_names()
            + self.portfolio_engineer.get_feature_names()
            + self.risk_engineer.get_feature_names()
        )
        return out

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "enable_behavioral_momentum": self.enable_behavioral_momentum,
            "enable_customer_friction": self.enable_customer_friction,
            "enable_product_bundling": self.enable_product_bundling,
            "enable_risk_interactions": self.enable_risk_interactions,
            "momentum_params": self.momentum_engineer.get_params(),
            "friction_params": self.friction_engineer.get_params(),
            "portfolio_params": self.portfolio_engineer.get_params(),
            "risk_params": self.risk_engineer.get_params(),
        })
        return params
