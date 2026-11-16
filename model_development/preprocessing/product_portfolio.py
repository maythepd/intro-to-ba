"""
Product Portfolio & Account Relationship Feature Engineering Module.
Engineers portfolio diversification, account utilization, dynamic market-regime classification
(Sweetspot vs Danger Zone), and savings stickiness without data leakage.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .base import BaseFeatureTransformer


class ProductPortfolioEngineer(BaseFeatureTransformer):
    """
    Engineers banking product bundling, utilization regimes, and relationship depth features.

    Features created:
      - portfolio_utilization: Monthly transaction intensity per banking product held
      - balance_per_product: Committed deposit balance per banking product held
      - is_product_sweetspot: Dynamic Bull-Regime indicator (engaged multi-product relationship)
      - is_product_danger_zone: Dynamic Bear-Regime indicator (over-bundled, dormant, or high-friction)
      - portfolio_health_score: Continuous composite momentum oscillator measuring portfolio vibrancy
      - has_savings_account: Binary flag indicating customer holds term savings
      - has_closed_account: Binary flag indicating customer has closed at least one deposit account
      - closed_account_ratio: Proportion of opened deposit accounts that were subsequently closed
      - savings_share: Share of customer total balance committed into savings accounts
      - multi_account_holder: Binary flag for holding > 1 deposit account
    """

    def __init__(
        self,
        enabled: bool = True,
        sweetspot_min_products: int = 2,
        danger_zone_min_products: int = 3,
        active_weight: float = 0.35,
        trend_weight: float = 0.25,
        util_weight: float = 0.20,
        friction_weight: float = 0.20,
    ):
        super().__init__(name="ProductPortfolioEngineer", enabled=enabled)
        self.sweetspot_min_products = sweetspot_min_products
        self.danger_zone_min_products = danger_zone_min_products
        self.active_weight = active_weight
        self.trend_weight = trend_weight
        self.util_weight = util_weight
        self.friction_weight = friction_weight

        # State learned strictly from training data during fit (Zero leakage)
        self.median_utilization_: float = 1.0

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ProductPortfolioEngineer":
        """
        Learns empirical baseline distribution statistics (median portfolio utilization)
        exclusively on the training split to eliminate data leakage.
        """
        if not self.enabled:
            self.is_fitted = True
            return self

        exp_m = X["exposure_months"] if "exposure_months" in X.columns else 6.0
        m_tx = (X["total_transactions"] / exp_m) if "total_transactions" in X.columns else 0.0
        prods = np.maximum(X["products_number"], 1) if "products_number" in X.columns else 1.0
        util = m_tx / prods
        self.median_utilization_ = float(util.median()) if len(X) > 0 else 1.0

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X.copy()

        out = X.copy()
        created: List[str] = []

        has_prods = "products_number" in out.columns
        prods = np.maximum(out["products_number"], 1) if has_prods else 1.0

        # 1. Dynamic portfolio utilization and balance depth
        if "monthly_transactions" in out.columns and has_prods:
            out["portfolio_utilization"] = out["monthly_transactions"] / prods
            created.append("portfolio_utilization")

        if "balance" in out.columns and has_prods:
            out["balance_per_product"] = np.maximum(0, out["balance"]) / prods
            created.append("balance_per_product")

        # 2. Dynamic Regime Segmentation (Bull Sweetspot vs Bear Danger Zone)
        if has_prods:
            has_act = "active_member" in out.columns
            has_sp_col = "is_spending_collapsing" in out.columns
            has_fric = "is_high_friction" in out.columns
            has_unres = "has_unresolved_complaint" in out.columns

            util = out["portfolio_utilization"] if "portfolio_utilization" in out.columns else 1.0
            med_util = getattr(self, "median_utilization_", 1.0)
            act = out["active_member"] if has_act else 1
            sp_col = out["is_spending_collapsing"] if has_sp_col else 0
            fric = out["is_high_friction"] if has_fric else 0
            unres = out["has_unresolved_complaint"] if has_unres else 0

            # Dynamic Sweetspot (Bull Regime):
            # Customer holds multiple products (>= sweetspot_min_products), is actively engaged,
            # achieves above-median utilization, has steady spending momentum, and experiences no service friction.
            out["is_product_sweetspot"] = (
                (out["products_number"] >= self.sweetspot_min_products)
                & (act == 1)
                & (util >= med_util)
                & (sp_col == 0)
                & (fric == 0)
            ).astype(int)

            # Dynamic Danger Zone (Bear Regime):
            # Customer holds heavy bundling (>= danger_zone_min_products) but is dormant or underutilizing products,
            # OR spending momentum collapsed while inactive, OR unresolved high-friction dispute exists.
            out["is_product_danger_zone"] = (
                ((out["products_number"] >= self.danger_zone_min_products) & ((act == 0) | (util < med_util)))
                | ((sp_col == 1) & (act == 0))
                | ((fric == 1) & (unres == 1))
            ).astype(int)

            # Continuous Portfolio Health Score
            trend_val = out["trans_trend_ratio"] if "trans_trend_ratio" in out.columns else 1.0
            trend_dev = np.log1p(np.clip(trend_val, 0.0, 10.0)) - np.log1p(1.0)
            out["portfolio_health_score"] = (
                self.active_weight * (act - 0.5) * 2.0
                + self.trend_weight * trend_dev
                + self.util_weight * np.log1p(util)
                - self.friction_weight * (fric * 1.5 + unres * 2.0)
            )
            created.extend([
                "is_product_sweetspot",
                "is_product_danger_zone",
                "portfolio_health_score",
            ])

        # 3. Account composition and relationship retention
        if "num_savings_accounts" in out.columns:
            out["has_savings_account"] = (out["num_savings_accounts"] > 0).astype(int)
            created.append("has_savings_account")

        if "num_closed_accounts" in out.columns:
            out["has_closed_account"] = (out["num_closed_accounts"] > 0).astype(int)
            created.append("has_closed_account")
            if "total_accounts" in out.columns:
                out["closed_account_ratio"] = out["num_closed_accounts"] / np.maximum(out["total_accounts"], 1)
                created.append("closed_account_ratio")

        if "savings_balance" in out.columns and "balance" in out.columns:
            out["savings_share"] = out["savings_balance"] / (np.maximum(0, out["balance"]) + 1.0)
            created.append("savings_share")

        if "total_accounts" in out.columns:
            out["multi_account_holder"] = (out["total_accounts"] > 1).astype(int)
            created.append("multi_account_holder")

        self.created_features = created
        return out

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "sweetspot_min_products": self.sweetspot_min_products,
            "danger_zone_min_products": self.danger_zone_min_products,
            "active_weight": self.active_weight,
            "trend_weight": self.trend_weight,
            "util_weight": self.util_weight,
            "friction_weight": self.friction_weight,
            "median_utilization_": self.median_utilization_,
        })
        return params

