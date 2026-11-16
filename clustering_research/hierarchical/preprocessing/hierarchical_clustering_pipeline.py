"""Unsupervised customer features; no labels, SMOTE or target encoding."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

GROUPS = {
    "rfm": ["recency_days", "initiated_transactions_monthly", "trans_amount_3m"],
    "digital": ["successful_logins_monthly", "login_recency_days"],
    "momentum": ["trans_daily_trend", "login_daily_trend"],
    "friction": ["customer_service_complaints", "unresolved_complaint_ratio", "trans_failure_rate"],
    "portfolio": ["products_number", "savings_share", "total_accounts"],
}
LOG_FEATURES = {"recency_days", "initiated_transactions_monthly", "trans_amount_3m",
                "successful_logins_monthly", "login_recency_days", "trans_daily_trend", "login_daily_trend",
                "customer_service_complaints", "total_accounts"}

class HierarchicalClusteringPipeline:
    def __init__(self, config):
        self.config = config

    def prepare(self, df, groups):
        if df["customer_id"].isna().any() or df["customer_id"].duplicated().any():
            raise ValueError("customer_id must be non-null and unique")
        out = df.copy()
        exposure = out["exposure_months"].where(out["exposure_months"] > 0)
        out["initiated_transactions_monthly"] = out["customer_initiated_tx_count"] / exposure
        out["successful_logins_monthly"] = out["successful_logins"] / exposure
        out["unresolved_complaint_ratio"] = out["unresolved_complaints_count"].div(out["customer_service_complaints"].replace(0, np.nan)).fillna(0)
        out["trans_failure_rate"] = out["failed_transactions"].div(out["total_tx_attempts"].replace(0, np.nan)).fillna(0)
        out["savings_share"] = out["savings_balance"].div(out["balance"].replace(0, np.nan)).fillna(0)
        q2 = out["days_observed_q2"].where(out["days_observed_q2"] > 0)
        q3 = out["days_observed_q3"].where(out["days_observed_q3"] > 0)
        for name, recent, prior in [("trans_daily_trend", "trans_amount_3m", "trans_amount_prior_3m"),
                                     ("login_daily_trend", "login_count_q3", "login_count_q2")]:
            out[name] = ((out[recent] / q3) / (out[prior] / q2 + 0.01)).clip(0, 10)
        self.profile_features = out
        columns = [c for g in groups for c in GROUPS[g]]
        x = out[columns].replace([np.inf, -np.inf], np.nan).copy()
        if (x < 0).any().any():
            raise ValueError("Clustering features must be nonnegative")
        rows = []
        for col in columns:
            log = self.config.get("log_transform", True) and col in LOG_FEATURES
            if log:
                x[col] = np.log1p(x[col])
            rows.append({"feature": col, "group": next(g for g in groups if col in GROUPS[g]),
                         "transform": "log1p" if log else "none", "imputation": "median",
                         "scaler": self.config.get("scaler", "standard"), "status": "selected"})
        constant = [c for c in columns if x[c].nunique(dropna=True) <= 1]
        x = x.drop(columns=constant)
        if x.empty:
            raise ValueError("No nonconstant clustering features")
        self.imputer = SimpleImputer(strategy="median")
        x = pd.DataFrame(self.imputer.fit_transform(x), columns=x.columns, index=x.index)
        threshold = self.config.get("correlation_threshold", 0.9)
        dropped = []
        if threshold is not None:
            corr = x.corr().abs()
            kept = []
            for col in x:
                if any(corr.loc[col, other] >= threshold for other in kept):
                    dropped.append(col)
                else:
                    kept.append(col)
            x = x[kept]
        if self.config.get("scaler", "standard") not in ("standard", "robust"):
            raise ValueError("scaler must be standard or robust")
        self.scaler = StandardScaler() if self.config.get("scaler", "standard") == "standard" else RobustScaler()
        scaled = pd.DataFrame(self.scaler.fit_transform(x), columns=x.columns, index=x.index)
        if not np.isfinite(scaled.to_numpy()).all():
            raise ValueError("Nonfinite clustering input")
        for row in rows:
            if row["feature"] in constant:
                row["status"] = "removed_constant_or_empty"
            elif row["feature"] in dropped:
                row["status"] = "removed_correlation"
        self.feature_report = pd.DataFrame(rows)
        return scaled
