"""Explicit feature lineage: no automatic inclusion of unreviewed columns."""
import numpy as np
import pandas as pd
from preprocessing.behavioral_momentum import BehavioralMomentumEngineer

ORIGINAL_FEATURES = ["credit_score", "country", "gender", "age", "tenure", "balance",
                     "products_number", "credit_card", "active_member", "estimated_salary"]
SYNTHETIC_FEATURES = [
    "recency_days", "login_recency_days", "complaint_recency_days",
    "monthly_transactions", "monthly_logins", "app_logins_monthly",
    "trans_amount_3m", "trans_amount_prior_3m", "total_outflow",
    "trans_trend_ratio", "trans_count_trend_ratio", "login_trend_ratio",
    "is_spending_collapsing", "is_login_collapsing",
    "customer_service_complaints", "unresolved_complaints_count",
    "login_failure_rate", "trans_failure_rate", "num_savings_accounts", "num_savings_cycles",
    "rfm_monetary_monthly", "prior_history_observed", "has_initiated_activity", "has_successful_login",
]
# This ablation removes transactions, savings, monetary endpoints and explicit
# tenure-derived history flags. Remaining simulation fields are NOT proven independent.
REDUCED_FEATURES = ["monthly_logins", "app_logins_monthly", "customer_service_complaints",
                    "unresolved_complaints_count", "login_failure_rate", "has_successful_login"]
CATEGORICAL = ["country", "gender"]
LOG_FEATURES = ["balance", "estimated_salary", *[c for c in SYNTHETIC_FEATURES
                 if not c.startswith(("is_", "has_")) and c not in
                 ("login_failure_rate", "trans_failure_rate", "prior_history_observed")]]


def provenance():
    result = {c: {"origin": "original", "source": "kaggle_bank_churn", "dependency": "source input"}
              for c in ORIGINAL_FEATURES}
    for col in SYNTHETIC_FEATURES:
        source = "transactions + accounts + customer_context"
        if "login" in col:
            source = "login + customer_context"
        elif "complaint" in col:
            source = "complaints + customer_context"
        elif "savings" in col:
            source = "accounts + savings_cycles"
        elif col == "prior_history_observed":
            source = "customer_context"
        result[col] = {"origin": "synthetic", "source": source,
            "dependency": "Simulated, not independent: activity conditioned on active_member; "
                          "balances hard endpoints; funding uses salary; dates consistent with tenure"}
    result.update({c: {"origin": "mixed_or_unverified", "source": "excluded",
                       "dependency": reason} for c, reason in {
        "savings_balance": "Component of total snapshot account balances constrained to original balance",
        "everyday_balance": "Component of original balance",
        "relationship_tenure_days": "Generated consistently with original tenure",
        "wealth_deceleration_risk": "Original balance times synthetic trend; supervised risk design",
        "is_new_customer": "Generated relationship dates consistent with original tenure",
        "rfm_composite_index": "Proposed in docs/new_feats.md; not implemented or used",
        "proxy_clv_3yr": "Proposed in docs/new_feats.md; not implemented or used",
    }.items()})
    return result


def build_features(base):
    required = set(ORIGINAL_FEATURES + ["customer_id", "churn", "exposure_days", "exposure_months",
        "customer_initiated_tx_count", "initiated_tx_attempts", "initiated_failed_transactions", "initiated_outflow"])
    if required - set(base):
        raise ValueError(f"Missing guarded ETL inputs: {sorted(required - set(base))}")
    if base.customer_id.isna().any() or base.customer_id.duplicated().any():
        raise ValueError("Customer IDs must be non-null and unique")
    # Stateless, label-independent transformer. No domain risk interactions.
    out = BehavioralMomentumEngineer().fit_transform(base.drop(columns="churn"))
    out["churn"] = base.churn
    # Correct inherited total-transactions denominator to exclude passive postings.
    months = out.exposure_days / 30.4375
    out["monthly_transactions"] = out.customer_initiated_tx_count / months
    out["monthly_logins"] = out.successful_logins / months
    out["app_logins_monthly"] = out.app_logins_total / months
    out["total_outflow"] = out.initiated_outflow
    out["rfm_monetary_monthly"] = out.total_outflow / months
    denom = out.initiated_tx_attempts.replace(0, np.nan)
    out["trans_failure_rate"] = out.initiated_failed_transactions / denom
    out["login_failure_rate"] = out.failed_logins / out.total_logins.replace(0, np.nan)
    out["prior_history_observed"] = out.days_observed_q2.gt(0).astype(int)
    out["has_initiated_activity"] = out.customer_initiated_tx_count.gt(0).astype(int)
    out["has_successful_login"] = out.successful_logins.gt(0).astype(int)
    for ratio, flag, prior in (("trans_trend_ratio", "is_spending_collapsing", "trans_amount_prior_3m"),
                              ("trans_count_trend_ratio", None, "trans_count_prior_3m"),
                              ("login_trend_ratio", "is_login_collapsing", "login_count_q2")):
        unknown = out.days_observed_q2.le(0) | out[prior].le(0)
        out.loc[unknown, ratio] = np.nan
        if flag:
            out.loc[unknown, flag] = np.nan
    # Unobserved quarters remain missing, distinct from observed zero activity.
    out.loc[out.days_observed_q2.le(0), "trans_amount_prior_3m"] = np.nan
    out.loc[out.days_observed_q3.le(0), "trans_amount_3m"] = np.nan
    for col in ORIGINAL_FEATURES:
        if col not in CATEGORICAL:
            out[col] = pd.to_numeric(out[col], errors="raise")
    missing = set(SYNTHETIC_FEATURES) - set(out)
    if missing:
        raise ValueError(f"Unavailable synthetic features: {sorted(missing)}")
    return out.replace([np.inf, -np.inf], np.nan)


def feature_sets(frame):
    """Same indexed population for every experiment; IDs remain outside matrices."""
    original = frame[ORIGINAL_FEATURES].copy()
    synthetic = frame[SYNTHETIC_FEATURES].copy()
    return {"A": original, "B": synthetic, "C": pd.concat([original, synthetic], axis=1)}
