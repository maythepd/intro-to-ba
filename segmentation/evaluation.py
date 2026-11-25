"""Post-fit descriptions, distribution differences and held-out churn associations."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .features import ORIGINAL_FEATURES, SYNTHETIC_FEATURES, CATEGORICAL


def churn_summary(labels, churn):
    frame = pd.DataFrame({"cluster": np.asarray(labels), "churn": np.asarray(churn)})
    summary = frame.groupby("cluster").churn.agg(customer_count="size", churn_count="sum").reset_index()
    prevalence = frame.churn.mean()
    n = summary.customer_count
    rate = summary.churn_count / n
    z = 1.959963984540054
    center = (rate + z*z / (2*n)) / (1 + z*z/n)
    margin = z * np.sqrt(rate * (1-rate)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return summary.assign(customer_share=n / n.sum(), churn_rate=rate, overall_prevalence=prevalence,
                          lift=rate / prevalence if prevalence > 0 else np.nan,
                          ci_low=(center-margin).clip(0, 1), ci_high=(center+margin).clip(0, 1))


def risk_separation(labels, churn):
    """Fraction of binary target variance between clusters; descriptive, not predictive."""
    summary = churn_summary(labels, churn)
    p = summary.overall_prevalence.iloc[0]
    if p in (0, 1):
        return np.nan
    return float(np.average(np.square(summary.churn_rate - p), weights=summary.customer_count) / (p * (1-p)))


def agreement(left, right):
    return {"ari": adjusted_rand_score(left, right), "nmi": normalized_mutual_info_score(left, right)}


def align_clusters(train_a, train_c, labels_c):
    """Learn an optimal display-label matching from training overlaps only."""
    table = pd.crosstab(np.asarray(train_a), np.asarray(train_c))
    rows, cols = linear_sum_assignment(-table.to_numpy())
    mapping = {int(table.columns[c]): f"A_{table.index[r]}" for r, c in zip(rows, cols)}
    return np.array([mapping.get(int(c), f"new_C_{c}") for c in labels_c]), mapping


def distribution_profiles(frame, labels, reference, columns=None):
    """Cluster vs population means/proportions, scaled by training dispersion."""
    columns = columns or ORIGINAL_FEATURES + SYNTHETIC_FEATURES
    records = []
    labels = np.asarray(labels)
    for col in columns:
        if col in CATEGORICAL:
            levels = sorted(set(frame[col].dropna()) | set(reference[col].dropna()))
            for level in levels:
                baseline = frame[col].eq(level).mean()
                train_p = reference[col].eq(level).mean()
                scale = np.sqrt(train_p * (1-train_p))
                for cluster in sorted(np.unique(labels)):
                    selected = frame.loc[labels == cluster, col].eq(level)
                    value = selected.mean()
                    records.append({"cluster": cluster, "feature": col, "level": str(level), "count": len(selected),
                        "mean": value, "population_mean": baseline,
                        "standardized_difference": (value-baseline)/scale if scale > 0 else np.nan})
        else:
            baseline, scale = frame[col].mean(), reference[col].std(ddof=0)
            for cluster in sorted(np.unique(labels)):
                selected = frame.loc[labels == cluster, col].dropna()
                rest = frame.loc[labels != cluster, col].dropna()
                records.append({"cluster": cluster, "feature": col, "level": "", "count": len(selected),
                    "mean": selected.mean(), "median": selected.median(), "q25": selected.quantile(.25), "q75": selected.quantile(.75),
                    "population_mean": baseline, "standardized_difference": (selected.mean()-baseline)/scale if scale > 0 else np.nan,
                    "ks_vs_rest": ks_2samp(selected, rest).statistic if len(selected) and len(rest) else np.nan})
    return pd.DataFrame(records)


def personas(profiles, experiment):
    records = []
    for cluster, group in profiles.groupby("cluster"):
        numeric = group.loc[group.level.eq("")].set_index("feature")
        z = numeric.standardized_difference
        score = lambda name: float(z.get(name, 0))
        if experiment != "A" and score("customer_service_complaints") > .5:
            name, action = "Higher-friction customers", "Review unresolved cases and recurring service failures; offer case-owner follow-up."
        elif experiment != "A" and score("monthly_logins") > .5:
            name, action = "Frequent digital users", "Preserve app reliability and offer relevant self-service support."
        elif experiment != "A" and score("monthly_transactions") < -.35 and score("recency_days") > .35:
            name, action = "Less-engaged transactors", "Check onboarding and activity barriers before a targeted reactivation offer."
        elif experiment != "A" and score("monthly_transactions") > .35 and score("recency_days") < -.35:
            name, action = "Active transactors", "Maintain reliable everyday payments and review needs before proposing relevant products."
        elif experiment == "B" and score("has_initiated_activity") > .2:
            name, action = "Observed transactors", "Preserve everyday account utility and distinguish stable use from emerging activity barriers."
        elif score("balance") > .45:
            name, action = "Higher-balance customers", "Review savings and service needs; prioritize value-sensitive relationship support."
        elif score("age") > .45:
            name, action = "Older customer profiles", "Offer accessible service and contact-channel choice; validate needs directly."
        elif score("balance") < -.45:
            name, action = "Lower-balance customers", "Review account utility, fee burden and onboarding before recommending products."
        else:
            strongest = z.abs().nlargest(1)
            feature = strongest.index[0] if len(strongest) else "profile"
            name = ("Higher " if score(feature) >= 0 else "Lower ") + feature.replace("_", " ")
            action = "Validate the observed needs through customer research before selecting a retention action."
        high, low = z.nlargest(3), z.nsmallest(3)
        evidence = "; ".join(f"{f}={numeric.loc[f, 'mean']:.3g} ({v:+.2f} SD)" for f, v in pd.concat([high, low]).items())
        records.append({"experiment": experiment, "cluster": cluster, "persona": f"{name} ({experiment}{cluster})",
                        "higher_deviations": "; ".join(f"{f} {v:+.2f}" for f, v in high.items()),
                        "lower_deviations": "; ".join(f"{f} {v:+.2f}" for f, v in low.items()),
                        "training_evidence": evidence, "retention_action": action})
    return pd.DataFrame(records)


def synthetic_diagnostics(train):
    description = train[SYNTHETIC_FEATURES].describe(percentiles=[.01, .5, .99]).T
    description["missing_fraction"] = train[SYNTHETIC_FEATURES].isna().mean()
    description["unique_values"] = train[SYNTHETIC_FEATURES].nunique()
    correlations = []
    original_numeric = [c for c in ORIGINAL_FEATURES if c not in CATEGORICAL]
    corr = train[original_numeric + SYNTHETIC_FEATURES].corr(method="spearman")
    for original in original_numeric:
        for synthetic in SYNTHETIC_FEATURES:
            correlations.append({"original_feature": original, "synthetic_feature": synthetic,
                                 "spearman": corr.loc[original, synthetic]})
    return description, pd.DataFrame(correlations).sort_values("spearman", key=lambda s: s.abs(), ascending=False)
