"""Reproducible deployment adapter for KMeans and hierarchical clustering.
The upstream KMeans checkpoint is not committed, while the upstream Ward linkage
has no customer ID/leaf mapping. Refit both on the official demo snapshot.
Hierarchical inference outside its Ward-fit sample is explicitly APPROXIMATE.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ORIGINAL_NUMERIC = ("credit_score", "age", "tenure", "balance", "products_number",
                    "credit_card", "active_member", "estimated_salary")
ORIGINAL_CATEGORICAL = ("country", "gender")
BEHAVIORAL_NUMERIC = ("recency_days", "monthly_transactions", "trans_amount_3m",
                      "trans_trend_ratio", "app_logins_monthly",
                      "customer_service_complaints", "has_unresolved_complaint")
FEATURES = ORIGINAL_NUMERIC + ORIGINAL_CATEGORICAL + BEHAVIORAL_NUMERIC
ALGORITHMS = ("kmeans", "hierarchical")
MODEL_DIR = Path(os.getenv("SEGMENTATION_MODEL_DIR", "artifacts/segmentation"))
DATA_PATH = Path(os.getenv("CUSTOMERS_CSV_PATH", "data/official/customer_features.csv"))


def signed_log1p(values):
    a = np.asarray(values, dtype=float)
    return np.sign(a) * np.log1p(np.abs(a))


def make_group(numeric, categoricals=()):
    steps = []
    if numeric:
        steps.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), list(numeric)))
    if categoricals:
        steps.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(categoricals)))
    return ColumnTransformer(steps, remainder="drop", sparse_threshold=0)


class BalancedFeatures(BaseEstimator, TransformerMixin):
    """Source-inspired group-weighted preprocessing; fit on features only."""

    def __init__(self, original_weight=.5):
        self.original_weight = original_weight

    def fit(self, X, y=None):
        if y is not None:
            raise ValueError("Clustering cannot learn from churn labels")
        if set(X.columns) != set(FEATURES):
            raise ValueError("Unsupported or missing segmentation inputs")
        if not 0 < self.original_weight < 1:
            raise ValueError("Invalid group weighting")
        self.original_ = make_group(ORIGINAL_NUMERIC, ORIGINAL_CATEGORICAL).fit(X)
        self.behavior_ = make_group(BEHAVIORAL_NUMERIC).fit(X)
        a = np.asarray(self.original_.transform(X), dtype=float)
        b = np.asarray(self.behavior_.transform(X), dtype=float)
        ea, eb = a.var(axis=0).sum(), b.var(axis=0).sum()
        if min(ea, eb) <= 0:
            raise ValueError("Insufficient training variability")
        self.scales_ = (float(np.sqrt(self.original_weight / ea)),
                        float(np.sqrt((1 - self.original_weight) / eb)))
        return self

    def transform(self, X):
        if set(X.columns) != set(FEATURES):
            raise ValueError("Segmentation input schema mismatch")
        result = np.hstack([
            np.asarray(self.original_.transform(X), dtype=float) * self.scales_[0],
            np.asarray(self.behavior_.transform(X), dtype=float) * self.scales_[1],
        ])
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite model features")
        return result


def feature_frame(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing official segmentation features: {sorted(missing)}")
    x = frame.loc[:, list(FEATURES)].copy()
    for col in ORIGINAL_NUMERIC + BEHAVIORAL_NUMERIC:
        x[col] = pd.to_numeric(x[col], errors="raise")
    for col in ORIGINAL_CATEGORICAL:
        x[col] = x[col].astype("string").astype(object)
    return x


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(data_path=DATA_PATH, model_dir=MODEL_DIR, clusters=4, hier_sample=1500):
    data_path, model_dir = Path(data_path), Path(model_dir)
    if not data_path.is_file():
        raise FileNotFoundError(f"Official customer feature table missing: {data_path}")
    frame = pd.read_csv(data_path)
    if ("customer_id" not in frame or frame.customer_id.isna().any()
            or frame.customer_id.duplicated().any()):
        raise ValueError("customer_id must be unique and non-null")
    if len(frame) < max(50, clusters * 5) or not 2 <= clusters <= 10:
        raise ValueError("Insufficient customers or invalid cluster count")
    frame = frame.sort_values("customer_id").reset_index(drop=True)
    x = feature_frame(frame)
    preprocess = BalancedFeatures().fit(x)
    values = preprocess.transform(x)
    kmeans = KMeans(n_clusters=clusters, n_init=10, max_iter=300, random_state=42).fit(values)
    rng = np.random.default_rng(42)
    sample = np.sort(rng.choice(len(frame), size=min(hier_sample, len(frame)), replace=False))
    ward = AgglomerativeClustering(n_clusters=clusters, linkage="ward")
    ward_labels = ward.fit_predict(values[sample]).astype(int)
    centers = np.vstack([values[sample][ward_labels == k].mean(axis=0) for k in range(clusters)])
    # Ward has no predict(). For customers beyond the Ward-fit sample, use
    # centroid distance as a clearly identified approximation.
    hier_labels = np.square(values[:, None, :] - centers[None, :, :]).sum(axis=2).argmin(axis=1)
    hier_labels[sample] = ward_labels
    info = {}
    rng = np.random.default_rng(123)
    eval_indices = np.sort(rng.choice(len(frame), size=min(600, len(frame)), replace=False))
    for name, labels in (("kmeans", kmeans.labels_), ("hierarchical", hier_labels)):
        profiles = []
        for k in range(clusters):
            group = frame.loc[labels == k]
            row = {
                "id": int(k), "name": f"Cluster {k}", "customers": int(len(group)),
                "share": float(len(group) / len(frame)),
                "avg_balance": float(pd.to_numeric(group["balance"]).mean()),
                "avg_monthly_transactions": float(pd.to_numeric(group["monthly_transactions"]).mean()),
                "avg_app_logins_monthly": float(pd.to_numeric(group["app_logins_monthly"]).mean()),
            }
            if "churn" in group and group.churn.notna().all():
                row["observed_churn_rate"] = float(pd.to_numeric(group.churn).mean())
            profiles.append(row)
        sampled = labels[eval_indices]
        sil = (float(silhouette_score(values[eval_indices], sampled))
               if 1 < len(np.unique(sampled)) < len(sampled) else None)
        info[name] = {"labels": np.asarray(labels, dtype=int), "clusters": profiles,
                      "silhouette_sample": sil}
    bundle = {
        "version": 1, "source": "official_customer_features",
        "data_sha256": sha256(data_path), "customer_ids": frame.customer_id.astype(str).tolist(),
        "features": list(FEATURES), "preprocessor": preprocess, "kmeans": kmeans,
        "hierarchical_centers": centers, "hierarchical_sample": set(int(x) for x in sample),
        "models": info, "n_clusters": clusters, "customer_count": len(frame),
        "sample_size": len(sample),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "cluster_bundle.joblib")
    (model_dir / "metadata.json").write_text(json.dumps({
        "snapshot_sha256": bundle["data_sha256"],
        "n_customers": len(frame), "n_clusters": clusters,
        "ward_fit_customers": len(sample), "silhouette_sample": {
            name: info[name]["silhouette_sample"] for name in ALGORITHMS},
        "hierarchical_warning": "Nearest-centroid assignment outside the Ward fit sample is approximate",
    }, indent=2), encoding="utf-8")
    return bundle


def load(data_path=DATA_PATH, model_dir=MODEL_DIR):
    path = Path(model_dir) / "cluster_bundle.joblib"
    if not path.is_file() or not Path(data_path).is_file():
        return None
    bundle = joblib.load(path)  # Load only trusted artifacts built by this project.
    if bundle.get("version") != 1 or bundle.get("data_sha256") != sha256(data_path):
        return None
    return bundle


def summary(bundle, algorithm):
    if algorithm not in ALGORITHMS:
        raise ValueError("Unknown clustering model")
    model = bundle["models"][algorithm]
    return {
        "algorithm": algorithm, "source": bundle["source"],
        "customer_count": bundle["customer_count"], "n_clusters": bundle["n_clusters"],
        "silhouette_sample": model["silhouette_sample"], "features": bundle["features"],
        "clusters": model["clusters"], "cluster_labels_are_risk_rankings": False,
        "assignment_method": ("Direct fitted KMeans prediction" if algorithm == "kmeans"
            else "Ward on representative subset; nearest-centroid approximation for other customers"),
        "study_note": "Synthetic-extension study, 2026-10-01 snapshot; not live bank operations.",
    }


def customer_membership(bundle, customer_id):
    try:
        idx = bundle["customer_ids"].index(str(customer_id))
    except ValueError:
        return None
    return {
        "customer_id": str(customer_id),
        "kmeans": int(bundle["models"]["kmeans"]["labels"][idx]),
        "hierarchical": int(bundle["models"]["hierarchical"]["labels"][idx]),
        "hierarchical_assignment_method": ("fitted_ward" if idx in bundle["hierarchical_sample"]
            else "nearest_centroid_approximation"),
    }


def predict(bundle, values, algorithm):
    if algorithm not in ALGORITHMS:
        raise ValueError("Unknown clustering model")
    encoded = bundle["preprocessor"].transform(feature_frame(pd.DataFrame([values])))
    if algorithm == "kmeans":
        cluster = int(bundle["kmeans"].predict(encoded)[0])
        method = "kmeans_predict"
    else:
        cluster = int(np.square(bundle["hierarchical_centers"] - encoded[0]).sum(axis=1).argmin())
        method = "nearest_centroid_surrogate_not_ward_prediction"
    return {"algorithm": algorithm, "cluster_id": cluster, "method": method,
        "warning": (None if algorithm == "kmeans" else
            "Ward agglomerative clustering cannot assign new points directly; nearest-centroid approximation used.")}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train demo segmentation on official data")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_DIR)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--hierarchical-sample", type=int, default=1500)
    args = parser.parse_args()
    result = train(args.data, args.output, args.clusters, args.hierarchical_sample)
    print(f"Exported KMeans and Ward adapter for {result['customer_count']} customers")


if __name__ == "__main__":
    # CLI entry points run as "__main__", but pickle/joblib must reference the
    # importable module so a separate Uvicorn process can deserialize pipelines
    # containing BalancedFeatures and signed_log1p.
    from api.segmentation_service import main as canonical_main
    canonical_main()
