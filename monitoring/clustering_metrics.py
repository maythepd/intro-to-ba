"""Fixed assignments in one frozen encoded space; churn is never a clustering target."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from monitoring import drift


def input_reference(frame, cfg):
    return {name: drift.numeric_reference(frame[name], cfg["histogram_bins"])
            if pd.api.types.is_numeric_dtype(frame[name]) else drift.categorical_reference(frame[name]) for name in frame}


def evaluate(z, labels, centers, reference, cfg, *, indices=None):
    z, labels, centers = np.asarray(z, dtype=float), np.asarray(labels), np.asarray(centers, dtype=float)
    if (z.ndim != 2 or not len(z) or centers.ndim != 2 or centers.shape[1] != z.shape[1] or len(centers) < 2
            or labels.shape != (len(z),) or not np.issubdtype(labels.dtype, np.integer)
            or not np.isfinite(z).all() or not np.isfinite(centers).all()
            or (labels < 0).any() or (labels >= len(centers)).any()):
        raise ValueError("Invalid clustering space, assignments or frozen centers")
    k = len(centers)
    shares = np.asarray(reference["shares"], dtype=float)
    p95 = np.asarray([np.nan if x is None else x for x in reference["p95"]], dtype=float)
    if shares.shape != (k,) or p95.shape != (k,) or not np.isfinite(shares).all() or (shares < 0).any() or not np.isclose(shares.sum(), 1):
        raise ValueError("Invalid fixed clustering reference")
    indices = np.arange(len(z)) if indices is None else np.asarray(indices)
    if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer) or len(set(indices)) != len(indices) or (indices < 0).any() or (indices >= len(z)).any():
        raise ValueError("Invalid clustering evaluation sample")
    distances = np.linalg.norm(z - centers[labels], axis=1)
    all_distances = np.linalg.norm(z[:, None, :] - centers[None, :, :], axis=2)
    ordered = np.sort(all_distances, axis=1)
    counts = np.bincount(labels, minlength=k)
    sampled, sampled_labels = z[indices], labels[indices]
    active = int(np.count_nonzero(counts))
    sample_active = len(np.unique(sampled_labels))
    valid = 2 <= sample_active < len(sampled) and np.any(np.var(sampled, axis=0) > 1e-15)
    enough = len(z) >= cfg["min_drift_samples"]
    quality = {"silhouette": None, "dbi": None, "chi": None}
    if valid and enough:
        quality = {"silhouette": float(silhouette_score(sampled, sampled_labels)),
                   "dbi": float(davies_bouldin_score(sampled, sampled_labels)),
                   "chi": float(calinski_harabasz_score(sampled, sampled_labels))}
        quality = {key: value if np.isfinite(value) else None for key, value in quality.items()}
    categorical = {"kind": "categorical", "categories": [str(i) for i in range(k)],
                   "proportions": shares.tolist(), "missing_rate": 0.0}
    return {"status": "insufficient_data" if not enough else "ready" if valid else "insufficient_clusters",
            "N": len(z), "eval_N": len(indices), "k": k, "active_clusters": active,
            "intrinsic_metric_sample": "Same deterministic sample for silhouette, DBI and CHI; fixed-distance metrics use all N",
            **quality, "mean_distance": float(distances.mean()), "median_distance": float(np.median(distances)),
            "p95_distance": float(np.quantile(distances, .95)), "distortion": float(np.mean(distances ** 2)),
            "outlier_share": float(np.mean(distances > p95[labels])) if np.isfinite(p95[labels]).all() else None,
            "margin": float(np.mean(ordered[:, 1] - ordered[:, 0])), "counts": counts.tolist(),
            "cluster_shares": (counts / len(z)).tolist(),
            "assignment_drift": drift.compare(labels, categorical, cfg, name="cluster_assignment")}


def frozen_reference(frame, encoded, labels, centers, cfg):
    d = np.linalg.norm(encoded - centers[labels], axis=1)
    k = len(centers)
    counts = np.bincount(labels, minlength=k)
    return {"origin": "Frozen fitted official segmentation snapshot; descriptive in-sample reference",
            "shares": (counts / len(labels)).tolist(),
            "p95": [float(np.quantile(d[labels == i], .95)) if counts[i] else None for i in range(k)],
            "profiles": {str(i): {"sample_count": int(counts[i]), "features": input_reference(frame.loc[labels == i], cfg)}
                         for i in range(k) if counts[i] >= cfg.get("min_cluster_profile_samples", 50)}}


def profile_drift(frame, labels, reference, cfg):
    result = []
    for cluster in range(len(reference["shares"])):
        current = frame.loc[labels == cluster]
        baseline = reference["profiles"].get(str(cluster))
        enough = len(current) >= cfg["min_cluster_profile_samples"] and baseline is not None
        result.append({"cluster_id": cluster, "sample_count": len(current),
                       "status": "ready" if enough else "insufficient_data",
                       "features": [drift.compare(current[name], ref, cfg, name=name)
                                    for name, ref in baseline["features"].items()] if enough else []})
    return result


def compare_baseline(report, baseline, cfg, *, building):
    report["baseline"] = baseline
    report["baseline_building"] = building
    report["drops"] = {key: ((baseline[key] - report[key]) if key in ("silhouette", "chi") else (report[key] - baseline[key]))
                       for key in baseline if baseline[key] is not None and report.get(key) is not None}
    drops = report["drops"]
    report["trigger_candidate"] = not building and report["status"] == "ready" and (
        drops.get("silhouette", 0) >= cfg["silhouette_drop"] or
        (baseline.get("dbi") not in (None, 0) and drops.get("dbi", 0) / abs(baseline["dbi"]) >= cfg["dbi_relative_increase"]) or
        (baseline.get("p95_distance") not in (None, 0) and drops.get("p95_distance", 0) / abs(baseline["p95_distance"]) >= cfg["p95_relative_increase"]) or
        drops.get("outlier_share", 0) >= cfg["outlier_share_increase"])
    report["relative_drops"] = {key: value / abs(baseline[key]) if abs(baseline[key]) > 1e-12 else None for key, value in drops.items()}
    return report
