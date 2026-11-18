"""Fixed-reference PSI and base-2 Jensen-Shannon distance; missingness is separate."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def numeric_reference(values, bins: int = 10) -> dict:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    good = array[np.isfinite(array)]
    if not len(good):
        return {"kind": "numeric", "cuts": [], "proportions": [1.0], "missing_rate": 1.0}
    cuts = np.unique(np.quantile(good, np.linspace(0, 1, bins + 1)[1:-1]))
    if good.min() == good.max():
        value = float(good[0])
        epsilon = max(abs(value) * 1e-6, 1e-6)
        cuts = np.array([value - epsilon, value + epsilon])
    counts = np.bincount(np.searchsorted(cuts, good, side="right"), minlength=len(cuts) + 1)
    return {"kind": "numeric", "cuts": cuts.tolist(), "proportions": (counts / counts.sum()).tolist(),
            "missing_rate": float(1 - len(good) / len(array)) if len(array) else 0.0}


def categorical_reference(values) -> dict:
    series = pd.Series(values)
    good = series.dropna().astype(str)
    counts = good.value_counts(normalize=True).sort_index()
    return {"kind": "categorical", "categories": counts.index.tolist(), "proportions": counts.tolist(),
            "missing_rate": float(series.isna().mean()) if len(series) else 0.0}


def compare(values, ref: dict, cfg: dict, *, name: str) -> dict:
    series = pd.Series(values)
    labels = []
    new_rate = 0.0
    if ref["kind"] == "numeric":
        array = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        good = array[np.isfinite(array)]
        counts = np.bincount(np.searchsorted(ref["cuts"], good, side="right"), minlength=len(ref["proportions"]))
        current = counts / max(1, counts.sum())
        expected = np.asarray(ref["proportions"], dtype=float)
        edges = ["-infinity"] + [f"{value:.5g}" for value in ref["cuts"]] + ["+infinity"]
        labels = [f"[{edges[i]}, {edges[i + 1]})" for i in range(len(edges) - 1)]
        eps = 1e-6
        p = np.clip(expected, eps, None); p /= p.sum()
        q = np.clip(current, eps, None); q /= q.sum()
        score = float(np.sum((q - p) * np.log(q / p))) if len(good) else None
        method, warning, critical = "PSI", cfg["psi_warning"], cfg["psi_critical"]
    else:
        good = series.dropna().astype(str)
        labels = sorted(set(ref["categories"]) | set(good.unique()))
        lookup = dict(zip(ref["categories"], ref["proportions"]))
        expected = np.array([lookup.get(label, 0.0) for label in labels])
        frequencies = good.value_counts(normalize=True)
        current = np.array([frequencies.get(label, 0.0) for label in labels])
        score = float(jensenshannon(expected, current, base=2)) if len(good) and expected.sum() else None
        new_rate = float((~good.isin(ref["categories"])).mean()) if len(good) else 0.0
        method, warning, critical = "Jensen-Shannon distance (base 2)", cfg["js_warning"], cfg["js_critical"]
    missing = float(1 - len(good) / len(series)) if len(series) else 0.0
    state = "missing_data" if score is None else "critical" if score >= critical else "warning" if score >= warning else "stable"
    # Only aggregate distributions are exposed; never return individual observations.
    distribution = [{"bin": label, "reference": float(p), "current": float(q)}
                    for label, p, q in zip(labels, expected, current)]
    if len(distribution) > 20:
        largest = sorted(distribution, key=lambda row: row["reference"] + row["current"], reverse=True)[:19]
        largest.append({"bin": "Other categories (aggregated)",
                        "reference": max(0.0, 1 - sum(row["reference"] for row in largest)),
                        "current": max(0.0, 1 - sum(row["current"] for row in largest))})
        distribution = largest
    return {"name": name, "kind": ref["kind"], "method": method, "score": score, "status": state,
            "valid_samples": len(good), "missing_rate": missing,
            "missing_excess": max(0.0, missing - ref["missing_rate"]), "new_category_rate": new_rate,
            "distribution": distribution}


def evaluate(frame: pd.DataFrame, reference: dict, cfg: dict) -> dict:
    if len(frame) < cfg["min_drift_samples"]:
        return {"status": "insufficient_data", "sample_count": len(frame), "minimum_samples": cfg["min_drift_samples"],
                "features": [], "critical_share": None, "trigger_candidate": False}
    rows = [compare(frame.get(name, pd.Series([None] * len(frame))), ref, cfg, name=name)
            for name, ref in reference["features"].items()]
    share = sum(row["status"] == "critical" for row in rows) / len(rows)
    key_shift = any(row["name"] in cfg["critical_features"] and row["status"] == "critical" for row in rows)
    candidate = share >= cfg["drift_feature_share"] or key_shift
    rows.sort(key=lambda row: (row["score"] is None, -(row["score"] or 0)))
    return {"status": "critical" if candidate else "warning" if any(r["status"] in ("warning", "critical") for r in rows) else "stable",
            "sample_count": len(frame), "minimum_samples": cfg["min_drift_samples"], "features": rows,
            "critical_share": share, "trigger_candidate": candidate}
