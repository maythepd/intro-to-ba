"""Shared churn scoring path for single-record and batch inference.

Never train, fit preprocessing, or take churn labels from the scoring population.
The trained feature order comes exclusively from the fitted bundle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def score_probabilities(bundle: dict, inputs: pd.DataFrame) -> np.ndarray:
    if not isinstance(bundle, dict) or not {"model", "pipeline", "features"}.issubset(bundle):
        raise ValueError("Incomplete fitted model bundle")
    features = list(bundle["features"])
    if not features or "churn" in features or "customer_id" in features:
        raise ValueError("Target/identity leakage or empty model feature list")
    if inputs.empty:
        raise ValueError("No customers to score")

    # Do not pass observed ground-truth labels to preprocessing at inference time.
    rows = inputs.drop(columns=["churn"], errors="ignore").copy()
    rows["churn"] = 0  # legacy pipeline expects this column, but must not use it in X
    transformed = bundle["pipeline"].transform(rows)
    missing = set(features) - set(transformed.columns)
    if missing:
        raise ValueError(f"Missing transformed model features: {sorted(missing)}")
    X = transformed.loc[:, features]
    raw = np.asarray(bundle["model"].predict_proba(X), dtype=float)
    if raw.ndim == 2 and raw.shape == (len(inputs), 2):
        probabilities = raw[:, 1]
    elif raw.ndim == 1 and raw.shape[0] == len(inputs):
        probabilities = raw
    else:
        raise ValueError("Incompatible probability shape")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid model probabilities")
    return probabilities


def risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.3:
        return "medium"
    return "low"
