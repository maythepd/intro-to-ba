"""Shared churn scoring path for single-record and batch inference.

Never train, fit preprocessing, or take churn labels from the scoring population.
The serving adapter owns the Module 5 preprocessing and checkpoint feature order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def score_probabilities(model, inputs: pd.DataFrame) -> np.ndarray:
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Module 5 prediction model unavailable")
    if inputs.empty:
        raise ValueError("No customers to score")

    # Do not pass observed ground-truth labels to preprocessing at inference time.
    rows = inputs.drop(columns=["churn"], errors="ignore").copy()
    raw = np.asarray(model.predict_proba(rows), dtype=float)
    if raw.ndim == 2 and raw.shape == (len(inputs), 2):
        probabilities = raw[:, 1]
    elif raw.ndim == 1 and raw.shape[0] == len(inputs):
        probabilities = raw
    else:
        raise ValueError("Incompatible probability shape")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid model probabilities")
    return probabilities


def risk_level(probability: float, threshold: float = 0.5) -> str:
    """
    Map churn probability to risk band relative to decision threshold.
    When threshold = 0.5, preserves standard benchmark bands (low < 0.3, med 0.3-0.7, high >= 0.7).
    """
    if threshold <= 0.0 or threshold >= 1.0:
        threshold = 0.5

    # If probability reaches high absolute level or significantly exceeds decision threshold:
    high_cutoff = min(0.85, max(0.65, threshold + 0.15))
    low_cutoff = max(0.15, min(0.30, threshold - 0.20))

    if probability >= high_cutoff:
        return "high"
    if probability >= low_cutoff:
        return "medium"
    return "low"
