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


def risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.3:
        return "medium"
    return "low"
