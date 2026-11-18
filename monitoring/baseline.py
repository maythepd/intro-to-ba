"""Build compact references only from the checkpoint's original training cohort."""
from __future__ import annotations

import numpy as np

from monitoring.drift import categorical_reference, numeric_reference
from monitoring.settings import digest, feature_config, load_config


def build_reference(train, preprocessor, classifier, metrics: dict, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    inputs = train.drop(columns=["customer_id", "churn"], errors="ignore")
    profiles = preprocessor.domain_engineer.transform(inputs)
    features = {}
    for name in cfg["numeric_features"]:
        if name not in profiles:
            raise ValueError(f"Missing numeric reference feature: {name}")
        features[name] = numeric_reference(profiles[name], cfg["histogram_bins"])
    for name in cfg["categorical_features"]:
        if name not in profiles:
            raise ValueError(f"Missing categorical reference feature: {name}")
        features[name] = categorical_reference(profiles[name])
    transformed = preprocessor.transform(inputs)
    probabilities = classifier.predict_proba(transformed[classifier.get_booster().feature_names])[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError("Invalid baseline probabilities")
    return {"version": 1, "origin": "Module 5 original training split, before resampling",
            "sample_count": len(train), "training_subjects_sha256": digest(sorted(train.customer_id.astype(str).tolist())),
            "feature_config_sha256": digest(feature_config(cfg)), "features": features,
            "prediction": numeric_reference(probabilities, cfg["histogram_bins"]),
            "performance": metrics, "decision_threshold": .5}
