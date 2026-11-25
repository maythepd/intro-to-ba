"""Approved Module 5 challenger checkpoints. Never deserialize request-supplied files."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import warnings

import numpy as np
import pandas as pd

from api import module5
from model_development.config import load_unified_config
from model_development.preprocessing.splitting import DataSplitter
from monitoring import drift, performance
from monitoring.settings import ROOT, digest, load_config

ARTIFACTS = ROOT / "artifacts/challengers"
APPROVED = {
    "lr": "ad5b1f0e5837bf616e7767465dcb8065ef8c20be5e9ad2142ad3708a075e243d",
    "rf": "8f4581fa984e6fcd7508630e95143b4a9e37617a28dfbeb48517ab57921b603a",
}
METRICS = {"ROC_AUC": "roc_auc", "PR_AUC": "average_precision", "F1_Score": "f1",
           "Precision": "precision", "Recall": "recall", "Accuracy": "accuracy",
           "Balanced_Accuracy": "balanced_accuracy", "Log_Loss": "log_loss"}


class TrustedCheckpointLoader(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("models.", "preprocessing.", "config.")) or module in ("models", "preprocessing", "config"):
            module = "model_development." + module
        return super().find_class(module, name)


def checkpoint(algorithm):
    if algorithm not in APPROVED:
        raise ValueError("Unsupported challenger")
    path = ROOT / "model_development/models/saved" / f"{algorithm}_churn.pkl"
    if module5.file_hash(path) != APPROVED[algorithm]:
        raise ValueError("Unapproved challenger checkpoint digest")
    return path


@dataclass
class ClassifierAdapter:
    champion: object
    estimator: object
    algorithm: str

    def predict_encoded(self, values):
        if list(values.columns) != self.champion.manifest["features"]:
            raise ValueError("Challenger encoded feature order mismatch")
        classes = list(self.estimator.classes_)
        if classes != [0, 1]:
            raise ValueError("Challenger class mapping must be [0, 1]")
        output = np.asarray(self.estimator.predict_proba(values), dtype=float)
        if output.shape != (len(values), 2):
            raise ValueError("Challenger probability shape mismatch")
        p = output[:, classes.index(1)]
        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError("Invalid challenger probabilities")
        return p


def load_checkpoint(algorithm, champion):
    path = checkpoint(algorithm)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with path.open("rb") as handle:
            wrapper = TrustedCheckpointLoader(handle).load()
    if not wrapper.is_fitted or list(wrapper.feature_names_) != champion.manifest["features"]:
        raise ValueError("Challenger feature provenance/order mismatch")
    estimator = wrapper.model
    if getattr(estimator, "n_features_in_", len(wrapper.feature_names_)) != len(wrapper.feature_names_):
        raise ValueError("Challenger feature count mismatch")
    if hasattr(estimator, "feature_names_in_") and list(estimator.feature_names_in_) != wrapper.feature_names_:
        raise ValueError("Challenger estimator feature order mismatch")
    if hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1  # Stable RF summation order as well as bounded replay resource use.
    return ClassifierAdapter(champion, estimator, algorithm), sorted({str(item.message) for item in recorded})


def original_split():
    data = pd.read_csv(module5.DATA_PATH)
    cfg = load_unified_config(module5.CONFIG_PATH).features
    return DataSplitter(test_size=cfg.test_size, val_size=cfg.val_size, stratify=cfg.stratify,
                        random_state=cfg.random_state).train_val_test_split(data)


def prepare(champion, algorithms=("lr", "rf"), *, output=ARTIFACTS):
    cfg = load_config()
    split = original_split()
    test = champion.transform(split.test_df[champion.input_columns])
    train = champion.transform(split.train_df[champion.input_columns])
    base = json.loads(module5.REFERENCE_PATH.read_text(encoding="utf-8"))
    study = pd.read_csv(module5.REPORT_PATH).set_index("Model")
    result = []
    for algorithm in algorithms:
        adapter, compatibility_warnings = load_checkpoint(algorithm, champion)
        measured = performance.measured_labels(split.test_df.churn, adapter.predict_encoded(test), base, cfg)["metrics"]
        expected = study.loc[algorithm.upper()]
        comparisons = {key: {"expected": float(expected[key]), "actual": measured[value],
                             "difference": measured[value] - float(expected[key])} for key, value in METRICS.items()}
        actual_counts = dict(zip(("TN", "FP", "FN", "TP"), np.asarray(measured["confusion_matrix"]).ravel().tolist()))
        if any(abs(item["difference"]) > 5e-5 for item in comparisons.values()) or any(
                actual_counts[name] != int(expected[name]) for name in actual_counts):
            raise ValueError(f"{algorithm.upper()} holdout reproduction failed: {comparisons}")
        reference = {**base, "prediction": drift.numeric_reference(adapter.predict_encoded(train), cfg["histogram_bins"]),
                     "performance": {key: measured[value] for key, value in METRICS.items()},
                     "checkpoint_sha256": APPROVED[algorithm]}
        reference_hash = digest(reference)
        manifest = {"version": 1, "algorithm": algorithm, "role": "challenger", "study_reproduced": True,
                    "checkpoint_sha256": APPROVED[algorithm], "preprocessor_sha256": champion.manifest["preprocessor_sha256"],
                    "features": champion.manifest["features"], "input_columns": champion.input_columns,
                    "class_mapping": [0, 1], "threshold": base["decision_threshold"],
                    "data_sha256": champion.manifest["data_sha256"], "config_sha256": champion.manifest["config_sha256"],
                    "split": champion.manifest["split"], "reference_sha256": reference_hash, "reference": reference,
                    "holdout_metrics": measured, "comparisons": comparisons,
                    "compatibility_warnings": compatibility_warnings,
                    "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                                    for name in ("numpy", "scikit-learn", "joblib")}}}
        manifest["model_id"] = digest(["challenger_adapter_v1", algorithm, APPROVED[algorithm],
                                       manifest["preprocessor_sha256"], reference_hash, manifest["threshold"]])
        folder = Path(output) / algorithm
        folder.mkdir(parents=True, exist_ok=True)
        staging = folder / "manifest.json.tmp"
        staging.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        staging.replace(folder / "manifest.json")
        result.append(manifest)
        del adapter
    return result


def load(algorithm, champion, *, directory=ARTIFACTS):
    manifest = json.loads((Path(directory) / algorithm / "manifest.json").read_text(encoding="utf-8"))
    for name in ("preprocessor_sha256", "data_sha256", "config_sha256"):
        if manifest[name] != champion.manifest[name]:
            raise ValueError("Challenger preparation belongs to another serving preprocessor/data/config")
    if (manifest["checkpoint_sha256"] != APPROVED[algorithm] or not manifest["study_reproduced"] or
            manifest["features"] != champion.manifest["features"] or manifest["class_mapping"] != [0, 1] or
            manifest["algorithm"] != algorithm or manifest["input_columns"] != champion.input_columns or
            manifest["threshold"] != manifest["reference"]["decision_threshold"] or
            manifest["reference"]["checkpoint_sha256"] != APPROVED[algorithm] or
            digest(manifest["reference"]) != manifest["reference_sha256"]):
        raise ValueError("Invalid prepared challenger manifest")
    identity = digest(["challenger_adapter_v1", algorithm, APPROVED[algorithm], manifest["preprocessor_sha256"],
                       manifest["reference_sha256"], manifest["threshold"]])
    if identity != manifest["model_id"]:
        raise ValueError("Challenger identity mismatch")
    adapter, _ = load_checkpoint(algorithm, champion)
    return adapter, manifest
