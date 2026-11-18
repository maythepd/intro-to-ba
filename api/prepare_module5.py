"""Reproduce Module 5 preprocessing; load the saved checkpoint without retraining it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

from api import module5
from model_development.config import load_unified_config
from model_development.preprocessing.etl import BankDataETL
from model_development.preprocessing.pipeline import FeatureEngineeringPipeline
from model_development.preprocessing.splitting import DataSplitter
from monitoring.baseline import build_reference

DATASET = "maythepd/full-dataset-intro2ba/versions/3"


def prepare(data: pd.DataFrame, *, checkpoint: Path = module5.CHECKPOINT_PATH,
            config: Path = module5.CONFIG_PATH, report: Path = module5.REPORT_PATH,
            preprocessor_path: Path = module5.PREPROCESSOR_PATH,
            manifest_path: Path = module5.MANIFEST_PATH,
            data_path: Path = module5.DATA_PATH) -> dict:
    if "customer_id" not in data or data.customer_id.isna().any() or data.customer_id.duplicated().any():
        raise ValueError("Module 5 requires unique non-null customer IDs")
    cfg = load_unified_config(config).features
    split = DataSplitter(test_size=cfg.test_size, val_size=cfg.val_size,
                         stratify=cfg.stratify, random_state=cfg.random_state).train_val_test_split(data)
    preprocessor = FeatureEngineeringPipeline(cfg).fit(split.train_df)
    classifier = XGBClassifier(n_jobs=2)
    classifier.load_model(checkpoint)
    features = classifier.get_booster().feature_names
    transformed = preprocessor.transform(split.test_df.drop(columns=["churn", "customer_id"]))
    if preprocessor.extract_model_features(transformed) != features:
        raise ValueError("Module 5 preprocessing feature names/order do not match the saved checkpoint")
    probabilities = classifier.predict_proba(transformed.loc[:, features])[:, 1]
    y = split.test_df.churn
    predicted = probabilities >= .5
    metrics = {
        "ROC_AUC": float(roc_auc_score(y, probabilities)),
        "PR_AUC": float(average_precision_score(y, probabilities)),
        "F1_Score": float(f1_score(y, predicted, zero_division=0)),
        "Precision": float(precision_score(y, predicted, zero_division=0)),
        "Recall": float(recall_score(y, predicted, zero_division=0)),
        "Accuracy": float(accuracy_score(y, predicted)),
    }
    reports = pd.read_csv(report)
    expected = reports.loc[reports.Model.str.upper() == "XGBOOST"].iloc[0]
    mismatches = {key: {"actual": value, "study": float(expected[key])}
                  for key, value in metrics.items() if abs(value - float(expected[key])) > .00005}
    if mismatches:
        raise ValueError(f"Saved Module 5 checkpoint/preprocessing do not reproduce the study: {mismatches}")
    input_columns = [key for key in data.columns if key not in ("churn", "customer_id")]
    manifest = {
        "source": "module5_checkpoint", "model_name": "XGBoost (Module 5)",
        "checkpoint_sha256": module5.file_hash(checkpoint), "config_sha256": module5.file_hash(config),
        "report_sha256": module5.file_hash(report), "study_reproduced": True,
        "preprocessing_origin": "Reconstructed on the original Module 5 training split; checkpoint unchanged",
        "dataset": DATASET, "features": features, "input_columns": input_columns,
        "input_dtypes": {key: "numeric" if pd.api.types.is_numeric_dtype(data[key]) else "categorical"
                         for key in input_columns},
        "decision_threshold": .5, "validation_metrics": metrics,
        "split": {"train": len(split.train_df), "validation": len(split.val_df), "test": len(split.test_df),
                  "random_state": cfg.random_state},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    reference = build_reference(split.train_df, preprocessor, classifier, metrics)
    # Publish only a fully validated export. The manifest is written last.
    output = preprocessor_path.parent
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as folder:
        staging = Path(folder)
        joblib.dump(preprocessor, staging / "preprocessor.joblib")
        data.to_csv(staging / "customers.csv", index=False)
        manifest["preprocessor_sha256"] = module5.file_hash(staging / "preprocessor.joblib")
        manifest["data_sha256"] = module5.file_hash(staging / "customers.csv")
        reference.update(checkpoint_sha256=manifest["checkpoint_sha256"], preprocessor_sha256=manifest["preprocessor_sha256"])
        (staging / "monitoring_reference.json").write_text(json.dumps(reference, allow_nan=False), encoding="utf-8")
        split.test_df.to_csv(staging / "monitoring_replay.csv", index=False)
        manifest["monitoring_reference_sha256"] = module5.file_hash(staging / "monitoring_reference.json")
        manifest["monitoring_replay_sha256"] = module5.file_hash(staging / "monitoring_replay.csv")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (staging / "preprocessor.joblib").replace(preprocessor_path)
        (staging / "customers.csv").replace(data_path)
        (staging / "monitoring_reference.json").replace(output / "monitoring_reference.json")
        (staging / "monitoring_replay.csv").replace(output / "monitoring_replay.csv")
        (staging / "manifest.json").replace(manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--source", type=Path, help="Directory with the seven original Module 5 CSV tables")
    parser.add_argument("--data", type=Path, help="Previously materialized Module 5 raw customer table")
    args = parser.parse_args()
    if args.data:
        data = pd.read_csv(args.data)
    else:
        source = args.source
        if args.download:
            import kagglehub
            source = Path(kagglehub.dataset_download(DATASET))
        if source is None:
            parser.error("Provide --download, --source, or --data")
        data = BankDataETL(raw_data_dir=source).process()
    manifest = prepare(data)
    print(f"Module 5 checkpoint ready: {len(manifest['features'])} features; study metrics reproduced")
    print(json.dumps(manifest["validation_metrics"]))


if __name__ == "__main__":
    main()
