"""Serve the committed Module 5 XGBoost JSON with its separate preprocessor."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = Path(os.getenv("MODULE5_CHECKPOINT_PATH", str(ROOT / "model_development/models/saved/xgboost_churn.json")))
PREPROCESSOR_PATH = Path(os.getenv("MODULE5_PREPROCESSOR_PATH", str(ROOT / "artifacts/module5/preprocessor.joblib")))
MANIFEST_PATH = Path(os.getenv("MODULE5_MANIFEST_PATH", str(ROOT / "artifacts/module5/manifest.json")))
DATA_PATH = Path(os.getenv("CUSTOMERS_CSV_PATH", str(ROOT / "data/official/module5_customer_features.csv")))
CONFIG_PATH = ROOT / "model_development/config/config.yaml"
REPORT_PATH = ROOT / "model_development/results/comparison/model_comparison_metrics.csv"
PROFILE_COLUMNS = (
    "customer_id", "credit_score", "country", "gender", "age", "tenure", "balance",
    "products_number", "credit_card", "active_member", "estimated_salary", "recency_days",
    "monthly_transactions", "trans_amount_3m", "trans_trend_ratio", "app_logins_monthly",
    "customer_service_complaints", "has_unresolved_complaint", "preferred_channel", "primary_device",
)


class Module5InputError(ValueError):
    """A request cannot provide the raw features required by the checkpoint."""


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Module5Model:
    classifier: XGBClassifier
    preprocessor: object
    manifest: dict

    @property
    def input_columns(self) -> list[str]:
        return self.manifest["input_columns"]

    def transform(self, inputs: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.input_columns) - set(inputs.columns)
        if missing:
            raise Module5InputError("Module 5 raw aggregates missing: " + ", ".join(sorted(missing)))
        # Select raw inputs in training order. Neither identity nor target reaches preprocessing.
        rows = inputs.loc[:, self.input_columns].copy()
        for name, dtype in self.manifest["input_dtypes"].items():
            if dtype == "numeric":
                try:
                    rows[name] = pd.to_numeric(rows[name], errors="raise")
                except (TypeError, ValueError) as exc:
                    raise Module5InputError(f"Module 5 input must be numeric: {name}") from exc
                if np.isinf(rows[name].to_numpy(dtype=float)).any():
                    raise Module5InputError(f"Non-finite input: {name}")
            elif rows[name].isna().any():
                raise Module5InputError(f"Missing categorical input: {name}")
        transformed = self.preprocessor.transform(rows)
        features = self.manifest["features"]
        if not set(features).issubset(transformed):
            raise ValueError("Module 5 transformed feature schema differs from checkpoint")
        return transformed.loc[:, features]

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(self.transform(inputs))

    def profiles(self, rows: pd.DataFrame) -> pd.DataFrame:
        # Display derived behavioral fields using the same Module 5 formulas.
        return self.preprocessor.domain_engineer.transform(rows.drop(columns=["churn"], errors="ignore"))


def load() -> Module5Model | None:
    if not all(path.is_file() for path in (CHECKPOINT_PATH, PREPROCESSOR_PATH, MANIFEST_PATH, DATA_PATH)):
        return None
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for key, path in (("checkpoint_sha256", CHECKPOINT_PATH), ("preprocessor_sha256", PREPROCESSOR_PATH),
                      ("data_sha256", DATA_PATH), ("config_sha256", CONFIG_PATH), ("report_sha256", REPORT_PATH)):
        if manifest.get(key) != file_hash(path):
            raise ValueError(f"Module 5 artifact mismatch: {key}. Run python -m api.prepare_module5")
    if manifest.get("source") != "module5_checkpoint" or not manifest.get("study_reproduced"):
        raise ValueError("Module 5 preprocessor has not reproduced the committed study")
    classifier = XGBClassifier(n_jobs=2)
    classifier.load_model(CHECKPOINT_PATH)
    if classifier.get_booster().feature_names != manifest["features"]:
        raise ValueError("Module 5 checkpoint feature order mismatch")
    return Module5Model(classifier, joblib.load(PREPROCESSOR_PATH), manifest)


def prediction_inputs(model: Module5Model, payload: dict, customers: pd.DataFrame | None) -> pd.DataFrame:
    row = {}
    identifier = payload.get("customer_id")
    if customers is not None and identifier is not None:
        match = customers.loc[customers.customer_id == identifier]
        if not match.empty:
            row.update(match.iloc[0].to_dict())
    extra = payload.get("raw_features") or {}
    unknown = set(extra) - set(model.input_columns)
    if unknown:
        raise Module5InputError("Unsupported Module 5 raw features: " + ", ".join(sorted(unknown)))
    row.update(extra)
    row.update({key: value for key, value in payload.items() if key in model.input_columns})
    missing = set(model.input_columns) - set(row)
    if missing:
        raise Module5InputError("Module 5 requires a customer from the snapshot or complete raw_features. Missing: "
                         + ", ".join(sorted(missing)))
    return pd.DataFrame([row]).loc[:, model.input_columns]
