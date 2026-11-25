"""Instrument predictions without making classifier availability depend on telemetry."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

import numpy as np
import pandas as pd

from api import module5
from monitoring.drift import categorical_reference
from monitoring.runner import closed_window, run_window, summarize
from monitoring.settings import Settings, digest, feature_config, now_utc, timestamp
from monitoring.store import Store


class MonitoringService:
    def __init__(self, model, *, segmentation=None, settings: Settings | None = None):
        self.model = model
        self.settings = settings or Settings.from_env()
        self.config = self.settings.config
        self.store = None
        self.model_id = None
        self.last_error = None
        self.last_error_at = None
        self.last_write_at = None
        self.lock = RLock()
        self.segmentation = segmentation
        self.segment_lookup = {}
        self.segment_model_id = None
        self.reference = None
        self.registered = False
        self.registration = None
        try:
            if model is None:
                self.last_error = "model_unavailable"
                return
            if not module5.REFERENCE_PATH.is_file():
                self.last_error = "reference_unavailable_run_prepare_module5"
                return
            expected = model.manifest.get("monitoring_reference_sha256")
            if not expected or module5.file_hash(module5.REFERENCE_PATH) != expected:
                raise ValueError("Monitoring reference digest mismatch")
            reference = json.loads(module5.REFERENCE_PATH.read_text(encoding="utf-8"))
            for name in ("checkpoint_sha256", "preprocessor_sha256"):
                if reference.get(name) != model.manifest[name]:
                    raise ValueError("Reference belongs to a different serving model")
            if reference["feature_config_sha256"] != digest(feature_config(self.config)):
                raise ValueError("Reference feature config changed; rerun prepare_module5")
            if segmentation:
                labels = [str(int(value)) for value in segmentation["models"]["kmeans"]["labels"]]
                self.segment_model_id = digest([segmentation["data_sha256"], segmentation["customer_ids"], labels])
                self.segment_lookup = dict(zip(segmentation["customer_ids"], labels))
                reference["segments"] = {"model_id": self.segment_model_id, "reference": categorical_reference(labels)}
            self.reference = reference
            reference_hash = digest(reference)
            self.model_id = digest([reference["checkpoint_sha256"], reference["preprocessor_sha256"], reference_hash])
            if not self.settings.database_url:
                self.last_error = "database_not_configured"
                return
            details = {"name": "XGBoost (Module 5)", "checkpoint_sha256": reference["checkpoint_sha256"],
                       "preprocessor_sha256": reference["preprocessor_sha256"], "feature_count": len(model.manifest["features"]),
                       "reference_sample_count": reference["sample_count"], "baseline_metrics": reference["performance"],
                       "segment_model_id": self.segment_model_id, "study_reproduced": True}
            self.registration = (self.model_id, reference, details, reference_hash)
            self.store = Store(self.settings.database_url)
            self.store.ping()
            self.store.register_model(self.model_id, reference, details, reference_hash)
            self.registered = True
        except Exception as exc:
            self.failed(exc)

    def failed(self, error):
        # Do not expose connection strings, passwords, SQL parameters or raw inputs.
        self.last_error = type(error).__name__
        self.last_error_at = timestamp(now_utc())

    @classmethod
    def default(cls):
        from api import segmentation_service
        settings = Settings.from_env()
        segmentation = segmentation_service.load(Path(os.getenv("SEGMENTATION_CUSTOMERS_CSV_PATH", "data/official/customer_features.csv")))
        return cls(module5.load(), settings=settings, segmentation=segmentation)

    def close(self):
        if self.store:
            self.store.close()

    def status(self) -> dict:
        ready = False
        if self.store:
            try:
                ready = self.store.ping()
                if ready and not self.registered and self.registration:
                    self.store.register_model(*self.registration)
                    self.registered = True
                ready = ready and self.registered
            except Exception as exc:
                self.failed(exc)
        return {"storage_ready": ready, "storage_backend": "sqlite_local" if self.store and self.store.is_sqlite else "postgresql" if self.store else "not_configured",
                "storage_persistent": bool(ready and not self.store.is_sqlite), "model_id": self.model_id,
                "last_ingest_error": self.last_error, "last_error_at": self.last_error_at,
                "last_write_at": self.last_write_at, "reference_available": self.reference is not None,
                "window_hours": self.config["window_hours"], "outcome_horizon_days": self.config["outcome_horizon_days"],
                "minimum_drift_samples": self.config["min_drift_samples"],
                "demo_enabled": self.settings.demo_enabled, "admin_operations_configured": bool(self.settings.admin_token)}

    def event_rows(self, inputs: pd.DataFrame, probabilities, *, source: str, customer_ids=None,
                   occurred_at=None, scenario="", namespace=None, latency_ms=0.0) -> list[dict]:
        if self.model is None or self.reference is None:
            return []
        date = now_utc() if occurred_at is None else pd.Timestamp(occurred_at).to_pydatetime()
        profiles = self.model.profiles(inputs)
        identifiers = list(customer_ids) if customer_ids is not None else inputs.get("customer_id", pd.Series([None] * len(inputs))).tolist()
        rows = []
        feature_names = list(self.reference["features"])
        for index, (_, profile) in enumerate(profiles.iterrows()):
            person = identifiers[index]
            event_id = digest([self.model_id, namespace, str(person) if person is not None else index]) if namespace else uuid4().hex
            subject = digest(["customer", str(person)]) if person is not None else event_id
            values = {}
            for name in feature_names:
                value = profile.get(name)
                values[name] = None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value
            rows.append({"id": event_id, "model_id": self.model_id, "occurred_at": timestamp(date),
                "horizon_end": timestamp(date + timedelta(days=self.config["outcome_horizon_days"])),
                "subject": subject, "source": source, "scenario": scenario, "features": values,
                "probability": float(probabilities[index]), "latency_ms": float(latency_ms), "valid": 1,
                "error_type": None, "segment": self.segment_lookup.get(str(person)), "segment_model_id": self.segment_model_id})
        return rows

    def record(self, inputs, probabilities, **kwargs) -> list[dict | None]:
        if not self.store or self.reference is None:
            return [None] * len(inputs)
        try:
            rows = self.event_rows(inputs, probabilities, **kwargs)
            self.store.append_events(rows)
            self.last_write_at = timestamp(now_utc())
            self.last_error = None
            return [{"prediction_id": row["id"], "prediction_timestamp": row["occurred_at"],
                     "outcome_horizon_end": row["horizon_end"]} for row in rows]
        except Exception as exc:
            self.failed(exc)
            return [None] * len(inputs)

    def record_failure(self, kind: str, latency_ms: float):
        if not self.store or not self.model_id:
            return
        date = now_utc()
        identifier = uuid4().hex
        row = {"id": identifier, "model_id": self.model_id, "occurred_at": timestamp(date),
               "horizon_end": timestamp(date + timedelta(days=self.config["outcome_horizon_days"])),
               "subject": identifier, "source": "live", "scenario": "", "features": {}, "probability": None,
               "latency_ms": latency_ms, "valid": 0, "error_type": kind, "segment": None, "segment_model_id": None}
        try:
            self.store.append_events([row])
        except Exception as exc:
            self.failed(exc)

    def summary(self, *, source="live", scenario="", model_id=None, since=None) -> dict:
        status = self.status()
        result = {**status, "source": source, "scenario": scenario, "model_id": model_id or self.model_id,
                  "latest": None, "history": [], "models": [], "drift_available": False,
                  "performance_tracking_available": False, "stale": True}
        if not status["storage_ready"] or not result["model_id"]:
            return result
        reports = self.store.list_runs(result["model_id"], source, scenario, since=since)
        result["models"] = self.store.list_models()
        result["scenarios"] = self.store.list_scenarios(result["model_id"])
        result["scenario_details"] = self.store.scenario_details(result["model_id"])
        result["history"] = [summarize(report) for report in reversed(reports)]
        result["latest"] = reports[0] if reports else None
        if reports:
            newest = reports[0]
            result["drift_available"] = bool(newest["drift"]["features"])
            result["performance_tracking_available"] = newest["performance"]["metrics"] is not None
            result["stale"] = now_utc() - pd.Timestamp(newest["computed_at"]).to_pydatetime() > timedelta(hours=self.config["window_hours"] * 2)
        return result

    def run(self, **kwargs):
        if not self.store or not self.model_id:
            raise ValueError("Monitoring storage/reference is unavailable")
        with self.lock:
            return run_window(self.store, kwargs.pop("model_id", None) or self.model_id, self.config, **kwargs)

    def refresh_labels(self, identifiers: list[str]) -> list[str]:
        """Delayed labels refresh their original windows, including downstream streaks."""
        windows = set()
        duration = timedelta(hours=self.config["window_hours"])
        for row in self.store.prediction_scopes(identifiers):
            start, end = closed_window(datetime.fromisoformat(row["occurred_at"]) + duration, self.config["window_hours"])
            windows.add((row["model_id"], row["source"], row["scenario"], start, end))
            for step in range(1, self.config["consecutive_windows"]):
                next_start = timestamp(datetime.fromisoformat(start) + duration * step)
                next_end = timestamp(datetime.fromisoformat(end) + duration * step)
                if self.store.previous_run(row["model_id"], row["source"], row["scenario"], next_end, digest(self.config)):
                    windows.add((row["model_id"], row["source"], row["scenario"], next_start, next_end))
        identifiers = []
        for model_id, source, scenario, start, end in sorted(windows):
            report = self.run(model_id=model_id, source=source, scenario=scenario, start=start, end=end)
            identifiers.append(report["id"])
        return identifiers
