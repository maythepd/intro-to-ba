from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("live", "snapshot_batch", "study_replay")


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def timestamp(value: str | datetime) -> str:
    date = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if date.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return date.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_config(path: Path | None = None) -> dict:
    cfg = yaml.safe_load((path or ROOT / "config/monitoring.yaml").read_text(encoding="utf-8"))
    for name in ("window_hours", "min_drift_samples", "min_labeled_samples", "min_class_samples",
                 "min_quality_samples", "histogram_bins", "consecutive_windows", "max_window_events",
                 "outcome_horizon_days", "retention_days", "max_job_windows"):
        if not isinstance(cfg[name], int) or cfg[name] <= 0:
            raise ValueError(f"Invalid monitoring setting: {name}")
    for name in ("min_label_coverage", "drift_feature_share", "performance_drop", "quality_error_rate"):
        if not 0 < cfg[name] <= 1:
            raise ValueError(f"Invalid monitoring setting: {name}")
    if not (0 < cfg["psi_warning"] < cfg["psi_critical"] and 0 < cfg["js_warning"] < cfg["js_critical"] <= 1):
        raise ValueError("Invalid drift thresholds")
    names = cfg["numeric_features"] + cfg["categorical_features"]
    if len(names) != len(set(names)) or {"churn", "customer_id"}.intersection(names):
        raise ValueError("Monitoring features must be unique and exclude identity/target")
    if not set(cfg["critical_features"]).issubset(cfg["numeric_features"]):
        raise ValueError("Critical features must be monitored numeric inputs")
    return cfg


def feature_config(cfg: dict) -> dict:
    return {name: cfg[name] for name in ("numeric_features", "categorical_features", "histogram_bins")}


@dataclass
class Settings:
    database_url: str | None = field(repr=False)
    admin_token: str = field(default="", repr=False)
    demo_enabled: bool = False
    config: dict = field(default_factory=load_config)

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env", override=False)
        database = os.getenv("DATABASE_URL")
        if not database and not os.getenv("RENDER"):
            path = ROOT / "artifacts/monitoring/monitoring.sqlite3"
            database = "sqlite:///" + path.as_posix()
        return cls(database, os.getenv("MONITORING_ADMIN_TOKEN", ""),
                   os.getenv("ENABLE_MONITORING_DEMO", "false").lower() == "true")
