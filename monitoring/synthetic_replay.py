"""Replay five validated synthetic cohorts through the existing monitoring service."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from api import module5
from demo_data.generate import BATCHES, OUTPUT
from demo_data.labels import read_labels
from demo_data.validate import check_dataset, model_paths, read_outputs, score_batch
from monitoring import performance
from monitoring.runner import summarize
from monitoring.settings import digest, now_utc, timestamp

VERSION = "synthetic_replay_v1"


def replay(service, *, data_dir: Path = OUTPUT, batch: str | None = None,
           labels_path: Path | None = None, anchor: str | None = None,
           scenario: str | None = None) -> dict:
    if not service.store or not service.model_id or service.model is None or service.segmentation is None:
        raise ValueError("Prepared classifier, segmentation, monitoring reference and database are required")
    if batch is not None and batch not in BATCHES:
        raise ValueError("Unknown synthetic batch")
    if service.config["window_hours"] != 24:
        raise ValueError("Synthetic replay v1 requires 24-hour windows")
    data_dir = Path(data_dir)
    if (data_dir / ".publishing").exists():
        raise ValueError("Demo artifacts are being published; retry after generation completes")
    manifest, frame, examples = read_outputs(data_dir)
    if (manifest["input_columns"] != service.model.input_columns or
            manifest["input_dtypes"] != service.model.manifest["input_dtypes"]):
        raise ValueError("Demo input schema differs from the fixed serving model")
    for name, path in model_paths().items():
        if not path.is_file() or module5.file_hash(path) != manifest["models"][name]["sha256"]:
            raise ValueError(f"Demo model artifact hash mismatch: {name}")
    if manifest["inference"]["status"] != "passed":
        raise ValueError("Generate and validate with both fitted models before replay")
    source_ids = set(manifest["source_customer_id_by_demo_id"].values())
    if len(source_ids) != manifest["N_base"]:
        source_ids = pd.read_csv(module5.DATA_PATH, usecols=["customer_id"]).customer_id
    check_dataset(frame, examples, manifest, source_ids)
    if set(frame.batch_id) != set(BATCHES):
        raise ValueError("Replay requires all five nonempty synthetic batches")
    if any(int(frame.batch_id.eq(name).sum()) > service.config["max_window_events"] for name in BATCHES):
        raise ValueError("A batch exceeds max_window_events")
    dataset_hash = manifest["output_sha256"]["customer_features.csv"]
    labels, label_metadata = (read_labels(Path(labels_path), frame, dataset_hash)
                              if labels_path is not None else (None, None))
    model = service.store.get_model(service.model_id)
    if model is None or model["reference_sha256"] != digest(service.reference):
        raise ValueError("Monitoring model/reference registration mismatch")
    as_of = now_utc()
    if anchor:
        first = datetime.fromisoformat(timestamp(anchor))
    else:
        first = datetime.fromisoformat(model["created_at"]).replace(hour=0, minute=0, second=0, microsecond=0)
        first -= timedelta(days=service.config["outcome_horizon_days"] + len(BATCHES) + 2)
    if first.timestamp() % (24 * 3600):
        raise ValueError("Replay anchor must be a UTC midnight")
    last_end = first + timedelta(days=len(BATCHES))
    if first < as_of - timedelta(days=service.config["retention_days"]) or last_end > as_of:
        raise ValueError("Replay windows must be closed and within retention; choose an explicit versioned --anchor")
    if labels is not None and last_end + timedelta(days=service.config["outcome_horizon_days"]) > as_of:
        raise ValueError("All simulated outcome horizons must be mature before importing labels")
    identity = {"version": VERSION, "dataset_sha256": dataset_hash, "generation_seed": manifest["seed"],
                "model_id": service.model_id, "reference_sha256": model["reference_sha256"],
                "segmentation_bundle_sha256": manifest["models"]["clustering_bundle"]["sha256"],
                "config_sha256": digest(service.config), "anchor": timestamp(first),
                "label_policy_version": label_metadata["label_policy_version"] if label_metadata else None,
                "label_policy_sha256": label_metadata["policy_sha256"] if label_metadata else None,
                "label_seed": label_metadata["label_seed"] if label_metadata else None,
                "labels_sha256": label_metadata["output_sha256"] if label_metadata else None}
    run_identity = digest(identity)
    scenario = scenario or f"synthetic_demo_{run_identity[:12]}"
    if not re.fullmatch(r"synthetic_demo(?:_[a-zA-Z0-9_-]+)?", scenario) or len(scenario) > 60:
        raise ValueError("Scenario must start with synthetic_demo, use letters/digits/_/-, and be at most 60 characters")

    # Complete all validation, scoring, segmentation and label joins before the first database mutation.
    probabilities, clusters = score_batch(frame, manifest["input_columns"], service.model, service.segmentation)
    if (len(probabilities) != len(frame) or not np.isfinite(probabilities).all() or
            not ((probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError("Classifier returned invalid probabilities")
    membership = clusters["kmeans"]
    if len(membership) != len(frame) or not set(map(str, membership)).issubset(set(service.reference["segments"]["reference"]["categories"])):
        raise ValueError("KMeans memberships differ from the registered reference")
    baseline = None
    if labels is not None:
        normal = frame.batch_id.str.startswith("normal_")
        baseline_events = [{"horizon_end": timestamp(first), "observed_at": timestamp(first),
                            "label": int(labels.loc[person, "churn"]), "label_source": "synthetic_stress",
                            "probability": float(probabilities[index])}
                           for index, person in enumerate(frame.customer_id) if normal.iloc[index]]
        measured = performance.evaluate(baseline_events, service.reference, service.config, as_of=timestamp(as_of))
        if measured["status"] == "ready":
            baseline = {"batches": list(BATCHES[:3]), "sample_count": measured["labeled_samples"],
                        "label_policy_version": label_metadata["label_policy_version"], "metrics": measured["metrics"]}
    prepared = []
    for index, name in enumerate(BATCHES):
        if batch is not None and name != batch:
            continue
        mask = frame.batch_id.eq(name)
        cohort = frame.loc[mask]
        start, end = first + timedelta(days=index), first + timedelta(days=index + 1)
        namespace = digest([run_identity, name])
        rows = service.event_rows(cohort[manifest["input_columns"]], probabilities[mask], source="study_replay",
                                  scenario=scenario, namespace=namespace, occurred_at=start + timedelta(hours=1),
                                  customer_ids=cohort.customer_id.tolist())
        for row, cluster in zip(rows, membership[mask]):
            row.update(segment=str(int(cluster)), segment_model_id=service.segment_model_id)
        outcomes = [] if labels is None else [
            {"event_id": row["id"], "label": int(labels.loc[person, "churn"]),
             "horizon_end": row["horizon_end"], "observed_at": row["horizon_end"], "source": "synthetic_stress"}
            for row, person in zip(rows, cohort.customer_id)]
        metadata = {**identity, "run_identity": run_identity, "batch_id": name, "batch_index": index + 1,
                    "batch_count": len(BATCHES), "namespace": namespace, "feature_snapshot": manifest["snapshot"],
                    "time_kind": "simulated", "outcome_kind": "simulated" if labels is not None else "unavailable",
                    "execution_kind": "offline_or_admin_replay_not_deployed_latency",
                    "synthetic_baseline": baseline if index >= 3 else None,
                    "label_policy": label_metadata["policy"] if label_metadata else None}
        prepared.append((name, timestamp(start), timestamp(end), rows, outcomes, metadata))

    created_events = created_outcomes = 0
    reports = []
    with service.lock:
        # Each batch commits as one unit. A failed batch can be safely retried; earlier completed batches remain visible.
        for name, start, end, rows, outcomes, metadata in prepared:
            with service.store.transaction():
                service.store.lock_scope(f"{service.model_id}:{scenario}")
                history = service.store.list_runs(service.model_id, "study_replay", scenario)
                if any(item.get("synthetic_replay", {}).get("run_identity") != run_identity for item in history):
                    raise ValueError("Scenario already belongs to another replay identity; use a new scenario")
                existing = service.store.window_events(service.model_id, "study_replay", scenario, start, end,
                                                       service.config["max_window_events"])
                if {row["id"] for row in existing} - {row["id"] for row in rows}:
                    raise ValueError("Replay window contains events from another identity")
                inserted_events = service.store.append_events(rows)
                inserted_outcomes = service.store.add_outcomes(outcomes, as_of=timestamp(as_of)) if outcomes else 0
                report = service.run(source="study_replay", scenario=scenario, start=start, end=end,
                                     as_of=timestamp(as_of), replay_metadata=metadata)
                if report["sample_count"] != len(rows):
                    raise ValueError("Stored replay cohort does not match the validated batch")
            created_events += inserted_events
            created_outcomes += inserted_outcomes
            reports.append(summarize(report))
    return {"source": "study_replay", "scenario": scenario, "run_identity": run_identity,
            "created_events": created_events, "created_outcomes": created_outcomes, "runs": reports}


def export_reports(service, result: dict, output: Path):
    """Export actual measurements, full provenance and feature distributions."""
    from monitoring.report import render_html
    output.mkdir(parents=True, exist_ok=True)
    summaries, features = [], []
    for item in result["runs"]:
        report = service.store.get_run(item["id"])
        name = report["synthetic_replay"]["batch_id"]
        (output / f"{name}.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        (output / f"{name}.html").write_text(render_html(report), encoding="utf-8")
        summaries.append({"batch_id": name, "N": report["sample_count"], "critical_share": report["drift"]["critical_share"],
                          "prediction_drift": (report["prediction_drift"] or {}).get("score"),
                          "drift_status": report["drift"]["status"], "label_coverage": report["performance"]["label_coverage"],
                          **{key: (report["performance"]["metrics"] or {}).get(key) for key in
                             ("roc_auc", "average_precision", "accuracy", "precision", "recall", "f1")},
                          "status": report["status"], "alerts": ",".join(alert["rule"] for alert in report["alerts"])})
        features.extend({"batch_id": name, **row, "distribution": json.dumps(row["distribution"])} for row in report["drift"]["features"])
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(features).to_csv(output / "features.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=OUTPUT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch", choices=BATCHES)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--labels", type=Path, help="Omit for drift-only; requires the CSV and its .csv.json provenance")
    parser.add_argument("--anchor", help="Explicit versioned UTC-midnight first window, within retention")
    parser.add_argument("--scenario", help="Default: synthetic_demo_<identity digest>")
    parser.add_argument("--export-dir", type=Path)
    args = parser.parse_args()
    from monitoring.service import MonitoringService
    service = MonitoringService.default()
    try:
        result = replay(service, data_dir=args.data_dir, batch=args.batch, labels_path=args.labels,
                        anchor=args.anchor, scenario=args.scenario)
        if args.export_dir:
            export_reports(service, result, args.export_dir)
        print(json.dumps(result, indent=2, allow_nan=False))
    finally:
        service.close()


if __name__ == "__main__":
    main()
