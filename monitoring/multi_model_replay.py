"""Validated five-model replay, atomic model/batch persistence, exports and resume."""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
from datetime import datetime, timedelta
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

from api import module5, segmentation_service
from api.challenger_models import TrustedCheckpointLoader  # Historical offline import compatibility.
from demo_data.generate import BATCHES, OUTPUT
from monitoring import campaign, performance
from monitoring.campaign_store import CampaignStore
from monitoring.settings import Settings, digest, now_utc, timestamp


def classification_report(service, prepared, algorithm, batch):
    spec = prepared.models[algorithm]
    cohort = prepared.frame.loc[prepared.frame.batch_id.eq(batch)]
    mask = prepared.frame.batch_id.eq(batch).to_numpy()
    index = BATCHES.index(batch)
    window = prepared.manifest["batches"][index]
    probabilities = prepared.probabilities[algorithm][mask]
    measured = (performance.measured_labels(prepared.labels.loc[cohort.customer_id, "churn"], probabilities,
                                           spec["reference"], service.config) if prepared.labels is not None else
                {"status": "awaiting_labels", "metrics": None, "label_coverage": 0.0, "labeled_samples": 0,
                 "sample_count": len(cohort), "mature_samples": len(cohort), "positive_samples": 0,
                 "negative_samples": 0, "threshold": spec["reference"]["decision_threshold"],
                 "baseline": spec["reference"]["performance"], "simulated": False, "churn_rate": None,
                 "trigger_candidate": False, "drops": {}})
    baseline = prepared.class_baselines[algorithm]
    measured.update(synthetic_baseline=baseline, baseline_building=index < 3, trigger_candidate=False, drops={})
    if index >= 3 and baseline and measured["status"] == "ready":
        measured["drops"] = {key: baseline["metrics"][key] - measured["metrics"][key]
                             for key in ("roc_auc", "average_precision", "f1", "recall")
                             if baseline["metrics"][key] is not None and measured["metrics"][key] is not None}
        measured["trigger_candidate"] = any(measured["drops"].get(key, 0) >= service.config["performance_drop"]
                                             for key in ("recall", "average_precision"))
    from monitoring import drift
    report = {"id": digest([prepared.manifest["campaign_id"], algorithm, batch]),
              "campaign_id": prepared.manifest["campaign_id"], "algorithm": algorithm, "task": "classification",
              "role": spec["role"], "model_id": spec["model_id"], "batch_id": batch,
              "window_start": window["window_start"], "window_end": window["window_end"],
              "cohort_sha256": window["cohort_sha256"], "sample_count": len(cohort),
              "performance": measured, "drift": prepared.input_reports["classification"][batch],
              "prediction_drift": drift.compare(probabilities, spec["reference"]["prediction"], service.config,
                                                  name="churn_probability") if len(cohort) >= service.config["min_drift_samples"] else None,
              "status": measured["status"], "alerts": []}
    return report


def persist(service, prepared, *, algorithm=None, batch=None, execution_origin="python_cli"):
    if service.store is None:
        raise ValueError("Campaign persistence requires a configured monitoring database")
    store = CampaignStore(service.store)
    identifier = prepared.manifest["campaign_id"]
    scenario = "multi_" + identifier[:48]
    selected = [algorithm] if algorithm else list(campaign.ALGORITHMS)
    if any(name not in campaign.ALGORITHMS for name in selected) or (batch and batch not in BATCHES):
        raise ValueError("Invalid campaign algorithm/batch")
    inserted = {"events": 0, "outcomes": 0, "assignments": 0}
    with service.lock:
        with service.store.transaction():
            service.store.lock_scope(identifier)
            store.register(prepared.manifest)
            for name, spec in prepared.models.items():
                if spec["task"] == "classification":
                    service.store.register_model(spec["model_id"], spec["reference"],
                        {"name": {"xgboost": "XGBoost", "lr": "Logistic Regression", "rf": "Random Forest"}[name] + " (Module 5)",
                         "algorithm": name, "role": spec["role"], "study_reproduced": True}, spec["reference_sha256"])
                else:
                    store.register_cluster(spec)
            for space, reports in prepared.input_reports.items():
                spec = prepared.models["xgboost" if space == "classification" else "kmeans"]
                reference_hash = (digest(spec["reference"]["features"]) if space == "classification"
                                  else spec["input_reference_sha256"])
                for name, report in reports.items():
                    store.save_input(identifier, name, spec["feature_space_id"], reference_hash,
                                     {"space": space, "reference_sha256": reference_hash, **report})
        for name in selected:
            spec = prepared.models[name]
            previous = {item["batch_id"]: item for item in store.reports(identifier) if item["algorithm"] == name}
            # Recompute already persisted downstream windows if an earlier unit is refreshed.
            chosen = list(BATCHES) if batch is None else [b for b in BATCHES if b == batch or
                (BATCHES.index(b) > BATCHES.index(batch) and b in previous)]
            for name_batch in chosen:
                index = BATCHES.index(name_batch)
                window = prepared.manifest["batches"][index]
                mask = prepared.frame.batch_id.eq(name_batch).to_numpy()
                cohort = prepared.frame.loc[mask]
                with service.store.transaction():
                    service.store.lock_scope(identifier + name)
                    if spec["task"] == "classification":
                        adapter_service = copy(service)
                        adapter_service.model_id, adapter_service.reference = spec["model_id"], spec["reference"]
                        events = adapter_service.event_rows(cohort[service.model.input_columns], prepared.probabilities[name][mask],
                            source="study_replay", scenario=scenario, namespace=digest([identifier, name, name_batch]),
                            occurred_at=datetime.fromisoformat(window["window_start"]) + timedelta(hours=1),
                            customer_ids=cohort.customer_id.tolist())
                        outcomes = [] if prepared.labels is None else [{"event_id": row["id"], "label": int(prepared.labels.loc[person, "churn"]),
                            "horizon_end": row["horizon_end"], "observed_at": row["horizon_end"], "source": "synthetic_stress"}
                            for row, person in zip(events, cohort.customer_id)]
                        event_count = service.store.append_events(events)
                        outcome_count = service.store.add_outcomes(outcomes) if outcomes else 0
                        metadata = {"campaign_id": identifier, "batch_id": name_batch, "batch_index": index + 1,
                            "batch_count": 5, "dataset_sha256": prepared.manifest["dataset_sha256"],
                            "generation_seed": prepared.manifest["generation_seed"], "feature_snapshot": prepared.manifest["feature_snapshot"],
                            "anchor": prepared.manifest["anchor"], "label_policy_version": prepared.manifest["label_policy_version"],
                            "label_seed": prepared.manifest["label_seed"], "time_kind": "simulated",
                            "outcome_kind": "simulated" if outcomes else "unavailable", "execution_origin": execution_origin,
                            "synthetic_baseline": prepared.class_baselines[name] if index >= 3 else None}
                        report = adapter_service.run(source="study_replay", scenario=scenario,
                            start=window["window_start"], end=window["window_end"], replay_metadata=metadata,
                            input_drift=prepared.input_reports["classification"][name_batch])
                        report.update(campaign_id=identifier, algorithm=name, role=spec["role"], task="classification",
                                      batch_id=name_batch, cohort_sha256=window["cohort_sha256"])
                        measured = classification_report(service, prepared, name, name_batch)["performance"]
                        report["performance"].update({key: measured[key] for key in ("baseline_building", "drops", "synthetic_baseline")})
                        # Preserve AP/recall alert convention; AUC/F1 drops remain diagnostics.
                        service.store.save_run(report)
                        assignment_count = 0
                    else:
                        report = deepcopy(prepared.cluster_reports[name][name_batch])
                        report.update(campaign_id=identifier, algorithm=name, task="clustering", model_id=spec["model_id"],
                            id=digest([identifier, name, name_batch]), batch_id=name_batch, sample_count=len(cohort),
                            assignment_method=spec["assignment_method"], reference_sha256=spec["reference_sha256"],
                            cohort_sha256=window["cohort_sha256"], execution_origin=execution_origin,
                            window_start=window["window_start"], window_end=window["window_end"], computed_at=timestamp(now_utc()))
                        earlier = store.reports(identifier, task="clustering")
                        prior = next((item["report"] for item in earlier if item["algorithm"] == name and
                                      index > 0 and item["batch_id"] == BATCHES[index - 1]), None)
                        candidate = report["trigger_candidate"] or (not report["baseline_building"] and
                            report["assignment_drift"]["status"] == "critical")
                        report["review_candidate"] = candidate
                        report["consecutive_candidates"] = (1 + prior.get("consecutive_candidates", 0)
                            if candidate and prior and prior.get("review_candidate") else int(candidate))
                        report["review_status"] = "review_required" if report["consecutive_candidates"] >= service.config["consecutive_windows"] else "warning" if candidate else report["status"]
                        report["action"] = "Synthetic demonstration only; inspect segmentation quality and collect independent evidence before refitting"
                        rows = [{"campaign_id": identifier, "model_id": spec["model_id"], "batch_id": name_batch,
                                 "subject": digest(["customer", str(person)]), "cluster_id": int(label), "distance": float(distance), "margin": float(margin)}
                                for person, label, distance, margin in zip(cohort.customer_id,
                                    prepared.assignments[name][mask], prepared.distances[name][mask], prepared.margins[name][mask])]
                        assignment_count = store.save_assignments(rows)
                        event_count = outcome_count = 0
                    store.save_report(identifier, name, name_batch, spec["task"], spec["model_id"], window["cohort_sha256"], report)
                inserted["events"] += event_count
                inserted["outcomes"] += outcome_count
                inserted["assignments"] += assignment_count
    return {"campaign_id": identifier, "inserted": inserted, "status": store.get(identifier)["status"],
            "report_count": store.get(identifier)["report_count"]}


def export(prepared, output, reports=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dry_run = reports is None
    if reports is None:
        # Dry-run reports use already validated probabilities/assignments and never open a database.
        from types import SimpleNamespace
        service = SimpleNamespace(config=prepared.manifest["config"])
        reports = []
        for name in campaign.ALGORITHMS:
            for batch in BATCHES:
                report = (classification_report(service, prepared, name, batch) if name in campaign.ALGORITHMS[:3]
                          else {**prepared.cluster_reports[name][batch], "algorithm": name, "batch_id": batch,
                                "model_id": prepared.models[name]["model_id"], "assignment_method": prepared.models[name]["assignment_method"]})
                reports.append({"algorithm": name, "batch_id": batch, "task": prepared.models[name]["task"], "report": report})
    classifier, clusters, profile_rows, distributions = [], [], [], []
    for unit in reports:
        report, name, batch = unit["report"], unit["algorithm"], unit["batch_id"]
        folder = output / "reports" / name
        folder.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(report, indent=2, allow_nan=False)
        (folder / (batch + ".json")).write_text(encoded, encoding="utf-8")
        (folder / (batch + ".html")).write_text("<!doctype html><meta charset='utf-8'><title>BaBank synthetic campaign</title>"
            "<style>body{font:14px system-ui;margin:2rem}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
            "<h1>" + escape(name + " / " + batch) + "</h1><p>Simulated study replay; not real-world accuracy.</p><pre>" + escape(encoded) + "</pre>", encoding="utf-8")
        common = {"campaign_id": prepared.manifest["campaign_id"], "algorithm": name, "batch": batch,
                  "model_id": report["model_id"], "N": report.get("sample_count", report.get("N")), "status": report["status"]}
        if unit["task"] == "classification":
            measured = report["performance"]
            matrix = (measured.get("metrics") or {}).get("confusion_matrix")
            classifier.append({**common, "status": measured["status"], "threshold": measured["threshold"],
                "mature_N": measured["mature_samples"], "labeled_N": measured["labeled_samples"], "positive_N": measured["positive_samples"],
                "prevalence": measured["churn_rate"], "coverage": measured["label_coverage"],
                "probability_psi": (report.get("prediction_drift") or {}).get("score"),
                **{key: value for key, value in (measured.get("metrics") or {}).items() if key != "confusion_matrix"},
                **(dict(zip(("TN", "FP", "FN", "TP"), np.asarray(matrix).ravel().tolist())) if matrix else {}),
                **{"baseline_" + key: value for key, value in ((measured.get("synthetic_baseline") or {}).get("metrics") or {}).items() if key != "confusion_matrix"},
                **{"drop_" + key: value for key, value in measured.get("drops", {}).items()}})
        else:
            clusters.append({**common, **{key: report.get(key) for key in ("assignment_method", "eval_N", "k", "active_clusters",
                "silhouette", "dbi", "chi", "mean_distance", "p95_distance", "distortion", "outlier_share", "margin")},
                "cluster_js": report["assignment_drift"]["score"], **{"baseline_" + k: v for k, v in report["baseline"].items()},
                **{"drop_" + k: v for k, v in report["drops"].items()}})
            distributions.extend({**common, "cluster_id": i, "count": count, "share": report["cluster_shares"][i]}
                                 for i, count in enumerate(report["counts"]))
            profile_rows.extend({**common, "cluster_id": item["cluster_id"], "cluster_N": item["sample_count"], **feature}
                                for item in report["profiles"] for feature in item["features"])
    input_rows = [{"batch": batch, "space": space, **feature} for space, rows in prepared.input_reports.items()
                  for batch, report in rows.items() for feature in report["features"]]
    validation = [{"algorithm": name, "checkpoint_sha256": spec.get("checkpoint_sha256"),
                   "study_reproduced": spec.get("study_reproduced"), "model_id": spec["model_id"]}
                  for name, spec in prepared.models.items()]
    robustness = []
    for task, rows, keys in (("classification", classifier, ("roc_auc", "average_precision", "f1", "recall")),
                             ("clustering", clusters, ("silhouette", "dbi", "chi", "p95_distance"))):
        for name in {row["algorithm"] for row in rows}:
            group = [row for row in rows if row["algorithm"] == name]
            stress = [row for row in group if row["batch"] in BATCHES[3:]]
            summary = {"algorithm": name, "task": task, "reports": len(group), "complete": len(group) == 5}
            for key in keys:
                values = [row[key] for row in stress if row.get(key) is not None]
                all_values = [row[key] for row in group if row.get(key) is not None]
                summary.update({key + "_stress_mean": float(np.mean(values)) if values else None,
                                key + "_worst": (max(values) if key in ("dbi", "p95_distance") else min(values)) if values else None,
                                key + "_five_batch_std": float(np.std(all_values)) if all_values else None})
            robustness.append(summary)
    for filename, rows in (("classifier_metrics", classifier), ("clustering_metrics", clusters), ("input_drift", input_rows),
                           ("cluster_distributions", distributions), ("cluster_profiles", profile_rows),
                           ("artifact_validation", validation), ("robustness_summary", robustness)):
        pd.DataFrame(rows).to_csv(output / (filename + ".csv"), index=False)
    (output / "campaign_manifest.json").write_text(json.dumps(prepared.manifest, indent=2, allow_nan=False), encoding="utf-8")
    result = {"campaign_id": prepared.manifest["campaign_id"], "classifier_results": len(classifier), "clustering_results": len(clusters),
              "status": "complete" if len(classifier) == 15 and len(clusters) == 10 else "partial", "blockers": {},
              "mode": prepared.manifest["mode"], "dry_run": dry_run}
    (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=OUTPUT)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--all", action="store_true", help="Default scope is all five models and all five batches")
    parser.add_argument("--batch", choices=BATCHES)
    parser.add_argument("--algorithm", choices=campaign.ALGORITHMS)
    parser.add_argument("--anchor")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--campaign-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume != bool(args.campaign_id) or args.dry_run and args.resume:
        parser.error("Resume requires campaign-id and persistent mode")
    from monitoring.service import MonitoringService
    from api.main import SEGMENTATION_DATA_PATH
    settings = Settings(None) if args.dry_run else Settings.from_env()
    service = MonitoringService(module5.load(), segmentation=segmentation_service.load(SEGMENTATION_DATA_PATH), settings=settings)
    try:
        anchor, labels = args.anchor, args.labels
        if args.resume:
            if service.store is None:
                parser.error("Resume requires a configured monitoring database")
            old = CampaignStore(service.store).get(args.campaign_id)
            if old is None:
                parser.error("Campaign not found")
            anchor = old["manifest"]["anchor"]
            labels = args.data_dir / "synthetic_outcomes.csv" if old["manifest"]["mode"] == "simulated_labels" else None
        prepared = campaign.prepare(service, data_dir=args.data_dir, labels_path=labels, anchor=anchor)
        if args.campaign_id and prepared.manifest["campaign_id"] != args.campaign_id:
            raise ValueError("Current artifacts/config differ from stored campaign; cannot resume")
        result = {"campaign_id": prepared.manifest["campaign_id"], "dry_run": True}
        reports = None
        if not args.dry_run:
            result = persist(service, prepared, algorithm=args.algorithm, batch=args.batch)
            reports = CampaignStore(service.store).reports(prepared.manifest["campaign_id"])
        if args.export_dir:
            result.update(export(prepared, args.export_dir, reports))
        print(json.dumps(result, indent=2, allow_nan=False))
    finally:
        service.close()


if __name__ == "__main__":
    main()
