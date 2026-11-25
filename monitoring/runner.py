"""Compute one closed, versioned window; adjacent windows establish alert streaks."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd

from monitoring import drift, performance
from monitoring.settings import SOURCES, digest, now_utc, timestamp


def closed_window(at: datetime, hours: int) -> tuple[str, str]:
    seconds = hours * 3600
    end = datetime.fromtimestamp((int(at.timestamp()) // seconds) * seconds, tz=at.tzinfo)
    return timestamp(end - timedelta(hours=hours)), timestamp(end)


def summarize(report: dict) -> dict:
    return {key: report[key] for key in ("id", "model_id", "source", "scenario", "window_start", "window_end",
            "computed_at", "status", "sample_count", "drift", "performance", "quality", "alerts", "synthetic_replay")
            if key not in ("drift", "performance") and key in report} | {
        "drift": {key: value for key, value in report["drift"].items() if key != "features"},
        "performance": report["performance"]}


def run_window(store, model_id: str, cfg: dict, *, source: str, scenario: str = "",
               start: str, end: str, as_of: str | None = None, replay_metadata: dict | None = None,
               input_drift: dict | None = None) -> dict:
    if source not in SOURCES or (source == "study_replay" and not scenario) or (source != "study_replay" and scenario):
        raise ValueError("Use an explicit scenario only for study_replay")
    start, end, as_of = timestamp(start), timestamp(end), timestamp(as_of or now_utc())
    if datetime.fromisoformat(end) - datetime.fromisoformat(start) != timedelta(hours=cfg["window_hours"]):
        raise ValueError("Monitoring windows must use the configured fixed duration")
    if end > as_of:
        raise ValueError("Only closed windows can be evaluated")
    model = store.get_model(model_id)
    if model is None:
        raise ValueError("Unknown model version")
    reference = model["reference"]
    config_hash = digest(cfg)
    identifier = digest([model_id, source, scenario, start, end, config_hash])
    existing = store.get_run(identifier)
    replay_metadata = replay_metadata or (existing or {}).get("synthetic_replay")
    window = store.window_events(model_id, source, scenario, start, end, cfg["max_window_events"])
    # Request-level quality includes rejections; feature/performance cohorts take the
    # last successful prediction per subject, not repeated lookups of one customer.
    latest = {event["subject"]: event for event in window if event["valid"]}
    cohort = list(latest.values())
    frame = pd.DataFrame([event["features"] for event in cohort])
    if input_drift is not None and input_drift["sample_count"] != len(cohort):
        raise ValueError("Shared input drift cohort mismatch")
    data_drift = input_drift if input_drift is not None else drift.evaluate(frame, reference, cfg)
    prediction_drift = (drift.compare([event["probability"] for event in cohort], reference["prediction"], cfg,
                                      name="churn_probability")
                        if len(cohort) >= cfg["min_drift_samples"] else None)
    quality = {"request_count": len(window), "valid_requests": sum(event["valid"] for event in window),
               "invalid_requests": sum(not event["valid"] for event in window),
               "error_rate": sum(not event["valid"] for event in window) / len(window) if window else 0.0,
               "max_missing_excess": max((row["missing_excess"] for row in data_drift["features"]), default=0.0),
               "eligible": len(window) >= cfg["min_quality_samples"]}
    quality["trigger_candidate"] = quality["eligible"] and max(quality["error_rate"], quality["max_missing_excess"]) > cfg["quality_error_rate"]
    measured = performance.evaluate(cohort, reference, cfg, as_of=as_of)
    if replay_metadata:
        # Compare synthetic batches with earlier normal batches using the same outcome policy.
        # A study-label AP/recall difference is not a measured real-world degradation.
        baseline = replay_metadata.get("synthetic_baseline")
        measured["synthetic_baseline"] = baseline
        measured["drops"] = {}
        measured["trigger_candidate"] = False
        measured["baseline_note"] = "Study baseline has different labels; alerts use eligible normal synthetic batches only"
        if baseline and measured["status"] == "ready":
            measured["drops"] = {name: baseline["metrics"][name] - measured["metrics"][name]
                                  for name in ("recall", "average_precision")}
            measured["trigger_candidate"] = any(value >= cfg["performance_drop"] for value in measured["drops"].values())
    durations = [event["latency_ms"] for event in window]
    report = {"id": identifier, "model_id": model_id,
              "source": source, "scenario": scenario, "window_start": start, "window_end": end,
              "config_sha256": config_hash, "computed_at": as_of, "sample_count": len(cohort),
              "policy": cfg,
              "raw_successful_requests": quality["valid_requests"], "reference_sha256": model["reference_sha256"],
              "deduplication": "Last successful prediction per subject in this window",
              "data_kind": "Simulated study replay" if source == "study_replay" else "Study snapshot batch" if source == "snapshot_batch" else "Incoming study-model API requests; not live bank operations",
              "drift": data_drift, "prediction_drift": prediction_drift, "performance": measured,
              "quality": quality, "latency": {"p50_ms": float(np.percentile(durations, 50)) if durations else None,
                                             "p95_ms": float(np.percentile(durations, 95)) if durations else None},
              "segments": {"status": "unavailable", "coverage": 0.0}, "alerts": []}
    if replay_metadata:
        report["synthetic_replay"] = replay_metadata
        report["data_kind"] = "Synthetic demo customers, simulated UTC windows and optional independent simulated outcomes"
        report["quality"]["note"] = "Validated successful replay rows only; excludes real request rejection/error rates"
        report["latency"] = {"p50_ms": None, "p95_ms": None,
                             "note": "Offline replay; not a measurement of deployed serving latency"}
    segment_reference = reference.get("segments")
    if segment_reference:
        eligible = [event for event in cohort if event["segment"] is not None
                    and event["segment_model_id"] == segment_reference["model_id"]]
        report["segments"] = {"status": "insufficient_data", "coverage": len(eligible) / len(cohort) if cohort else 0.0,
                              "model_id": segment_reference["model_id"], "sample_count": len(eligible),
                              "note": ("Fitted KMeans prediction for new synthetic customers; fixed bundle, no refit" if replay_metadata
                                       else "Fixed KMeans memberships from the separate segmentation snapshot; unknown customers excluded")}
        if len(eligible) >= cfg["min_drift_samples"]:
            report["segments"].update(drift.compare([event["segment"] for event in eligible], segment_reference["reference"],
                                                    cfg, name="KMeans cluster distribution"))
    streak = {"drift": int(data_drift["trigger_candidate"]), "performance": int(measured["trigger_candidate"])}
    cursor = start
    for _ in range(cfg["consecutive_windows"] - 1):
        previous = store.previous_run(model_id, source, scenario, cursor, config_hash)
        if previous is None or previous["reference_sha256"] != report["reference_sha256"]:
            break
        for name in streak:
            if streak[name] and previous[name]["trigger_candidate"]:
                streak[name] += 1
            else:
                streak[name] = 0
        cursor = previous["window_start"]
    report["consecutive_candidates"] = streak
    candidates = []
    if streak["drift"] >= cfg["consecutive_windows"]:
        candidates.append(("data_drift", "review_required", "Inspect shifted inputs and data quality, collect labels, then assess retraining.",
                           {"critical_share": data_drift["critical_share"], "consecutive_windows": streak["drift"]}))
    if streak["performance"] >= cfg["consecutive_windows"]:
        candidates.append(("performance_drop", "retraining_required", "Review labels, train a candidate, evaluate independently and approve a version before deployment.",
                           {"drops": measured["drops"], "labeled_samples": measured["labeled_samples"], "consecutive_windows": streak["performance"]}))
    if quality["trigger_candidate"]:
        candidates.append(("data_quality", "data_quality_alert", "Inspect rejected inputs and unexpected missingness before retraining.", quality))
    for rule, status, action, evidence in candidates:
        if replay_metadata:
            action = "Simulation demonstration only. " + action
        report["alerts"].append({"id": digest([report["id"], rule]), "rule": rule, "status": status,
                                 "action": action, "evidence": evidence})
    report["status"] = ("retraining_required" if any(item[1] == "retraining_required" for item in candidates)
                        else "review_required" if candidates else "warning" if measured["trigger_candidate"] else data_drift["status"])
    store.save_run(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, default="live")
    parser.add_argument("--scenario", default="")
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--prune", action="store_true", help="Explicitly remove raw events older than retention; keep reports")
    args = parser.parse_args()
    from monitoring.service import MonitoringService
    service = MonitoringService.default()
    if not service.store or not service.model_id:
        raise SystemExit("Monitoring storage/reference is unavailable; configure DATABASE_URL and prepare Module 5")
    if not 1 <= args.days <= service.config["max_job_windows"]:
        parser.error("days exceeds the configured job limit")
    start, _ = closed_window(now_utc(), service.config["window_hours"])
    first = datetime.fromisoformat(start) - timedelta(hours=service.config["window_hours"] * (args.days - 1))
    for index in range(args.days):
        left = first + timedelta(hours=service.config["window_hours"] * index)
        report = run_window(service.store, service.model_id, service.config, source=args.source, scenario=args.scenario,
                            start=timestamp(left), end=timestamp(left + timedelta(hours=service.config["window_hours"])))
        print(json.dumps({"id": report["id"], "status": report["status"], "sample_count": report["sample_count"]}))
    if args.prune:
        removed = service.store.prune(timestamp(now_utc() - timedelta(days=service.config["retention_days"])))
        print(json.dumps({"pruned_raw_events": removed}))
    service.close()


if __name__ == "__main__":
    main()
