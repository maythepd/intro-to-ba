"""Reproducible study replay. Synthetic timestamps/labels never enter the live scope."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd

from api import module5
from api.scoring import score_probabilities
from monitoring.settings import digest, now_utc, timestamp

SCENARIOS = ("stable", "behavior_drift", "performance_drop")


def simulate(service, *, seed: int = 42) -> dict:
    if not service.store or service.model is None or not service.model_id:
        raise ValueError("Monitoring storage and the prepared Module 5 model are required")
    if not module5.REPLAY_PATH.is_file() or module5.file_hash(module5.REPLAY_PATH) != service.model.manifest.get("monitoring_replay_sha256"):
        raise ValueError("Replay cohort hash mismatch; run prepare_module5")
    frame = pd.read_csv(module5.REPLAY_PATH)
    # The frozen Module 5 test cohort is disjoint from train. Reusing that cohort
    # across stress scenarios is explicit; each scenario has two disjoint windows.
    order = np.random.default_rng(seed).permutation(len(frame))
    frame = frame.iloc[order].reset_index(drop=True)
    size = service.config["min_drift_samples"]
    if len(frame) < 2 * size:
        raise ValueError("Replay needs two non-overlapping cohorts of minimum_drift_samples")
    created = 0
    reports = []
    for name in SCENARIOS:
        scenario = name if seed == 42 else f"{name}_seed{seed}"
        # Fix synthetic dates from the preparation date, so reruns remain idempotent.
        anchor = datetime.fromisoformat(service.store.get_model(service.model_id)["created_at"])
        anchor = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        anchor -= timedelta(days=service.config["outcome_horizon_days"] + 5)
        for index in range(2):
            cohort = frame.iloc[index * size:(index + 1) * size].copy()
            labels = cohort.churn.to_numpy(dtype=int)
            if name == "behavior_drift":
                cohort["recency_days"] += 90
                for feature_name in ("trans_amount_3m", "trans_count_3m", "total_transactions", "total_logins", "app_logins_total"):
                    cohort[feature_name] = cohort[feature_name] * .05
            probabilities = score_probabilities(service.model, cohort)
            if name == "performance_drop":
                # Deliberately adversarial, balanced synthetic outcomes, not real
                # churn labels. AP falls through calculation, never a hardcoded metric.
                labels = np.zeros(len(cohort), dtype=int)
                labels[np.argsort(probabilities)[:len(cohort) // 2]] = 1
            start = anchor + timedelta(hours=service.config["window_hours"] * index)
            end = start + timedelta(hours=service.config["window_hours"])
            namespace = digest(["study_replay_v1", scenario, seed, index, service.model_id])
            rows = service.event_rows(cohort, probabilities, source="study_replay", scenario=scenario,
                namespace=namespace, occurred_at=start + timedelta(hours=1), customer_ids=cohort.customer_id.tolist())
            created += service.store.append_events(rows)
            outcomes = [{"event_id": row["id"], "label": int(label), "horizon_end": row["horizon_end"],
                         "observed_at": row["horizon_end"],
                         "source": "synthetic_stress" if name == "performance_drop" else "study_snapshot"}
                        for row, label in zip(rows, labels)]
            service.store.add_outcomes(outcomes)
            report = service.run(source="study_replay", scenario=scenario, start=timestamp(start), end=timestamp(end))
            reports.append({"id": report["id"], "scenario": scenario, "status": report["status"],
                            "samples": report["sample_count"], "alerts": report["alerts"]})
    return {"source": "study_replay", "created_events": created, "runs": reports,
            "notes": ["Synthetic timestamps; not real operational time series",
                      "Frozen held-out customers reused across scenarios, disjoint within each scenario",
                      "performance_drop replaces study labels with deliberately adversarial synthetic labels"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from monitoring.service import MonitoringService
    service = MonitoringService.default()
    print(json.dumps(simulate(service, seed=args.seed), indent=2))
    service.close()


if __name__ == "__main__":
    main()
