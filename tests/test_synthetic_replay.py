"""Synthetic outcomes, transaction retries, scope isolation and real-model integration."""
from datetime import timedelta
import os
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sqlalchemy import func, select

from api import module5, segmentation_service
from api.main import SEGMENTATION_DATA_PATH
from api.monitoring_routes import router
from demo_data.generate import BATCHES
from demo_data.labels import generate_labels, read_labels, simulate_labels
from demo_data.validate import read_outputs
from monitoring.service import MonitoringService
from monitoring.settings import Settings, load_config, now_utc, timestamp
from monitoring.store import Store, events, outcomes, runs
from monitoring.synthetic_replay import replay
from test_demo_data import demo, source  # Reuse the audited two-customer relational fixture.


@pytest.fixture
def service(tmp_path):
    settings = Settings("sqlite:///" + (tmp_path / "replay.db").as_posix(), admin_token="test-secret", demo_enabled=True,
                        config={**load_config(), "min_drift_samples": 2, "min_labeled_samples": 2, "min_class_samples": 1})
    service = MonitoringService(module5.load(), segmentation=segmentation_service.load(SEGMENTATION_DATA_PATH), settings=settings)
    assert service.status()["storage_ready"], service.status()
    yield service
    service.close()


def counts(service):
    with service.store.engine.connect() as connection:
        return tuple(connection.execute(select(func.count()).select_from(table)).scalar_one()
                     for table in (events, outcomes, runs))


def label_file(out):
    path = out / "synthetic_outcomes.csv"
    generate_labels(out / "customer_features.csv", path)
    return path


def test_labels_independent_of_predictions_donors_and_order(demo):
    out, _ = demo
    _, frame, _ = read_outputs(out)
    expected = simulate_labels(frame).set_index("customer_id").sort_index()
    poisoned = frame.assign(churn=1, predicted_churn=0, churn_probability=.999, source_customer_id=7)
    actual = simulate_labels(poisoned.sample(frac=1, random_state=7)).set_index("customer_id").sort_index()
    pd.testing.assert_frame_equal(actual, expected)
    subset = simulate_labels(frame.iloc[:3]).set_index("customer_id")
    pd.testing.assert_frame_equal(subset, expected.loc[subset.index])
    assert not simulate_labels(frame, seed=5).equals(simulate_labels(frame, seed=6))
    with pytest.raises(ValueError, match="finite"):
        simulate_labels(frame.assign(recency_days=np.inf))


def test_label_checksums_and_one_to_one_policy_are_enforced(demo):
    out, manifest = demo
    _, frame, _ = read_outputs(out)
    path = label_file(out)
    found, metadata = read_labels(path, frame, manifest["output_sha256"]["customer_features.csv"])
    assert len(found) == len(frame) and metadata["label_policy_version"] == "synthetic_outcome_v1"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        read_labels(path, frame, manifest["output_sha256"]["customer_features.csv"])


def test_drift_only_five_batches_dedup_and_persistence(demo, service):
    out, _ = demo
    result = replay(service, data_dir=out)
    assert result["created_events"] == 20 and result["created_outcomes"] == 0
    assert [item["synthetic_replay"]["batch_id"] for item in result["runs"]] == list(BATCHES)
    assert counts(service) == (20, 0, 5)
    again = replay(service, data_dir=out)
    assert again["scenario"] == result["scenario"] and again["created_events"] == 0
    assert counts(service) == (20, 0, 5)
    for item in result["runs"]:
        report = service.store.get_run(item["id"])
        assert report["sample_count"] == 4
        assert report["performance"]["metrics"] is None and report["performance"]["status"] == "awaiting_labels"
        assert report["segments"]["coverage"] == 1
        assert report["segments"]["model_id"] == service.segment_model_id
        assert report["latency"]["p95_ms"] is None
        refreshed = service.run(source="study_replay", scenario=result["scenario"],
                                start=report["window_start"], end=report["window_end"])
        assert refreshed["synthetic_replay"] == report["synthetic_replay"]
    assert service.store.list_runs(service.model_id, "live", "") == []
    with service.store.engine.connect() as connection:
        assert set(connection.execute(select(events.c.source)).scalars()) == {"study_replay"}
    reopened = Store(service.settings.database_url)
    try:
        assert len(reopened.list_runs(service.model_id, "study_replay", result["scenario"])) == 5
    finally:
        reopened.close()


def test_simulated_metrics_match_sklearn_and_rerun_is_immutable(demo, service):
    out, _ = demo
    path = label_file(out)
    result = replay(service, data_dir=out, labels_path=path)
    assert result["created_outcomes"] == 20 and counts(service) == (20, 20, 5)
    for item in result["runs"]:
        report = service.store.get_run(item["id"])
        cohort = service.store.window_events(service.model_id, "study_replay", result["scenario"],
                                              report["window_start"], report["window_end"], 100)
        y = [row["label"] for row in cohort]
        p = [row["probability"] for row in cohort]
        predicted = np.asarray(p) >= service.reference["decision_threshold"]
        metrics = report["performance"]["metrics"]
        assert metrics["accuracy"] == accuracy_score(y, predicted)
        assert metrics["f1"] == f1_score(y, predicted, zero_division=0)
        assert metrics["confusion_matrix"] == confusion_matrix(y, predicted, labels=[0, 1]).tolist()
        if len(set(y)) == 2:
            assert metrics["roc_auc"] == roc_auc_score(y, p)
            assert metrics["average_precision"] == average_precision_score(y, p)
        assert report["performance"]["label_sources"] == ["synthetic_stress"]
        assert report["performance"]["simulated"] and report["performance"]["label_coverage"] == 1
        if report["synthetic_replay"]["batch_index"] <= 3:
            assert not report["performance"]["trigger_candidate"]  # No comparison to different study labels.
    repeat = replay(service, data_dir=out, labels_path=path)
    assert repeat["created_events"] == repeat["created_outcomes"] == 0
    assert counts(service) == (20, 20, 5)
    drift = replay(service, data_dir=out)
    assert drift["scenario"] != result["scenario"] and counts(service) == (40, 20, 10)


def test_report_failure_rolls_back_events_and_outcomes_then_retries(demo, service):
    out, _ = demo
    path = label_file(out)
    with patch.object(service.store, "save_run", side_effect=RuntimeError("interrupted")):
        with pytest.raises(RuntimeError, match="interrupted"):
            replay(service, data_dir=out, labels_path=path, batch="normal_01")
    assert counts(service) == (0, 0, 0)
    assert replay(service, data_dir=out, labels_path=path, batch="normal_01")["created_events"] == 4


def test_invalid_score_or_label_or_time_never_writes_events(demo, service):
    out, _ = demo
    with patch("monitoring.synthetic_replay.score_batch", return_value=(np.full(20, np.nan), {})):
        with pytest.raises(ValueError, match="probabilities"):
            replay(service, data_dir=out)
    with pytest.raises(ValueError, match="retention"):
        replay(service, data_dir=out, anchor="2000-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="timezone"):
        replay(service, data_dir=out, anchor="2026-01-01")
    recent = now_utc().replace(hour=0, minute=0, second=0) - timedelta(days=8)
    with pytest.raises(ValueError, match="mature"):
        replay(service, data_dir=out, labels_path=label_file(out), anchor=timestamp(recent))
    assert counts(service) == (0, 0, 0)


def test_single_batch_anchor_is_stable_and_custom_scenario_rejects_mixing(demo, service):
    out, _ = demo
    first = replay(service, data_dir=out, batch="friction_shift", scenario="synthetic_demo_test")
    all_batches = replay(service, data_dir=out, scenario="synthetic_demo_test")
    assert first["runs"][0]["id"] == all_batches["runs"][-1]["id"]
    with pytest.raises(ValueError, match="another replay identity"):
        replay(service, data_dir=out, labels_path=label_file(out), scenario="synthetic_demo_test")
    assert counts(service) == (20, 0, 5)


def test_admin_route_guards_and_read_metadata(demo, service, monkeypatch):
    out, _ = demo
    app = FastAPI()
    app.include_router(router)
    app.state.monitoring, app.state.churn_model, app.state.inference_count = service, service.model, 0
    monkeypatch.setenv("DEMO_DATA_DIR", str(out))
    headers = {"Authorization": "Bearer test-secret"}
    with TestClient(app) as client:
        assert client.post("/api/monitoring/synthetic-replay", json={"all": True}).status_code == 401
        assert client.post("/api/monitoring/synthetic-replay", headers={"Authorization": "Bearer wrong"}, json={"all": True}).status_code == 401
        assert client.post("/api/monitoring/synthetic-replay", headers=headers, json={}).status_code == 422
        service.settings.demo_enabled = False
        assert client.post("/api/monitoring/synthetic-replay", headers=headers, json={"all": True}).status_code == 403
        service.settings.demo_enabled = True
        response = client.post("/api/monitoring/synthetic-replay", headers=headers, json={"all": True})
        assert response.status_code == 200, response.text
        scenario = response.json()["scenario"]
        result = client.get(f"/api/monitoring?source=study_replay&scenario={scenario}").json()
        assert len(result["history"]) == 5 and scenario in result["scenarios"]
        assert result["history"][0]["synthetic_replay"]["batch_id"] == "normal_01"
        report_id = result["latest"]["id"]
        assert client.get(f"/api/monitoring/runs/{report_id}/report?format=html").status_code == 200
        stored = service.store.window_events(service.model_id, "study_replay", scenario,
                                             result["latest"]["window_start"], result["latest"]["window_end"], 100)[0]
        rejected = client.post("/api/monitoring/labels", headers=headers, json={"items": [{
            "event_id": stored["id"], "label": 1, "horizon_end": stored["horizon_end"],
            "observed_at": stored["horizon_end"], "source": "observed"}]})
        assert rejected.status_code == 422


@pytest.mark.skipif(not os.getenv("MONITORING_TEST_DATABASE_URL"), reason="Dedicated disposable PostgreSQL URL is not set")
def test_postgres_batch_transaction_and_retry(demo, tmp_path):
    settings = Settings(os.environ["MONITORING_TEST_DATABASE_URL"], config=load_config())
    # The test service uses the normal model but a unique scope; never use production DATABASE_URL here.
    migration = Store(settings.database_url, migrate=True)
    migration.close()
    service = MonitoringService(module5.load(), segmentation=segmentation_service.load(SEGMENTATION_DATA_PATH), settings=settings)
    try:
        import uuid
        scenario = "synthetic_demo_ci_" + uuid.uuid4().hex[:12]
        out, _ = demo
        first = replay(service, data_dir=out, scenario=scenario)
        repeat = replay(service, data_dir=out, scenario=scenario)
        assert first["created_events"] == 20 and repeat["created_events"] == 0
        assert len(service.store.list_runs(service.model_id, "study_replay", scenario)) == 5
    finally:
        service.close()
