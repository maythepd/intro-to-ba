"""Checkpoint compatibility, fair campaign cohorts, fixed clusters, transactions and public API."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
import os
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sqlalchemy import func, select, text

from api import module5, segmentation_service as seg
from api.challenger_models import APPROVED, ClassifierAdapter, load, checkpoint
from api.campaign_routes import router as campaign_router
from api.monitoring_routes import router as old_router
from demo_data.labels import generate_labels
from demo_data.validate import read_outputs, segmentation_frame
from monitoring import campaign, clustering_metrics as cm
from monitoring.campaign_store import CampaignStore, campaigns, assignments, units, inputs
from monitoring.multi_model_replay import persist, export, main
from monitoring.settings import digest
from monitoring.store import Store, events, outcomes
from test_demo_data import source, demo
from test_synthetic_replay import service


@pytest.fixture
def prepared(demo, service):
    out, _ = demo
    labels = out / "synthetic_outcomes.csv"
    generate_labels(out / "customer_features.csv", labels)
    return campaign.prepare(service, data_dir=out, labels_path=labels)


def counts(service):
    with service.store.engine.connect() as c:
        return tuple(c.execute(select(func.count()).select_from(table)).scalar_one()
                     for table in (events, outcomes, assignments, units, inputs))


def test_approved_challengers_reproduce_study_and_do_not_import_research_dependencies(service):
    import sys
    for name in ("lr", "rf"):
        adapter, manifest = load(name, service.model)
        assert manifest["study_reproduced"] and len(manifest["features"]) == 133
        assert all(abs(item["difference"]) <= 5e-5 for item in manifest["comparisons"].values())
        assert manifest["checkpoint_sha256"] == APPROVED[name]
        assert manifest["reference"]["checkpoint_sha256"] == APPROVED[name]
    assert "model_development.models.tuner" not in sys.modules
    with patch.dict(APPROVED, {"lr": "0" * 64}):
        with pytest.raises(ValueError, match="Unapproved"):
            checkpoint("lr")


def test_adapter_rejects_wrong_feature_order_or_class_mapping(service):
    class Estimator:
        classes_ = np.array([1, 0])
        def predict_proba(self, values):
            return np.tile([.9, .1], (len(values), 1))
    adapter = ClassifierAdapter(service.model, Estimator(), "lr")
    values = pd.DataFrame(np.zeros((1, 133)), columns=service.model.manifest["features"])
    with pytest.raises(ValueError, match="class mapping"):
        adapter.predict_encoded(values)
    with pytest.raises(ValueError, match="feature order"):
        adapter.predict_encoded(values.iloc[:, ::-1])


def cluster_fixture():
    z = np.array([[0., 0], [0, .2], [.2, 0], [5, 5], [5.2, 5], [5, 5.2]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    centers = np.array([[0., 0], [5, 5]])
    return z, labels, centers, {"shares": [.5, .5], "p95": [.1, .1]}


def test_clustering_metrics_match_sklearn_and_frozen_distances(service):
    z, labels, centers, ref = cluster_fixture()
    result = cm.evaluate(z, labels, centers, ref, service.config)
    assert result["silhouette"] == silhouette_score(z, labels)
    assert result["dbi"] == davies_bouldin_score(z, labels)
    assert result["chi"] == calinski_harabasz_score(z, labels)
    d = np.linalg.norm(z - centers[labels], axis=1)
    assert result["mean_distance"] == np.mean(d)
    assert result["distortion"] == np.mean(d ** 2)
    assert result["outlier_share"] == np.mean(d > .1)
    assert result["assignment_drift"]["score"] == jensenshannon([.5, .5], [.5, .5], base=2)


def test_clustering_degenerate_missing_threshold_and_invalid_inputs(service):
    z, labels, centers, ref = cluster_fixture()
    one = cm.evaluate(z, np.zeros(6, dtype=int), centers, ref, service.config)
    assert one["status"] == "insufficient_clusters" and one["silhouette"] is None
    missing = cm.evaluate(z, labels, centers, {**ref, "p95": [None, .1]}, service.config)
    assert missing["outlier_share"] is None
    with pytest.raises(ValueError, match="Invalid clustering"):
        cm.evaluate(z, labels.astype(float), centers, ref, service.config)
    with pytest.raises(ValueError, match="Invalid clustering"):
        cm.evaluate(z * np.nan, labels, centers, ref, service.config)


def test_campaign_joins_ids_and_preserves_ward_surrogate_without_refit(prepared, service):
    actual = seg.predict_clusters(service.segmentation, segmentation_frame(prepared.frame), "hierarchical")
    assert np.array_equal(actual, prepared.assignments["hierarchical"])
    assert prepared.models["xgboost"]["model_id"] == service.model_id
    assert len({model["model_id"] for model in prepared.models.values()}) == 5
    assert len({model["feature_space_id"] for name, model in prepared.models.items() if name in campaign.ALGORITHMS[:3]}) == 1
    assert prepared.models["kmeans"]["feature_space_id"] != prepared.models["xgboost"]["feature_space_id"]
    assert prepared.manifest["batches"][0]["cohort_sha256"] == digest(sorted(
        prepared.frame.loc[prepared.frame.batch_id.eq("normal_01"), "customer_id"].astype(str).tolist()))


def test_labels_change_identity_but_never_predictions_assignments_or_input_drift(demo, prepared, service):
    out, _ = demo
    labels = out / "other_outcomes.csv"
    generate_labels(out / "customer_features.csv", labels, seed=5)
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    with patch.object(KMeans, "fit", side_effect=AssertionError("No refit")), \
            patch.object(RandomForestClassifier, "fit", side_effect=AssertionError("No retrain")), \
            patch.object(LogisticRegression, "fit", side_effect=AssertionError("No retrain")):
        changed = campaign.prepare(service, data_dir=out, labels_path=labels)
    assert changed.manifest["campaign_id"] != prepared.manifest["campaign_id"]
    for name in campaign.ALGORITHMS[:3]:
        assert np.array_equal(changed.probabilities[name], prepared.probabilities[name])
    for name in campaign.ALGORITHMS[3:]:
        assert np.array_equal(changed.assignments[name], prepared.assignments[name])
    assert changed.input_reports == prepared.input_reports


def test_complete_persistence_rerun_and_restart(prepared, service, tmp_path):
    result = persist(service, prepared)
    assert result["status"] == "complete" and result["report_count"] == 25
    assert result["inserted"] == {"events": 60, "outcomes": 60, "assignments": 40}
    assert counts(service) == (60, 60, 40, 25, 10)
    rerun = persist(service, prepared)
    assert rerun["inserted"] == {"events": 0, "outcomes": 0, "assignments": 0}
    stored = CampaignStore(service.store).get(result["campaign_id"])
    ids = {item["report"]["id"] for item in stored["reports"]}
    assert len(ids) == 25
    assert all(item["report"]["performance"]["baseline_building"] for item in stored["reports"]
               if item["task"] == "classification" and item["batch_id"].startswith("normal_"))
    assert service.store.list_runs(service.model_id, "live", "") == []
    exported = export(prepared, tmp_path / "results", stored["reports"])
    assert exported["classifier_results"] == 15 and exported["clustering_results"] == 10
    reopened = Store(service.settings.database_url)
    try:
        assert {item["report"]["id"] for item in CampaignStore(reopened).get(result["campaign_id"])["reports"]} == ids
    finally:
        reopened.close()


def test_failed_unit_rolls_back_and_resumes(prepared, service):
    with patch.object(CampaignStore, "save_report", side_effect=RuntimeError("interrupted")):
        with pytest.raises(RuntimeError, match="interrupted"):
            persist(service, prepared, algorithm="lr", batch="normal_01")
    assert counts(service) == (0, 0, 0, 0, 10)
    assert CampaignStore(service.store).get(prepared.manifest["campaign_id"])["status"] == "partial"
    result = persist(service, prepared)
    assert result["report_count"] == 25 and counts(service) == (60, 60, 40, 25, 10)


def test_drift_only_and_labeled_campaigns_are_isolated(demo, prepared, service):
    out, _ = demo
    unlabelled = campaign.prepare(service, data_dir=out)
    result = persist(service, unlabelled)
    assert result["inserted"]["outcomes"] == 0
    assert prepared.manifest["campaign_id"] != unlabelled.manifest["campaign_id"]
    for item in CampaignStore(service.store).get(result["campaign_id"])["reports"]:
        if item["task"] == "classification":
            assert item["report"]["performance"]["metrics"] is None
        else:
            assert "silhouette" in item["report"]


def test_out_of_order_cluster_refresh_recomputes_downstream_streak(prepared, service):
    for name in ("engagement_shift", "friction_shift"):
        prepared.cluster_reports["kmeans"][name].update(trigger_candidate=True, status="ready")
    persist(service, prepared, algorithm="kmeans", batch="friction_shift")
    persist(service, prepared, algorithm="kmeans", batch="engagement_shift")
    report = next(item["report"] for item in CampaignStore(service.store).reports(prepared.manifest["campaign_id"])
                  if item["batch_id"] == "friction_shift")
    assert report["consecutive_candidates"] == 2 and report["review_status"] == "review_required"


def test_dry_run_never_opens_database(demo, service, monkeypatch, tmp_path, capsys):
    out, _ = demo
    monkeypatch.setattr("sys.argv", ["replay", "--dry-run", "--data-dir", str(out), "--export-dir", str(tmp_path / "dry")])
    with patch.object(Store, "__init__", side_effect=AssertionError("Dry run must not open DB")):
        main()
    result = json.loads((tmp_path / "dry/verification.json").read_text())
    assert result["dry_run"] and result["status"] == "complete"


def test_campaign_public_reads_exports_admin_guards_and_no_arbitrary_paths(demo, prepared, service, monkeypatch):
    out, _ = demo
    monkeypatch.setenv("DEMO_DATA_DIR", str(out))
    app = FastAPI()
    app.include_router(old_router)
    app.include_router(campaign_router)
    app.state.monitoring, app.state.churn_model, app.state.inference_count = service, service.model, 0
    headers = {"Authorization": "Bearer test-secret"}
    with TestClient(app) as client:
        assert client.post("/api/monitoring/multi-model-replay", json={"with_labels": True}).status_code == 401
        assert client.post("/api/monitoring/multi-model-replay", headers=headers, json={"data_dir": "C:/other"}).status_code == 422
        assert client.post("/api/monitoring/multi-model-replay", headers=headers, json={"algorithm": "untrusted"}).status_code == 422
        service.settings.demo_enabled = False
        assert client.post("/api/monitoring/multi-model-replay", headers=headers, json={}).status_code == 403
        service.settings.demo_enabled = True
        with patch("monitoring.campaign.prepare", return_value=prepared):
            response = client.post("/api/monitoring/multi-model-replay", headers=headers, json={"with_labels": True})
        assert response.status_code == 200, response.text
        identifier = response.json()["campaign_id"]
        assert len(client.get("/api/monitoring/campaigns").json()["items"]) == 1
        assert client.get(f"/api/monitoring/campaigns/{identifier}").json()["report_count"] == 25
        assert len(client.get(f"/api/monitoring/campaigns/{identifier}/classification").json()["items"]) == 15
        assert len(client.get(f"/api/monitoring/campaigns/{identifier}/clustering").json()["items"]) == 10
        assert len(client.get(f"/api/monitoring/campaigns/{identifier}/input-drift").json()["items"]) == 10
        for fmt in ("json", "csv", "html"):
            result = client.get(f"/api/monitoring/campaigns/{identifier}/export?format={fmt}")
            assert result.status_code == 200
            assert "customer_id" not in result.text and "test-secret" not in result.text
        assert client.get("/api/monitoring/campaigns/invalid").status_code == 422


@pytest.mark.skipif(not os.getenv("MONITORING_TEST_DATABASE_URL"), reason="Disposable PostgreSQL test URL required")
def test_postgres_additive_schema_campaign_role_and_idempotency(prepared, service):
    database = Store(os.environ["MONITORING_TEST_DATABASE_URL"], migrate=True)
    original = service.store
    try:
        service.store = database
        result = persist(service, prepared)
        assert result["status"] == "complete"
        assert persist(service, prepared)["inserted"] == {"events": 0, "outcomes": 0, "assignments": 0}
        with database.engine.connect() as connection:
            assert connection.execute(text("select current_user")).scalar_one() == "bank_monitoring_writer"
            rows = connection.execute(text("select relrowsecurity from pg_class c join pg_namespace n on n.oid=c.relnamespace where n.nspname='bank_monitoring' and c.relname in ('monitoring_campaigns','campaign_reports','cluster_assignments','input_drift_reports','cluster_model_registry')")).scalars().all()
            assert len(rows) == 5 and all(rows)
    finally:
        service.store = original
        database.close()
