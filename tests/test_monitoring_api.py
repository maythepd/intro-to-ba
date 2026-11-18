import os
import json
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from api import module5
from api.main import app
from monitoring.settings import now_utc, timestamp
from monitoring.store import events
from monitoring.service import MonitoringService


@unittest.skipUnless(module5.REFERENCE_PATH.is_file(), "Prepare Module 5 monitoring artifacts first")
class MonitoringApiTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        environment = patch.dict(os.environ, {"DATABASE_URL": "sqlite:///" + (Path(self.folder.name) / "monitoring.db").as_posix(),
            "MONITORING_ADMIN_TOKEN": "unit-test-monitoring-token", "ENABLE_MONITORING_DEMO": "true"})
        environment.start()
        self.addCleanup(environment.stop)
        self.context = TestClient(app)
        self.client = self.context.__enter__()
        self.addCleanup(lambda: self.context.__exit__(None, None, None))
        self.service = app.state.monitoring
        self.headers = {"Authorization": "Bearer unit-test-monitoring-token"}

    def profile(self):
        row = self.client.get("/api/customers?limit=1").json()["items"][0]
        row["customer_id"] = int(row["customer_id"])
        return row

    def count_events(self):
        with self.service.store.engine.connect() as connection:
            return connection.execute(select(func.count()).select_from(events)).scalar_one()

    def test_admin_auth_and_default_live_empty_state(self):
        response = self.client.get("/api/monitoring")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source"], "live")
        self.assertTrue(response.json()["storage_ready"])
        self.assertIsNone(response.json()["latest"])
        self.assertEqual(self.client.post("/api/monitoring/runs", json={}).status_code, 401)
        self.assertEqual(self.client.post("/api/monitoring/demo", headers={"Authorization":"Bearer wrong"}).status_code, 401)
        self.assertEqual(self.client.post("/api/monitoring/demo", headers=[(b"authorization",b"Bearer \xff")]).status_code, 401)
        self.assertEqual(self.client.get("/api/monitoring?source=study_replay").status_code, 422)

    def test_single_success_and_validation_error_are_logged(self):
        result = self.client.post("/api/predict", json=self.profile())
        self.assertEqual(result.status_code, 200, result.text)
        self.assertTrue(result.json()["monitoring_recorded"])
        self.assertTrue(result.json()["prediction_id"])
        self.assertTrue(result.json()["outcome_horizon_end"])
        self.assertEqual(self.client.post("/api/predict", json={}).status_code, 422)
        self.assertEqual(self.count_events(), 2)

    def test_delayed_labels_refresh_the_original_prediction_window(self):
        frame = app.state.customers.iloc[:1]
        row = self.service.event_rows(frame, [.7], source="live", occurred_at=now_utc() - timedelta(days=100))[0]
        self.service.store.append_events([row])
        result = self.client.post("/api/monitoring/labels", headers=self.headers, json={"items":[{
            "event_id":row["id"], "label":1, "horizon_end":row["horizon_end"],
            "observed_at":row["horizon_end"], "source":"observed"}]})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(len(result.json()["reports_updated"]), 1)
        report = self.client.get("/api/monitoring?days=180").json()["latest"]
        self.assertEqual(report["performance"]["labeled_samples"], 1)
        self.assertEqual(report["performance"]["status"], "single_class")

    def test_cached_and_forced_batch_do_not_duplicate_snapshot_events(self):
        for suffix in ("batch_size=1000", "batch_size=500", "force=true&batch_size=333"):
            response = self.client.post("/api/predict/batch?" + suffix)
            self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.count_events(), 10000)

    def test_logging_failure_preserves_prediction_and_exposes_no_secret(self):
        with patch.object(self.service.store, "append_events", side_effect=RuntimeError("postgresql://secret-password")):
            result = self.client.post("/api/predict", json=self.profile())
        self.assertEqual(result.status_code, 200)
        self.assertFalse(result.json()["monitoring_recorded"])
        summary = self.client.get("/api/monitoring")
        self.assertEqual(summary.json()["last_ingest_error"], "RuntimeError")
        self.assertNotIn("secret-password", summary.text)

    def test_reference_from_another_checkpoint_is_rejected(self):
        wrong = {**self.service.reference, "checkpoint_sha256":"f" * 64}
        path = Path(self.folder.name) / "wrong-reference.json"
        path.write_text(json.dumps(wrong), encoding="utf-8")
        model = self.service.model
        altered = module5.Module5Model(model.classifier, model.preprocessor,
            {**model.manifest, "monitoring_reference_sha256":module5.file_hash(path)})
        with patch.object(module5, "REFERENCE_PATH", path):
            invalid = MonitoringService(altered, settings=self.service.settings)
        self.assertFalse(invalid.status()["storage_ready"])
        self.assertEqual(invalid.last_error, "ValueError")

    def test_replay_alerts_reports_and_history_survive_api_restart(self):
        first = self.client.post("/api/monitoring/demo", headers=self.headers)
        self.assertEqual(first.status_code, 200, first.text)
        items = first.json()["runs"]
        self.assertEqual([row["status"] for row in items[:2]], ["stable", "stable"])
        self.assertEqual(items[3]["status"], "review_required")
        self.assertEqual(items[5]["status"], "retraining_required")
        second = self.client.post("/api/monitoring/demo", headers=self.headers)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["created_events"], 0)
        self.assertEqual(self.count_events(), 3000)
        identifier = items[3]["id"]
        html = self.client.get(f"/api/monitoring/runs/{identifier}/report")
        self.assertEqual(html.status_code, 200)
        self.assertIn("attachment", html.headers["Content-Disposition"])
        self.assertIn("default-src 'none'", html.headers["Content-Security-Policy"])
        self.assertNotIn("unit-test-monitoring-token", html.text)
        self.context.__exit__(None, None, None)
        self.context = TestClient(app)
        self.client = self.context.__enter__()
        summary = self.client.get("/api/monitoring?source=study_replay&scenario=behavior_drift")
        self.assertEqual(summary.status_code, 200, summary.text)
        self.assertEqual(len(summary.json()["history"]), 2)
        self.assertEqual(summary.json()["latest"]["id"], identifier)
        self.assertIsNone(self.client.get("/api/monitoring").json()["latest"])


if __name__ == "__main__":
    unittest.main()
