"""Batch scoring API contract tests with deterministic, fake model outputs only."""
import unittest
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app


class ProbabilityModel:
    is_fitted = True

    def __init__(self):
        self.calls = 0

    def predict_proba(self, x):
        self.calls += 1
        return ((x["age"].to_numpy() - 18.0) / 82.0).clip(0, 1)


def customer(cid, age):
    return {
        "customer_id": cid, "credit_score": 650, "country": "France", "gender": "Female",
        "age": age, "tenure": 4, "balance": 1000, "products_number": 2,
        "credit_card": 1, "active_member": 1, "estimated_salary": 55000,
        "recency_days": 15, "monthly_transactions": 8, "trans_amount_3m": 100,
        "trans_trend_ratio": 1.0, "app_logins_monthly": 7,
        "customer_service_complaints": 0, "has_unresolved_complaint": 0,
        "preferred_channel": "Mobile App", "primary_device": "iOS",
    }


class BatchRiskTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        environment = patch.dict(os.environ, {"DATABASE_URL": "sqlite:///" + (Path(folder.name) / "monitoring.db").as_posix()})
        environment.start()
        self.addCleanup(environment.stop)
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.model = ProbabilityModel()
        app.state.churn_model = self.model
        app.state.customers = pd.DataFrame([customer(1, 18), customer(2, 45), customer(3, 80), customer(4, 90)])
        app.state.risk_summary = None
        app.state.batch_predictions = None

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_counts_match_individual_predictions_and_are_cached(self):
        self.assertEqual(self.client.get("/api/dashboard/risk-summary").status_code, 503)
        result = self.client.post("/api/predict/batch?include_predictions=true&batch_size=2")
        self.assertEqual(result.status_code, 200, result.text)
        payload = result.json()
        self.assertEqual(payload["counts"], {"low": 1, "medium": 1, "high": 2})
        self.assertEqual(payload["customers_requiring_attention"], 2)
        self.assertEqual(payload["total_customers"], 4)
        self.assertEqual(len(payload["predictions"]), 4)
        self.assertEqual(self.client.get("/api/dashboard/risk-summary").json()["counts"], payload["counts"])
        for person, batch_item in zip(app.state.customers.to_dict("records"), payload["predictions"]):
            one = self.client.post("/api/predict", json=person).json()
            self.assertAlmostEqual(one["churn_probability"], batch_item["churn_probability"])
            self.assertEqual(one["risk_level"], batch_item["risk_level"])
        before = self.model.calls
        cached = self.client.post("/api/predict/batch").json()
        self.assertTrue(cached["cached"])
        self.assertNotIn("predictions", cached)
        self.assertEqual(self.model.calls, before)
        forced = self.client.post("/api/predict/batch?force=true").json()
        self.assertFalse(forced["cached"])
        self.assertEqual(sum(forced["counts"].values()), 4)
        self.assertGreater(self.model.calls, before)

    def test_missing_model_and_dataset_fail_without_fake_summary(self):
        app.state.churn_model = None
        self.assertEqual(self.client.post("/api/predict/batch").status_code, 503)
        self.assertEqual(self.client.get("/api/dashboard/risk-summary").status_code, 503)
        app.state.churn_model = self.model
        app.state.customers = None
        self.assertEqual(self.client.post("/api/predict/batch").status_code, 503)

    def test_invalid_rows_fail_without_partial_cache(self):
        app.state.customers.loc[1, "country"] = "Atlantis"
        response = self.client.post("/api/predict/batch?batch_size=2")
        self.assertEqual(response.status_code, 503)
        self.assertIsNone(app.state.risk_summary)

    def test_duplicate_ids_rejected(self):
        app.state.customers.loc[1, "customer_id"] = 1
        self.assertEqual(self.client.post("/api/predict/batch").status_code, 503)
        self.assertIsNone(app.state.risk_summary)

    def test_invalid_model_scores_cannot_be_published(self):
        class BrokenModel(ProbabilityModel):
            def predict_proba(self, x):
                return np.full(len(x), float("nan"))
        app.state.churn_model = BrokenModel()
        self.assertEqual(self.client.post("/api/predict/batch").status_code, 500)
        self.assertEqual(self.client.get("/api/dashboard/risk-summary").status_code, 503)

    def test_ground_truth_churn_not_used_for_inference(self):
        app.state.customers["churn"] = [0, 1, 0, 1]
        a = self.client.post("/api/predict/batch?include_predictions=true").json()
        app.state.customers["churn"] = [1, 0, 1, 0]
        b = self.client.post("/api/predict/batch?include_predictions=true&force=true").json()
        self.assertEqual(a["predictions"], b["predictions"])


if __name__ == "__main__":
    unittest.main()
