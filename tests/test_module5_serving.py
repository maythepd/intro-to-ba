"""Serving contract for the real, committed Module 5 checkpoint and preprocessor."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api import module5
from api.main import app
from api.scoring import score_probabilities


@unittest.skipUnless(module5.MANIFEST_PATH.is_file(), "Run python -m api.prepare_module5 --download first")
class Module5ServingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        environment = patch.dict(os.environ, {"DATABASE_URL": "sqlite:///" + (Path(folder.name) / "monitoring.db").as_posix()})
        environment.start()
        cls.addClassCleanup(environment.stop)
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()
        cls.model = app.state.churn_model
        cls.frame = pd.read_csv(module5.DATA_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def profile(self, index=0):
        identifier = int(self.frame.iloc[index].customer_id)
        response = self.client.get(f"/api/customers/{identifier}")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        payload["customer_id"] = identifier
        return payload

    def test_checkpoint_identity_and_reproduced_metrics(self):
        health = self.client.get("/health").json()
        self.assertTrue(health["model_loaded"])
        self.assertEqual(health["model_source"], "module5_checkpoint")
        self.assertEqual(health["feature_count"], 133)
        self.assertEqual(health["checkpoint_sha256"], module5.file_hash(module5.CHECKPOINT_PATH))
        self.assertTrue(health["study_reproduced"])
        self.assertAlmostEqual(self.model.manifest["validation_metrics"]["ROC_AUC"], .8372, places=4)
        self.assertNotIn("churn", self.model.input_columns)
        self.assertNotIn("customer_id", self.model.input_columns)

    def test_single_predictions_equal_direct_checkpoint_with_complete_features(self):
        for index in (0, 7, 53):
            payload = self.profile(index)
            response = self.client.post("/api/predict", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            raw = self.frame.iloc[[index]]
            transformed = self.model.preprocessor.transform(raw.drop(columns=["customer_id", "churn"]))
            expected = self.model.classifier.predict_proba(transformed[self.model.manifest["features"]])[0, 1]
            self.assertAlmostEqual(response.json()["churn_probability"], float(expected), places=7)

    def test_unknown_customer_needs_complete_raw_aggregates(self):
        payload = self.profile()
        payload["customer_id"] = 999999999
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("raw_features", response.json()["detail"])
        payload["raw_features"] = self.frame.iloc[0][self.model.input_columns].where(
            pd.notna(self.frame.iloc[0][self.model.input_columns]), None).to_dict()
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200, response.text)

    def test_target_and_extra_transformed_features_cannot_override_scoring(self):
        a = self.frame.iloc[:3].copy()
        b = a.copy()
        b["churn"] = 1 - b.churn
        b["robust_recency_days"] = 999999
        np.testing.assert_array_equal(score_probabilities(self.model, a), score_probabilities(self.model, b))
        payload = self.profile()
        payload["raw_features"] = {"churn": 1}
        self.assertEqual(self.client.post("/api/predict", json=payload).status_code, 422)

    def test_corrupt_checkpoint_cannot_reuse_preprocessor(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "changed.json"
            path.write_bytes(module5.CHECKPOINT_PATH.read_bytes() + b"\n")
            with patch.object(module5, "CHECKPOINT_PATH", path):
                with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                    module5.load()

    def test_no_legacy_fallback_if_module5_preprocessor_is_missing(self):
        with patch.object(module5, "PREPROCESSOR_PATH", Path("missing-module5-preprocessor.joblib")):
            self.assertIsNone(module5.load())

    def test_invalid_raw_values_return_input_errors(self):
        payload = self.profile()
        payload["raw_features"] = {"total_accounts": "not-a-number"}
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("total_accounts", response.json()["detail"])

    def test_fractional_recency_and_no_activity_customers(self):
        index = int(self.frame.index[self.frame.preferred_channel == "No_Channel"][0])
        payload = self.profile(index)
        self.assertEqual(payload["preferred_channel"], "No_Channel")
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200, response.text)

    def test_batch_uses_the_same_checkpoint_as_lookup_predictions(self):
        response = self.client.post("/api/predict/batch?include_predictions=true&force=true")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["scored_customers"], 10000)
        expected = self.client.post("/api/predict", json=self.profile()).json()
        self.assertAlmostEqual(result["predictions"][0]["churn_probability"], expected["churn_probability"], places=7)


if __name__ == "__main__":
    unittest.main()
