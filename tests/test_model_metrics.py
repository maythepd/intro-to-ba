"""
Unit tests for model metrics discovery, priority scanning, and availability detection.
Verifies that individual model results directories (results/{model}/) are prioritized
over legacy results/comparison/ files, and unavailable models are properly flagged.
"""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import artifacts_inspector
from api.main import metrics as api_metrics_endpoint


class ModelComparisonMetricsTest(unittest.TestCase):
    """Test model metrics discovery, priority scanning, and availability."""

    def test_xgboost_priority_scan_from_individual_results(self):
        """XGBoost metrics should be scanned directly from results/xgboost/."""
        metrics, source = artifacts_inspector.scan_model_metrics("xgboost")
        self.assertIsNotNone(metrics, "XGBoost metrics should be discovered")
        self.assertIn("results/xgboost", source.replace("\\", "/"), "Should prioritize results/xgboost/")
        self.assertGreater(metrics["roc_auc"], 0.80, "XGBoost ROC-AUC should exceed baseline threshold")
        self.assertGreater(metrics["pr_auc"], 0.60, "XGBoost PR-AUC should exceed baseline threshold")
        self.assertAlmostEqual(metrics["roc_auc"], 0.8513, places=3)

    def test_logistic_regression_priority_scan(self):
        """Logistic regression metrics should be scanned directly from results/lr/."""
        metrics, source = artifacts_inspector.scan_model_metrics("logistic_regression")
        self.assertIsNotNone(metrics, "Logistic Regression metrics should be discovered")
        self.assertIn("results/lr", source.replace("\\", "/"), "Should prioritize results/lr/")
        self.assertGreater(metrics["roc_auc"], 0.80, "Logistic Regression ROC-AUC should exceed 0.80")
        self.assertAlmostEqual(metrics["roc_auc"], 0.8306, places=3)

    def test_benchmark_comparison_catalog_availability(self):
        """All models in comparison catalog should report valid availability and train commands."""
        comparison_list = artifacts_inspector.get_benchmark_comparison_list()
        self.assertGreaterEqual(len(comparison_list), 3, "Catalog must include at least 3 benchmark models")

        model_ids = {m["model_id"] for m in comparison_list}
        self.assertTrue({"xgboost", "logistic_regression"}.issubset(model_ids))

        for row in comparison_list:
            self.assertIn("available", row)
            self.assertIn("status", row)
            self.assertIn("train_command", row)
            self.assertTrue(row["train_command"].startswith("python main.py --mode train"))

            if row["available"]:
                self.assertIsNotNone(row["roc_auc"])
                self.assertNotEqual(row["source_location"], "unavailable")
            else:
                self.assertEqual(row["status"], "unavailable")

    def test_model_decile_lift_tables(self):
        """Decile lift tables should prioritize model-specific result CSVs."""
        xgb_lift = artifacts_inspector.get_decile_lift_table("xgboost")
        self.assertTrue(xgb_lift["available"])
        self.assertEqual(len(xgb_lift["deciles"]), 10, "XGBoost should have 10 deciles")
        self.assertIn("results/xgboost", xgb_lift["source"].replace("\\", "/"))

        lr_lift = artifacts_inspector.get_decile_lift_table("lr")
        self.assertTrue(lr_lift["available"])
        self.assertEqual(len(lr_lift["deciles"]), 10, "Logistic Regression should have 10 deciles")
        self.assertIn("results/lr", lr_lift["source"].replace("\\", "/"))

    def test_api_metrics_endpoint_response_structure(self):
        """GET /api/metrics should return champion metrics and models list without crashing."""
        resp = api_metrics_endpoint()
        self.assertEqual(resp["model"], "XGBoost")
        self.assertGreater(resp["roc_auc"], 0.80)
        self.assertIsInstance(resp["models"], list)
        self.assertGreaterEqual(len(resp["models"]), 3)


if __name__ == "__main__":
    unittest.main()
