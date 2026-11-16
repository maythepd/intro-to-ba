"""Check committed model study metrics against the demo's displayed reference figures."""
import csv
import unittest
from pathlib import Path

METRICS = Path(__file__).resolve().parents[1] / "model_development/results/comparison/model_comparison_metrics.csv"

class ModelComparisonMetricsTest(unittest.TestCase):
    def test_three_models_are_present(self):
        with METRICS.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({row["Model"] for row in rows}, {"LR", "RF", "XGBOOST"})

    def test_xgboost_metrics_match_committed_report(self):
        with METRICS.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        xgb = next(row for row in rows if row["Model"] == "XGBOOST")
        for col, value in {"ROC_AUC": .8372, "PR_AUC": .6060, "F1_Score": .3389, "Precision": .204, "Recall": 1.0}.items():
            self.assertAlmostEqual(float(xgb[col]), value, places=4)

if __name__ == "__main__":
    unittest.main()
