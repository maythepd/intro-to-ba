"""Checks distance features do not depend on churn, IDs, or row order."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from preprocessing.hierarchical_clustering_pipeline import HierarchicalClusteringPipeline, GROUPS
from models.hierarchical import HierarchicalModel
from utils.clustering_experiments import business_metrics

class HierarchicalChecks(unittest.TestCase):
    def test_business_metrics(self):
        df = pd.DataFrame({"customer_id": [1,2,3,4], "churn": [1,1,0,0], "balance": [10.,20.,30.,40.]})
        table = business_metrics(df, [0,0,1,1])
        self.assertEqual(table.loc[0, "churn_lift"], 2)
        self.assertEqual(table.loc[0, "churn_share"], 1)
        self.assertAlmostEqual(table.loc[0, "balance_share"], .3)
        self.assertEqual(table.loc[0, "total_balance"], 30)
        for col in ["customer_share", "churn_share", "balance_share"]:
            self.assertAlmostEqual(table[col].sum(), 1)
        zeros = business_metrics(df.assign(churn=0, balance=0), [0,0,1,1])
        self.assertTrue(zeros[["churn_lift", "churn_share", "balance_share"]].isna().all().all())

    def test_real_features_exclude_labels_and_preserve_scaling(self):
        df = pd.read_csv(Path(__file__).resolve().parent / "data/full_bank_data.csv").head(300)
        cfg = {"scaler": "standard", "correlation_threshold": 0.9}
        first = HierarchicalClusteringPipeline(cfg)
        x = first.prepare(df, list(GROUPS))
        changed = df.copy()
        changed["churn"] = 1 - changed["churn"]
        changed["customer_id"] = np.arange(len(df)) + 900000
        y = HierarchicalClusteringPipeline(cfg).prepare(changed, list(GROUPS))
        np.testing.assert_allclose(x, y)
        self.assertNotIn("churn", x)
        self.assertNotIn("customer_id", x)
        np.testing.assert_allclose(x.mean(), 0, atol=1e-10)
        np.testing.assert_allclose(x.std(ddof=0), 1, atol=1e-10)
        labels = HierarchicalModel().fit(x).labels(4)
        self.assertEqual(len(np.unique(labels)), 4)
        self.assertEqual(len(labels), len(df))
        shuffled = df.sample(frac=1, random_state=10)
        z = HierarchicalClusteringPipeline(cfg).prepare(shuffled, list(GROUPS)).sort_index()
        np.testing.assert_allclose(x, z, atol=1e-10)

    def test_robust_and_missing_prior_history(self):
        df = pd.read_csv(Path(__file__).resolve().parent / "data/full_bank_data.csv").head(300)
        df.loc[df.index[0], "days_observed_q2"] = 0
        prep = HierarchicalClusteringPipeline({"scaler": "robust", "correlation_threshold": None})
        x = prep.prepare(df, list(GROUPS))
        self.assertTrue(np.isfinite(x).all().all())
        self.assertTrue(pd.isna(prep.profile_features.loc[df.index[0], "trans_daily_trend"]))
        self.assertTrue((prep.feature_report.scaler == "robust").all())

if __name__ == "__main__":
    unittest.main()
