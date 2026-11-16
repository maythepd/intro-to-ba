"""Smoke tests for the segmentation API adapter, using fixture data only."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from api.segmentation_service import train, load, summary, customer_membership, predict, FEATURES


def fixture(path):
    rng = np.random.default_rng(12)
    n = 72
    values = {
        "customer_id": np.arange(100, 100+n),
        "credit_score": rng.integers(310,850,n),
        "age": rng.integers(18,80,n), "tenure": rng.integers(0,11,n),
        "balance": rng.gamma(3,12000,n), "products_number": rng.integers(1,5,n),
        "credit_card": rng.integers(0,2,n), "active_member": rng.integers(0,2,n),
        "estimated_salary": rng.gamma(5,19000,n),
        "country": np.resize(["France","Germany","Spain"],n),
        "gender": np.resize(["Male","Female"],n),
        "recency_days": rng.integers(1,180,n),
        "monthly_transactions": rng.integers(0,25,n),
        "trans_amount_3m": rng.gamma(2,600,n),
        "trans_trend_ratio": rng.random(n)*2,
        "app_logins_monthly": rng.integers(0,40,n),
        "customer_service_complaints": rng.integers(0,5,n),
        "has_unresolved_complaint": rng.integers(0,2,n),
        "churn": rng.integers(0,2,n),
    }
    data = pd.DataFrame(values)
    data.to_csv(path,index=False)
    return data


class SegmentationAdapterTest(unittest.TestCase):
    def test_train_serve_both_models_and_detect_stale_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, output = Path(tmp)/"official.csv",Path(tmp)/"export"
            frame = fixture(data)
            train(data,output,clusters=4,hier_sample=35)
            bundle = load(data,output)
            self.assertIsNotNone(bundle)
            for algo in ("kmeans","hierarchical"):
                s = summary(bundle,algo)
                self.assertEqual(sum(c["customers"] for c in s["clusters"]),len(frame))
                result = predict(bundle,frame.iloc[0][list(FEATURES)].to_dict(),algo)
                self.assertIn(result["cluster_id"],range(4))
            member = customer_membership(bundle,"100")
            self.assertIn(member["hierarchical_assignment_method"],("fitted_ward","nearest_centroid_approximation"))
            self.assertIsNone(customer_membership(bundle,"99999"))
            frame.loc[0,"balance"] += 1
            frame.to_csv(data,index=False)
            self.assertIsNone(load(data,output))

    def test_churn_not_used_to_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,p1,p2 = Path(tmp)/"official.csv",Path(tmp)/"v1",Path(tmp)/"v2"
            frame = fixture(data)
            a = train(data,p1,4,35)
            frame["churn"] = 1 - frame["churn"]
            frame.to_csv(data,index=False)
            b = train(data,p2,4,35)
            for name in ("kmeans","hierarchical"):
                np.testing.assert_array_equal(a["models"][name]["labels"],b["models"][name]["labels"])


if __name__ == "__main__":
    unittest.main()
