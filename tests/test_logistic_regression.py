"""
Unit test suite for Logistic Regression Classifier module:
Verifies automatic feature standardization (StandardScaler),
solver convergence, coefficient importances, and PR-AUC performance.
"""

import os
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.logistic_regression import LogisticRegressionChurnModel


class TestLogisticRegressionModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create synthetic dataset with wildly different scales (e.g. balance ~ 100k, binary ~ 0/1)
        X_raw, y_raw = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            weights=[0.8, 0.2],
            random_state=42,
        )
        self.df_X = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(20)])
        # Artificially blow up several features to simulate unscaled banking balances & transactions
        self.df_X["feat_0"] = self.df_X["feat_0"] * 50000.0 + 100000.0  # Balance scale
        self.df_X["feat_1"] = self.df_X["feat_1"] * 25000.0              # Net cash flow scale
        self.y = pd.Series(y_raw, name="churn")

        self.X_train = self.df_X.iloc[:800]
        self.y_train = self.y.iloc[:800]
        self.X_test = self.df_X.iloc[800:]
        self.y_test = self.y.iloc[800:]

    def test_01_automatic_standardization_prevents_solver_stalling(self):
        """Test 1: Internal scaler allows SAGA and L-BFGS to converge and achieve strong PR-AUC."""
        lr_scaled = LogisticRegressionChurnModel(
            params={"solver": "lbfgs", "C": 0.1, "max_iter": 500, "random_state": 42},
            scale_features=True,
        )
        lr_scaled.fit(self.X_train, self.y_train)
        metrics = lr_scaled.evaluate(self.X_test, self.y_test)

        self.assertGreater(metrics["PR_AUC"], 0.40)
        self.assertGreater(metrics["ROC_AUC"], 0.70)
        print(f"[PASS] Test 1: Standardized LR achieves PR_AUC={metrics['PR_AUC']:.4f}, ROC_AUC={metrics['ROC_AUC']:.4f}.")

    def test_02_feature_importances_standardized(self):
        """Test 2: Coefficient magnitudes reflect standardized impact, not raw feature scales."""
        lr = LogisticRegressionChurnModel(
            params={"solver": "lbfgs", "C": 0.1, "random_state": 42},
            scale_features=True,
        )
        lr.fit(self.X_train, self.y_train)
        df_imp = lr.get_feature_importances()

        self.assertEqual(len(df_imp), 20)
        self.assertIn("Feature", df_imp.columns)
        self.assertIn("Importance", df_imp.columns)
        # Check that feat_0 does not explode in importance just because its raw scale was 100,000
        feat_0_imp = df_imp.loc[df_imp["Feature"] == "feat_0", "Importance"].values[0]
        self.assertLess(feat_0_imp, 10.0)
        print(f"[PASS] Test 2: Standardized coefficient feature importances verified (feat_0 imp = {feat_0_imp:.4f}).")

    def test_03_serialization_roundtrip(self):
        """Test 3: Pickle serialization saves and loads fitted scaler and model seamlessly."""
        tmp_path = Path("results/test_lr_temp/lr_model.pkl")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            lr = LogisticRegressionChurnModel(scale_features=True)
            lr.fit(self.X_train, self.y_train)
            probs_orig = lr.predict_proba(self.X_test)

            lr.save(tmp_path)
            loaded_lr = LogisticRegressionChurnModel()
            loaded_lr.load(tmp_path)

            probs_loaded = loaded_lr.predict_proba(self.X_test)
            np.testing.assert_allclose(probs_orig, probs_loaded, rtol=1e-5)
            print("[PASS] Test 3: Model + Scaler serialization roundtrip verified.")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
            if tmp_path.parent.exists():
                import shutil
                shutil.rmtree(tmp_path.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

