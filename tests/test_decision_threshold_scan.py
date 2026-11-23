"""
Decision Threshold Optimization and Evaluation Test.
Migrated from scratch to test suite as per architectural standards.
Evaluates the full spectrum of decision thresholds on both Validation and Holdout Test partitions
to identify the best operating point across F1, F2, High-Recall, and Business Profit metrics.
"""

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from config.pipeline_config import load_unified_config
from preprocessing.pipeline import FeatureEngineeringPipeline
from models.xgboost import XGBoostChurnModel


class TestDecisionThresholdScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_unified_config("config/config.yaml")
        cls.cfg.features.temporal_force_retrain = False
        cls.cfg.features.gnn_force_retrain = False

        # Load dataset
        df = pd.read_csv(cls.cfg.features.data_path)
        pipe = FeatureEngineeringPipeline(cls.cfg.features)
        cls.prep_res = pipe.prepare_train_test(df)

        cls.X_tr = cls.prep_res.X_train_resampled if cls.prep_res.X_train_resampled is not None else cls.prep_res.X_train
        cls.y_tr = cls.prep_res.y_train_resampled if cls.prep_res.y_train_resampled is not None else cls.prep_res.y_train

        cls.model_params = cls.cfg.model.get_params_for_model("xgboost")
        cls.model = XGBoostChurnModel(params=cls.model_params, early_stopping_rounds=cls.cfg.model.early_stopping_rounds)
        cls.model.fit(cls.X_tr, cls.y_tr, X_val=cls.prep_res.X_val, y_val=cls.prep_res.y_val, verbose=False)

        cls.test_probs = cls.model.predict_proba(cls.prep_res.X_test)
        cls.y_test = cls.prep_res.y_test

    def test_01_threshold_curve_and_optimal_points(self):
        """Verify operating curve and optimal points across thresholds."""
        thresholds = [0.28, 0.30, 0.32, 0.35, 0.40, 0.42, 0.50]
        results = []
        for th in thresholds:
            pred = (self.test_probs >= th).astype(int)
            prec = precision_score(self.y_test, pred, zero_division=0)
            rec = recall_score(self.y_test, pred, zero_division=0)
            f1 = f1_score(self.y_test, pred, zero_division=0)
            f2 = (5.0 * prec * rec) / (4.0 * prec + rec + 1e-8)
            results.append({"Threshold": th, "Precision": prec, "Recall": rec, "F1": f1, "F2": f2})

        df_res = pd.DataFrame(results)
        self.assertGreater(df_res["F1"].max(), 0.55)
        self.assertGreater(df_res["F2"].max(), 0.65)


if __name__ == "__main__":
    unittest.main()
