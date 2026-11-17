"""
Unit and Integration Tests for Raw Event Log Feeder Mode & Decision Threshold Configuration.
Verifies:
1. Decision Threshold Propagation from YAML to ModelConfig, PipelineConfig, and ML Models.
2. Default prediction and evaluation adherence to configured decision_threshold across XGBoost, LR, and RF.
3. RawEventSequenceExtractor extraction schema: (N, max_seq_len, 26), no NaNs, correct padding.
4. TemporalEmbeddingManager dynamic mode switching between 'extractors' and 'raw_events'.
5. End-to-end training and inference with raw event sequence tensor through Bi-GRU Autoencoder.
"""

import os
import sys
import shutil
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from config.pipeline_config import UnifiedConfig, PipelineConfig, ModelConfig, load_unified_config
from models.xgboost import XGBoostChurnModel
from models.logistic_regression import LogisticRegressionChurnModel
from models.random_forest import RandomForestChurnModel
from preprocessing.temporal.raw_extractor import RawEventSequenceExtractor
from preprocessing.temporal.manager import TemporalEmbeddingManager

TEMP_TEST_DIR = Path("results/test_raw_temp")


class TestRawEventsAndThreshold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up isolated test workspace."""
        if TEMP_TEST_DIR.exists():
            shutil.rmtree(TEMP_TEST_DIR, ignore_errors=True)
        TEMP_TEST_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up test artifacts."""
        if TEMP_TEST_DIR.exists():
            shutil.rmtree(TEMP_TEST_DIR, ignore_errors=True)

    def test_01_decision_threshold_configuration_loading(self):
        """Test 1: Verify decision_threshold in config loads into ModelConfig and PipelineConfig."""
        yaml_content = """
dataset:
  data_path: "data/full_bank_data.csv"
model:
  name: "xgboost"
  decision_threshold: 0.35
features:
  enable_temporal_embeddings: true
  temporal_embeddings:
    temporal_input_mode: "raw_events"
    raw_sequence_length: 40
    manifold_strategy: "unsupervised"
"""
        yaml_path = TEMP_TEST_DIR / "test_config.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        cfg = load_unified_config(str(yaml_path))
        self.assertEqual(cfg.model.decision_threshold, 0.35)
        self.assertEqual(cfg.features.decision_threshold, 0.35)
        self.assertEqual(cfg.features.temporal_input_mode, "raw_events")
        self.assertEqual(cfg.features.raw_sequence_length, 40)

    def test_02_model_decision_threshold_adherence(self):
        """Test 2: Verify XGBoost, LR, and RF apply configured decision threshold by default."""
        np.random.seed(42)
        N = 100
        X = pd.DataFrame({
            "f1": np.random.randn(N),
            "f2": np.random.randn(N),
        })
        y = pd.Series(np.random.randint(0, 2, size=N))

        # 1. XGBoost with threshold 0.35
        xgb_model = XGBoostChurnModel(
            params={"n_estimators": 5, "max_depth": 2, "random_state": 42},
            decision_threshold=0.35,
        )
        xgb_model.fit(X, y)
        probs = xgb_model.predict_proba(X)
        preds_default = xgb_model.predict(X)
        expected_preds = (probs >= 0.35).astype(int)
        np.testing.assert_array_equal(preds_default, expected_preds)

        # Explicit threshold override works
        preds_override = xgb_model.predict(X, threshold=0.50)
        np.testing.assert_array_equal(preds_override, (probs >= 0.50).astype(int))

        # Evaluate uses default decision_threshold
        eval_res = xgb_model.evaluate(X, y)
        self.assertIn("F1_Score", eval_res)

        # 2. Logistic Regression with threshold 0.35
        lr_model = LogisticRegressionChurnModel(
            params={"C": 1.0, "max_iter": 50, "random_state": 42},
            decision_threshold=0.35,
        )
        lr_model.fit(X, y)
        lr_probs = lr_model.predict_proba(X)
        np.testing.assert_array_equal(lr_model.predict(X), (lr_probs >= 0.35).astype(int))

        # 3. Random Forest with threshold 0.35
        rf_model = RandomForestChurnModel(
            params={"n_estimators": 5, "max_depth": 2, "random_state": 42},
            decision_threshold=0.35,
        )
        rf_model.fit(X, y)
        rf_probs = rf_model.predict_proba(X)
        np.testing.assert_array_equal(rf_model.predict(X), (rf_probs >= 0.35).astype(int))

    def test_03_raw_event_sequence_extractor_schema(self):
        """Test 3: Verify RawEventSequenceExtractor extracts valid (N, L, 26) tensor without NaNs."""
        # Use synthetic event files in temp dir to ensure fast isolated unit test
        raw_dir = TEMP_TEST_DIR / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)

        cust_ids = [101, 102, 103]
        acct_ids = ["ACC_1", "ACC_2", "ACC_3"]
        pd.DataFrame({"account_id": acct_ids, "customer_id": cust_ids}).to_csv(
            raw_dir / "accounts.csv", index=False
        )

        pd.DataFrame({
            "account_id": ["ACC_1", "ACC_1", "ACC_2"],
            "transaction_datetime": ["2026-05-01 10:00:00", "2026-06-01 11:00:00", "2026-07-01 12:00:00"],
            "amount": [150.0, 50.0, 200.0],
            "direction": ["in", "out", "in"],
            "transaction_type": ["cash_deposit", "payment", "internal_transfer"],
            "transaction_status": ["success", "success", "failed"],
            "channel": ["branch", "mobile_app", "atm"],
        }).to_csv(raw_dir / "transactions.csv", index=False)

        pd.DataFrame({
            "customer_id": [101, 103],
            "sign_in_datetime": ["2026-05-15 09:00:00", "2026-08-01 10:00:00"],
            "login_status": ["success", "failed"],
            "device_type": ["ios", "desktop"],
        }).to_csv(raw_dir / "login.csv", index=False)

        pd.DataFrame({
            "customer_id": [102],
            "complaint_datetime": ["2026-06-15 14:00:00"],
        }).to_csv(raw_dir / "complaints.csv", index=False)

        extractor = RawEventSequenceExtractor(raw_data_dir=raw_dir, max_seq_len=10)
        tensor, extracted_ids = extractor.extract(cust_ids, verbose=False, save_cache=False)

        self.assertEqual(tensor.shape, (3, 10, 26))
        self.assertFalse(np.isnan(tensor).any())
        self.assertFalse(np.isinf(tensor).any())
        np.testing.assert_array_equal(extracted_ids, np.array(cust_ids, dtype=np.int64))

        # Check left-padding: customer 103 only has 1 event, so tensor[2, :-1, :] should be 0s
        self.assertTrue(np.all(tensor[2, :-1, :] == 0.0))
        # The last event at index -1 should be non-zero
        self.assertGreater(np.count_nonzero(tensor[2, -1, :]), 0)

    def test_04_temporal_embedding_manager_raw_mode_end_to_end(self):
        """Test 4: Verify TemporalEmbeddingManager in 'raw_events' mode fits and infers cleanly."""
        raw_dir = TEMP_TEST_DIR / "raw_data"
        ckpt_dir = TEMP_TEST_DIR / "checkpoints"
        cache_path = TEMP_TEST_DIR / "cache.csv"

        df_train = pd.DataFrame({
            "customer_id": [101, 102],
            "churn": [0, 1],
            "balance": [50000.0, 10000.0],
        })

        df_test = pd.DataFrame({
            "customer_id": [103],
            "churn": [0],
            "balance": [30000.0],
        })

        mgr = TemporalEmbeddingManager(
            enabled=True,
            raw_data_dir=raw_dir,
            temporal_input_mode="raw_events",
            raw_sequence_length=10,
            latent_dim=8,
            hidden_dim=16,
            num_layers=1,
            epochs=2,
            batch_size=2,
            checkpoint_dir=ckpt_dir,
            cache_path=cache_path,
            manifold_strategy="unsupervised",
            verbose=False,
            force_retrain=True,
        )

        # Fit on train customers
        mgr.fit(df_train)
        self.assertIsNotNone(mgr.embeddings_df_)
        self.assertEqual(len(mgr.embeddings_df_), 2)
        for i in range(8):
            self.assertIn(f"temporal_z{i}", mgr.embeddings_df_.columns)

        # Transform train dataframe
        df_train_trans = mgr.transform(df_train)
        self.assertIn("temporal_z0", df_train_trans.columns)

        # Transform test dataframe (eval mode inference for unseen customer 103)
        df_test_trans = mgr.transform(df_test)
        self.assertIn("temporal_z0", df_test_trans.columns)
        self.assertFalse(df_test_trans["temporal_z0"].isna().any())
        self.assertEqual(len(df_test_trans), 1)


if __name__ == "__main__":
    unittest.main()
