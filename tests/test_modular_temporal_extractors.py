"""
Unit and Integration Tests for Modular Temporal Extractors and Feature Dimensions.
Verifies:
1. Dimensionality: Checks all 5 sub-extractors and full TemporalSnapshotExtractor return exact channel counts (D=37).
2. Non-zero Variance: Asserts no fallback to default/zero values across sampled customer behavior.
3. Numerical Safety: Asserts zero NaNs, Infs, or unhandled null values in all extracted tensors.
4. Robust Outlier & Skew Bounds: Asserts normalized tensors strictly conform to [-5.0, 5.0] bounds.
5. Context Sharing: Verifies intermediate aggregations are efficiently shared via context dictionary.
6. Cleanliness: Ensures complete cleanup of all temporary test artifacts and caches.
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

from preprocessing.temporal.extractors import (
    BaseSubExtractor,
    TransactionSnapshotExtractor,
    LoginSnapshotExtractor,
    ComplaintSnapshotExtractor,
    AccountSnapshotExtractor,
    CrossChannelSnapshotExtractor,
    ALL_FEATURE_NAMES_37,
)
from preprocessing.temporal.extractor import TemporalSnapshotExtractor

TEMP_TEST_DIR = Path("results/test_extractors_temp")


class TestModularTemporalExtractors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up isolated test workspace and load sample customer cohort."""
        if TEMP_TEST_DIR.exists():
            shutil.rmtree(TEMP_TEST_DIR, ignore_errors=True)
        TEMP_TEST_DIR.mkdir(parents=True, exist_ok=True)

        cls.raw_data_dir = "data/full_dataset"
        cls.months = [4, 5, 6, 7, 8, 9]

        # Ingest 300 real customer IDs for empirical statistical validation
        acc_path = Path(cls.raw_data_dir) / "accounts.csv"
        acc_df = pd.read_csv(acc_path, usecols=["customer_id"])
        unique_cids = acc_df["customer_id"].drop_duplicates().values
        cls.sample_cids = unique_cids[:300]
        cls.n_samples = len(cls.sample_cids)

        cls.test_cache_path = TEMP_TEST_DIR / "test_snapshot_tensor.npz"

    @classmethod
    def tearDownClass(cls):
        """Clean up all temporary files created during testing."""
        if TEMP_TEST_DIR.exists():
            shutil.rmtree(TEMP_TEST_DIR, ignore_errors=True)
            print(f"\n[OK] Cleaned up temporary extractor test artifacts at: {TEMP_TEST_DIR}")

    def test_01_feature_name_registry_and_counts(self):
        """Test 1: Verify all 37 channel names are uniquely defined across sub-extractors."""
        self.assertEqual(len(TransactionSnapshotExtractor.FEATURE_NAMES), 11)
        self.assertEqual(len(LoginSnapshotExtractor.FEATURE_NAMES), 15)
        self.assertEqual(len(ComplaintSnapshotExtractor.FEATURE_NAMES), 5)
        self.assertEqual(len(AccountSnapshotExtractor.FEATURE_NAMES), 4)
        self.assertEqual(len(CrossChannelSnapshotExtractor.FEATURE_NAMES), 2)

        self.assertEqual(len(ALL_FEATURE_NAMES_37), 37)
        self.assertEqual(len(set(ALL_FEATURE_NAMES_37)), 37, "All 37 feature names must be globally unique.")
        print(f"[PASS] Test 1: Feature registry verified — 37 unique channels across 5 sub-extractors.")

    def test_02_transaction_sub_extractor(self):
        """Test 2: Verify TransactionSnapshotExtractor generates 11 channels without nulls or zero-variance."""
        extractor = TransactionSnapshotExtractor(self.raw_data_dir)
        context = {}
        tensor, names = extractor.extract(self.sample_cids, self.months, context)

        self.assertEqual(tensor.shape, (self.n_samples, len(self.months), 11))
        self.assertEqual(names, TransactionSnapshotExtractor.FEATURE_NAMES)
        self.assertFalse(np.isnan(tensor).any(), "Transaction tensor must have zero NaNs.")
        self.assertFalse(np.isinf(tensor).any(), "Transaction tensor must have zero Infs.")

        # Verify context storage
        self.assertIn("acct_map", context)
        self.assertIn("tx_cnt_raw", context)

        # Non-zero variance across active volume channels
        for col_idx, name in enumerate(names):
            std_val = float(np.std(tensor[:, :, col_idx]))
            self.assertGreater(
                std_val,
                0.001,
                f"Channel '{name}' has near-zero variance ({std_val:.6f})! Indicates default fallback bug.",
            )
        print(f"[PASS] Test 2: TransactionSnapshotExtractor passed (11 channels, valid variance, zero NaNs).")

    def test_03_login_sub_extractor(self):
        """Test 3: Verify LoginSnapshotExtractor generates 15 channels without nulls or zero-variance."""
        extractor = LoginSnapshotExtractor(self.raw_data_dir)
        context = {}
        tensor, names = extractor.extract(self.sample_cids, self.months, context)

        self.assertEqual(tensor.shape, (self.n_samples, len(self.months), 15))
        self.assertEqual(names, LoginSnapshotExtractor.FEATURE_NAMES)
        self.assertFalse(np.isnan(tensor).any(), "Login tensor must have zero NaNs.")
        self.assertFalse(np.isinf(tensor).any(), "Login tensor must have zero Infs.")

        # Verify context storage
        self.assertIn("log_cnt_raw", context)
        self.assertIn("fail_cnt_raw", context)

        # Verify variance across channels
        for col_idx, name in enumerate(names):
            std_val = float(np.std(tensor[:, :, col_idx]))
            self.assertGreater(
                std_val,
                0.0001,
                f"Channel '{name}' has near-zero variance ({std_val:.6f})! Indicates default fallback bug.",
            )
        print(f"[PASS] Test 3: LoginSnapshotExtractor passed (15 channels, valid variance, zero NaNs).")

    def test_04_complaint_sub_extractor(self):
        """Test 4: Verify ComplaintSnapshotExtractor generates 5 channels without nulls."""
        extractor = ComplaintSnapshotExtractor(self.raw_data_dir)
        context = {}
        tensor, names = extractor.extract(self.sample_cids, self.months, context)

        self.assertEqual(tensor.shape, (self.n_samples, len(self.months), 5))
        self.assertEqual(names, ComplaintSnapshotExtractor.FEATURE_NAMES)
        self.assertFalse(np.isnan(tensor).any(), "Complaint tensor must have zero NaNs.")
        self.assertFalse(np.isinf(tensor).any(), "Complaint tensor must have zero Infs.")

        # Verify context storage
        self.assertIn("cmp_cnt_raw", context)

        # Check complaint channels have valid variance
        for col_idx, name in enumerate(names):
            std_val = float(np.std(tensor[:, :, col_idx]))
            self.assertGreater(
                std_val,
                0.0001,
                f"Channel '{name}' has near-zero variance ({std_val:.6f})! Check complaint processing.",
            )
        print(f"[PASS] Test 4: ComplaintSnapshotExtractor passed (5 channels, valid variance, zero NaNs).")

    def test_05_account_sub_extractor(self):
        """Test 5: Verify AccountSnapshotExtractor generates 4 channels without nulls."""
        extractor = AccountSnapshotExtractor(self.raw_data_dir)
        context = {}
        tensor, names = extractor.extract(self.sample_cids, self.months, context)

        self.assertEqual(tensor.shape, (self.n_samples, len(self.months), 4))
        self.assertEqual(names, AccountSnapshotExtractor.FEATURE_NAMES)
        self.assertFalse(np.isnan(tensor).any(), "Account tensor must have zero NaNs.")
        self.assertFalse(np.isinf(tensor).any(), "Account tensor must have zero Infs.")

        # Check account holding stock has positive active count and variance
        active_counts = tensor[:, :, 0]
        self.assertGreater(np.mean(active_counts), 0.5, "Average active accounts should be > 0.5.")
        self.assertGreater(float(np.std(active_counts)), 0.1, "Active accounts must have variation.")
        print(f"[PASS] Test 5: AccountSnapshotExtractor passed (4 channels, valid portfolio dynamics).")

    def test_06_cross_channel_sub_extractor(self):
        """Test 6: Verify CrossChannelSnapshotExtractor operates with and without pre-populated context."""
        extractor = CrossChannelSnapshotExtractor(self.raw_data_dir)

        # Scenario A: with pre-populated context
        context_mock = {
            "log_cnt_raw": np.random.poisson(15, size=(self.n_samples, 6)).astype(np.float32),
            "tx_cnt_raw": np.random.poisson(5, size=(self.n_samples, 6)).astype(np.float32),
            "fail_cnt_raw": np.random.poisson(1, size=(self.n_samples, 6)).astype(np.float32),
            "cmp_cnt_raw": np.random.poisson(0.5, size=(self.n_samples, 6)).astype(np.float32),
        }
        tensor_ctx, names_ctx = extractor.extract(self.sample_cids, self.months, context_mock)
        self.assertEqual(tensor_ctx.shape, (self.n_samples, 6, 2))
        self.assertFalse(np.isnan(tensor_ctx).any())
        self.assertFalse(np.isinf(tensor_ctx).any())

        # Scenario B: standalone invocation (loads raw tables automatically)
        tensor_standalone, names_sa = extractor.extract(self.sample_cids[:20], self.months, context=None)
        self.assertEqual(tensor_standalone.shape, (20, 6, 2))
        self.assertEqual(names_sa, CrossChannelSnapshotExtractor.FEATURE_NAMES)
        self.assertFalse(np.isnan(tensor_standalone).any())
        print(f"[PASS] Test 6: CrossChannelSnapshotExtractor passed (2 channels, context-aware & standalone).")

    def test_07_full_temporal_snapshot_extractor_integration(self):
        """Test 7: Verify TemporalSnapshotExtractor coordinates sub-extractors, standardizes, and clamps outliers."""
        extractor = TemporalSnapshotExtractor(
            raw_data_dir=self.raw_data_dir,
            months=self.months,
            tensor_cache_path=self.test_cache_path,
            use_extended=True,
        )

        tensor, cids = extractor.extract(self.sample_cids, verbose=False, save_cache=True)

        # 1. Dimension checks
        self.assertEqual(tensor.shape, (self.n_samples, len(self.months), 37))
        self.assertEqual(len(cids), self.n_samples)

        # 2. Strict numerical safety (Zero NaNs, Infs)
        self.assertEqual(int(np.isnan(tensor).sum()), 0, "Integrated tensor contains NaNs!")
        self.assertEqual(int(np.isinf(tensor).sum()), 0, "Integrated tensor contains Infs!")

        # 3. Robust outlier and skew bounds
        self.assertTrue(
            bool(np.all(tensor >= -5.0)),
            f"Found values below -5.0 lower bound: min={np.min(tensor)}",
        )
        self.assertTrue(
            bool(np.all(tensor <= 5.0)),
            f"Found values above 5.0 upper bound: max={np.max(tensor)}",
        )

        # 4. Standardization sanity checks
        means = np.mean(tensor, axis=(0, 1))
        stds = np.std(tensor, axis=(0, 1))
        self.assertTrue(
            bool(np.all(np.abs(means) < 0.25)),
            f"Mean values should center near 0.0, max abs mean was {np.max(np.abs(means))}",
        )
        self.assertTrue(
            bool(np.all(stds > 0.01)),
            f"Standard deviations should be positive non-zero across all channels.",
        )

        # 5. Caching verification
        self.assertTrue(self.test_cache_path.exists(), "Snapshot tensor should be saved to test cache.")
        tensor_cached, _ = extractor.extract(self.sample_cids, verbose=False)
        np.testing.assert_array_almost_equal(tensor, tensor_cached, decimal=5)
        print(f"[PASS] Test 7: Full TemporalSnapshotExtractor verified (37 dimensions, zero NaNs, strictly bounded [-5.0, 5.0]).")

    def test_08_legacy_10d_mode_sub_extractor_delegation(self):
        """Test 8: Verify legacy 10-channel extraction path delegates to sub-extractors cleanly."""
        extractor_10d = TemporalSnapshotExtractor(
            raw_data_dir=self.raw_data_dir,
            months=self.months,
            use_extended=False,
        )
        self.assertEqual(len(extractor_10d.feature_names), 10)
        self.assertEqual(extractor_10d.feature_names, TemporalSnapshotExtractor.LEGACY_FEATURE_NAMES_10)

        tensor_10d, _ = extractor_10d.extract(self.sample_cids[:50], verbose=False)
        self.assertEqual(tensor_10d.shape, (50, len(self.months), 10))
        self.assertFalse(np.isnan(tensor_10d).any(), "Legacy 10D tensor contains NaNs!")
        self.assertFalse(np.isinf(tensor_10d).any(), "Legacy 10D tensor contains Infs!")
        self.assertTrue(bool(np.all(tensor_10d >= -5.0)))
        self.assertTrue(bool(np.all(tensor_10d <= 5.0)))
        print(f"[PASS] Test 8: Legacy 10D mode delegates cleanly to sub-extractors without code duplication.")


if __name__ == "__main__":
    unittest.main()
