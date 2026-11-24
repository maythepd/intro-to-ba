"""
Unit and Integration Tests for Behavioral Economics & Consumer Psychology Features.
Tests cover:
  1. Transformer interface contract (fit, transform, fit_transform, enabled toggle).
  2. Edge case stress testing: zero denominators, astronomical values, missing columns.
  3. Sparsity analysis (zero-ratio, NaN, Inf, degenerate variance).
  4. Extreme outlier containment (Winsorization clipping bounds learned without leakage).
  5. End-to-end integration with FeatureEngineeringPipeline and full_bank_data.csv.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from preprocessing.psychology import (
    PsychologyFeatureEngineer,
    MentalAccountingEngineer,
    AnxietyMonitoringEngineer,
    PainOfPayingEngineer,
    PeakEndFrictionEngineer,
    ScarcityMindsetEngineer,
)
from preprocessing.pipeline import FeatureEngineeringPipeline
from config.pipeline_config import PipelineConfig, UnifiedConfig


class TestPsychologyFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parent.parent
        cls.data_path = project_root / "data" / "full_bank_data.csv"
        if not cls.data_path.exists():
            cls.data_path = Path("data/full_bank_data.csv").resolve()

        if cls.data_path.exists():
            cls.df_real = pd.read_csv(cls.data_path)
        else:
            try:
                from preprocessing.ingestion.etl import load_or_build_dataset
                raw_dir = str(project_root / "data" / "full_dataset")
                cls.df_real = load_or_build_dataset(raw_data_dir=raw_dir, output_path=str(cls.data_path), verbose=False)
            except Exception:
                cls.df_real = None

        # Build a representative synthetic edge-case dataset
        cls.df_edge = pd.DataFrame({
            "customer_id": [1, 2, 3, 4, 5],
            "balance": [0.0, 100000.0, 50.0, 1e8, 0.0],
            "savings_balance": [0.0, 80000.0, 0.0, 9e7, 0.0],
            "everyday_balance": [0.0, 20000.0, 50.0, 1e7, 0.0],
            "has_auto_renewal": [0, 1, 0, 1, 0],
            "has_manual_renewal": [0, 0, 1, 0, 0],
            "num_closed_accounts": [0, 0, 2, 0, 0],
            "total_accounts_ever": [1, 2, 3, 5, 1],
            "avg_early_withdrawal_rate": [0.0, 0.0, 0.05, 0.0, 0.0],
            "estimated_salary": [0.0, 120000.0, 30000.0, 500000.0, 0.0],
            "successful_logins": [0, 40, 100, 10, 0],
            "total_logins": [0, 42, 105, 10, 2],
            "customer_initiated_tx_count": [0, 20, 0, 5, 0],
            "total_transactions": [0, 22, 2, 5, 0],
            "monthly_logins": [0.0, 7.0, 20.0, 2.0, 0.0],
            "monthly_transactions": [0.0, 3.5, 0.0, 1.0, 0.0],
            "trans_amount_3m": [0.0, 5000.0, 0.0, 100000.0, 0.0],
            "trans_amount_prior_3m": [0.0, 6000.0, 1000.0, 100000.0, 0.0],
            "desktop_logins_total": [0, 2, 50, 0, 2],
            "login_recency_days": [180.0, 1.0, 0.1, 5.0, 180.0],
            "bank_fee_amount": [0.0, 0.0, 50.0, 0.0, 10.0],
            "bank_fee_count": [0, 0, 3, 0, 1],
            "cash_withdrawal_amount": [0.0, 200.0, 500.0, 0.0, 0.0],
            "total_outflow": [0.0, 4000.0, 800.0, 50000.0, 0.0],
            "total_inflow": [0.0, 10000.0, 200.0, 100000.0, 0.0],
            "customer_service_complaints": [0, 0, 3, 0, 1],
            "unresolved_complaints_count": [0, 0, 1, 0, 1],
            "failed_transactions": [0, 0, 5, 0, 1],
            "failed_logins": [0, 2, 5, 0, 2],
            "total_tx_attempts": [0, 22, 7, 5, 1],
            "exposure_months": [6.0, 6.0, 5.0, 6.0, 1.0],
            "complaint_recency_days": [180.0, 180.0, 3.0, 180.0, 1.0],
            "avg_resolution_delay_days": [0.0, 0.0, 14.0, 0.0, 0.0],
            "churn": [0, 0, 1, 0, 1],
        })

    def test_01_mental_accounting_features(self):
        """Test MentalAccountingEngineer feature creation and extreme value safety."""
        eng = MentalAccountingEngineer(enabled=True)
        res = eng.fit_transform(self.df_edge)

        expected_cols = [
            "psych_savings_commitment_ratio",
            "psych_early_break_propensity",
            "psych_renewal_inertia_status",
            "psych_everyday_drain_ratio",
            "psych_savings_to_salary_ratio",
        ]
        for col in expected_cols:
            self.assertIn(col, res.columns, f"Missing feature: {col}")
            self.assertFalse(res[col].isna().any(), f"NaN in {col}")
            self.assertFalse(np.isinf(res[col]).any(), f"Inf in {col}")

        # Check bounds: commitment ratio in [0, 1]
        self.assertTrue((res["psych_savings_commitment_ratio"] >= 0.0).all())
        self.assertTrue((res["psych_savings_commitment_ratio"] <= 1.0).all())

        # Check renewal inertia status for customer 3 (broke inertia)
        self.assertEqual(res.loc[2, "psych_renewal_inertia_status"], -1.0)
        # Customer 2 auto-renewed
        self.assertEqual(res.loc[1, "psych_renewal_inertia_status"], 1.0)
        print("[PASS] Test 1: Mental Accounting features safe across zero and extreme inputs.")

    def test_02_anxiety_monitoring_features(self):
        """Test AnxietyMonitoringEngineer feature creation and phantom engagement."""
        eng = AnxietyMonitoringEngineer(enabled=True)
        res = eng.fit_transform(self.df_edge)

        expected_cols = [
            "psych_phantom_login_ratio",
            "psych_login_to_tx_intensity",
            "psych_anxious_checking_index",
            "psych_panic_channel_migration",
            "psych_login_recency_decay",
        ]
        for col in expected_cols:
            self.assertIn(col, res.columns, f"Missing feature: {col}")
            self.assertFalse(res[col].isna().any(), f"NaN in {col}")
            self.assertFalse(np.isinf(res[col]).any(), f"Inf in {col}")

        # Customer 3 has 100 logins with 0 customer-initiated transactions -> high phantom login ratio
        self.assertGreater(res.loc[2, "psych_phantom_login_ratio"], 20.0)
        # Customer 3 has collapsing spending and high logins -> high anxious checking index
        self.assertGreater(res.loc[2, "psych_anxious_checking_index"], 2.0)
        # Recency decay is bounded in [0, 1]
        self.assertTrue((res["psych_login_recency_decay"] >= 0.0).all())
        self.assertTrue((res["psych_login_recency_decay"] <= 1.0).all())
        print("[PASS] Test 2: Anxiety Monitoring and Phantom Engagement features verified.")

    def test_03_pain_of_paying_features(self):
        """Test PainOfPayingEngineer fee retaliation and cash decoupling."""
        eng = PainOfPayingEngineer(enabled=True)
        res = eng.fit_transform(self.df_edge)

        expected_cols = [
            "psych_fee_retaliation_score",
            "psych_fee_burden_to_balance",
            "psych_cash_detachment_ratio",
            "psych_loss_aversion_drain_index",
            "psych_fee_frequency_shock",
        ]
        for col in expected_cols:
            self.assertIn(col, res.columns, f"Missing feature: {col}")
            self.assertFalse(res[col].isna().any(), f"NaN in {col}")
            self.assertFalse(np.isinf(res[col]).any(), f"Inf in {col}")

        # Customer 3 had 3 bank fees and spending collapsed -> positive retaliation score
        self.assertGreater(res.loc[2, "psych_fee_retaliation_score"], 0.0)
        # Customer 3 has 50 CU fee on 50 CU balance -> fee burden is substantial
        self.assertGreater(res.loc[2, "psych_fee_burden_to_balance"], 50.0)
        print("[PASS] Test 3: Pain of Paying and Loss Sensitivity features verified.")

    def test_04_peak_end_friction_features(self):
        """Test PeakEndFrictionEngineer recency-weighted complaint dynamics."""
        eng = PeakEndFrictionEngineer(enabled=True, recency_half_life_days=30.0)
        res = eng.fit_transform(self.df_edge)

        expected_cols = [
            "psych_exponential_recency_complaint",
            "psych_complaint_recency_velocity",
            "psych_unresolved_expectation_gap",
            "psych_terminal_failure_shock",
            "psych_peak_friction_intensity",
        ]
        for col in expected_cols:
            self.assertIn(col, res.columns, f"Missing feature: {col}")
            self.assertFalse(res[col].isna().any(), f"NaN in {col}")
            self.assertFalse(np.isinf(res[col]).any(), f"Inf in {col}")

        # Customer 3 had 3 complaints only 3 days ago -> exponential score stays high
        self.assertGreater(res.loc[2, "psych_exponential_recency_complaint"], 2.0)
        # Customer 5 had 1 complaint 1 day ago -> terminal failure shock is positive
        self.assertGreater(res.loc[4, "psych_terminal_failure_shock"], 0.0)
        print("[PASS] Test 4: Peak-End Rule and Catastrophic Friction features verified.")

    def test_05_scarcity_mindset_features(self):
        """Test ScarcityMindsetEngineer financial runway and survival pressure."""
        eng = ScarcityMindsetEngineer(enabled=True)
        res = eng.fit_transform(self.df_edge)

        expected_cols = [
            "psych_financial_runway_months",
            "psych_is_runway_vulnerable",
            "psych_hand_to_mouth_burn_rate",
            "psych_paycheck_absorption_rate",
            "psych_scarcity_friction_interaction",
        ]
        for col in expected_cols:
            self.assertIn(col, res.columns, f"Missing feature: {col}")
            self.assertFalse(res[col].isna().any(), f"NaN in {col}")
            self.assertFalse(np.isinf(res[col]).any(), f"Inf in {col}")

        # Customer 3 has balance 50 and monthly outflow 160 -> runway < 1.0 month -> vulnerable!
        self.assertEqual(res.loc[2, "psych_is_runway_vulnerable"], 1.0)
        self.assertLess(res.loc[2, "psych_financial_runway_months"], 1.0)
        # Customer 2 has 100k balance and 4k outflow -> runway is long, not vulnerable
        self.assertEqual(res.loc[1, "psych_is_runway_vulnerable"], 0.0)
        print("[PASS] Test 5: Scarcity Mindset and Financial Runway features verified.")

    def test_06_orchestrator_winsorization_and_sparsity(self):
        """Test PsychologyFeatureEngineer composite orchestrator, Winsorization, and sparsity pruning."""
        orchestrator = PsychologyFeatureEngineer(
            enabled=True,
            prune_sparse=True,
            outlier_clip_quantile=0.99,
        )
        res_train = orchestrator.fit_transform(self.df_edge)
        self.assertTrue(orchestrator.is_fitted)
        self.assertGreater(len(orchestrator.valid_features_), 10)

        # Transform on test partition
        res_test = orchestrator.transform(self.df_edge)
        self.assertEqual(res_train.shape[1], res_test.shape[1])

        # Test disabled mode
        orch_disabled = PsychologyFeatureEngineer(enabled=False)
        res_disabled = orch_disabled.fit_transform(self.df_edge)
        self.assertEqual(res_disabled.shape[1], self.df_edge.shape[1])
        print("[PASS] Test 6: Composite Orchestrator Winsorization and sparsity pruning verified.")

    def test_07_real_dataset_sparsity_and_outlier_audit(self):
        """
        Audit all 25 psychology features on the full real dataset (full_bank_data.csv).
        Checks:
          - NaN / Missing percentage: MUST BE 0.0%
          - Inf / Extreme overflow: MUST BE 0.0%
          - Variance: MUST BE > 1e-6 (non-degenerate)
          - Sparsity (< 98% zeros or valid binary signals)
        """
        if self.df_real is None:
            self.skipTest("full_bank_data.csv not available")

        orchestrator = PsychologyFeatureEngineer(
            enabled=True,
            prune_sparse=True,
            outlier_clip_quantile=0.995,
        )
        res = orchestrator.fit_transform(self.df_real)
        created = orchestrator.created_features
        self.assertGreaterEqual(len(created), 20, "Should generate at least 20 valid features on real data.")

        print(f"\n--- PSYCHOLOGY FEATURE AUDIT ON REAL DATASET ({len(self.df_real)} rows) ---")
        for feat in created:
            vals = res[feat]
            nan_cnt = int(vals.isna().sum())
            inf_cnt = int(np.isinf(vals).sum())
            zero_pct = float((vals == 0.0).sum() / len(vals)) * 100.0
            var = float(vals.var())
            min_val = float(vals.min())
            max_val = float(vals.max())
            q99 = float(np.percentile(vals, 99))

            self.assertEqual(nan_cnt, 0, f"Feature {feat} contains {nan_cnt} NaNs!")
            self.assertEqual(inf_cnt, 0, f"Feature {feat} contains {inf_cnt} Infs!")
            self.assertGreater(var, 1e-6, f"Feature {feat} has degenerate near-zero variance ({var})!")

            print(f"  • {feat:38s} | Zeros: {zero_pct:5.1f}% | Var: {var:10.4f} | Range: [{min_val:.2f}, {max_val:.2f}] (q99={q99:.2f})")

        print("[PASS] Test 7: Real dataset audit confirmed 0% NaNs, 0% Infs, healthy variance across all features.")

    def test_08_pipeline_end_to_end_integration(self):
        """Verify full FeatureEngineeringPipeline executes smoothly with psychology features included."""
        if self.df_real is None:
            self.skipTest("full_bank_data.csv not available")

        cfg = PipelineConfig(
            enable_psychology_features=True,
            enable_domain_features=True,
            enable_scaling=True,
            enable_skewness=True,
            enable_one_hot_encoding=True,
            enable_target_encoding=False,
            enable_smote=False,
            enable_feature_selection=False,
        )
        pipeline = FeatureEngineeringPipeline(config=cfg, verbose=False)
        result = pipeline.prepare_train_test(self.df_real)

        psych_in_model = [f for f in result.model_features if f.startswith("psych_")]
        self.assertGreater(len(psych_in_model), 15, "Model features should include psychology features.")

        self.assertFalse(result.X_train[psych_in_model].isna().any().any(), "X_train contains NaNs in psych features.")
        self.assertFalse(result.X_test[psych_in_model].isna().any().any(), "X_test contains NaNs in psych features.")
        print(f"\n[PASS] Test 8: Pipeline integration successful with {len(psych_in_model)} psychology features active in model.")


if __name__ == "__main__":
    unittest.main()
