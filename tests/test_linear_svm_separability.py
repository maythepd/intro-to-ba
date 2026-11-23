"""
Test and Benchmark Suite for Linear SVM and Feature Space Separability Analysis.
Evaluates:
  1. Unit tests for LinearSVMChurnModel (Platt calibration, standardized hyperplane, separability metrics).
  2. Geometric margin diagnostics: gamma = 1 / ||w||_2, slack violations xi_i > 0, severe violations xi_i > 1.
  3. Empirical linear separability across 3 feature spaces on the Bank Churn Dataset:
     - Space 1: Baseline Tabular (Raw Kaggle + Domain Features)
     - Space 2: Baseline + Multiplicative Interactions (quant_inter_*)
     - Space 3: Baseline + Full Multi-Scale Quant Signals & Interactions
  4. Benchmark comparison between Linear SVM, Logistic Regression, and XGBoost.
  5. Exports results to results/linear_svm_separability_comparison.csv.
"""

from pathlib import Path
import sys
import unittest
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.pipeline_config import PipelineConfig, ModelConfig
from models.linear_svm import LinearSVMChurnModel
from models.logistic_regression import LogisticRegressionChurnModel
from models.xgboost import XGBoostChurnModel
from preprocessing import pipeline, FeatureEngineeringPipeline


class TestLinearSVMUnit(unittest.TestCase):
    """Unit tests for LinearSVMChurnModel contract and mathematical properties."""

    def setUp(self):
        np.random.seed(42)
        # 1. Strictly linearly separable 2D dataset
        X_sep, y_sep = make_blobs(
            n_samples=200,
            centers=[[-3.0, -3.0], [3.0, 3.0]],
            cluster_std=0.8,
            random_state=42,
        )
        self.df_sep_X = pd.DataFrame(X_sep, columns=["f1", "f2"])
        self.y_sep = pd.Series(y_sep, name="target")

        # 2. Overlapping, non-separable dataset with disparate scales
        X_nonsep, y_nonsep = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            weights=[0.8, 0.2],
            flip_y=0.15,
            random_state=42,
        )
        self.df_nonsep_X = pd.DataFrame(X_nonsep, columns=[f"feat_{i}" for i in range(10)])
        self.df_nonsep_X["feat_0"] = self.df_nonsep_X["feat_0"] * 100000.0 + 50000.0
        self.y_nonsep = pd.Series(y_nonsep, name="churn")

    def test_01_separable_data_diagnostics(self):
        """Linearly separable data should achieve near-zero violations."""
        model = LinearSVMChurnModel(
            params={"C": 1.0, "random_state": 42},
            scale_features=True,
            calibrate_probabilities=False,
        )
        model.fit(self.df_sep_X, self.y_sep)
        sep_metrics = model.compute_separability_metrics(self.df_sep_X, self.y_sep)

        self.assertEqual(sep_metrics["misclassification_rate"], 0.0)
        self.assertEqual(sep_metrics["severe_violation_rate"], 0.0)
        self.assertGreater(sep_metrics["geometric_margin"], 0.0)
        print(f"[PASS] Separable diagnostics verified: margin={sep_metrics['geometric_margin']}, misclass={sep_metrics['misclassification_rate']}")

    def test_02_platt_calibration_output(self):
        """Platt scaling via CalibratedClassifierCV must output valid probabilities in [0, 1]."""
        model = LinearSVMChurnModel(
            params={"C": 0.1, "random_state": 42},
            scale_features=True,
            calibrate_probabilities=True,
            calibration_cv=3,
        )
        model.fit(self.df_nonsep_X.iloc[:400], self.y_nonsep.iloc[:400])
        probs = model.predict_proba(self.df_nonsep_X.iloc[400:])

        self.assertEqual(len(probs), 100)
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        eval_metrics = model.evaluate(self.df_nonsep_X.iloc[400:], self.y_nonsep.iloc[400:])
        self.assertIn("PR_AUC", eval_metrics)
        self.assertIn("ROC_AUC", eval_metrics)
        self.assertGreater(eval_metrics["PR_AUC"], 0.0)
        print(f"[PASS] Platt calibration verified: PR_AUC={eval_metrics['PR_AUC']:.4f}, ROC_AUC={eval_metrics['ROC_AUC']:.4f}")

    def test_03_serialization_roundtrip(self):
        """Verifies model artifact saving and loading preserves state and prediction outputs."""
        tmp_dir = Path("results/test_linear_svm_temp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        save_file = str(tmp_dir / "svm_model.pkl")

        try:
            model = LinearSVMChurnModel(params={"C": 0.1, "random_state": 42}, scale_features=True)
            model.fit(self.df_nonsep_X.iloc[:400], self.y_nonsep.iloc[:400])
            orig_probs = model.predict_proba(self.df_nonsep_X.iloc[400:])
            model.save(save_file)

            loaded_model = LinearSVMChurnModel()
            loaded_model.load(save_file)
            loaded_probs = loaded_model.predict_proba(self.df_nonsep_X.iloc[400:])

            np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-5)
            print("[PASS] Linear SVM serialization roundtrip verified.")
        finally:
            if Path(save_file).exists():
                Path(save_file).unlink()
            if tmp_dir.exists():
                tmp_dir.rmdir()


class TestLinearSVMBankChurnSeparability(unittest.TestCase):
    """
    Empirical benchmark of Linear SVM, Logistic Regression, and XGBoost
    across multiple feature spaces of the Bank Customer Churn dataset.
    """

    @classmethod
    def setUpClass(cls):
        cls.data_path = Path("data/full_bank_data.csv")
        if not cls.data_path.exists():
            raise unittest.SkipTest("data/full_bank_data.csv not found, skipping empirical benchmark.")

        raw_df = pd.read_csv(cls.data_path)
        cls.df = raw_df

    def test_04_benchmark_separability_across_feature_spaces(self):
        """
        Runs empirical comparison across 3 feature spaces:
          1. Baseline Tabular (Raw Kaggle + Domain Features)
          2. Baseline + Multiplicative Interactions (quant_inter_*)
          3. Baseline + Full Quant Multi-Scale Signals + Interactions
        Measures separability metrics and compares Linear SVM with LR and XGBoost.
        """
        print("\n" + "=" * 80)
        print("EMPIRICAL LINEAR SEPARABILITY & BENCHMARK ON BANK CHURN DATASET")
        print("=" * 80)

        feature_space_configs = [
            {
                "space_name": "1. Baseline Tabular",
                "enable_quant_signal_features": False,
                "quant_enable_interactions": False,
                "quant_enable_momentum": False,
                "quant_enable_macd": False,
                "quant_enable_volatility": False,
                "quant_enable_fourier": False,
                "quant_enable_exogenous": False,
                "quant_enable_lags": False,
                "quant_enable_rolling": False,
                "quant_enable_cyclic_time": False,
                "quant_enable_multiscale_rolling": False,
            },
            {
                "space_name": "2. Baseline + Interactions",
                "enable_quant_signal_features": True,
                "quant_enable_interactions": True,
                "quant_enable_momentum": False,
                "quant_enable_macd": False,
                "quant_enable_volatility": False,
                "quant_enable_fourier": False,
                "quant_enable_exogenous": False,
                "quant_enable_lags": False,
                "quant_enable_rolling": False,
                "quant_enable_cyclic_time": False,
                "quant_enable_multiscale_rolling": False,
            },
            {
                "space_name": "3. Baseline + Full Quant & Multi-Scale",
                "enable_quant_signal_features": True,
                "quant_enable_interactions": True,
                "quant_enable_momentum": True,
                "quant_enable_macd": True,
                "quant_enable_volatility": True,
                "quant_enable_fourier": True,
                "quant_enable_exogenous": True,
                "quant_enable_lags": True,
                "quant_enable_rolling": True,
                "quant_enable_cyclic_time": True,
                "quant_enable_multiscale_rolling": True,
            },
        ]

        results = []

        for space in feature_space_configs:
            space_name = space["space_name"]
            print(f"\n--- Testing Feature Space: {space_name} ---")

            p_cfg = PipelineConfig(
                data_path=str(self.data_path),
                enable_skewness=False,  # Keep Yeo-Johnson disabled as per pipeline specs
                enable_temporal_embeddings=False,
                enable_gnn_embeddings=False,
                enable_smote=False,  # Unbiased separability diagnosis on real geometric space
                enable_quant_signal_features=space["enable_quant_signal_features"],
                quant_enable_interactions=space["quant_enable_interactions"],
                quant_enable_momentum=space["quant_enable_momentum"],
                quant_enable_macd=space["quant_enable_macd"],
                quant_enable_volatility=space["quant_enable_volatility"],
                quant_enable_fourier=space["quant_enable_fourier"],
                quant_enable_exogenous=space["quant_enable_exogenous"],
                quant_enable_lags=space["quant_enable_lags"],
                quant_enable_rolling=space["quant_enable_rolling"],
                quant_enable_cyclic_time=space["quant_enable_cyclic_time"],
                quant_enable_multiscale_rolling=space["quant_enable_multiscale_rolling"],
            )

            pipe = FeatureEngineeringPipeline(p_cfg)
            prep_res = pipe.prepare_train_test(self.df)
            X_train = prep_res.X_train
            y_train = prep_res.y_train
            X_test = prep_res.X_test
            y_test = prep_res.y_test

            n_features = X_train.shape[1]
            print(f"Feature count: {n_features}")

            # 1. Fit Linear SVM
            svm_model = LinearSVMChurnModel(
                params={
                    "C": 0.1,
                    "penalty": "l2",
                    "loss": "squared_hinge",
                    "dual": "auto",
                    "class_weight": "balanced",
                    "random_state": 42,
                    "max_iter": 5000,
                },
                scale_features=True,
                calibrate_probabilities=True,
                decision_threshold=0.35,
            )
            svm_model.fit(X_train, y_train)

            # Compute Separability Diagnostics on Training Data
            sep_train = svm_model.compute_separability_metrics(X_train, y_train)
            svm_eval = svm_model.evaluate(X_test, y_test, threshold=0.35)

            results.append({
                "Feature_Space": space_name,
                "Num_Features": n_features,
                "Model": "Linear SVM",
                "PR_AUC": svm_eval["PR_AUC"],
                "ROC_AUC": svm_eval["ROC_AUC"],
                "F1_Score": svm_eval["F1_Score"],
                "Precision": svm_eval["Precision"],
                "Recall": svm_eval["Recall"],
                "Accuracy": svm_eval["Accuracy"],
                "Geometric_Margin": sep_train["geometric_margin"],
                "Margin_Violation_Rate": sep_train["margin_violation_rate"],
                "Severe_Violation_Rate": sep_train["severe_violation_rate"],
                "Train_Misclass_Rate": sep_train["misclassification_rate"],
            })

            # 2. Fit Logistic Regression
            lr_model = LogisticRegressionChurnModel(
                params={
                    "C": 0.05,
                    "solver": "lbfgs",
                    "penalty": "l2",
                    "class_weight": "balanced",
                    "max_iter": 1000,
                    "random_state": 42,
                },
                scale_features=True,
                decision_threshold=0.35,
            )
            lr_model.fit(X_train, y_train)
            lr_eval = lr_model.evaluate(X_test, y_test, threshold=0.35)

            results.append({
                "Feature_Space": space_name,
                "Num_Features": n_features,
                "Model": "Logistic Regression",
                "PR_AUC": lr_eval["PR_AUC"],
                "ROC_AUC": lr_eval["ROC_AUC"],
                "F1_Score": lr_eval["F1_Score"],
                "Precision": lr_eval["Precision"],
                "Recall": lr_eval["Recall"],
                "Accuracy": lr_eval["Accuracy"],
                "Geometric_Margin": np.nan,
                "Margin_Violation_Rate": np.nan,
                "Severe_Violation_Rate": np.nan,
                "Train_Misclass_Rate": np.nan,
            })

            # 3. Fit XGBoost (Non-linear decision trees benchmark)
            xgb_model = XGBoostChurnModel(
                params={
                    "n_estimators": 250,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "scale_pos_weight": 2.0,
                    "random_state": 42,
                    "n_jobs": -1,
                },
                decision_threshold=0.35,
            )
            xgb_model.fit(X_train, y_train)
            xgb_eval = xgb_model.evaluate(X_test, y_test, threshold=0.35)

            results.append({
                "Feature_Space": space_name,
                "Num_Features": n_features,
                "Model": "XGBoost",
                "PR_AUC": xgb_eval["PR_AUC"],
                "ROC_AUC": xgb_eval["ROC_AUC"],
                "F1_Score": xgb_eval["F1_Score"],
                "Precision": xgb_eval["Precision"],
                "Recall": xgb_eval["Recall"],
                "Accuracy": xgb_eval["Accuracy"],
                "Geometric_Margin": np.nan,
                "Margin_Violation_Rate": np.nan,
                "Severe_Violation_Rate": np.nan,
                "Train_Misclass_Rate": np.nan,
            })

            print(f"  -> Linear SVM: PR-AUC={svm_eval['PR_AUC']:.4f}, ROC-AUC={svm_eval['ROC_AUC']:.4f}, Margin Violations={sep_train['margin_violation_rate']*100:.1f}%")
            print(f"  -> Logistic Reg: PR-AUC={lr_eval['PR_AUC']:.4f}, ROC-AUC={lr_eval['ROC_AUC']:.4f}")
            print(f"  -> XGBoost:    PR-AUC={xgb_eval['PR_AUC']:.4f}, ROC-AUC={xgb_eval['ROC_AUC']:.4f}")

        df_bench = pd.DataFrame(results)

        # Export results table
        out_csv = Path("results/linear_svm_separability_comparison.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_bench.to_csv(out_csv, index=False)
        print(f"\n[OK] Benchmark saved to: {out_csv.resolve()}")

        # Print structured summary
        print("\n" + "=" * 90)
        print("EMPIRICAL BENCHMARK SUMMARY TABLE:")
        print("=" * 90)
        display_cols = ["Feature_Space", "Model", "Num_Features", "PR_AUC", "ROC_AUC", "F1_Score", "Precision", "Recall"]
        print(df_bench[display_cols].to_string(index=False))

        # Assertions
        # 1. Verify that on raw baseline tabular data, margin violations are high (> 80%)
        base_svm = df_bench[(df_bench["Feature_Space"] == "1. Baseline Tabular") & (df_bench["Model"] == "Linear SVM")].iloc[0]
        self.assertGreater(base_svm["Margin_Violation_Rate"], 0.80, "Data should have high margin violation rate on raw space")

        # 2. Verify that multiplicative interactions dramatically increase Linear SVM PR-AUC
        inter_svm = df_bench[(df_bench["Feature_Space"] == "2. Baseline + Interactions") & (df_bench["Model"] == "Linear SVM")].iloc[0]
        self.assertGreater(inter_svm["PR_AUC"], base_svm["PR_AUC"] + 0.10, "Interactions must boost Linear SVM PR-AUC significantly")


if __name__ == "__main__":
    unittest.main()
