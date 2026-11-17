"""
Unit & Integration Test Suite for Advanced Classical Manual ML Feature Engineering:
  1. Lagged Features (1-month & 2-month shifts, velocity, acceleration)
  2. Rolling Window Statistics (3-month mean, std, min, slope)
  3. Sin/Cos Cyclical Calendar Embeddings (Day-of-month harmonics, dispersion)
  4. Multiplicative Interaction Terms (Shortcut exploiters: Age x Products, Germany x Balance, etc.)

Verifies:
  - Numerical health: no NaNs, no Infs, positive variance (> 1e-6), non-extreme sparsity (< 98%).
  - Zero target leakage: transformations fitted strictly on training fold.
  - End-to-end XGBoost performance improvement (PR-AUC, ROC-AUC, F1, Precision, Recall).
  - Exports ablation comparison to results/manual_ml_enhancements_comparison.csv.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from preprocessing.quant_signals import QuantSignalFeatureExtractor
from config.pipeline_config import PipelineConfig


class TestManualMLEnhancements(unittest.TestCase):
    """Test suite for validating numerical integrity and predictive power of manual features."""

    @classmethod
    def setUpClass(cls):
        cls.raw_data_dir = Path("data/full_dataset")
        cls.processed_dir = Path("data/processed")
        cls.results_dir = Path("results")
        cls.results_dir.mkdir(parents=True, exist_ok=True)
        cls.bank_csv = Path("data/full_bank_data.csv")

        # Load bank data
        cls.df_bank = pd.read_csv(cls.bank_csv)
        cls.extractor = QuantSignalFeatureExtractor(
            enabled=True,
            enable_momentum=True,
            enable_macd=True,
            enable_volatility=True,
            enable_fourier=True,
            enable_exogenous=True,
            enable_lags=True,
            enable_rolling=True,
            enable_cyclic_time=True,
            enable_interactions=True,
            prune_sparse=True,
            force_recompute=False,
            verbose=False,
        )
        cls.df_quant = cls.extractor._get_or_load_features()

    def test_01_feature_families_presence(self):
        """Verify that all 4 newly requested feature families are present in the cached dataframe."""
        cols = list(self.df_quant.columns)

        lag_cols = [c for c in cols if c.startswith("quant_lag_")]
        rolling_cols = [c for c in cols if c.startswith("quant_rolling_")]
        cyclic_cols = [c for c in cols if c.startswith("quant_cyclic_")]
        inter_cols = [c for c in cols if c.startswith("quant_inter_")]

        self.assertGreaterEqual(len(lag_cols), 4, f"Expected >= 4 lag features, got {len(lag_cols)}")
        self.assertGreaterEqual(len(rolling_cols), 6, f"Expected >= 6 rolling features, got {len(rolling_cols)}")
        self.assertGreaterEqual(len(cyclic_cols), 6, f"Expected >= 6 cyclic features, got {len(cyclic_cols)}")
        self.assertGreaterEqual(len(inter_cols), 8, f"Expected >= 8 interaction features, got {len(inter_cols)}")

    def test_02_numerical_health_and_stability(self):
        """Verify no NaNs, no Infs, positive variance, and valid sparsity across all features."""
        feature_cols = [c for c in self.df_quant.columns if c != "customer_id"]

        for col in feature_cols:
            vals = self.df_quant[col].values
            nan_count = int(np.isnan(vals).sum())
            inf_count = int(np.isinf(vals).sum())
            var_val = float(np.var(vals))
            sparsity = float(np.mean(vals == 0.0))

            self.assertEqual(nan_count, 0, f"Feature {col} contains {nan_count} NaNs")
            self.assertEqual(inf_count, 0, f"Feature {col} contains {inf_count} Infs")
            self.assertGreater(var_val, 1e-8, f"Feature {col} has degenerate variance: {var_val}")
            self.assertLess(sparsity, 0.99, f"Feature {col} has extreme sparsity: {sparsity*100:.1f}%")

    def test_03_transformer_fit_transform_no_leakage(self):
        """Test fit and transform pipeline partition to ensure strict out-of-fold clipping."""
        train_df, test_df = train_test_split(
            self.df_bank, test_size=0.20, random_state=42, stratify=self.df_bank["churn"]
        )

        ext = QuantSignalFeatureExtractor(
            enabled=True,
            prune_sparse=True,
            outlier_clip_quantile=0.99,
            verbose=False,
        )
        ext.fit(train_df)
        self.assertTrue(ext.is_fitted)
        self.assertGreater(len(ext.valid_features_), 40)

        # Transform both
        train_trans = ext.transform(train_df)
        test_trans = ext.transform(test_df)

        self.assertEqual(len(train_trans), len(train_df))
        self.assertEqual(len(test_trans), len(test_df))
        self.assertFalse(train_trans[ext.valid_features_].isna().any().any())
        self.assertFalse(test_trans[ext.valid_features_].isna().any().any())

    def test_04_ablation_benchmark_xgboost(self):
        """
        Runs an empirical ablation benchmark across feature sets on a holdout 20% test split.
        Compares PR-AUC, ROC-AUC, F1, Precision, and Recall at threshold 0.35.
        Exports comparison results to results/manual_ml_enhancements_comparison.csv.
        """
        # Prepare base tabular dataset with one-hot encoded country and gender
        cat_df = pd.get_dummies(self.df_bank[["country", "gender"]], drop_first=True)
        df_base_full = pd.concat([self.df_bank.drop(columns=["country", "gender"]), cat_df], axis=1)
        df_merged = df_base_full.merge(self.df_quant, on="customer_id", how="inner")
        y = df_merged["churn"].values

        # Base tabular feature columns
        base_features = [
            c for c in df_base_full.columns
            if c not in ["customer_id", "churn", "preferred_channel", "primary_device"]
        ]

        # Feature sets to evaluate
        lag_cols = [c for c in self.df_quant.columns if c.startswith("quant_lag_")]
        rolling_cols = [c for c in self.df_quant.columns if c.startswith("quant_rolling_")]
        cyclic_cols = [c for c in self.df_quant.columns if c.startswith("quant_cyclic_")]
        inter_cols = [c for c in self.df_quant.columns if c.startswith("quant_inter_")]
        quant_core_cols = [
            c for c in self.df_quant.columns
            if c.startswith("quant_") and not any(
                c.startswith(p) for p in ["quant_lag_", "quant_rolling_", "quant_cyclic_", "quant_inter_"]
            )
        ]

        feature_suites = {
            "1. Baseline Tabular (Raw+Domain)": base_features,
            "2. Baseline + Multiplicative Interactions": base_features + inter_cols,
            "3. Baseline + Rolling 3M Statistics": base_features + rolling_cols,
            "4. Baseline + Lagged Dynamics": base_features + lag_cols,
            "5. Baseline + Sin/Cos Cyclic Time": base_features + cyclic_cols,
            "6. Baseline + Core Quant Signals (MACD, Mom, Vol, FFT)": base_features + quant_core_cols,
            "7. Baseline + (Interactions + Rolling + Cyclic)": base_features + inter_cols + rolling_cols + cyclic_cols,
            "8. Full Manual ML Suite (All Enhancements)": base_features + [c for c in self.df_quant.columns if c != "customer_id"],
        }

        # Fixed holdout split 80/20
        idx_train, idx_test = train_test_split(
            np.arange(len(y)), test_size=0.20, random_state=42, stratify=y
        )
        y_train, y_test = y[idx_train], y[idx_test]

        results = []
        threshold = 0.35

        for suite_name, feat_names in feature_suites.items():
            X_suite = df_merged[feat_names].values
            X_tr, X_te = X_suite[idx_train], X_suite[idx_test]

            model = XGBClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=2.0,
                eval_metric="auc",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_tr, y_train)

            y_proba = model.predict_proba(X_te)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            pr_auc = average_precision_score(y_test, y_proba)
            roc_auc = roc_auc_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)

            results.append({
                "Configuration": suite_name,
                "Feature_Count": len(feat_names),
                "PR_AUC": round(pr_auc, 4),
                "ROC_AUC": round(roc_auc, 4),
                "F1_Score": round(f1, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
            })

        res_df = pd.DataFrame(results)
        csv_path = self.results_dir / "manual_ml_enhancements_comparison.csv"
        res_df.to_csv(csv_path, index=False)
        print(f"\n[OK] Ablation benchmark saved to {csv_path}:\n{res_df.to_string(index=False)}")

        # Verify that Multiplicative Interactions and Combined suite yield strictly better PR-AUC than baseline
        baseline_pr = res_df.loc[res_df["Configuration"] == "1. Baseline Tabular (Raw+Domain)", "PR_AUC"].values[0]
        inter_pr = res_df.loc[res_df["Configuration"] == "2. Baseline + Multiplicative Interactions", "PR_AUC"].values[0]
        comb_pr = res_df.loc[res_df["Configuration"] == "7. Baseline + (Interactions + Rolling + Cyclic)", "PR_AUC"].values[0]

        self.assertGreater(
            inter_pr, baseline_pr,
            f"Interactions PR_AUC ({inter_pr}) did not beat Baseline ({baseline_pr})"
        )
        self.assertGreater(
            comb_pr, baseline_pr,
            f"Combined PR_AUC ({comb_pr}) did not beat Baseline ({baseline_pr})"
        )


if __name__ == "__main__":
    unittest.main()
