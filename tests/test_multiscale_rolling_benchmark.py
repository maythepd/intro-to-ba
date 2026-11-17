"""
Benchmark & Ablation Test Suite for Multi-Resolution Rolling Windows (7d, 14d, 30d, 60d, 90d).
Evaluates whether adding high-resolution short-term windows improves PR-AUC, ROC-AUC, F1, Precision, and Recall.
Pure Classical ML (Zero Deep Learning).
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from preprocessing.quant_signals import QuantSignalFeatureExtractor


class TestMultiScaleRollingBenchmark(unittest.TestCase):
    """Test suite for validating predictive improvement of multi-resolution rolling features."""

    @classmethod
    def setUpClass(cls):
        cls.results_dir = Path("results")
        cls.results_dir.mkdir(parents=True, exist_ok=True)
        cls.bank_csv = Path("data/full_bank_data.csv")
        cls.df_bank = pd.read_csv(cls.bank_csv)

        cls.extractor = QuantSignalFeatureExtractor(
            enabled=True,
            force_recompute=False,
            verbose=False,
        )
        cls.df_quant = cls.extractor._get_or_load_features()

    def test_01_multiscale_features_numerical_health(self):
        """Verifies no NaNs, no Infs, positive variance, and valid sparsity across multiscale features."""
        multiscale_cols = [c for c in self.df_quant.columns if "multiscale" in c]
        self.assertGreaterEqual(len(multiscale_cols), 35, f"Expected >= 35 multiscale features, got {len(multiscale_cols)}")

        for col in multiscale_cols:
            vals = self.df_quant[col].values
            self.assertEqual(int(np.isnan(vals).sum()), 0, f"{col} has NaNs")
            self.assertEqual(int(np.isinf(vals).sum()), 0, f"{col} has Infs")
            self.assertGreater(float(np.var(vals)), 1e-8, f"{col} has zero variance")

    def test_02_multiscale_ablation_benchmark(self):
        """
        Runs empirical benchmark comparing Baseline, Multiplicative Interactions,
        and Multi-Scale Rolling Windows. Exports to results/multiscale_rolling_comparison.csv.
        """
        cat_df = pd.get_dummies(self.df_bank[["country", "gender"]], drop_first=True)
        df_base_full = pd.concat([self.df_bank.drop(columns=["country", "gender"]), cat_df], axis=1)
        df_merged = df_base_full.merge(self.df_quant, on="customer_id", how="inner")
        y = df_merged["churn"].values

        base_features = [
            c for c in df_base_full.columns
            if c not in ["customer_id", "churn", "preferred_channel", "primary_device"]
        ]

        multiscale_cols = [c for c in self.df_quant.columns if "multiscale" in c]
        inter_cols = [c for c in self.df_quant.columns if c.startswith("quant_inter_")]
        cyclic_cols = [c for c in self.df_quant.columns if c.startswith("quant_cyclic_")]
        lag_cols = [c for c in self.df_quant.columns if c.startswith("quant_lag_")]
        quant_rolling_cols = [c for c in self.df_quant.columns if c.startswith("quant_rolling_")]

        feature_suites = {
            "1. Baseline Tabular (Raw+Domain)": base_features,
            "2. Baseline + Multiplicative Interactions": base_features + inter_cols,
            "3. Baseline + Multi-Scale High-Res Rolling (7d..90d)": base_features + multiscale_cols,
            "4. Baseline + Multi-Scale + Interactions": base_features + multiscale_cols + inter_cols,
            "5. Baseline + Multi-Scale + Interactions + Cyclic": base_features + multiscale_cols + inter_cols + cyclic_cols,
            "6. Baseline + Multi-Scale + Interactions + Cyclic + Lags": base_features + multiscale_cols + inter_cols + cyclic_cols + lag_cols,
            "7. Full Classical Suite (All 108 Quant & Multi-Scale)": base_features + [c for c in self.df_quant.columns if c != "customer_id"],
        }

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
        csv_path = self.results_dir / "multiscale_rolling_comparison.csv"
        res_df.to_csv(csv_path, index=False)
        print(f"\n[OK] Multi-Scale Benchmark saved to {csv_path}:\n{res_df.to_string(index=False)}")

        # Verify that Multi-Scale + Interactions beats baseline
        baseline_pr = res_df.loc[res_df["Configuration"] == "1. Baseline Tabular (Raw+Domain)", "PR_AUC"].values[0]
        comb_pr = res_df.loc[res_df["Configuration"] == "5. Baseline + Multi-Scale + Interactions + Cyclic", "PR_AUC"].values[0]
        self.assertGreater(comb_pr, baseline_pr)


if __name__ == "__main__":
    unittest.main()
