"""
Test Suite for Classical Quantitative & Signal Processing Feature Engineering.
Validates:
  1. Feature extraction completeness and mathematical health (no NaNs, no Infs).
  2. Pruning logic for degenerate/sparse features and robust outlier clipping.
  3. Flexible sub-toggles (momentum, macd, volatility, fourier, exogenous).
  4. End-to-end FeatureEngineeringPipeline integration and downstream XGBoost evaluation.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score

from config.pipeline_config import load_unified_config
from preprocessing.pipeline import FeatureEngineeringPipeline
from preprocessing.quant_signals import QuantSignalFeatureExtractor
from models.xgboost import XGBoostChurnModel


class TestQuantSignalFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_unified_config("config/config.yaml")
        cls.df = pd.read_csv(cls.cfg.features.data_path)

    def test_01_quant_feature_extraction_completeness_and_health(self):
        """Test 1: Verify all 37 quant features extract cleanly with zero NaNs and valid distributions."""
        extractor = QuantSignalFeatureExtractor(
            enabled=True,
            force_recompute=False,
            verbose=False,
        )
        full_df = extractor._get_or_load_features()
        self.assertGreaterEqual(full_df.shape[0], 10000)
        self.assertIn("customer_id", full_df.columns)

        quant_cols = [c for c in full_df.columns if c.startswith("quant_")]
        self.assertGreaterEqual(len(quant_cols), 30)

        # Health checks: No NaNs, No Infs
        self.assertFalse(full_df[quant_cols].isnull().any().any(), "Quant features contain NaNs!")
        self.assertFalse(np.isinf(full_df[quant_cols].values).any(), "Quant features contain Infinite values!")

        # Verify variance of every feature is positive
        for col in quant_cols:
            v = full_df[col].values
            var_val = np.var(v)
            self.assertGreater(var_val, 0.0, f"Feature {col} has zero variance!")

        print(f"\n[PASS] Test 1: All {len(quant_cols)} Quant features verified mathematically healthy (zero NaNs, positive variance).")

    def test_02_pruning_and_outlier_clipping_logic(self):
        """Test 2: Verify degenerate/sparse features are pruned and training bounds clip outliers."""
        extractor = QuantSignalFeatureExtractor(
            enabled=True,
            prune_sparse=True,
            max_sparsity_threshold=0.90,
            min_variance_threshold=1e-5,
            outlier_clip_quantile=0.99,
            verbose=False,
        )
        # Mock cached DataFrame with a healthy feature, a near-constant feature, and a 99% sparse feature
        N = 1000
        mock_cids = np.arange(1001, 1001 + N)
        mock_df = pd.DataFrame({
            "customer_id": mock_cids,
            "quant_healthy_feature": np.random.normal(10.0, 2.0, N),
            "quant_constant_feature": np.ones(N) * 5.0,  # var = 0 -> MUST BE PRUNED
            "quant_sparse_feature": np.concatenate([np.zeros(990), np.ones(10)]),  # 99% zeros -> MUST BE PRUNED
        })
        extractor.cached_df_ = mock_df

        dummy_train = pd.DataFrame({"customer_id": mock_cids[:800], "churn": np.random.randint(0, 2, 800)})
        extractor.fit(dummy_train)

        self.assertIn("quant_healthy_feature", extractor.valid_features_)
        self.assertNotIn("quant_constant_feature", extractor.valid_features_)
        self.assertNotIn("quant_sparse_feature", extractor.valid_features_)

        # Test clipping in transform
        dummy_test = pd.DataFrame({"customer_id": mock_cids[800:]})
        # Inject an extreme outlier in test data
        extractor.cached_df_.loc[extractor.cached_df_["customer_id"] == 1801, "quant_healthy_feature"] = 9999.0
        transformed = extractor.transform(dummy_test)

        high_bound = extractor.clip_bounds_["quant_healthy_feature"][1]
        test_val = transformed.loc[transformed["customer_id"] == 1801, "quant_healthy_feature"].values[0]
        self.assertLessEqual(test_val, high_bound + 1e-5, "Outlier was not clipped to training quantile bound!")

        print("[PASS] Test 2: Pruning and quantile clipping logic verified cleanly.")

    def test_03_flexible_sub_toggles(self):
        """Test 3: Verify disabling sub-modules turns off corresponding feature families."""
        extractor = QuantSignalFeatureExtractor(
            enabled=True,
            enable_momentum=False,  # Momentum disabled
            enable_macd=True,
            enable_volatility=True,
            enable_fourier=True,
            enable_exogenous=True,
            force_recompute=True,
            verbose=False,
        )
        feats_df = extractor._extract_all_features()
        cols = feats_df.columns.tolist()

        has_momentum = any(c.startswith("quant_momentum_") for c in cols)
        has_macd = any(c.startswith("quant_macd_") for c in cols)
        self.assertFalse(has_momentum, "Momentum features should not exist when enable_momentum=False!")
        self.assertTrue(has_macd, "MACD features should exist when enable_macd=True!")

        print("[PASS] Test 3: Sub-toggle granularity verified cleanly.")

    def test_04_pipeline_integration_and_downstream_xgboost_benchmark(self):
        """Test 4: Evaluate pipeline with Quant Signal Features on XGBoost vs Baseline."""
        # 1. Baseline Run (Quant Features Disabled)
        cfg_base = load_unified_config("config/config.yaml")
        cfg_base.features.enable_quant_signal_features = False
        cfg_base.features.enable_temporal_embeddings = False
        cfg_base.features.enable_gnn_embeddings = False

        pipe_base = FeatureEngineeringPipeline(cfg_base.features, verbose=False)
        prep_base = pipe_base.prepare_train_test(self.df)

        xgb_base = XGBoostChurnModel(params=cfg_base.model.get_params_for_model("xgboost"))
        X_tr_b = prep_base.X_train_resampled if prep_base.X_train_resampled is not None else prep_base.X_train
        y_tr_b = prep_base.y_train_resampled if prep_base.y_train_resampled is not None else prep_base.y_train
        xgb_base.fit(X_tr_b, y_tr_b, X_val=prep_base.X_val, y_val=prep_base.y_val, verbose=False)

        th = getattr(cfg_base.model, "decision_threshold", 0.35)
        probs_b = xgb_base.predict_proba(prep_base.X_test)
        preds_b = (probs_b >= th).astype(int)

        pr_b = average_precision_score(prep_base.y_test, probs_b)
        roc_b = roc_auc_score(prep_base.y_test, probs_b)
        f1_b = f1_score(prep_base.y_test, preds_b, zero_division=0)
        rec_b = recall_score(prep_base.y_test, preds_b, zero_division=0)
        prec_b = precision_score(prep_base.y_test, preds_b, zero_division=0)

        # 2. Augmented Run (Quant Features Enabled)
        cfg_quant = load_unified_config("config/config.yaml")
        cfg_quant.features.enable_quant_signal_features = True
        cfg_quant.features.enable_temporal_embeddings = False
        cfg_quant.features.enable_gnn_embeddings = False

        pipe_quant = FeatureEngineeringPipeline(cfg_quant.features, verbose=False)
        prep_quant = pipe_quant.prepare_train_test(self.df)

        xgb_quant = XGBoostChurnModel(params=cfg_quant.model.get_params_for_model("xgboost"))
        X_tr_q = prep_quant.X_train_resampled if prep_quant.X_train_resampled is not None else prep_quant.X_train
        y_tr_q = prep_quant.y_train_resampled if prep_quant.y_train_resampled is not None else prep_quant.y_train
        xgb_quant.fit(X_tr_q, y_tr_q, X_val=prep_quant.X_val, y_val=prep_quant.y_val, verbose=False)

        probs_q = xgb_quant.predict_proba(prep_quant.X_test)
        preds_q = (probs_q >= th).astype(int)

        pr_q = average_precision_score(prep_quant.y_test, probs_q)
        roc_q = roc_auc_score(prep_quant.y_test, probs_q)
        f1_q = f1_score(prep_quant.y_test, preds_q, zero_division=0)
        rec_q = recall_score(prep_quant.y_test, preds_q, zero_division=0)
        prec_q = precision_score(prep_quant.y_test, preds_q, zero_division=0)

        results = [
            {
                "Configuration": "1. Baseline (Tabular Only)",
                "Features_Count": len(prep_base.model_features),
                "PR_AUC": round(pr_b, 4),
                "ROC_AUC": round(roc_b, 4),
                "F1_th35": round(f1_b, 4),
                "Recall_th35": round(rec_b, 4),
                "Prec_th35": round(prec_b, 4),
            },
            {
                "Configuration": "2. Tabular + Quant Signal Features",
                "Features_Count": len(prep_quant.model_features),
                "PR_AUC": round(pr_q, 4),
                "ROC_AUC": round(roc_q, 4),
                "F1_th35": round(f1_q, 4),
                "Recall_th35": round(rec_q, 4),
                "Prec_th35": round(prec_q, 4),
            },
        ]
        df_comp = pd.DataFrame(results)
        print("\n" + "=" * 95)
        print("QUANTITATIVE & SIGNAL PROCESSING FEATURES: XGBOOST ABLATION BENCHMARK (Threshold = 0.35)")
        print("=" * 95)
        print(df_comp.to_string(index=False))
        print("=" * 95)

        out_csv = Path("results/quant_signals_xgboost_comparison.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_comp.to_csv(out_csv, index=False)
        print(f"[OK] Saved comparison results to: {out_csv}")

        # Assertions
        self.assertGreater(len(prep_quant.model_features), len(prep_base.model_features))
        self.assertGreater(pr_q, 0.60)
        self.assertGreater(roc_q, 0.84)


if __name__ == "__main__":
    unittest.main()
