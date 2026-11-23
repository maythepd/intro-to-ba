"""
Unit tests for PCA & LDA Feature Extraction modules.
Verifies:
  1. PCAFeatureExtractor: disabled passthrough, component projection, reconstruction error, Mahalanobis distance.
  2. LDAFeatureExtractor: disabled passthrough, Fisher canonical projection, posterior probabilities,
     log-odds, centroid distances, and anti-leakage protection.
  3. FeatureEngineeringPipeline integration with PCA and LDA steps.
  4. Config loading & CLI argument parsing for PCA and LDA flags.
"""

import unittest
import numpy as np
import pandas as pd

from config import load_unified_config, UnifiedConfig
from preprocessing.projections.pca_features import PCAFeatureExtractor
from preprocessing.projections.lda_features import LDAFeatureExtractor
from preprocessing.pipeline import FeatureEngineeringPipeline
from main import build_arg_parser


class TestPCALDAFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load a small slice of real dataset
        cls.df = pd.read_csv("data/full_bank_data.csv", nrows=100)
        cls.target = cls.df["churn"]

    def test_01_pca_disabled_passthrough(self):
        """When disabled, PCA extractor returns dataframe unmodified."""
        ext = PCAFeatureExtractor(enabled=False)
        ext.fit(self.df)
        out = ext.transform(self.df)
        self.assertEqual(out.shape, self.df.shape)
        self.assertEqual(list(out.columns), list(self.df.columns))

    def test_02_pca_feature_generation(self):
        """Verify PCA extractor generates components, reconstruction error, and Mahalanobis distance."""
        ext = PCAFeatureExtractor(
            enabled=True,
            n_components=5,
            whiten=False,
            enable_reconstruction_error=True,
            enable_mahalanobis=True,
        )
        ext.fit(self.df)
        out = ext.transform(self.df)
        added_cols = ext.get_feature_names_out()

        self.assertIn("pca_component_1", added_cols)
        self.assertIn("pca_component_5", added_cols)
        self.assertIn("pca_reconstruction_error", added_cols)
        self.assertIn("pca_relative_recon_error", added_cols)
        self.assertIn("pca_mahalanobis_dist", added_cols)

        out_features = out[added_cols]
        self.assertFalse(out_features.isna().any().any(), "Detected NaNs in PCA features!")
        self.assertTrue(np.all(np.isfinite(out_features.to_numpy())), "Detected non-finite values in PCA features!")

    def test_03_lda_disabled_passthrough(self):
        """When disabled, LDA extractor returns dataframe unmodified."""
        ext = LDAFeatureExtractor(enabled=False)
        ext.fit(self.df, self.target)
        out = ext.transform(self.df)
        self.assertEqual(out.shape, self.df.shape)
        self.assertEqual(list(out.columns), list(self.df.columns))

    def test_04_lda_feature_generation(self):
        """Verify supervised Fisher LDA feature extraction generates projection, probs, log-odds, and distances."""
        ext = LDAFeatureExtractor(
            enabled=True,
            solver="lsqr",
            shrinkage="auto",
            enable_probabilities=True,
            enable_log_odds=True,
            enable_centroid_distances=True,
        )
        ext.fit(self.df, self.target)
        out = ext.transform(self.df)
        added_cols = ext.get_feature_names_out()

        self.assertIn("lda_projection_churn", added_cols)
        self.assertIn("lda_prob_churn", added_cols)
        self.assertIn("lda_log_odds", added_cols)
        self.assertIn("lda_dist_to_loyal_centroid", added_cols)
        self.assertIn("lda_dist_to_churn_centroid", added_cols)
        self.assertIn("lda_relative_churn_proximity", added_cols)

        out_features = out[added_cols]
        self.assertFalse(out_features.isna().any().any(), "Detected NaNs in LDA features!")
        self.assertTrue(np.all(np.isfinite(out_features.to_numpy())), "Detected non-finite values in LDA features!")

    def test_05_lda_missing_labels_anti_leakage(self):
        """When labels y are not passed to fit, LDA skips gracefully and transform passes unmodified data."""
        ext = LDAFeatureExtractor(enabled=True)
        # Attempt fit with y=None
        ext.fit(self.df, y=None)
        out = ext.transform(self.df)
        self.assertEqual(out.shape, self.df.shape)
        self.assertEqual(len(ext.get_feature_names_out()), 0)

    def test_06_pipeline_integration(self):
        """Verify FeatureEngineeringPipeline end-to-end with PCA and LDA features enabled."""
        cfg = load_unified_config("config/config.yaml")
        cfg.features.enable_pca_features = True
        cfg.features.pca_n_components = 3
        cfg.features.enable_lda_features = True
        cfg.features.lda_solver = "lsqr"
        cfg.features.lda_shrinkage = "auto"
        cfg.features.enable_feature_selection = False

        pipeline = FeatureEngineeringPipeline(cfg.features)
        transformed = pipeline.fit_transform(self.df)

        self.assertIn("pca_component_1", transformed.columns)
        self.assertIn("pca_reconstruction_error", transformed.columns)
        self.assertIn("lda_projection_churn", transformed.columns)
        self.assertIn("lda_prob_churn", transformed.columns)
        self.assertIn("lda_log_odds", transformed.columns)

        # Test leak-free prepare_train_test
        res = pipeline.prepare_train_test(self.df)
        self.assertIn("pca_component_1", res.X_train.columns)
        self.assertIn("pca_component_1", res.X_test.columns)
        self.assertIn("lda_projection_churn", res.X_train.columns)
        self.assertIn("lda_projection_churn", res.X_test.columns)
        self.assertEqual(list(res.X_train.columns), list(res.X_test.columns))

    def test_07_cli_arguments(self):
        """Verify CLI parser correctly parses PCA and LDA command-line options."""
        parser = build_arg_parser()
        args = parser.parse_args([
            "--enable-pca-features",
            "--pca-components", "7",
            "--enable-lda-features",
            "--lda-solver", "eigen",
            "--lda-shrinkage", "auto",
        ])
        self.assertTrue(args.enable_pca_features)
        self.assertEqual(args.pca_components, 7)
        self.assertTrue(args.enable_lda_features)
        self.assertEqual(args.lda_solver, "eigen")
        self.assertEqual(args.lda_shrinkage, "auto")


if __name__ == "__main__":
    unittest.main()
