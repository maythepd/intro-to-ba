"""
Unit tests for Cluster-Based Feature Extraction (Module 5 Segmentation Integration).
Verifies:
  1. ClusterFeatureExtractor disabled state passthrough
  2. Sub-feature group generation (centroid dists, soft probs, boundary margins, risk affinity, peer ratios, one-hot)
  3. Hierarchical clustering linkage features
  4. Dynamic fitting fallback (when saved bundle is bypassed)
  5. YAML config & CLI flag propagation
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone

from config import load_unified_config, UnifiedConfig
from preprocessing.clustering import ClusterFeatureExtractor, ClusterFeatureConfig
from preprocessing.pipeline import FeatureEngineeringPipeline
from main import build_arg_parser


class TestClusterFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load a small slice of real dataset
        cls.df = pd.read_csv("data/full_bank_data.csv", nrows=100)
        cls.target = cls.df["churn"]

    def test_01_disabled_passthrough(self):
        """When disabled, extractor must return dataframe unmodified."""
        ext = ClusterFeatureExtractor(enabled=False)
        ext.fit(self.df, self.target)
        out = ext.transform(self.df)
        self.assertEqual(out.shape, self.df.shape)
        self.assertEqual(list(out.columns), list(self.df.columns))

    def test_02_all_kmeans_subgroups_generation(self):
        """Verify that all KMeans cluster feature groups generate cleanly without NaNs."""
        ext = ClusterFeatureExtractor(
            enabled=True,
            algorithm="kmeans",
            n_clusters=4,
            use_saved_bundle=True,
            enable_centroid_distances=True,
            enable_soft_probabilities=True,
            enable_inertia_boundary=True,
            enable_risk_affinity=True,
            enable_relative_ratios=True,
            enable_cluster_one_hot=True,
            enable_hierarchical_features=False,
        )
        ext.fit(self.df, self.target)
        out = ext.transform(self.df)
        added_cols = ext.get_feature_names_out()

        self.assertGreater(len(added_cols), 30)
        self.assertIn("cluster_dist_0", added_cols)
        self.assertIn("cluster_prob_0", added_cols)
        self.assertIn("cluster_entropy", added_cols)
        self.assertIn("cluster_silhouette_margin", added_cols)
        self.assertIn("cluster_risk_affinity_ratio", added_cols)
        self.assertIn("cluster_rel_balance", added_cols)
        self.assertIn("is_cluster_0", added_cols)

        # Check for zero NaNs and zero Infs
        out_features = out[added_cols]
        self.assertFalse(out_features.isna().any().any(), "Detected NaNs in cluster features!")
        self.assertTrue(np.all(np.isfinite(out_features.to_numpy())), "Detected non-finite values in cluster features!")

    def test_03_hierarchical_subgroup(self):
        """Verify hierarchical features generation when enabled."""
        ext = ClusterFeatureExtractor(
            enabled=True,
            algorithm="both",
            enable_hierarchical_features=True,
        )
        ext.fit(self.df, self.target)
        out = ext.transform(self.df)
        added_cols = ext.get_feature_names_out()

        self.assertIn("hier_dist_0", added_cols)
        self.assertIn("hier_min_dist", added_cols)
        self.assertIn("hier_cluster_id", added_cols)
        self.assertIn("hier_kmeans_agreement", added_cols)

    def test_04_dynamic_fitting_without_bundle(self):
        """Verify that extractor can dynamically fit on training data without relying on pre-saved bundle."""
        ext = ClusterFeatureExtractor(
            enabled=True,
            n_clusters=3,
            use_saved_bundle=False,
        )
        ext.fit(self.df, self.target)
        out = ext.transform(self.df)
        added_cols = ext.get_feature_names_out()

        self.assertIn("cluster_dist_0", added_cols)
        self.assertIn("cluster_dist_1", added_cols)
        self.assertIn("cluster_dist_2", added_cols)
        self.assertEqual(len([c for c in added_cols if c.startswith("is_cluster_")]), 3)
        self.assertFalse(out[added_cols].isna().any().any())

    def test_05_config_yaml_loading(self):
        """Verify unified configuration YAML loads clustering block correctly."""
        cfg = load_unified_config("config/config.yaml")
        self.assertIsNotNone(cfg.features)
        self.assertTrue(hasattr(cfg.features, "enable_clustering_features"))
        self.assertTrue(hasattr(cfg.features, "clustering_algorithm"))
        self.assertTrue(hasattr(cfg.features, "cluster_count"))
        self.assertTrue(hasattr(cfg.features, "cluster_enable_centroid_distances"))

    def test_06_cli_parser_aliases_and_flags(self):
        """Verify main.py CLI parser supports --model, --metric, and clustering flags."""
        parser = build_arg_parser()
        args = parser.parse_args([
            "--model", "xgboost",
            "--mode", "tune",
            "--metric", "PR_AUC",
            "--enable-clustering-features",
            "--clustering-algorithm", "both",
            "--cluster-count", "4",
        ])
        self.assertEqual(args.model_type, "xgboost")
        self.assertEqual(args.mode, "tune")
        self.assertEqual(args.tuning_metric, "PR_AUC")
        self.assertTrue(args.enable_clustering_features)
        self.assertEqual(args.clustering_algorithm, "both")
        self.assertEqual(args.cluster_count, 4)

    def test_07_psychology_augmented_clustering(self):
        """Verify that ClusterFeatureExtractor incorporates psychology dimensions into its vector space."""
        from preprocessing.psychology import PsychologyFeatureEngineer

        psych_eng = PsychologyFeatureEngineer(enabled=True)
        df_psych = psych_eng.fit_transform(self.df)

        ext = ClusterFeatureExtractor(
            enabled=True,
            algorithm="both",
            n_clusters=4,
            enable_hierarchical_features=True,
            enable_relative_ratios=True,
        )
        ext.fit(df_psych, self.target)
        out = ext.transform(df_psych)

        # 1. Verify psychology dimensions entered the clustering space
        psych_in_cluster = [f for f in ext.selected_features_ if f.startswith("psych_")]
        self.assertGreater(len(psych_in_cluster), 5, "Psychology dimensions should be included in clustering space.")
        self.assertIn("psych_financial_runway_months", ext.selected_features_)
        self.assertIn("psych_phantom_login_ratio", ext.selected_features_)

        # 2. Verify output columns
        added = ext.get_feature_names_out()
        self.assertIn("cluster_dist_0", added)
        self.assertIn("hier_dist_0", added)
        self.assertIn("hier_kmeans_agreement", added)
        self.assertIn("cluster_rel_psych_financial_runway_months", added)
        self.assertIn("cluster_rel_psych_phantom_login_ratio", added)

        # 3. Cleanliness: no NaNs, finite values
        self.assertFalse(out[added].isna().any().any())
        self.assertTrue(np.all(np.isfinite(out[added].to_numpy())))
        print(f"\n[PASS] Test 7: Independent clustering successfully enriched with {len(psych_in_cluster)} psychology dimensions.")


if __name__ == "__main__":
    unittest.main()
