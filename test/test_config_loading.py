"""
Unit tests for configuration loading and validation.
Verifies that all sections and parameters in config/config.yaml are loaded accurately:
1. Dataset & Partition Settings (12 fields)
2. High-level Feature Toggles (8 fields)
3. Domain Features, Skewness, Scaling, Categorical & Balancing (27 fields)
4. Advanced Feature Modules: Sample Weighting, Feature Selection & Temporal Embeddings (23 fields)
5. Model Configuration: XGBoost, LR, RF & Focal Loss
6. Experiment & Reporting Paths & Templates
7. Optuna Hyperparameter Tuning Settings
8. Roundtrip YAML Serialization/Deserialization
9. Standalone PipelineConfig Loading Compatibility
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

import yaml
from config.pipeline_config import (
    UnifiedConfig,
    PipelineConfig,
    ModelConfig,
    ExperimentConfig,
    TuningConfig,
    load_config,
    load_unified_config,
)

TEMP_TEST_DIR = Path("results/test_config_temp")


class TestConfigLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load raw YAML and UnifiedConfig instance."""
        if TEMP_TEST_DIR.exists():
            shutil.rmtree(TEMP_TEST_DIR, ignore_errors=True)
        TEMP_TEST_DIR.mkdir(parents=True, exist_ok=True)

        cls.config_path = PROJECT_ROOT / "config" / "config.yaml"
        with open(cls.config_path, "r", encoding="utf-8") as f:
            cls.raw = yaml.safe_load(f)

        cls.cfg = load_unified_config(cls.config_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test artifacts."""
        if TEMP_TEST_DIR.exists():
            shutil.rmtree(TEMP_TEST_DIR, ignore_errors=True)
            print(f"\n[OK] Cleaned up temporary test artifacts at: {TEMP_TEST_DIR}")

    def test_01_dataset_section_loading(self):
        """Test 1: Verify all 12 dataset & partition settings match YAML exactly."""
        raw_ds = self.raw["dataset"]
        feat = self.cfg.features

        self.assertEqual(feat.data_path, raw_ds["data_path"])
        self.assertEqual(feat.raw_full_dataset_dir, raw_ds["raw_full_dataset_dir"])
        self.assertEqual(feat.rebuild_full_dataset, raw_ds["rebuild_full_dataset"])
        self.assertEqual(feat.target_col, raw_ds["target_col"])
        self.assertEqual(feat.group_col, raw_ds["group_col"])
        self.assertEqual(feat.test_size, raw_ds["test_size"])
        self.assertEqual(feat.val_size, raw_ds["val_size"])
        self.assertEqual(feat.enable_cv, raw_ds["enable_cv"])
        self.assertEqual(feat.cv_folds, raw_ds["cv_folds"])
        self.assertEqual(feat.random_state, raw_ds["random_state"])
        self.assertEqual(feat.stratify, raw_ds["stratify"])
        self.assertEqual(feat.exclude_cols, raw_ds["exclude_cols"])
        print("[PASS] Test 1: Dataset & Partition configurations (12 fields) loaded accurately.")

    def test_02_features_high_level_toggles(self):
        """Test 2: Verify all 8 high-level feature engineering toggles match YAML."""
        raw_feat = self.raw["features"]
        feat = self.cfg.features

        toggles = [
            "enable_skewness",
            "enable_scaling",
            "enable_domain_features",
            "enable_one_hot_encoding",
            "enable_target_encoding",
            "enable_smote",
            "enable_feature_selection",
            "enable_temporal_embeddings",
        ]
        for t in toggles:
            self.assertEqual(
                getattr(feat, t),
                raw_feat[t],
                f"Mismatch for toggle: {t}",
            )
        print(f"[PASS] Test 2: High-level feature toggles ({len(toggles)} fields) loaded accurately.")

    def test_03_features_submodules_parameters(self):
        """Test 3: Verify Skewness, Scaling, Domain Features, Categorical & Balancing submodules."""
        raw_f = self.raw["features"]
        feat = self.cfg.features

        # Skewness transform
        for k, v in raw_f["skewness_transform"].items():
            self.assertEqual(getattr(feat, k), v, f"Mismatch in skewness_transform.{k}")

        # Scaling
        self.assertEqual(feat.apply_robust_scaling, raw_f["scaling"]["apply_robust_scaling"])
        self.assertEqual(feat.apply_standard_scaling, raw_f["scaling"]["apply_standard_scaling"])
        self.assertEqual(feat.robust_features, raw_f["scaling"]["robust_features"])
        self.assertEqual(feat.standard_features, raw_f["scaling"]["standard_features"])

        # Domain features
        for k, v in raw_f["domain_features"].items():
            self.assertEqual(getattr(feat, k), v, f"Mismatch in domain_features.{k}")

        # Categorical encoding
        self.assertEqual(feat.categorical_cols, raw_f["categorical_encoding"]["categorical_cols"])
        self.assertEqual(feat.target_encoding_m_weight, raw_f["categorical_encoding"]["target_encoding_m_weight"])
        self.assertEqual(feat.ohe_drop_first, raw_f["categorical_encoding"]["ohe_drop_first"])

        # Balancing
        self.assertEqual(feat.smote_sampling_strategy, raw_f["balancing"]["smote_sampling_strategy"])
        self.assertEqual(feat.smote_k_neighbors, raw_f["balancing"]["smote_k_neighbors"])
        print("[PASS] Test 3: Skewness, Scaling, Domain, Categorical & Balancing (27 fields) loaded accurately.")

    def test_04_advanced_feature_modules(self):
        """Test 4: Verify Sample Weighting, Feature Selection, and Temporal Embeddings."""
        raw_f = self.raw["features"]
        feat = self.cfg.features

        # Sample weighting
        raw_sw = raw_f["sample_weighting"]
        self.assertEqual(feat.enable_sample_weighting, raw_sw["enabled"])
        self.assertEqual(feat.weight_decay_mode, raw_sw["decay_mode"])
        self.assertEqual(feat.weight_recency_col, raw_sw["recency_col"])
        self.assertEqual(feat.weight_decay_half_life_days, raw_sw["decay_half_life_days"])
        self.assertEqual(feat.weight_enable_value_weighting, raw_sw["enable_value_weighting"])
        self.assertEqual(feat.weight_balance_col, raw_sw["balance_col"])
        self.assertEqual(feat.weight_value_weight_alpha, raw_sw["value_weight_alpha"])
        self.assertEqual(feat.weight_churn_multiplier, raw_sw["churn_weight_multiplier"])
        self.assertEqual(feat.weight_normalize, raw_sw["normalize"])

        # Feature selection
        raw_fs = raw_f["feature_selection"]
        self.assertEqual(feat.top_k_features, raw_fs["top_k_features"])
        self.assertEqual(feat.feature_selection_metric, raw_fs["feature_selection_metric"])
        self.assertEqual(feat.hybrid_pearson_weight, raw_fs["hybrid_pearson_weight"])
        self.assertEqual(feat.compute_feature_importance, raw_fs["compute_feature_importance"])

        # Temporal embeddings
        raw_temp = raw_f["temporal_embeddings"]
        self.assertEqual(feat.enable_temporal_embeddings, raw_temp["enabled"])
        self.assertEqual(feat.temporal_latent_dim, raw_temp["latent_dim"])
        self.assertEqual(feat.temporal_hidden_dim, raw_temp["hidden_dim"])
        self.assertEqual(feat.temporal_num_layers, raw_temp["num_layers"])
        self.assertEqual(feat.temporal_epochs, raw_temp["epochs"])
        self.assertEqual(feat.temporal_batch_size, raw_temp["batch_size"])
        self.assertEqual(feat.temporal_learning_rate, raw_temp["learning_rate"])
        self.assertEqual(feat.temporal_dropout, raw_temp["dropout"])
        self.assertEqual(feat.temporal_checkpoint_dir, raw_temp["checkpoint_dir"])
        self.assertEqual(feat.temporal_cache_path, raw_temp["cache_path"])
        if "manifold_strategy" in raw_temp:
            self.assertEqual(feat.temporal_manifold_strategy, raw_temp["manifold_strategy"])
            self.assertEqual(feat.temporal_margin_churn, raw_temp["margin_churn"])
            self.assertEqual(feat.temporal_margin_loyal, raw_temp["margin_loyal"])
            self.assertEqual(feat.temporal_angular_scale, raw_temp["angular_scale"])
            self.assertEqual(feat.temporal_lambda_angular, raw_temp["lambda_angular"])
        print("[PASS] Test 4: Sample Weighting, Feature Selection & Temporal Embeddings (28 fields) loaded accurately.")

    def test_05_model_section_loading(self):
        """Test 5: Verify Model parameters (XGBoost, LR, RF) and Focal Loss settings."""
        raw_m = self.raw["model"]
        model = self.cfg.model

        self.assertEqual(model.name, raw_m["name"])
        self.assertEqual(model.early_stopping_rounds, raw_m["early_stopping_rounds"])

        # Focal Loss
        self.assertEqual(model.use_focal_loss, raw_m["focal_loss"]["enabled"])
        self.assertEqual(model.focal_gamma, raw_m["focal_loss"]["gamma"])
        self.assertEqual(model.focal_alpha, raw_m["focal_loss"]["alpha"])

        # Hyperparameter dicts
        self.assertEqual(model.xgboost_params, raw_m["xgboost_params"])
        self.assertEqual(model.lr_params, raw_m["lr_params"])
        self.assertEqual(model.rf_params, raw_m["rf_params"])

        # Param resolver helper
        xgb_resolved = model.get_params_for_model("xgboost")
        self.assertEqual(xgb_resolved["n_estimators"], raw_m["xgboost_params"]["n_estimators"])
        self.assertEqual(xgb_resolved["max_depth"], raw_m["xgboost_params"]["max_depth"])

        lr_resolved = model.get_params_for_model("lr")
        self.assertEqual(lr_resolved["C"], raw_m["lr_params"]["C"])
        self.assertEqual(lr_resolved["solver"], raw_m["lr_params"]["solver"])

        rf_resolved = model.get_params_for_model("rf")
        self.assertEqual(rf_resolved["n_estimators"], raw_m["rf_params"]["n_estimators"])
        self.assertEqual(rf_resolved["max_depth"], raw_m["rf_params"]["max_depth"])
        print("[PASS] Test 5: Model configurations & parameter unpacking (XGBoost, LR, RF, Focal Loss) verified.")

    def test_06_experiment_section_loading(self):
        """Test 6: Verify Experiment paths, templates and dynamic resolution methods."""
        raw_e = self.raw["experiment"]
        exp = self.cfg.experiment

        self.assertEqual(exp.output_dir, raw_e["output_dir"])
        self.assertEqual(exp.save_model, raw_e["save_model"])
        self.assertEqual(exp.model_save_path, raw_e["model_save_path"])
        self.assertEqual(exp.report_filename, raw_e["report_filename"])

        # Dynamic template resolution
        self.assertTrue(exp.get_model_save_path("xgboost").endswith(".json"))
        self.assertTrue(exp.get_model_save_path("rf").endswith(".pkl"))
        self.assertEqual(exp.get_report_filename("compare"), "model_comparison_report.md")
        self.assertEqual(exp.get_report_filename("tune", "xgboost"), "tune_xgboost_report.md")
        print("[PASS] Test 6: Experiment paths and dynamic formatters verified.")

    def test_07_tuning_section_loading(self):
        """Test 7: Verify Optuna Tuning settings."""
        raw_t = self.raw["tuning"]
        tune = self.cfg.tuning

        self.assertEqual(tune.enabled, raw_t["enabled"])
        self.assertEqual(tune.n_trials, raw_t["n_trials"])
        self.assertEqual(tune.timeout, raw_t["timeout"])
        self.assertEqual(tune.metric, raw_t["metric"])
        self.assertEqual(tune.cv_folds, raw_t["cv_folds"])
        self.assertEqual(tune.study_name, raw_t["study_name"])
        self.assertEqual(tune.direction, raw_t["direction"])
        self.assertEqual(tune.random_state, raw_t["random_state"])
        print("[PASS] Test 7: Optuna tuning configurations (8 fields) loaded accurately.")

    def test_08_roundtrip_yaml_serialization(self):
        """Test 8: Verify lossless roundtrip serialization (to_yaml -> from_yaml)."""
        roundtrip_file = TEMP_TEST_DIR / "roundtrip_test.yaml"
        self.cfg.to_yaml(roundtrip_file)
        self.assertTrue(roundtrip_file.exists(), "Exported YAML file should exist.")

        loaded_cfg = UnifiedConfig.from_yaml(roundtrip_file)

        # Verify key fields preserved
        self.assertEqual(loaded_cfg.features.data_path, self.cfg.features.data_path)
        self.assertEqual(loaded_cfg.features.test_size, self.cfg.features.test_size)
        self.assertEqual(loaded_cfg.features.enable_temporal_embeddings, self.cfg.features.enable_temporal_embeddings)
        self.assertEqual(loaded_cfg.features.temporal_latent_dim, self.cfg.features.temporal_latent_dim)
        self.assertEqual(loaded_cfg.features.top_k_features, self.cfg.features.top_k_features)
        self.assertEqual(loaded_cfg.model.xgboost_params, self.cfg.model.xgboost_params)
        self.assertEqual(loaded_cfg.experiment.output_dir, self.cfg.experiment.output_dir)
        self.assertEqual(loaded_cfg.tuning.metric, self.cfg.tuning.metric)
        self.assertEqual(loaded_cfg.tuning.cv_folds, self.cfg.tuning.cv_folds)

        roundtrip_file.unlink()
        print("[PASS] Test 8: Lossless roundtrip YAML serialization verified.")

    def test_09_standalone_pipeline_config_loading(self):
        """Test 9: Verify load_config() directly loads PipelineConfig with complete fidelity."""
        pipe_cfg = load_config(self.config_path)
        self.assertIsInstance(pipe_cfg, PipelineConfig)
        self.assertEqual(pipe_cfg.data_path, self.raw["dataset"]["data_path"])
        self.assertEqual(pipe_cfg.test_size, self.raw["dataset"]["test_size"])
        self.assertEqual(pipe_cfg.val_size, self.raw["dataset"]["val_size"])
        expected_temporal = self.raw["features"].get("enable_temporal_embeddings", True)
        self.assertEqual(pipe_cfg.enable_temporal_embeddings, expected_temporal)
        self.assertEqual(pipe_cfg.temporal_latent_dim, 16)
        self.assertEqual(pipe_cfg.top_k_features, self.raw["features"]["feature_selection"]["top_k_features"])
        self.assertEqual(pipe_cfg.feature_selection_metric, self.raw["features"]["feature_selection"]["feature_selection_metric"])
        self.assertEqual(pipe_cfg.temporal_manifold_strategy, self.raw["features"]["temporal_embeddings"]["manifold_strategy"])
        print("[PASS] Test 9: Standalone PipelineConfig loading (via load_config) verified.")


if __name__ == "__main__":
    unittest.main()

