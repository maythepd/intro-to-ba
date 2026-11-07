"""
Verification and demonstration script for the FeatureEngineeringPipeline and Config module.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
try:
    import joblib
    from joblib.externals.loky.backend import context
    context._count_physical_cores = lambda: 4
except Exception:
    pass

from pathlib import Path
import shutil
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from config import PipelineConfig, load_config
from preprocessing import pipeline, FeatureEngineeringPipeline
from utils import resolve_data_path


def cleanup_test_artifacts():
    """Purge temporary test files and directories created during test runs."""
    test_paths = [
        Path("results/test_feature_importance.png"),
        Path("results/test_tune"),
    ]
    for p in test_paths:
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                p.unlink(missing_ok=True)
        except Exception:
            pass


def run_checks():
    cleanup_test_artifacts()
    try:
        # 1. Test Config loading from YAML & Dataset loading via pipeline
        cfg = load_config()
        print(f"[OK] Loaded config from YAML: enable_domain_features={cfg.enable_domain_features}, enable_smote={cfg.enable_smote}")

        df = pipeline.load_dataset(cfg, verbose=False)
        print(f"[OK] Ingested dataset via pipeline.load_dataset: {df.shape[0]} rows, {df.shape[1]} columns")

        # 2. Test Default Full Pipeline using YAML
        print("\n--- Running Full Pipeline (Loaded from config/config.yaml) ---")
        pipe = pipeline.FeatureEngineeringPipeline(cfg)
        res = pipe.prepare_train_test(df, compute_feature_importance=True)

        print(f"Train features shape (X_train): {res.X_train.shape}")
        if res.X_val is not None:
            print(f"Val features shape   (X_val):   {res.X_val.shape}")
        print(f"Test features shape  (X_test):  {res.X_test.shape}")
        print(f"SMOTE Train shape    (X_smote): {res.X_train_resampled.shape}")
        print(f"Total model features: {len(res.model_features)}")

        total_rows = df.shape[0]
        expected_test = int(round(total_rows * cfg.test_size))
        expected_val = int(round(total_rows * getattr(cfg, "val_size", 0.0))) if getattr(cfg, "val_size", 0.0) > 0 else 0
        expected_train = total_rows - expected_test - expected_val

        assert abs(res.X_train.shape[0] - expected_train) <= 2, f"Expected ~{expected_train} train rows, got {res.X_train.shape[0]}"
        if getattr(cfg, "val_size", 0.0) > 0:
            assert abs(res.X_val.shape[0] - expected_val) <= 2, f"Validation set should have ~{expected_val} rows"
        assert abs(res.X_test.shape[0] - expected_test) <= 2, f"Expected ~{expected_test} test rows, got {res.X_test.shape[0]}"
        assert res.X_train.shape[1] == len(res.model_features), "X_train columns should match model_features"
        assert res.X_train_resampled.shape[0] > res.X_train.shape[0], "SMOTE should oversample minority class"
        assert res.feature_importance is not None, "Feature importance should be computed"
        print("[OK] Pipeline feature generation, validation split, and SMOTE verified successfully!")

        # 3. Test Ablation via Config (e.g. disabling domain features & smote)
        print("\n--- Running Ablation via Config Override ---")
        ablation_cfg = PipelineConfig.from_dict({
            "modules": {
                "enable_domain_features": False,
                "enable_smote": False,
            }
        })
        ablation_pipe = FeatureEngineeringPipeline(ablation_cfg)
        res_ablation = ablation_pipe.prepare_train_test(df)

        print(f"Ablation Train shape (X_train): {res_ablation.X_train.shape}")
        print(f"Ablation Total model features: {len(res_ablation.model_features)}")
        assert len(res_ablation.model_features) < len(res.model_features), "Ablation should reduce feature count"
        abl_test = int(round(total_rows * getattr(ablation_cfg, "test_size", cfg.test_size)))
        abl_val = int(round(total_rows * getattr(ablation_cfg, "val_size", getattr(cfg, "val_size", 0.0)))) if getattr(ablation_cfg, "val_size", getattr(cfg, "val_size", 0.0)) > 0 else 0
        expected_abl_train = total_rows - abl_test - abl_val
        assert abs(res_ablation.X_train_resampled.shape[0] - expected_abl_train) <= 2, f"SMOTE disabled, shape should remain ~{expected_abl_train}"
        print(f"[OK] Domain ablation verified: {len(res.model_features)} -> {len(res_ablation.model_features)} features!")

        # 3b. Test Individual Feature Group Ablations
        print("\n--- Testing Individual Feature Group Ablations ---")
        for group_name, group_cfg in [
            ("Behavioral Momentum", {"enable_behavioral_momentum": False}),
            ("Customer Friction", {"enable_customer_friction": False}),
            ("Product Portfolio", {"enable_product_bundling": False}),
            ("Risk Interactions", {"enable_risk_interactions": False}),
        ]:
            g_pipe = FeatureEngineeringPipeline(PipelineConfig.from_dict(group_cfg))
            g_res = g_pipe.prepare_train_test(df)
            print(f"Group Ablation [{group_name} Disabled]: {len(g_res.model_features)} features")
            assert len(g_res.model_features) < len(res.model_features), f"{group_name} ablation should reduce feature count"
        print("[OK] Individual feature group ablations verified successfully!")

        # 4. Test Raw Kaggle Baseline
        print("\n--- Testing Raw Kaggle Baseline (use_raw_kaggle_features_only) ---")
        base_cfg = PipelineConfig.from_dict({"use_raw_kaggle_features_only": True, "enable_domain_features": False})
        base_pipe = FeatureEngineeringPipeline(base_cfg)
        base_res = base_pipe.prepare_train_test(df)
        print(f"Raw Kaggle Baseline features: {len(base_res.model_features)}")
        assert len(base_res.model_features) == 21, f"Expected 21 baseline features, got {len(base_res.model_features)}"
        print(f"[OK] Raw Kaggle baseline verified: {len(base_res.model_features)} features!")

        # 5. Test Feature Selection with MI_Score, Pearson, and MI_Pearson_Hybrid
        print("\n--- Testing Feature Selection with Pearson, MI, and Hybrid ---")
        assert "MI_Score" in res.feature_importance.columns, "MI_Score column should be in feature_importance"
        assert "Pearson" in res.feature_importance.columns, "Pearson column should be in feature_importance"
        assert "MI_Pearson_Hybrid" in res.feature_importance.columns, "MI_Pearson_Hybrid column should be in feature_importance"
        top_feat = res.feature_importance.iloc[0]
        print(f"[OK] FeatureSelector verified: top feature is {top_feat['Feature']} (MI={top_feat['MI_Score']:.4f}, Pearson={top_feat['Pearson']:.4f}, Hybrid={top_feat['MI_Pearson_Hybrid']:.4f})")

        # 6. Test Visualization Plotting
        print("\n--- Testing Visualization Plotting ---")
        from visualization import plot_feature_importance, plot_model_comparison
        test_plot_path = Path("results/test_feature_importance.png")
        test_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_feature_importance(res.feature_importance, test_plot_path, model_name="TestModel", top_k=10)
        assert test_plot_path.exists() and test_plot_path.stat().st_size > 0, "Feature importance plot should be saved"
        test_plot_path.unlink()
        print("[OK] Feature importance plotting verified and test plot cleaned up!")

        # 7. Test DataSplitter K-Fold CV Generator
        print("\n--- Testing DataSplitter Stratified K-Fold CV ---")
        from preprocessing import DataSplitter
        splitter = DataSplitter(n_splits=5, stratify=True, random_state=42)
        folds = list(splitter.get_cv_folds(df, target_col="churn"))
        assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"
        for fold in folds:
            assert fold.train_df.shape[0] == 8000, f"Expected 8000 train rows per fold, got {fold.train_df.shape[0]}"
            assert fold.val_df.shape[0] == 2000, f"Expected 2000 val rows per fold, got {fold.val_df.shape[0]}"
        print("[OK] DataSplitter 5-fold CV splits verified successfully!")

        # 8. Test OptunaTuner Smoke Test (1 trial, 2 folds)
        print("\n--- Testing OptunaTuner Smoke Run ---")
        from models import OptunaTuner
        tuner = OptunaTuner(model_type="xgboost", cv_folds=2, n_trials=1, random_state=42)
        res_tune = tuner.tune(df.head(1000), n_trials=1, output_dir="results/test_tune", verbose=False)
        assert res_tune.best_cv_score > 0, "Tuning best score should be > 0"
        assert len(res_tune.best_params) > 0, "Best params should not be empty"
        print("[OK] OptunaTuner smoke run verified successfully!")

        # 9. Test & Export Processed Data to data/processed
        print("\n--- Exporting Main Processed Dataset to data/processed ---")
        exported = pipe.export_processed_data(res, output_dir="data/processed", verbose=True)
        assert Path("data/processed/train.csv").exists(), "data/processed/train.csv should exist"
        assert Path("data/processed/test.csv").exists(), "data/processed/test.csv should exist"
        assert Path("data/processed/train_smote.csv").exists(), "data/processed/train_smote.csv should exist"
        assert Path("data/processed/full_transformed.csv").exists(), "data/processed/full_transformed.csv should exist"
        assert Path("data/processed/feature_metadata.json").exists(), "data/processed/feature_metadata.json should exist"
        print("[OK] Main pipeline processed dataset exported cleanly to data/processed!")

        # 10. Test Temporal Sequence Event Representation Sub-package (preprocessing.temporal)
        print("\n--- Testing Temporal Behavioral Sequence Representation Package ---")
        from preprocessing.temporal import (
            TemporalSnapshotExtractor,
            BiGRUTemporalAutoencoder,
            TemporalAutoencoderTrainer,
            TemporalEmbeddingManager,
        )

        test_cust_ids = df["customer_id"].head(200).values
        extractor = TemporalSnapshotExtractor(raw_data_dir="data/full_dataset")
        tensor, extracted_ids = extractor.extract(test_cust_ids, verbose=False)
        assert tensor.shape == (200, 6, 10), f"Expected tensor shape (200, 6, 10), got {tensor.shape}"
        assert len(extracted_ids) == 200, f"Expected 200 customer IDs, got {len(extracted_ids)}"
        print(f"[OK] TemporalSnapshotExtractor verified: extracted 6-month sequence tensor {tensor.shape}")

        # Test BiGRUTemporalAutoencoder
        test_ckpt = Path("results/test_tune/checkpoints/test_temporal_encoder.pt")
        trainer = TemporalAutoencoderTrainer(
            in_features=10,
            seq_len=6,
            hidden_dim=32,
            latent_dim=8,
            epochs=2,
            batch_size=64,
            checkpoint_path=test_ckpt,
            verbose=False,
        )
        trainer.fit(tensor)
        latents = trainer.transform(tensor)
        assert latents.shape == (200, 8), f"Expected latents shape (200, 8), got {latents.shape}"
        assert test_ckpt.exists(), "Temporal checkpoint should be saved to disk"
        print(f"[OK] BiGRUTemporalAutoencoder verified: encoded sequence tensor into latent embeddings {latents.shape}")

        # Test TemporalEmbeddingManager
        temp_mgr = TemporalEmbeddingManager(
            enabled=True,
            raw_data_dir="data/full_dataset",
            latent_dim=8,
            epochs=2,
            batch_size=64,
            checkpoint_dir="results/test_tune/checkpoints",
            cache_path="results/test_tune/test_temporal_embeddings.csv",
            verbose=False,
        )
        sample_df = df.head(100)[["customer_id", "credit_score", "age"]]
        temp_mgr.fit(sample_df)
        trans_df = temp_mgr.transform(sample_df)
        assert "temporal_z0" in trans_df.columns, "temporal_z0 should be in augmented dataframe"
        assert "temporal_z7" in trans_df.columns, "temporal_z7 should be in augmented dataframe"
        assert trans_df.shape[1] == sample_df.shape[1] + 8, f"Expected {sample_df.shape[1] + 8} columns, got {trans_df.shape[1]}"
        print(f"[OK] TemporalEmbeddingManager verified: augmented customer table to {trans_df.shape}")

        # Test true forward-pass inference on unseen validation customers
        unseen_df = df.iloc[100:150][["customer_id", "credit_score", "age"]]
        trans_unseen = temp_mgr.transform(unseen_df)
        assert trans_unseen.shape[1] == unseen_df.shape[1] + 8
        assert not (trans_unseen["temporal_z0"] == 0.0).all(), "Unseen validation customers should receive inferred embeddings, not zeros!"
        print(f"[OK] TemporalEmbeddingManager verified: inferred latent embeddings on unseen validation customers {trans_unseen.shape}")

        print("\n[SUCCESS] All pipeline, splitting, Pearson/MI hybrid selection, Optuna, and Temporal Representation tests passed!")

    finally:
        cleanup_test_artifacts()


if __name__ == "__main__":
    run_checks()
