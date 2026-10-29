"""
Verification and demonstration script for the FeatureEngineeringPipeline and Config module.
"""

import os
from pathlib import Path
import shutil
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from config import PipelineConfig, load_config
from preprocessing import pipeline, FeatureEngineeringPipeline


def resolve_data_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return str(path)
    if path.exists():
        return str(path.resolve())
    script_dir = Path(__file__).resolve().parent
    candidate = (script_dir / path).resolve()
    if candidate.exists():
        return str(candidate)
    candidate_parent = (script_dir.parent / path).resolve()
    if candidate_parent.exists():
        return str(candidate_parent)
    raise FileNotFoundError(f"Data file not found: {path_str}")


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

        expected_train = 7000 if getattr(cfg, "val_size", 0.0) > 0 else 8000
        assert res.X_train.shape[0] == expected_train, f"Expected {expected_train} train rows, got {res.X_train.shape[0]}"
        if getattr(cfg, "val_size", 0.0) > 0:
            assert res.X_val is not None and res.X_val.shape[0] == 1000, "Validation set should have 1000 rows"
        assert res.X_test.shape[0] == 2000, f"Expected 2000 test rows, got {res.X_test.shape[0]}"
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
        expected_abl_train = 7000 if getattr(ablation_cfg, "val_size", 0.0) > 0 else 8000
        assert res_ablation.X_train_resampled.shape[0] == expected_abl_train, f"SMOTE disabled, shape should remain {expected_abl_train}"
        print(f"[OK] Domain ablation verified: {len(res.model_features)} -> {len(res_ablation.model_features)} features!")

        # 4. Test Raw Kaggle Baseline
        print("\n--- Testing Raw Kaggle Baseline (use_raw_kaggle_features_only) ---")
        base_cfg = PipelineConfig.from_dict({"use_raw_kaggle_features_only": True, "enable_domain_features": False})
        base_pipe = FeatureEngineeringPipeline(base_cfg)
        base_res = base_pipe.prepare_train_test(df)
        print(f"Raw Kaggle Baseline features: {len(base_res.model_features)}")
        assert len(base_res.model_features) == 21, f"Expected 21 baseline features, got {len(base_res.model_features)}"
        print(f"[OK] Raw Kaggle baseline verified: {len(base_res.model_features)} features!")

        # 5. Test Feature Selection with MI_Score
        print("\n--- Testing Feature Selection with MI_Score ---")
        assert cfg.feature_selection_metric == "MI_Score", f"Expected MI_Score, got {cfg.feature_selection_metric}"
        assert "MI_Score" in res.feature_importance.columns, "MI_Score column should be in feature_importance"
        print(f"[OK] FeatureSelector MI_Score verified: top feature is {res.feature_importance.iloc[0]['Feature']} (MI={res.feature_importance.iloc[0]['MI_Score']:.4f})")

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

        print("\n[SUCCESS] All pipeline, splitting, MI ranking, visualization, and Optuna tests passed!")

    finally:
        cleanup_test_artifacts()


if __name__ == "__main__":
    run_checks()
