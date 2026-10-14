"""
Verification and demonstration script for the FeatureEngineeringPipeline and Config module.
"""

import os
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from config import PipelineConfig, load_config
from preprocessing import FeatureEngineeringPipeline


def run_checks():
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Week 5", "Project", "artifacts", "cleaned_bank_data.csv")
    )
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"[OK] Ingested dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Test Config loading from YAML
    cfg = load_config()
    print(f"[OK] Loaded config from YAML: enable_domain_features={cfg.enable_domain_features}, enable_smote={cfg.enable_smote}")

    # 2. Test Default Full Pipeline using YAML
    print("\n--- Running Full Pipeline (Loaded from config/feature_engineering.yaml) ---")
    pipe = FeatureEngineeringPipeline(cfg)
    res = pipe.prepare_train_test(df, compute_feature_importance=True)

    print(f"Train features shape (X_train): {res.X_train.shape}")
    print(f"Test features shape  (X_test):  {res.X_test.shape}")
    print(f"SMOTE Train shape    (X_smote): {res.X_train_resampled.shape}")
    print(f"Total model features: {len(res.model_features)}")

    assert res.X_train.shape == (8000, 55), f"Expected X_train shape (8000, 55), got {res.X_train.shape}"
    assert res.X_test.shape == (2000, 55), f"Expected X_test shape (2000, 55), got {res.X_test.shape}"
    assert res.X_train_resampled.shape[0] > 8000, "SMOTE should oversample minority class"
    assert res.feature_importance is not None, "Feature importance should be computed"
    print("[OK] Pipeline shapes match 04_Data_Transformation exactly!")

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
    assert len(res_ablation.model_features) == 44, f"Expected 44 features without domain features, got {len(res_ablation.model_features)}"
    assert res_ablation.X_train_resampled.shape[0] == 8000, "SMOTE was disabled, shape should remain 8000"
    print("[OK] Config-driven ablation successful!")

    print("\n[SUCCESS] All pipeline and config tests passed!")


if __name__ == "__main__":
    run_checks()
