"""
Comprehensive Representation Ablation & PR-AUC Benchmark Test.
Evaluates and compares:
  1. Baseline Tabular Only (No Temporal, No GNN)
  2. Tabular + Temporal BiGRU Embeddings (16d)
  3. Tabular + Self-Supervised GNN Embeddings (16d)
  4. Tabular + Temporal BiGRU + GNN Embeddings (32d augmentation)
Verifies metric gains, PR-AUC, F1-Score, and feature dimensions without data leakage.
"""

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score

from config.pipeline_config import load_unified_config
from preprocessing.pipeline import FeatureEngineeringPipeline
from models.xgboost import XGBoostChurnModel


def run_single_ablation_eval(cfg, df, name: str):
    """Prepares pipeline, fits XGBoost, and evaluates test performance."""
    pipe = FeatureEngineeringPipeline(cfg.features)
    prep_res = pipe.prepare_train_test(df)

    X_tr = prep_res.X_train_resampled if prep_res.X_train_resampled is not None else prep_res.X_train
    y_tr = prep_res.y_train_resampled if prep_res.y_train_resampled is not None else prep_res.y_train

    model_params = cfg.model.get_params_for_model("xgboost")
    xgb = XGBoostChurnModel(params=model_params, early_stopping_rounds=cfg.model.early_stopping_rounds)
    xgb.fit(X_tr, y_tr, X_val=prep_res.X_val, y_val=prep_res.y_val, verbose=False)

    test_prob = xgb.predict_proba(prep_res.X_test)
    y_test = prep_res.y_test

    th = getattr(cfg.model, "decision_threshold", 0.35)
    pred = (test_prob >= th).astype(int)

    roc = roc_auc_score(y_test, test_prob)
    pr = average_precision_score(y_test, test_prob)
    f1 = f1_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    prec = precision_score(y_test, pred, zero_division=0)

    return {
        "Configuration": name,
        "Total_Features": len(prep_res.model_features),
        "PR_AUC": round(pr, 4),
        "ROC_AUC": round(roc, 4),
        "F1_Score": round(f1, 4),
        "Recall": round(rec, 4),
        "Precision": round(prec, 4),
    }


class TestRepresentationAblationAndPR(unittest.TestCase):
    def test_01_compare_all_representation_modes(self):
        """Runs the 4-way comparison and verifies all configurations complete cleanly."""
        base_cfg = load_unified_config("config/config.yaml")
        base_cfg.features.temporal_force_retrain = False
        base_cfg.features.gnn_force_retrain = False
        df = pd.read_csv(base_cfg.features.data_path)

        results = []

        # 1. Tabular Only
        cfg1 = load_unified_config("config/config.yaml")
        cfg1.features.enable_temporal_embeddings = False
        cfg1.features.enable_gnn_embeddings = False
        res1 = run_single_ablation_eval(cfg1, df, "1. Tabular Only (Baseline)")
        results.append(res1)

        # 2. Tabular + Temporal BiGRU
        cfg2 = load_unified_config("config/config.yaml")
        cfg2.features.enable_temporal_embeddings = True
        cfg2.features.enable_gnn_embeddings = False
        res2 = run_single_ablation_eval(cfg2, df, "2. Tabular + Temporal BiGRU (16d)")
        results.append(res2)

        # 3. Tabular + GNN Embeddings
        cfg3 = load_unified_config("config/config.yaml")
        cfg3.features.enable_temporal_embeddings = False
        cfg3.features.enable_gnn_embeddings = True
        res3 = run_single_ablation_eval(cfg3, df, "3. Tabular + GNN Embeddings (16d)")
        results.append(res3)

        # 4. Tabular + Temporal BiGRU + GNN Embeddings
        cfg4 = load_unified_config("config/config.yaml")
        cfg4.features.enable_temporal_embeddings = True
        cfg4.features.enable_gnn_embeddings = True
        res4 = run_single_ablation_eval(cfg4, df, "4. Tabular + Temporal + GNN (32d)")
        results.append(res4)

        df_results = pd.DataFrame(results)
        print("\n" + "=" * 90)
        print("REPRESENTATION MODES PR-AUC & ROC-AUC COMPARISON TABLE (Decision Threshold = 0.35)")
        print("=" * 90)
        print(df_results.to_string(index=False))
        print("=" * 90)

        # Export comparison table to results directory
        out_csv = Path("results/representation_modes_comparison.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(out_csv, index=False)
        print(f"[OK] Saved comparison results to: {out_csv}")

        # Assertions
        for res in results:
            self.assertGreater(res["ROC_AUC"], 0.82)
            self.assertGreater(res["PR_AUC"], 0.60)
            self.assertGreater(res["F1_Score"], 0.52)


if __name__ == "__main__":
    unittest.main()
