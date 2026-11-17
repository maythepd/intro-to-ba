"""
Benchmark Test: Representation Learning Ablation for Logistic Regression (LR).
Evaluates the impact of:
  1. Tabular Only Baseline (LR)
  2. Tabular + Temporal BiGRU Embeddings (16d) (LR)
  3. Tabular + Self-Supervised GNN Embeddings (16d) (LR)
  4. Tabular + Temporal + GNN Joint Embeddings (32d) (LR)

Analyzes whether non-linear manifold embeddings (BiGRU + GNN) alleviate
the linear separability deficit of Logistic Regression on churn prediction.
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
from models.logistic_regression import LogisticRegressionChurnModel


def run_single_lr_ablation(cfg, df, name: str, threshold: float = 0.35):
    """Fits FeatureEngineeringPipeline and evaluates Logistic Regression on test holdout."""
    pipe = FeatureEngineeringPipeline(cfg.features)
    prep_res = pipe.prepare_train_test(df)

    X_tr = prep_res.X_train_resampled if prep_res.X_train_resampled is not None else prep_res.X_train
    y_tr = prep_res.y_train_resampled if prep_res.y_train_resampled is not None else prep_res.y_train

    model_params = cfg.model.get_params_for_model("lr")
    lr = LogisticRegressionChurnModel(
        params=model_params,
        decision_threshold=threshold,
        scale_features=True,
    )
    lr.fit(X_tr, y_tr, X_val=prep_res.X_val, y_val=prep_res.y_val, verbose=False)

    test_prob = lr.predict_proba(prep_res.X_test)
    y_test = prep_res.y_test

    pred = (test_prob >= threshold).astype(int)

    roc = roc_auc_score(y_test, test_prob)
    pr = average_precision_score(y_test, test_prob)
    f1 = f1_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    prec = precision_score(y_test, pred, zero_division=0)

    # Also calculate metrics at default threshold 0.50 for comparison
    pred_50 = (test_prob >= 0.50).astype(int)
    f1_50 = f1_score(y_test, pred_50, zero_division=0)
    rec_50 = recall_score(y_test, pred_50, zero_division=0)
    prec_50 = precision_score(y_test, pred_50, zero_division=0)

    return {
        "Configuration": name,
        "Total_Features": len(prep_res.model_features),
        "PR_AUC": round(pr, 4),
        "ROC_AUC": round(roc, 4),
        "F1_th35": round(f1, 4),
        "Recall_th35": round(rec, 4),
        "Prec_th35": round(prec, 4),
        "F1_th50": round(f1_50, 4),
        "Recall_th50": round(rec_50, 4),
        "Prec_th50": round(prec_50, 4),
    }


class TestLRRepresentationBenchmark(unittest.TestCase):
    def test_01_lr_representation_ablation(self):
        """Runs the 4 representation modes on Logistic Regression and outputs comparison table."""
        base_cfg = load_unified_config("config/config.yaml")
        base_cfg.features.temporal_force_retrain = False
        base_cfg.features.gnn_force_retrain = False
        df = pd.read_csv(base_cfg.features.data_path)

        th = getattr(base_cfg.model, "decision_threshold", 0.35)
        results = []

        # Mode 1: Tabular Only
        cfg1 = load_unified_config("config/config.yaml")
        cfg1.features.enable_temporal_embeddings = False
        cfg1.features.enable_gnn_embeddings = False
        res1 = run_single_lr_ablation(cfg1, df, "1. Tabular Only (Baseline LR)", threshold=th)
        results.append(res1)

        # Mode 2: Tabular + Temporal BiGRU
        cfg2 = load_unified_config("config/config.yaml")
        cfg2.features.enable_temporal_embeddings = True
        cfg2.features.enable_gnn_embeddings = False
        res2 = run_single_lr_ablation(cfg2, df, "2. Tabular + Temporal BiGRU (16d)", threshold=th)
        results.append(res2)

        # Mode 3: Tabular + GNN Embeddings
        cfg3 = load_unified_config("config/config.yaml")
        cfg3.features.enable_temporal_embeddings = False
        cfg3.features.enable_gnn_embeddings = True
        res3 = run_single_lr_ablation(cfg3, df, "3. Tabular + GNN Embeddings (16d)", threshold=th)
        results.append(res3)

        # Mode 4: Tabular + Temporal + GNN Embeddings
        cfg4 = load_unified_config("config/config.yaml")
        cfg4.features.enable_temporal_embeddings = True
        cfg4.features.enable_gnn_embeddings = True
        res4 = run_single_lr_ablation(cfg4, df, "4. Tabular + Temporal + GNN (32d)", threshold=th)
        results.append(res4)

        df_res = pd.DataFrame(results)
        print("\n" + "=" * 110)
        print(f"LOGISTIC REGRESSION: 4-WAY REPRESENTATION LEARNING BENCHMARK (Operating Threshold theta={th})")
        print("=" * 110)
        print(df_res.to_string(index=False))
        print("=" * 110)

        out_csv = Path("results/lr_representation_modes_comparison.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(out_csv, index=False)
        print(f"[OK] Saved Logistic Regression comparison results to: {out_csv}")

        # Basic sanity checks
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertGreater(r["ROC_AUC"], 0.70)
            self.assertGreater(r["PR_AUC"], 0.40)


if __name__ == "__main__":
    unittest.main()
