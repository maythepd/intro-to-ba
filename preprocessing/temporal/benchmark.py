"""
Benchmarking Module for Temporal Behavioral Embeddings.
Compares downstream performance of baseline tabular features vs.
tabular features augmented with dense longitudinal temporal embeddings under strict inductive isolation.
"""

import copy
from pathlib import Path
import time
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from .manager import TemporalEmbeddingManager


def _make_benchmark_model(model_type: str, random_state: int = 42):
    m = model_type.lower()
    if m == "xgboost":
        from models.xgboost import XGBoostChurnModel
        return XGBoostChurnModel(params={"random_state": random_state, "n_jobs": -1})
    elif m in ["lr", "logistic_regression"]:
        from models.logistic_regression import LogisticRegressionChurnModel
        return LogisticRegressionChurnModel(params={"random_state": random_state, "max_iter": 1000})
    elif m in ["rf", "random_forest"]:
        from models.random_forest import RandomForestChurnModel
        return RandomForestChurnModel(params={"random_state": random_state, "n_jobs": -1})
    else:
        raise ValueError(f"Unknown model_type for temporal benchmark: {model_type}")


def run_temporal_representation_benchmark(
    df: pd.DataFrame,
    pipeline_obj: Any,
    output_dir: str = "results/temporal",
    model_type: str = "xgboost",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluates downstream model performance under strict inductive isolation:
      1. Baseline: Pure domain tabular features.
      2. Augmented: Domain tabular features + 16D Bi-GRU Temporal Embeddings
                    (encoder trained strictly on Train partition, inferred on Test partition).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print("TEMPORAL BEHAVIORAL SEQUENCE EMBEDDING BENCHMARK (STRICT INDUCTIVE)")
        print("Extracting 6-Month Event Sequences (Transactions, Logins, Complaints)...")
        print("=" * 70)

    cfg = pipeline_obj.config

    # 1. Benchmark Baseline (without temporal embeddings)
    if verbose:
        print("\n--- Training Baseline (Without Temporal Embeddings) ---")
    cfg_base = copy.deepcopy(cfg)
    cfg_base.enable_temporal_embeddings = False
    pipe_base = type(pipeline_obj)(cfg_base)
    res_base = pipe_base.prepare_train_test(df)

    m_base = _make_benchmark_model(model_type, random_state=cfg.random_state)
    m_base.fit(res_base.X_train_resampled, res_base.y_train_resampled, verbose=False)
    metrics_base = m_base.evaluate(res_base.X_test, res_base.y_test)

    # 2. Benchmark Augmented (with temporal embeddings strictly fit on Train & inferred on Test)
    latent_d = getattr(cfg, "temporal_latent_dim", 16)
    if verbose:
        print(f"\n--- Training Augmented Model (+ {latent_d}D Bi-GRU Temporal Embeddings z) ---")
    cfg_aug = copy.deepcopy(cfg)
    cfg_aug.enable_temporal_embeddings = True
    pipe_aug = type(pipeline_obj)(cfg_aug, fold_id="benchmark")
    res_aug = pipe_aug.prepare_train_test(df)

    m_aug = _make_benchmark_model(model_type, random_state=cfg.random_state)
    m_aug.fit(res_aug.X_train_resampled, res_aug.y_train_resampled, verbose=False)
    metrics_aug = m_aug.evaluate(res_aug.X_test, res_aug.y_test)

    # 3. Compare results
    model_disp = model_type.upper()
    rows = [
        {"Model": f"{model_disp} (Baseline Tabular - {len(res_base.model_features)} features)", **metrics_base},
        {"Model": f"{model_disp} (+ {latent_d}D Temporal Embeddings - {len(res_aug.model_features)} features)", **metrics_aug},
    ]
    df_comp = pd.DataFrame(rows)

    csv_path = out_path / "temporal_embedding_comparison.csv"
    df_comp.to_csv(csv_path, index=False)

    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK COMPARISON RESULTS:")
        print("=" * 70)
        print(df_comp[["Model", "ROC_AUC", "PR_AUC", "F1_Score", "Recall", "Precision"]])
        print(f"\n[OK] Results saved to: {csv_path}")

    return df_comp
