"""
Benchmark suite comparing non-linear and neural representation learning methods.
No entrypoint (__main__) is present here; invoked via main.py --mode representation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression

from .manager import NonlinearRepresentationManager


def run_representation_benchmark(
    df: pd.DataFrame,
    pipeline_obj: Any,
    output_dir: str = "results/visualizations",
    model_type: str = "lr",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluates baseline features vs. multiple non-linear representation paradigms:
      - Raw Baseline (Engineered Tabular Features)
      - Kernel Nystroem (RBF Hilbert space embedding)
      - Tabular Denoising Autoencoder (Bottleneck Latent)
      - Supervised TabNet Embedding (Class-discriminative)
      - Contrastive TabNet (Metric Hypersphere)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Prepare base train/val/test splits without non-linear representations
    if hasattr(pipeline_obj, "config"):
        orig_setting = getattr(pipeline_obj.config, "allow_nonlinear_representations", False)
        pipeline_obj.config.allow_nonlinear_representations = False
    
    res = pipeline_obj.prepare_train_test(df)
    X_tr = res.X_train
    y_tr = res.y_train.values if hasattr(res.y_train, "values") else np.asarray(res.y_train)
    X_te = res.X_test
    y_te = res.y_test.values if hasattr(res.y_test, "values") else np.asarray(res.y_test)

    methods = [
        ("Baseline (Engineered)", None),
        ("Nystroem Kernel (RBF)", "kernel"),
        ("Denoising Autoencoder (DAE)", "autoencoder"),
        ("Supervised TabNet Latent", "supervised"),
        ("Supervised Contrastive TabNet", "contrastive"),
    ]

    records: List[Dict[str, Any]] = []

    for label, method in methods:
        if verbose:
            print(f"\n>>> Benchmarking Representation: [{label}]...")

        if method is None:
            cur_X_tr = X_tr.values
            cur_X_te = X_te.values
        else:
            mgr = NonlinearRepresentationManager(
                enabled=True,
                method=method,
                mode="augment",
                latent_dim=16,
                kernel_components=50,
                epochs=15,
                checkpoint_dir=out_path / "checkpoints",
            )
            mgr.fit(X_tr, y_tr)
            cur_X_tr = mgr.transform(X_tr).values
            cur_X_te = mgr.transform(X_te).values

        # Evaluate with Logistic Regression
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(cur_X_tr, y_tr)
        probs = clf.predict_proba(cur_X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)

        roc = roc_auc_score(y_te, probs)
        pr = average_precision_score(y_te, probs)
        f1 = f1_score(y_te, preds)
        rec = recall_score(y_te, preds)
        prec = precision_score(y_te, preds)

        records.append({
            "Representation": label,
            "Method": method or "none",
            "Features_Count": cur_X_tr.shape[1],
            "ROC_AUC": round(float(roc), 4),
            "PR_AUC": round(float(pr), 4),
            "F1_Score": round(float(f1), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
        })

    df_bench = pd.DataFrame(records)
    csv_file = out_path / "representation_benchmark_results.csv"
    df_bench.to_csv(csv_file, index=False)

    if verbose:
        print("\n" + "=" * 70)
        print("REPRESENTATION LEARNING BENCHMARK SUMMARY")
        print("=" * 70)
        print(df_bench.to_string(index=False))
        print(f"\n[OK] Benchmark results saved to: {csv_file}")

    return df_bench

