"""
Main CLI Entrypoint for Digital Bank Churn Prediction.
Orchestrates training, ablation studies, model comparison, tuning, and representation benchmarking.
Core utilities, experiment runners, and report generation are modularized in utils/.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Environment defaults for multiprocessing and plotting
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import pandas as pd

from config import load_unified_config, UnifiedConfig
from models import OptunaTuner
from models.recall_eval import export_lift_artifacts
from preprocessing import pipeline
from utils import (
    resolve_data_path,
    MODEL_REGISTRY,
    AnyChurnModel,
    build_model,
    _build_model,
    run_single_experiment,
    run_model_comparison,
    run_ablation_suite,
    generate_report,
    generate_ablation_report,
    generate_comparison_report,
    generate_train_report,
    generate_markdown_report,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds and returns the command line argument parser."""
    parser = argparse.ArgumentParser(description="Digital Bank Churn Prediction Entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to unified config YAML file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="compare",
        choices=["ablation", "train", "compare", "tune", "recall", "separability", "eda", "representation"],
        help=(
            "Run mode: "
            "'compare' (LR vs RF vs XGBoost, same pipeline), "
            "'ablation' (feature ablation on one model), "
            "'tune' (Optuna hyperparameter tuning on CV), "
            "'train' (single model training), "
            "'recall' (threshold optimization, high-recall analysis & baseline lift comparison), "
            "'separability' (PCA, t-SNE, bar charts, boxplots, KDE distributions & separability metrics), "
            "'representation' (Bi-GRU Temporal Behavioral Sequence Embeddings benchmark)"
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=["xgboost", "lr", "rf"],
        help="Model to use in 'ablation', 'train', 'tune', or 'recall' mode (ignored in 'compare' mode)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of trials for Optuna hyperparameter optimization",
    )
    parser.add_argument(
        "--tuning-metric",
        type=str,
        default=None,
        choices=["PR_AUC", "ROC_AUC", "F1_Score", "Recall", "F2_Score", "Precision", "recall"],
        help="Target metric for Optuna tuning (PR_AUC | ROC_AUC | F1_Score | Recall | F2_Score | Precision)",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.75,
        help="Target minimum recall for threshold optimization in recall mode (default: 0.75)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Manually specified decision threshold for evaluation in recall mode",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (auto-routed to results/{rf,lr,xgboost,comparison}/ if omitted)",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default=None,
        help="Custom filename for the generated markdown report (auto-resolved from config if omitted)",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Force rebuild of full dataset from raw multi-table relational files in data/full_dataset/",
    )
    parser.add_argument(
        "--sample-weights",
        action="store_true",
        help="Enable sample weighting with exponential recency decay and financial value weighting (Phương án 1)",
    )
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        help="Enable Focal Loss objective for gradient models (XGBoost, Logistic Regression) (Phương án 2)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=None,
        help="Focusing parameter gamma for Focal Loss (default: 2.0)",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Positive class balance weight alpha for Focal Loss (default: 0.25)",
    )
    parser.add_argument(
        "--decay-half-life",
        type=float,
        default=None,
        help="Half-life days for exponential recency decay in sample weighting (default: 60.0)",
    )
    parser.add_argument(
        "--enable-temporal-embeddings",
        action="store_true",
        help="Enable multi-snapshot Bi-GRU event sequence embeddings (transactions, logins, complaints)",
    )
    parser.add_argument(
        "--temporal-epochs",
        type=int,
        default=None,
        help="Number of epochs for temporal sequence autoencoder training (default: 25)",
    )
    parser.add_argument(
        "--temporal-dim",
        type=int,
        default=None,
        help="Latent embedding dimension z for temporal representation (default: 16)",
    )
    parser.add_argument(
        "--force-retrain-temporal",
        action="store_true",
        help="Force re-training of Bi-GRU temporal sequence encoders from scratch, ignoring cached checkpoints",
    )
    parser.add_argument(
        "--feature-selection-metric",
        type=str,
        default=None,
        choices=["MI_Score", "Pearson", "MI_Pearson_Hybrid", "RF_Importance"],
        help="Metric for feature ranking/selection (MI_Score | Pearson | MI_Pearson_Hybrid | RF_Importance)",
    )
    parser.add_argument(
        "--hybrid-pearson-weight",
        type=float,
        default=None,
        help="Weight of Pearson vs MI in MI_Pearson_Hybrid (0.0 to 1.0, default: 0.50)",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=None,
        help="Top K features to retain when feature selection is enabled",
    )
    # Legacy alias for backward compatibility
    parser.add_argument(
        "--allow-nonlinear-representations",
        action="store_true",
        help="(Legacy alias for --enable-temporal-embeddings)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # 1. Load Unified Config
    cfg = load_unified_config(args.config)
    print(f"[OK] Loaded unified configuration from: {args.config}")

    # Apply Sample Weighting overrides if passed via CLI
    if args.sample_weights:
        cfg.features.enable_sample_weighting = True
        print("[INFO] Sample Weighting enabled (Exponential Recency Decay + Financial Value Weighting)")
    if args.decay_half_life is not None:
        cfg.features.weight_decay_half_life_days = args.decay_half_life

    # Apply Focal Loss overrides if passed via CLI
    if args.focal_loss:
        cfg.model.use_focal_loss = True
        print("[INFO] Focal Loss objective enabled for gradient models")
    if args.focal_gamma is not None:
        cfg.model.focal_gamma = args.focal_gamma
    if args.focal_alpha is not None:
        cfg.model.focal_alpha = args.focal_alpha

    # Apply Feature Selection overrides if passed via CLI
    if args.feature_selection_metric is not None:
        cfg.features.feature_selection_metric = args.feature_selection_metric
        print(f"[INFO] Feature Selection metric set to: {args.feature_selection_metric}")
    if args.hybrid_pearson_weight is not None:
        cfg.features.hybrid_pearson_weight = args.hybrid_pearson_weight
    if args.top_k_features is not None:
        cfg.features.top_k_features = args.top_k_features
        cfg.features.enable_feature_selection = True
        print(f"[INFO] Feature Selection enabled: Top-{args.top_k_features} features")

    # Apply Temporal Behavioral Embeddings overrides if passed via CLI
    if args.enable_temporal_embeddings or args.allow_nonlinear_representations:
        cfg.features.enable_temporal_embeddings = True
        print("[INFO] Temporal Behavioral Sequence Embeddings (Bi-GRU Autoencoder) enabled via CLI")
    if args.temporal_dim is not None:
        cfg.features.temporal_latent_dim = args.temporal_dim
    if args.temporal_epochs is not None:
        cfg.features.temporal_epochs = args.temporal_epochs
    if args.force_retrain_temporal:
        cfg.features.temporal_force_retrain = True
        print("[INFO] Force re-training enabled for Temporal Sequence Encoders (ignoring checkpoints)")

    # 2. Ingest / load dataset via pipeline (auto-orchestrates ETL if needed)
    force_rebuild = args.rebuild_dataset or getattr(cfg.features, "rebuild_full_dataset", False)
    df = pipeline.load_dataset(cfg.features, force_rebuild=force_rebuild)
    print(f"     Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 3. Determine structured output subfolder:
    # results/comparison/ for 'compare', results/{model_type}/ for 'ablation' and 'train'
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(cfg.experiment.get_output_dir(mode=args.mode, model_type=args.model_type))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve report filename dynamically based on mode and model_type
    report_filename = args.report_name or cfg.experiment.get_report_filename(
        mode=args.mode,
        model_type=args.model_type,
    )
    report_path = str(output_dir / report_filename)

    # 4. Dispatch based on mode
    if args.mode == "compare":
        print("\n=======================================================")
        print("STARTING MODEL COMPARISON: LR vs RF vs XGBoost")
        print("=======================================================")
        df_results, models_dict, importances_dict = run_model_comparison(df, cfg)

        best_row = df_results.sort_values(by=["PR_AUC", "ROC_AUC"], ascending=False).iloc[0]
        best_name = best_row["Experiment"]

        print("\n=======================================================")
        print(f"BEST MODEL: {best_name}")
        print(f"ROC-AUC: {best_row['ROC_AUC']:.4f} | PR-AUC: {best_row['PR_AUC']:.4f} | F1: {best_row['F1_Score']:.4f}")
        print("=======================================================")
        print("\nFull Comparison Table:")
        display_cols = [
            "Model", "Experiment", "ROC_AUC", "PR_AUC", "F1_Score",
            "Precision", "Recall", "Accuracy", "Balanced_Accuracy", "Log_Loss",
        ]
        print(df_results[display_cols].to_string(index=False))

        # Save artifacts
        if cfg.experiment.save_model:
            for name, mtype in [("Logistic Regression", "lr"), ("Random Forest", "rf"), ("XGBoost", "xgboost")]:
                model_inst = models_dict[name]
                save_path = cfg.experiment.get_model_save_path(mtype)
                saved_file = model_inst.save(save_path)
                print(f"[OK] Saved {name} model artifact to: {saved_file}")

        compare_csv = str(output_dir / "model_comparison_metrics.csv")
        df_results.to_csv(compare_csv, index=False)
        print(f"\n[OK] Model comparison metrics saved to: {compare_csv}")

        for name, df_imp in importances_dict.items():
            safe_name = name.lower().replace(" ", "_")
            df_imp.to_csv(str(output_dir / f"feature_importance_{safe_name}.csv"), index=False)
        print(f"[OK] Feature importances saved to: {output_dir}/feature_importance_*.csv")

        # Export Lift & Gains artifacts across all 3 models
        lift_res_dict = {}
        for name, mtype in [("Logistic Regression", "lr"), ("Random Forest", "rf"), ("XGBoost", "xgboost")]:
            m_inst = models_dict[name]
            if hasattr(m_inst, "last_test_y") and hasattr(m_inst, "last_test_prob"):
                l_res = export_lift_artifacts(
                    y_test=m_inst.last_test_y,
                    test_prob=m_inst.last_test_prob,
                    output_dir=output_dir,
                    model_type=mtype,
                    file_prefix=mtype,
                    y_val=getattr(m_inst, "last_val_y", None),
                    val_prob=getattr(m_inst, "last_val_prob", None),
                    quiet=True,
                )
                lift_res_dict[name] = l_res

        # Export champion model as generic lift_and_gains_curve.png
        best_inst = models_dict[best_name]
        if hasattr(best_inst, "last_test_y") and hasattr(best_inst, "last_test_prob"):
            export_lift_artifacts(
                y_test=best_inst.last_test_y,
                test_prob=best_inst.last_test_prob,
                output_dir=output_dir,
                model_type=best_inst.model_type if hasattr(best_inst, "model_type") else "xgboost",
                file_prefix="champion",
                y_val=getattr(best_inst, "last_val_y", None),
                val_prob=getattr(best_inst, "last_val_prob", None),
                quiet=False,
            )

        # Generate comprehensive markdown comparison report
        generate_report(
            mode="compare",
            output_filepath=report_path,
            config=cfg,
            df_results=df_results,
            models_dict=models_dict,
            importances_dict=importances_dict,
            lift_res_dict=lift_res_dict,
        )

    elif args.mode == "ablation":
        model_type = args.model_type
        if args.n_trials is not None:
            tune_metric = args.tuning_metric or "PR_AUC"
            print("\n" + "=" * 65)
            print(f"[TIP] You passed '--n-trials {args.n_trials}' and '--tuning-metric {tune_metric}'.")
            print("To run the Optuna Hyperparameter Optimization Tuner (with colored trial logs):")
            print(f"    python main.py --mode tune --model-type {model_type} --n-trials {args.n_trials} --tuning-metric {tune_metric}")
            print("=" * 65 + "\n")
        print("\n=======================================================")
        print(f"STARTING ABLATION BENCHMARK SUITE  [{model_type.upper()}]")
        print("=======================================================")
        df_results, models_dict, importances_dict = run_ablation_suite(df, cfg, model_type=model_type)

        # Select best model based on PR_AUC then ROC_AUC
        best_row = df_results.sort_values(by=["PR_AUC", "ROC_AUC"], ascending=False).iloc[0]
        best_name = best_row["Experiment"]
        best_model = models_dict[best_name]
        best_imp = importances_dict[best_name]

        print("\n=======================================================")
        print(f"BEST PERFORMING CONFIGURATION: {best_name}")
        print(f"ROC-AUC: {best_row['ROC_AUC']:.4f} | PR-AUC: {best_row['PR_AUC']:.4f} | F1: {best_row['F1_Score']:.4f}")
        print("=======================================================")

        # Save best model artifact
        if cfg.experiment.save_model:
            save_path = cfg.experiment.get_model_save_path(model_type)
            saved_file = best_model.save(save_path)
            print(f"[OK] Saved best {model_type.upper()} model artifact to: {saved_file}")

        # Save metrics CSV
        metrics_csv_path = str(output_dir / f"{model_type}_ablation_metrics.csv")
        df_results.to_csv(metrics_csv_path, index=False)
        print(f"[OK] Exported metrics CSV to: {metrics_csv_path}")

        # Export Lift & Gains artifacts for best performing ablation configuration
        lift_res = None
        if hasattr(best_model, "last_test_y") and hasattr(best_model, "last_test_prob"):
            lift_res = export_lift_artifacts(
                y_test=best_model.last_test_y,
                test_prob=best_model.last_test_prob,
                output_dir=output_dir,
                model_type=model_type,
                y_val=getattr(best_model, "last_val_y", None),
                val_prob=getattr(best_model, "last_val_prob", None),
                quiet=False,
            )

        # Generate comprehensive markdown ablation report
        generate_report(
            mode="ablation",
            output_filepath=report_path,
            config=cfg,
            df_results=df_results,
            best_exp_name=best_name,
            best_model=best_model,
            best_imp=best_imp,
            model_type=model_type,
            lift_res=lift_res,
        )

    elif args.mode == "train":
        model_type = args.model_type
        manual_th = args.threshold if hasattr(args, "threshold") and args.threshold is not None else 0.5
        print("\n=======================================================")
        print(f"TRAINING SINGLE [{model_type.upper()}] MODEL ACCORDING TO CONFIG")
        if manual_th != 0.5:
            print(f"Decision Threshold: {manual_th:.2f}")
        print("=======================================================")
        metrics, model, df_imp = run_single_experiment(
            df=df,
            features_cfg=cfg.features,
            model_cfg=cfg.model,
            experiment_name="Single Run",
            model_type=model_type,
            threshold=manual_th,
        )

        save_path = None
        if cfg.experiment.save_model:
            save_path = cfg.experiment.get_model_save_path(model_type)
            saved_file = model.save(save_path)
            print(f"[OK] {model_type.upper()} model saved to: {saved_file}")

        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(str(output_dir / f"{model_type}_train_metrics.csv"), index=False)
        df_imp.to_csv(str(output_dir / f"{model_type}_feature_importance.csv"), index=False)
        print(f"[OK] Saved training metrics and feature importances to: {output_dir}")

        # Export Lift & Gains artifacts for single model training run
        lift_res = None
        if hasattr(model, "last_test_y") and hasattr(model, "last_test_prob"):
            lift_res = export_lift_artifacts(
                y_test=model.last_test_y,
                test_prob=model.last_test_prob,
                output_dir=output_dir,
                model_type=model_type,
                y_val=getattr(model, "last_val_y", None),
                val_prob=getattr(model, "last_val_prob", None),
                manual_threshold=manual_th,
                quiet=False,
            )

        # Generate markdown train report
        generate_report(
            mode="train",
            output_filepath=report_path,
            config=cfg,
            metrics=metrics,
            model=model,
            df_imp=df_imp,
            model_type=model_type,
            model_save_path=save_path,
            lift_res=lift_res,
        )

    elif args.mode == "tune":
        model_type = args.model_type
        trials_count = args.n_trials or getattr(cfg.tuning, "n_trials", 30)
        tune_metric = args.tuning_metric or getattr(cfg.tuning, "metric", "PR_AUC")
        cv_folds = getattr(cfg.tuning, "cv_folds", 5)
        timeout = getattr(cfg.tuning, "timeout", None)
        manual_th = args.threshold if hasattr(args, "threshold") and args.threshold is not None else None

        print("\n=======================================================")
        print(f"HYPERPARAMETER OPTIMIZATION (OPTUNA): [{model_type.upper()}]")
        if manual_th is not None:
            print(f"Operating Threshold: {manual_th:.2f}")
        print("=======================================================")

        tuner = OptunaTuner(
            model_type=model_type,
            config=cfg,
            target_metric=tune_metric,
            cv_folds=cv_folds,
            n_trials=trials_count,
            timeout=timeout,
            random_state=getattr(cfg.tuning, "random_state", 42),
        )

        tuner.tune(
            df=df,
            n_trials=trials_count,
            timeout=timeout,
            output_dir=str(output_dir),
            manual_threshold=manual_th,
            verbose=True,
        )

    elif args.mode == "recall":
        from models.recall_eval import run_recall_optimization

        model_type = args.model_type
        target_rec = args.target_recall if hasattr(args, "target_recall") and args.target_recall is not None else 0.75
        manual_th = args.threshold if hasattr(args, "threshold") else None

        run_recall_optimization(
            df=df,
            cfg=cfg,
            model_type=model_type,
            target_recall=target_rec,
            manual_threshold=manual_th,
            output_dir=str(output_dir),
        )

    elif args.mode in ("separability", "eda"):
        from visualization.separability import run_all_separability_analyses

        target_out = args.output_dir or "results/visualizations"
        pipe = pipeline.FeatureEngineeringPipeline(cfg.features)
        run_all_separability_analyses(
            df=df,
            pipeline_obj=pipe,
            output_dir=target_out,
            sample_size_tsne=3000,
            verbose=True,
        )

    elif args.mode == "representation":
        from preprocessing.temporal.benchmark import run_temporal_representation_benchmark

        target_out = args.output_dir or "results/temporal"
        pipe = pipeline.FeatureEngineeringPipeline(cfg.features)
        run_temporal_representation_benchmark(
            df=df,
            pipeline_obj=pipe,
            output_dir=target_out,
            model_type=args.model_type,
            verbose=True,
        )


if __name__ == "__main__":
    main()
