# Digital Bank Customer Churn Prediction

> **Business Analytics Group Project — Module 5: Model Development**
>
> End-to-end classification pipeline predicting customer churn for a digital bank.
> Three models are benchmarked on a shared feature-engineering pipeline:
> **Logistic Regression**, **Random Forest**, and **XGBoost**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Quickstart](#quickstart)
4. [Run Modes](#run-modes)
5. [Models](#models)
6. [Configuration](#configuration)
7. [Outputs](#outputs)
8. [Workflow Diagram](#workflow-diagram)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Dependencies](#dependencies)

---

## Project Overview

This sub-project implements **Module 5 (Model Development)** of the Business Analytics milestone:

| Branch | Task |
|---|---|
| **Classification** | Logistic Regression, Random Forest, XGBoost for churn prediction |
| **Evaluation** | ROC-AUC, PR-AUC, F1-Score, Precision, Recall, Balanced Accuracy, Log-Loss |
| **Comparison** | Side-by-side benchmark of all three classifiers on the same preprocessed data |
| **Ablation** | Feature-engineering sensitivity analysis (SMOTE, domain features, encoding, scaling, selection) |

The feature engineering pipeline (`preprocessing/`) is **read-only** from this module's perspective — all experiments share the same preprocessing output.

---

## Repository Structure

```
intro-to-ba/
├── config/
│   ├── config.yaml                  # Unified config (dataset, features, model, experiment)
│   └── pipeline_config.py           # Dataclass config loaders & path routers
├── data/
│   └── cleaned_bank_data.csv        # Input dataset (after cleaning, Module 3)
├── models/
│   ├── __init__.py                  # Exports all three model classes
│   ├── logistic_regression.py       # LogisticRegressionChurnModel
│   ├── random_forest.py             # RandomForestChurnModel
│   └── xgboost.py                   # XGBoostChurnModel
├── preprocessing/                   # Feature Engineering Pipeline (Module 4, read-only)
├── results/                         # Auto-generated reports and CSVs grouped by model/mode
│   ├── comparison/                  # Head-to-head comparison reports & feature importances
│   ├── lr/                          # Logistic Regression ablation/train metrics & reports
│   ├── rf/                          # Random Forest ablation/train metrics & reports
│   └── xgboost/                     # XGBoost ablation/train metrics & reports
├── main.py                          # Pipeline entrypoint
├── test_pipeline.py                 # Smoke tests for feature pipeline and config
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt   # or: pip install xgboost scikit-learn pandas numpy pyyaml tabulate imbalanced-learn

# 2. Run the canonical model comparison (LR vs RF vs XGBoost)
python main.py --mode compare

# 3. Run XGBoost ablation study (feature sensitivity)
python main.py --mode ablation --model-type xgboost

# 4. Train a single model (e.g., Random Forest)
python main.py --mode train --model-type rf
```

---

## Run Modes

| Mode | Command | Description |
|---|---|---|
| **compare** | `python main.py --mode compare` | Trains LR, RF, XGBoost on the **same** pipeline; prints and saves a side-by-side metric table and executive Markdown report in `results/comparison/`. **Default mode.** |
| **ablation** | `python main.py --mode ablation --model-type {xgboost\|lr\|rf}` | Runs 7 feature-ablation experiments on the selected model. Generates a Markdown report and metrics CSV in `results/{model_type}/`. |
| **train** | `python main.py --mode train --model-type {xgboost\|lr\|rf}` | Trains a single model according to `config.yaml`. Saves serialized model artifact + metrics CSV + train report in `results/{model_type}/`. |

### All CLI Arguments

```
python main.py [--mode {compare,ablation,train}]
               [--model-type {xgboost,lr,rf}]
               [--config PATH]
               [--output-dir DIR]
               [--report-name FILENAME]
```

---

## Models

All three classifiers implement a **unified interface contract**:

```python
fit(X_train, y_train, X_val=None, y_val=None, verbose=False) -> self
predict(X, threshold=0.5) -> np.ndarray
predict_proba(X) -> np.ndarray
evaluate(X_test, y_test, threshold=0.5) -> Dict[str, Any]
get_feature_importances(importance_type) -> pd.DataFrame
save(path) -> str
load(path) -> self
```

### Logistic Regression — `models/logistic_regression.py`

| Parameter | Default | Notes |
|---|---|---|
| `C` | 0.1 | Regularization penalty for multi-dimensional tabular data |
| `solver` | `lbfgs` | Fast Quasi-Newton optimization for binary classification |
| `max_iter` | 2000 | Ensures full convergence across scaled & encoded features |
| `class_weight` | `balanced` | Automatically handles class imbalance (~4:1) |
| Feature importance | Absolute Coefficient \|β\| | Linear proxy for feature impact on log-odds |

### Random Forest — `models/random_forest.py`

| Parameter | Default | Notes |
|---|---|---|
| `n_estimators` | 300 | Stable ensemble variance reduction |
| `max_depth` | 15 | Explicit cap to prevent tree overfitting |
| `min_samples_split` | 10 | Regularization at node level |
| `min_samples_leaf` | 4 | Regularization at leaf level |
| `max_features` | `sqrt` | De-correlates trees |
| `class_weight` | `balanced_subsample` | Re-weights classes per bootstrap replica |
| `oob_score` | `True` | Out-of-bag validation |
| Feature importance | Gini MDI | Mean decrease in impurity |

### XGBoost — `models/xgboost.py`

| Parameter | Default | Notes |
|---|---|---|
| `n_estimators` | 300 | With early stopping (30 rounds) |
| `max_depth` | 5 | Conservative tree depth |
| `learning_rate` | 0.05 | Shrinkage step size |
| `scale_pos_weight` | 1.0 | Dynamically calculated when SMOTE is disabled |
| Feature importance | Gain | Average gain across all splits |

---

## Configuration

All settings live in `config/config.yaml`. Key sections:

```yaml
dataset:
  data_path: "data/cleaned_bank_data.csv"
  target_col: "churn"
  test_size: 0.20

features:
  enable_smote: true            # SMOTE on training set
  enable_domain_features: true  # Banking behavioral features
  enable_target_encoding: true  # Smoothed m-estimate encoding
  enable_feature_selection: false

model:
  name: "xgboost"               # Default single-run model: "xgboost" | "lr" | "rf"
  xgboost_params:
    n_estimators: 300
    max_depth: 5
    learning_rate: 0.05
  lr_params:
    C: 0.05
    solver: "saga"
    penalty: "l1"
    l1_ratio: 1.0
  rf_params:
    n_estimators: 500
    max_depth: 12

experiment:
  output_dir: "results"
  save_model: true
  model_save_path: "models/saved/{model}_churn.{ext}"
  report_filename: "{mode}_{model}_report.md"
```

---

## Outputs

All outputs are automatically routed into dedicated subdirectories:

| Directory | Mode | Key Artifacts |
|---|---|---|
| `results/comparison/` | `compare` | `model_comparison_report.md`, `model_comparison_metrics.csv`, `feature_importance_*.csv` |
| `results/lr/` | `ablation` / `train` | `ablation_lr_report.md`, `train_lr_report.md`, `lr_ablation_metrics.csv`, `lr_train_metrics.csv`, `lr_feature_importance.csv` |
| `results/rf/` | `ablation` / `train` | `ablation_rf_report.md`, `train_rf_report.md`, `rf_ablation_metrics.csv`, `rf_train_metrics.csv`, `rf_feature_importance.csv` |
| `results/xgboost/` | `ablation` / `train` | `ablation_xgboost_report.md`, `train_xgboost_report.md`, `xgboost_ablation_metrics.csv`, `xgboost_train_metrics.csv`, `xgboost_feature_importance.csv` |
| `models/saved/` | `train` / `ablation` | Serialized model files: `lr_churn.pkl`, `rf_churn.pkl`, `xgboost_churn.json` |

---

## Workflow Diagram

```
cleaned_bank_data.csv
        │
        ▼
FeatureEngineeringPipeline (preprocessing/)
  ├── Skewness transform (Log1p, Yeo-Johnson)
  ├── Scaling (Robust + Standard)
  ├── Domain features (behavioral, friction, bundling)
  ├── Categorical encoding (OHE + Target Encoding)
  ├── SMOTE (train set only)
  └── Feature selection (optional)
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │         Model Execution Engine (main.py)      │
  │   ├── Compare  ──► LR vs RF vs XGBoost        │
  │   ├── Ablation ──► 7 Feature Scenarios        │
  │   └── Train    ──► Single Tuned Architecture  │
  └───────────────────────────────────────────────┘
        │
        ▼
  Evaluation: ROC-AUC · PR-AUC · F1 · Recall · Precision · Log-Loss
        │
        ▼
  Structured Results: results/{comparison, lr, rf, xgboost}/
```

---

## Evaluation Metrics

| Metric | Business Rationale |
|---|---|
| **ROC-AUC** | Overall ranking ability across all probability thresholds |
| **PR-AUC** | Most informative on imbalanced data (~20.37% churn rate) |
| **F1-Score** | Harmonic mean of precision and recall |
| **Recall** | Critical: missing a churner has higher business cost than a false retention alert |
| **Balanced Accuracy** | Macro-averaged recall; insensitive to class imbalance |
| **Log-Loss** | Calibration quality of probability estimates |

---

## Dependencies

```
xgboost>=1.7
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
imbalanced-learn>=0.11
pyyaml>=6.0
tabulate>=0.9
```

---

*Auto-generated pipeline for Business Analytics Milestone Project — Module 5: Model Development.*
