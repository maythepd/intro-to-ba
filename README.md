# Digital Bank Customer Churn Prediction

> **Business Analytics Group Project — Module 5: Model Development**
>
> End-to-end machine learning pipeline predicting customer churn for a digital bank.
> Orchestrates multi-table ETL, leak-free feature engineering, Bayesian hyperparameter tuning (Optuna), and model evaluation across three classifiers:
> **Logistic Regression**, **Random Forest**, and **XGBoost**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation & Requirements](#installation--requirements)
4. [CLI Entrypoint & Run Modes](#cli-entrypoint--run-modes)
5. [Configuration Guide (`config/config.yaml`)](#configuration-guide-configconfigyaml)
6. [Preprocessing & Leak-Free Pipeline](#preprocessing--leak-free-pipeline)
7. [Models & Interface Contract](#models--interface-contract)
8. [Hyperparameter Optimization (Optuna)](#hyperparameter-optimization-optuna)
9. [Visualizations & Generated Reports](#visualizations--generated-reports)
10. [Evaluation Metrics & Business Rationale](#evaluation-metrics--business-rationale)

---

## Project Overview

This project delivers **Module 5 (Model Development)** of the Business Analytics churn prediction system:

| Component | Description |
|---|---|
| **Unified Entrypoint** | Single CLI controller (`main.py`) executing all ETL, data splitting, training, comparison, ablation, and tuning workflows. |
| **Relational ETL** | Automated multi-table aggregation (`preprocessing/etl.py`) joining 7 transactional and behavioral tables into customer-level analytical records. |
| **Leak-Free Preprocessing** | Strict partitioning (`preprocessing/splitting.py`): Scalers, encoders, and SMOTE fit **only** on the training fold, transforming validation and test portions without data leakage. |
| **Model Benchmarking** | Side-by-side comparison of Logistic Regression (L1/SAGA), Random Forest, and XGBoost on identical preprocessed datasets. |
| **Feature Ablation Suite** | 8 systematic ablation experiments, including a raw Kaggle baseline (10 original features only) to isolate incremental lift. |
| **Mutual Information Selection** | Feature filtering using `mutual_info_classif` (`"MI_Score"`) to preserve non-linear relationships while dropping noise. |
| **Optuna Tuning** | Automated Bayesian optimization with Stratified K-Fold Cross-Validation, Median Pruning, and visualization plots. |

---

## Repository Structure

```
intro-to-ba/
├── requirements.txt                 # Project dependencies with version bounds
├── main.py                          # Unified CLI entrypoint (compare, ablation, tune, train)
├── test_pipeline.py                 # Smoke & regression test suite
├── README.md                        # Project documentation & configuration guide
│
├── config/
│   ├── config.yaml                  # Master YAML configuration file
│   └── pipeline_config.py           # Dataclass loaders, schemas, and path routers
│
├── data/
│   ├── full_bank_data.csv           # Consolidated customer-level dataset (82 columns)
│   ├── cleaned_bank_data.csv        # Legacy simplified dataset (backward compatibility)
│   └── full_dataset/                # 7 raw relational tables (accounts, transactions, logins, etc.)
│
├── preprocessing/                   # Feature Engineering Pipeline
│   ├── __init__.py                  # Exposes FeatureEngineeringPipeline & load_dataset
│   ├── pipeline.py                  # Master pipeline orchestrator & load_dataset wrapper
│   ├── etl.py                       # Pure relational multi-table joiner (BankDataETL)
│   ├── splitting.py                 # Leak-free Train/Val/Test & Stratified K-Fold CV
│   ├── domain_features.py           # Behavioral momentum, friction index, product bundling
│   ├── skewness_transform.py        # Log1p, Zero-balance hurdle, Yeo-Johnson transforms
│   ├── scaling.py                   # RobustScaler (outliers) & StandardScaler
│   ├── categorical_encoding.py      # One-Hot Encoding & Smoothed Target Encoding
│   ├── balancing.py                 # SMOTE class balancing (train set only)
│   └── feature_selection.py         # Mutual Information (`MI_Score`) & ranking
│
├── models/                          # Classifier Architectures & Tuner
│   ├── __init__.py                  # Exports LR, RF, XGBoost, and OptunaTuner
│   ├── base_model.py                # Abstract BaseChurnModel interface contract
│   ├── logistic_regression.py       # Logistic Regression with SAGA solver & L1 penalty
│   ├── random_forest.py             # Random Forest with class_weight='balanced_subsample'
│   ├── xgboost.py                   # XGBoost with early stopping & dynamic pos_weight
│   ├── tuner.py                     # Optuna hyperparameter optimization engine
│   └── saved/                       # Serialized model artifacts (.json, .pkl)
│
├── visualization/                   # Plotting Engine (300 DPI Figures)
│   ├── __init__.py                  # Exports plotting utilities
│   └── plots.py                     # Feature importance, model comparison, Optuna history
│
└── results/                         # Generated artifacts, metrics CSVs, plots & reports
    ├── comparison/                  # Head-to-head comparison reports & comparison charts
    ├── lr/                          # Logistic Regression metrics, plots & ablation reports
    ├── rf/                          # Random Forest metrics, plots & ablation reports
    └── xgboost/                     # XGBoost metrics, plots & ablation reports
```

---

## Installation & Requirements

Dependencies are specified in [requirements.txt](file:///d:/Introduction%20to%20BA/intro-to-ba/requirements.txt) at the root of the workspace.

### 1. Install via pip
```bash
# Ensure Python 3.9+ is active
pip install -r requirements.txt
```

### 2. Dependency Overview

| Package | Minimum Version | Purpose |
|---|---|---|
| `numpy` | `>=1.24.0` | Numerical calculations & vectorized array operations |
| `pandas` | `>=2.0.0` | Dataframe manipulation, multi-table joins, aggregations |
| `scipy` | `>=1.10.0` | Statistical distributions & skewness calculations |
| `scikit-learn` | `>=1.3.0` | Classifiers, preprocessors, metrics, Mutual Information (`mutual_info_classif`) |
| `xgboost` | `>=1.7.0` | Extreme Gradient Boosting classifier with GPU/CPU multi-threading |
| `imbalanced-learn`| `>=0.11.0` | SMOTE minority class oversampling |
| `optuna` | `>=3.5.0` | Bayesian hyperparameter optimization with TPE sampler & pruning |
| `pyyaml` | `>=6.0` | YAML configuration file parsing |
| `tabulate` | `>=0.9.0` | Formatted Markdown table generation |
| `matplotlib` | `>=3.7.0` | High-resolution publication-grade figure rendering (300 DPI) |
| `seaborn` | `>=0.12.0` | Statistical visualization palettes and chart styling |

### 3. Verify Setup
Run the end-to-end smoke test to verify dependencies, config loading, and pipeline integrity:
```bash
python test_pipeline.py
```

---

## CLI Entrypoint & Run Modes

All operations are executed through the single root entrypoint: [`main.py`](file:///d:/Introduction%20to%20BA/intro-to-ba/main.py).

Sub-modules (such as `etl.py`, `splitting.py`, `tuner.py`) contain no standalone entrypoints. Instead, `main.py` ingests data via `preprocessing.pipeline.load_dataset()`, dynamically checking if `data/full_bank_data.csv` exists and compiling it from `data/full_dataset/` raw tables if needed.

### Execution Synopsis

```
python main.py [--mode {compare,ablation,tune,train}]
               [--model-type {xgboost,lr,rf}]
               [--config CONFIG_PATH]
               [--n-trials N_TRIALS]
               [--tuning-metric {PR_AUC,ROC_AUC,F1_Score}]
               [--rebuild-dataset]
               [--output-dir OUTPUT_DIR]
               [--report-name REPORT_FILENAME]
```

---

### Run Modes Breakdown

#### 1. Model Comparison (`--mode compare`)
Trains Logistic Regression, Random Forest, and XGBoost side-by-side on the identical feature pipeline. Evaluates performance on the blind holdout test set, exports metrics to CSV, creates a side-by-side metric comparison bar chart, and writes an executive Markdown report.

```bash
# Default mode (compares LR vs RF vs XGBoost)
python main.py --mode compare
```
*Outputs generated in:* `results/comparison/`
- `model_comparison_report.md`
- `model_comparison_metrics.csv`
- `model_comparison_chart.png` (300 DPI bar chart)
- `feature_importance_{model}.csv` and `.png`

---

#### 2. Feature Ablation Study (`--mode ablation`)
Evaluates model sensitivity across **8 distinct feature scenarios** on the selected architecture:
1. `0. Raw Kaggle Baseline`: Only the 10 original tabular features (21 encoded features, no domain features, no ETL aggregations).
2. `1. Full Pipeline`: All features enabled (ETL + Domain + Skewness + Scaling + OneHot + TargetEncoding + SMOTE).
3. `2. No SMOTE`: Pipeline without minority class oversampling (evaluates scale_pos_weight/class_weight).
4. `3. No Domain Features`: Drops banking behavioral, friction, bundling, and risk interaction features.
5. `4. No Skewness Transform`: Disables Log1p, Zero-balance hurdle, and Yeo-Johnson transforms.
6. `5. No Scaling`: Raw numerical distributions without Robust/Standard scaling.
7. `6. No Target Encoding`: Disables smoothed m-estimate categorical encoding (OHE only).
8. `7. With Feature Selection`: Applies Mutual Information (`MI_Score`) top-K filtering.

```bash
# XGBoost feature ablation
python main.py --mode ablation --model-type xgboost

# Random Forest feature ablation
python main.py --mode ablation --model-type rf

# Logistic Regression feature ablation
python main.py --mode ablation --model-type lr
```
*Outputs generated in:* `results/{model_type}/`
- `ablation_{model_type}_report.md`
- `{model_type}_ablation_metrics.csv`
- `{model_type}_feature_importance.png`

---

#### 3. Hyperparameter Optimization (`--mode tune`)
Launches an **Optuna** Bayesian optimization study using the Tree-structured Parzen Estimator (TPE) algorithm. Evaluates parameter candidates using **Stratified K-Fold Cross-Validation** strictly on the training partition. Early-stops unpromising trials via `MedianPruner`.

```bash
# Tune XGBoost for PR-AUC with 30 trials (default settings from config)
python main.py --mode tune --model-type xgboost

# Tune Random Forest for F1-Score with 50 trials
python main.py --mode tune --model-type rf --n-trials 50 --tuning-metric F1_Score

# Tune Logistic Regression for ROC-AUC with 40 trials
python main.py --mode tune --model-type lr --n-trials 40 --tuning-metric ROC_AUC
```
*Outputs generated in:* `results/{model_type}/`
- `optuna_{model_type}_trials.csv` (Full history of all trials, parameters, and CV metrics)
- `optuna_{model_type}_history.png` (Dual-panel optimization history & trial objective plot)
- `tune_{model_type}_report.md` (Executive tuning summary with best parameters and performance delta)

---

#### 4. Single Model Training (`--mode train`)
Trains a single model according to parameters in `config/config.yaml`, evaluates on the test partition, saves the serialized model artifact (`.json` for XGBoost, `.pkl` for scikit-learn models), and produces an evaluation report.

```bash
# Train and save XGBoost
python main.py --mode train --model-type xgboost

# Train and save Random Forest
python main.py --mode train --model-type rf

# Train and save Logistic Regression
python main.py --mode train --model-type lr
```
*Outputs generated in:*
- `results/{model_type}/train_{model_type}_report.md`
- `results/{model_type}/{model_type}_train_metrics.csv`
- `results/{model_type}/{model_type}_feature_importance.png`
- `models/saved/{model_type}_churn.{json|pkl}`

---

### Command-Line Arguments Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mode` | string | `compare` | Operational mode: `compare`, `ablation`, `tune`, or `train`. |
| `--model-type` | string | `xgboost` | Target classifier: `xgboost`, `lr`, or `rf`. (Ignored in `compare` mode). |
| `--config` | string | `config/config.yaml` | Path to custom YAML configuration file. |
| `--n-trials` | integer | `null` (uses config) | Number of Optuna optimization trials (in `tune` mode). |
| `--tuning-metric` | string | `null` (uses config) | Target metric for Optuna: `PR_AUC`, `ROC_AUC`, or `F1_Score`. |
| `--rebuild-dataset`| flag | `False` | Forces multi-table ETL rebuild from `data/full_dataset/`. |
| `--output-dir` | string | `null` (auto-routed) | Custom directory for outputs; defaults to `results/{comparison\|lr\|rf\|xgboost}`. |
| `--report-name` | string | `null` (auto-templated)| Custom report filename; defaults to `{mode}_{model}_report.md`. |

---

## Configuration Guide (`config/config.yaml`)

The entire pipeline is driven by [`config/config.yaml`](file:///d:/Introduction%20to%20BA/intro-to-ba/config/config.yaml). Edit this file to adjust data paths, toggle features, tune models, and control experiments without modifying code.

### 1. `dataset`: Data & Partition Settings

```yaml
dataset:
  data_path: "data/full_bank_data.csv"      # Path to consolidated dataset
  raw_full_dataset_dir: "data/full_dataset" # Folder containing 7 raw relational tables
  rebuild_full_dataset: false               # Set true to force ETL regeneration
  target_col: "churn"                       # Binary target column (0: active, 1: churned)
  test_size: 0.20                           # 20% holdout test set (completely blind)
  val_size: 0.10                            # 10% validation set (used for early stopping)
  enable_cv: true                           # Enable Stratified K-Fold Cross-Validation
  cv_folds: 5                               # Number of folds for CV (default: 5)
  random_state: 42                          # Seed for reproducibility
  stratify: true                            # Preserve churn ratio (~20.37%) across splits
  exclude_cols:                             # Columns excluded from feature matrices
    - customer_id
    - churn
    - country
    - gender
    - preferred_channel
    - primary_device
    - yj_trans_amount_3m
```

> **How Splitting Works:**
> With 10,000 customers:
> - **Test partition (20%):** 2,000 samples held out completely blind for final validation.
> - **Train partition (70%):** 7,000 samples used for model fitting and SMOTE balancing.
> - **Validation partition (10%):** 1,000 samples used for XGBoost early stopping.
> - **Cross-Validation:** When `enable_cv: true`, the 8,000 non-test samples are partitioned into 5 stratified folds (6,400 train / 1,600 val per fold).

---

### 2. `features`: Feature Engineering & Ablation Toggles

```yaml
features:
  # High-level Master Toggles (Ideal for ablation experiments)
  enable_skewness: true             # Log1p, Zero-balance hurdle, Yeo-Johnson
  enable_scaling: true              # RobustScaler & StandardScaler
  enable_domain_features: true      # Banking behavioral & risk features
  enable_one_hot_encoding: true     # One-Hot Encoding for categorical features
  enable_target_encoding: true      # Smoothed Target Encoding (m-estimate)
  enable_smote: true                # SMOTE oversampling on Train portion only
  enable_feature_selection: false   # Top-K feature selection

  # Distribution Skewness Transformations
  skewness_transform:
    transform_trans_amount_log: true
    transform_recency_log: true
    zero_balance_hurdle: true
    transform_yeo_johnson: true

  # Numerical Scaling
  scaling:
    apply_robust_scaling: true      # RobustScaler (median, IQR) for heavy-tailed features
    apply_standard_scaling: true    # StandardScaler (z-score) for bounded/normal features
    robust_features:
      - recency_days
      - monthly_transactions
      - trans_amount_3m
      - trans_trend_ratio
      - app_logins_monthly
    standard_features:
      - age
      - credit_score
      - estimated_salary
      - tenure

  # Domain Banking Features
  domain_features:
    enable_behavioral_momentum: true  # velocity, spending collapse, ticket size, login ratio
    enable_customer_friction: true    # friction_index, is_high_friction (complaints, declined trans)
    enable_product_bundling: true     # is_product_sweetspot (2 prods), is_product_danger_zone
    enable_risk_interactions: true    # wealth_deceleration_risk, germany_high_balance

  # Categorical Encoding
  categorical_encoding:
    categorical_cols:
      - country
      - gender
      - preferred_channel
      - primary_device
    target_encoding_m_weight: 10.0   # m-estimate smoothing weight
    ohe_drop_first: false            # Keep all dummies for tree models

  # Class Balancing (SMOTE)
  balancing:
    smote_sampling_strategy: "auto"  # Equalize minority class to 1:1 on train fold
    smote_k_neighbors: 5

  # Feature Selection (Mutual Information)
  feature_selection:
    top_k_features: null             # Set an integer (e.g., 40) or null to keep all
    feature_selection_metric: "MI_Score"  # Mutual Information Classif (mutual_info_classif)
    compute_feature_importance: false
```

---

### 3. `model`: Model Architectures & Hyperparameters

```yaml
model:
  name: "xgboost"                   # Default model: "xgboost" | "lr" | "rf"

  # XGBoost Hyperparameters
  xgboost_params:
    n_estimators: 300
    max_depth: 5
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    gamma: 0.1
    min_child_weight: 1
    scale_pos_weight: 1.0           # 1.0 when SMOTE is enabled; 3.9 if SMOTE disabled
    eval_metric: "auc"
    random_state: 42
    n_jobs: -1
  early_stopping_rounds: 30

  # Logistic Regression Hyperparameters (L1 + SAGA solver for sparsity)
  lr_params:
    C: 0.05                         # Stronger regularization for sparse tabular weights
    solver: "saga"                  # Supports L1 penalty
    penalty: "l1"                   # Zeroes out non-informative coefficients
    max_iter: 2000
    class_weight: "balanced"
    random_state: 42

  # Random Forest Hyperparameters
  rf_params:
    n_estimators: 500
    max_depth: 12
    min_samples_split: 15
    min_samples_leaf: 5
    max_features: "sqrt"
    class_weight: "balanced_subsample"
    oob_score: true
    random_state: 42
    n_jobs: -1
```

---

### 4. `tuning`: Optuna Optimization Settings

```yaml
tuning:
  enabled: false                    # Set true or run via `main.py --mode tune`
  n_trials: 30                      # Number of trials
  timeout: 600                      # Time limit in seconds (or null)
  metric: "PR_AUC"                  # Optimization objective ("PR_AUC" | "ROC_AUC" | "F1_Score")
  cv_folds: 5                       # Stratified K-Fold CV on training partition
  study_name: "bank_churn_tuning"
  direction: "maximize"
  random_state: 42
```

---

### 5. `experiment`: Results & Model Serialization

```yaml
experiment:
  output_dir: "results"             # Base output directory
  save_model: true                  # Save model artifacts after training
  model_save_path: "models/saved/{model}_churn.{ext}"
  report_filename: "{mode}_{model}_report.md"
```

---

## Preprocessing & Leak-Free Pipeline

### 1. Multi-Table Relational ETL (`preprocessing/etl.py`)
The digital banking raw data comprises 7 relational tables in `data/full_dataset/`:
- `customers.csv` (Demographics, country, age, gender)
- `accounts.csv` (Balance, credit score, tenure, products)
- `cards.csv` (Credit cards, card types, card limits)
- `complaints.csv` (Customer complaints & resolution status)
- `support.csv` (Customer service tickets and channels)
- `transactions.csv` (Transaction recency, frequency, amounts)
- `login.csv` (1,309,432 mobile and web login sessions)

`BankDataETL` aggregates all transactional and login histories to the `customer_id` level. All 1,309,432 login events are preserved into aggregated metrics (`app_logins_monthly`, `web_logins_monthly`, `login_recency_days`, `mobile_login_ratio`), ensuring **zero data loss** while joining into exactly 10,000 customer records.

---

### 2. Leak-Free Cross-Validation Guarantee
Data leakage occurs when parameters calculated across the entire dataset (such as mean/std for scaling, target frequency for target encoding, or synthetic samples for SMOTE) bleed into the validation or test evaluation.

The pipeline eliminates data leakage through strict design:
1. **Holdout Test Set Isolation:** The 20% test partition (2,000 samples) is separated immediately upon ingestion and never seen by any transformer until final evaluation.
2. **K-Fold CV Isolation:** In `evaluate_cv()` and `OptunaTuner`, an independent instance of `FeatureEngineeringPipeline` is constructed for every fold.
3. **Fit vs. Transform Discipline:**
   - **`fit_transform()`** is called **only** on `fold.train_df`.
   - **`transform()`** is called on `fold.val_df` and `test_df`.
   - **SMOTE** is applied exclusively to the training portion.

```
Raw Customer Data (10,000 records)
         │
         ├───► Holdout Test Set (2,000 records) — BLIND UNTIL FINAL TEST
         │
         ▼
Training Partition (8,000 records)
         │
         ▼
Stratified K-Fold CV (5 Folds: 6,400 Train / 1,600 Val)
   ├── Fold 1: Pipeline.fit_transform(Train) ──► Pipeline.transform(Val)
   ├── Fold 2: Pipeline.fit_transform(Train) ──► Pipeline.transform(Val)
   ├── Fold 3: Pipeline.fit_transform(Train) ──► Pipeline.transform(Val)
   ├── Fold 4: Pipeline.fit_transform(Train) ──► Pipeline.transform(Val)
   └── Fold 5: Pipeline.fit_transform(Train) ──► Pipeline.transform(Val)
```

---

## Models & Interface Contract

All three classifiers inherit from `BaseChurnModel` (`models/base_model.py`) and adhere to a unified interface contract:

```python
fit(X_train, y_train, X_val=None, y_val=None, verbose=False) -> self
predict(X, threshold=0.5) -> np.ndarray
predict_proba(X) -> np.ndarray
evaluate(X_test, y_test, threshold=0.5) -> Dict[str, Any]
get_feature_importances(importance_type) -> pd.DataFrame
save(path) -> str
load(path) -> self
```

### Classifier Comparison Matrix

| Model | Implementation | Key Regularization | Handling Class Imbalance | Feature Importance Proxy |
|---|---|---|---|---|
| **Logistic Regression** | `models/logistic_regression.py` | L1 penalty (`C=0.05`, `solver="saga"`) | `class_weight="balanced"` | Absolute Coefficient $\|w_i\|$ |
| **Random Forest** | `models/random_forest.py` | `max_depth=12`, `min_samples_leaf=5`, `max_features="sqrt"` | `class_weight="balanced_subsample"` | Gini Mean Decrease in Impurity (MDI) |
| **XGBoost** | `models/xgboost.py` | `max_depth=5`, `gamma=0.1`, `subsample=0.8`, `colsample_bytree=0.8` | `scale_pos_weight=1.0` (SMOTE) or dynamically tuned | Average Split Gain across Trees |

---

## Hyperparameter Optimization (Optuna)

The [`models/tuner.py`](file:///d:/Introduction%20to%20BA/intro-to-ba/models/tuner.py) module implements Bayesian search using Optuna:
- **Search Space:**
  - **XGBoost:** `n_estimators`, `max_depth` (3–9), `learning_rate` (0.01–0.2), `subsample`, `colsample_bytree`, `gamma`, `min_child_weight`, `scale_pos_weight`.
  - **Random Forest:** `n_estimators` (100–600), `max_depth` (6–20), `min_samples_split`, `min_samples_leaf`, `max_features` (`sqrt`, `log2`).
  - **Logistic Regression:** `C` ($10^{-3}$ to $10^2$, log scale), `penalty` (`l1`, `l2`), `solver` (`saga`, `lbfgs`).
- **Pruning:** Integrates `optuna.pruners.MedianPruner` to stop unpromising trials after early folds.
- **Reporting:** Automatically generates `optuna_{model}_trials.csv` and an optimization history chart.

---

## Visualizations & Generated Reports

All experiment artifacts are automatically stored in structured subdirectories under `results/`:

```
results/
├── comparison/
│   ├── model_comparison_report.md       # Executive comparison report
│   ├── model_comparison_metrics.csv     # Metrics across LR, RF, XGBoost
│   ├── model_comparison_chart.png       # 300 DPI comparative bar chart
│   └── feature_importance_*.png         # Top-20 features for each model
├── xgboost/
│   ├── ablation_xgboost_report.md       # 8-scenario ablation analysis
│   ├── xgboost_ablation_metrics.csv     # Numerical ablation results
│   ├── xgboost_feature_importance.png   # Top-20 features by Gain
│   ├── optuna_xgboost_trials.csv        # Optimization trial history
│   ├── optuna_xgboost_history.png       # Dual-panel Optuna plot
│   └── tune_xgboost_report.md           # Tuning report with best params
├── rf/
│   └── ...                              # Same artifacts for Random Forest
└── lr/
    └── ...                              # Same artifacts for Logistic Regression
```

### Visual Artifacts Highlights
- **Feature Importance Bar Charts (`visualization/plots.py`):** Horizontal bar charts plotting top-20 features styled with custom gradients (Viridis for XGBoost, Blues for RF, Magma for LR).
- **Model Comparison Chart:** Grouped bar charts comparing ROC-AUC, PR-AUC, F1-Score, and Recall side-by-side across models.
- **Optuna Tuning History:** Dual-panel figure displaying trial score trajectories and running best objective progression.

---

## Evaluation Metrics & Business Rationale

Customer churn in retail banking is an **imbalanced classification task** (~20.37% positive churn rate). Standard Accuracy is misleading (an 80% accuracy model can predict 0 churners).

We prioritize metrics based on business impact:

| Metric | Target | Business Justification |
|---|---|---|
| **PR-AUC** | $\ge 0.65$ | **Primary Metric:** Evaluates the trade-off between Precision and Recall specifically on the minority churn class. Less sensitive to large numbers of true negatives. |
| **ROC-AUC** | $\ge 0.85$ | Evaluates ranking discrimination across all possible decision thresholds. |
| **Recall** | $\ge 0.70$ | **Cost of False Negatives:** Missing an impending churner ($3,000–$5,000 CLV loss) is substantially costlier than sending a retention offer to an active customer. |
| **F1-Score** | $\ge 0.60$ | Harmonic balance between Precision (marketing budget efficiency) and Recall (churn capture). |
| **Balanced Accuracy** | $\ge 0.75$ | Macro-average of recall across both classes, uninfluenced by class imbalance. |
| **Log-Loss** | Lower is better | Measures the calibration of predicted churn probabilities, critical for tiered customer intervention. |

---

*Module 5: Model Development & Evaluation — Business Analytics Capstone Project.*
