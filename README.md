# Digital Bank Customer Churn Prediction

> **Business Analytics Group Project — Module 5: Model Development**


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
11. [Continuous Deployment (CI/CD) & Public Hosting Guide](#continuous-deployment-cicd--public-hosting-guide)
12. [System Telemetry & Monitoring Architecture](#system-telemetry--monitoring-architecture)

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
| **Recall & Lift Optimization** | Dynamic threshold scanning to resolve the Accuracy/Recall dilemma ($\ge 75\%-80\%$ recall), 10-decile lift tables, and Cumulative Gains curves. |

---

## Repository Structure

```
intro-to-ba/
├── requirements.txt                 # Project dependencies with version bounds
├── main.py                          # Unified CLI entrypoint (compare, ablation, tune, train, recall)
├── api.py                           # MLOps & API Serving CLI (runs FastAPI backend + Vite frontend)
├── README.md                        # Project documentation & configuration guide
│
├── api/                             # FastAPI Serving Backend & Inference Service
│   ├── __init__.py                  # Exposes FastAPI app instance
│   ├── main.py                      # FastAPI routes (/health, /predict, /segments, /dashboard)
│   ├── module5.py                   # XGBoost & preprocessing serving adapter
│   ├── official_data.py             # Official Kaggle data preparation pipeline
│   ├── scoring.py                   # Shared churn probability scoring & risk thresholds
│   └── segmentation_service.py      # KMeans & Hierarchical clustering inference service
│
├── frontend/                        # Modern React 19 + Vite + Tailwind CSS Dashboard
│   ├── package.json                 # Frontend dependencies (TanStack Router & Query, Recharts, Lucide)
│   ├── vite.config.ts               # Vite configuration with proxy to FastAPI
│   └── src/                         # Application source (routes, components, api client)
│
├── segmentation/                    # Customer Segmentation Engine (KMeans & Hierarchical Ward)
│   ├── data.py                      # Data loading & feature snapshot preparation
│   ├── kmeans.py                    # KMeans clustering model & evaluation
│   ├── runner.py                    # Training & cluster profiling coordinator
│   └── evaluation.py                # Silhouette score & business metrics evaluation
│
├── config/
│   ├── config.yaml                  # Master YAML configuration file
│   └── pipeline_config.py           # Dataclass loaders, schemas, and path routers
│
├── docs/                            # Deep Architectural & Feature Engineering Documentation
│   ├── manifold_representation_learning_strategies.md # Detailed specification of 4 manifold learning strategies
│   └── new_temporal_feats.md        # Comprehensive catalog of 37 dynamic temporal sequence channels
│
├── data/
│   ├── full_bank_data.csv           # Consolidated customer-level dataset (70+ columns)
│   ├── cleaned_bank_data.csv        # Legacy simplified dataset (backward compatibility)
│   └── full_dataset/                # 7 raw relational tables (accounts, transactions, logins, etc.)
│
├── utils/                           # Common Utilities, Experiment Runners & Reporting
│   ├── __init__.py                  # Exports all utilities, experiment runners & report generators
│   ├── common.py                    # Path resolution and environment setup
│   ├── experiments.py               # Model factory (build_model), experiment runners & ablation
│   └── reporting.py                 # Executive Markdown report generation & dispatching
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
│   ├── feature_selection.py         # Mutual Information (`MI_Score`) & ranking
│   └── temporal/                    # Temporal Behavioral Sequence Representation Package
│       ├── __init__.py              # Clean exports
│       ├── extractor.py             # TemporalSnapshotExtractor Coordinator (T=6, D=37)
│       ├── model.py                 # Bi-GRU Sequence Autoencoder & Loss Head Trainer
│       ├── manager.py               # TemporalEmbeddingManager (extraction, caching, merging)
│       ├── benchmark.py             # Representation benchmark comparison suite
│       ├── extractors/              # Domain-specific multi-snapshot Sub-Extractors (D=37)
│       │   ├── __init__.py          # Exports ALL_FEATURE_NAMES_37
│       │   ├── base.py              # BaseSubExtractor ABC with numerical safety helpers
│       │   ├── transaction_extractor.py # 11 Financial & cash flow channels (T1 - T11)
│       │   ├── login_extractor.py   # 15 Digital habits & authentication friction channels (L1 - L15)
│       │   ├── complaint_extractor.py # 5 Service dispute & SLA breach channels (C1 - C5)
│       │   ├── account_extractor.py # 4 Portfolio stock & account closure channels (A1 - A4)
│       │   └── cross_channel_extractor.py # 2 Cross-table interaction coupling channels (X1 - X2)
│       └── strategies/              # Pluggable Manifold Representation Learning Strategies
│           ├── __init__.py          # Strategy Registry & Factory (get_temporal_strategy)
│           ├── base.py              # BaseTemporalStrategy ABC
│           ├── unsupervised.py      # S0: Unsupervised Bi-GRU Reconstruction Baseline (MSE)
│           ├── asymmetric_arcface.py # S1: Asymmetric Angular Margin Contrastive Metric (ArcFace)
│           ├── hyperbolic_poincare.py # S2: Hyperbolic Poincaré Ball Geometry
│           ├── optimal_transport.py # S3: Optimal Transport & Wasserstein Drift Regularization
│           └── neyman_pearson.py    # S4: Constrained Neyman-Pearson False Positive Minimization
│
├── models/                          # Classifier Architectures & Tuner
│   ├── __init__.py                  # Exports LR, RF, XGBoost, and OptunaTuner
│   ├── base_model.py                # Abstract BaseChurnModel interface contract
│   ├── logistic_regression.py       # Logistic Regression with SAGA solver & L1 penalty
│   ├── random_forest.py             # Random Forest with class_weight='balanced_subsample'
│   ├── xgboost.py                   # XGBoost with early stopping & dynamic pos_weight
│   ├── tuner.py                     # Optuna hyperparameter optimization engine
│   ├── recall_eval.py               # Recall optimization, baseline benchmarking & Decile Lift
│   └── saved/                       # Serialized model artifacts (.json, .pkl)
│       └── temporal/                # PyTorch recurrent checkpoints (temporal_encoder.pt)
│
├── test/                            # Comprehensive Automated Test Suite
│   ├── test_config_loading.py       # Config validation and YAML serialization tests
│   ├── test_latent_importance.py    # Temporal latent feature importance & gain verification
│   ├── test_modular_temporal_extractors.py # Sub-extractors (D=37), variance & safety tests
│   ├── test_pipeline.py             # Full end-to-end preprocessing, Optuna, & model smoke tests
│   └── test_temporal_strategies.py  # Pluggable representation strategies & loss head tests
│
├── visualization/                   # Plotting Engine (300 DPI Figures)
│   ├── __init__.py                  # Exports plotting utilities
│   └── plots.py                     # Feature importance, model comparison, Optuna history
│
└── results/                         # Generated artifacts, metrics CSVs, plots & reports
    ├── comparison/                  # Head-to-head comparison reports & comparison charts
    ├── lr/                          # Logistic Regression metrics, plots & ablation reports
    ├── rf/                          # Random Forest metrics, plots & ablation reports
    ├── temporal/                    # Representation benchmark logs & strategy comparisons
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
Run the comprehensive test suite to verify dependencies, config loading, modular extractors, and end-to-end pipeline integrity:
```bash
# Run all unit and integration tests across test/
python -m unittest discover -s test -p "test_*.py"

# Or run the full end-to-end pipeline smoke test
python test/test_pipeline.py
```

---

## CLI Entrypoint & Run Modes

All operations are executed through two primary entrypoints:
- **[`main.py`](file:///d:/Introduction%20to%20BA/intro-to-ba/main.py):** Model training, ablation, tuning, comparison, and evaluation.
- **[`api.py`](file:///d:/Introduction%20to%20BA/intro-to-ba/api.py):** MLOps data/artifact preparation and web serving (FastAPI + Vite Frontend).

---

### 4.1 MLOps & Web Serving CLI (`api.py`)

[`api.py`](file:///d:/Introduction%20to%20BA/intro-to-ba/api.py) provides a unified command-line interface for running the live web service and managing production serving artifacts.

#### One-Command Full Stack Startup & Auto Browser Pop-up

Running `python api.py` (or `python api.py serve`) starts both the FastAPI backend on port 8000 AND the Vite frontend development server on port 5173, then **automatically opens the UI in your default web browser**:

```bash
# Starts both services and automatically pops open http://localhost:5173 in browser
python api.py serve

# Or run in headless mode (no auto-opening browser, e.g. for CI/remote servers)
python api.py serve --no-browser
```

> [!TIP]
> **Windows Browser Access Note:**
> The CLI automatically opens [`http://localhost:5173`](http://localhost:5173). In Windows browsers (Chrome, Edge), always navigate to:
> - **Frontend Web UI:** [`http://localhost:5173`](http://localhost:5173) (or `http://localhost:8000` when built)
> - **FastAPI Swagger Docs:** [`http://localhost:8000/docs`](http://localhost:8000/docs)
> - **API Health Check:** [`http://localhost:8000/health`](http://localhost:8000/health)
>
> Do **not** type `http://0.0.0.0:8000` into your browser address bar; `0.0.0.0` is an all-interfaces bind address that Windows browsers reject with `ERR_ADDRESS_INVALID`.

#### Serving CLI Commands & Options:

```bash
# 1. Start both FastAPI (port 8000) and Frontend UI (port 5173) with auto browser pop-up
python api.py serve --port 8000 --frontend-port 5173 --reload

# 2. Start only the FastAPI backend (headless / API-only mode)
python api.py serve --no-frontend --port 8000 --no-browser

# 3. Download and reproduce Module 5 XGBoost serving artifacts
python api.py prepare --download

# 4. Download and prepare official customer feature snapshot
python api.py official-data --download

# 5. Train and export customer segmentation models (KMeans & Hierarchical Ward)
python api.py segmentation --data data/official/customer_features.csv --output artifacts/segmentation
```

---

### 4.2 Model Development & Evaluation CLI (`main.py`)

All model research operations are executed through [`main.py`](file:///d:/Introduction%20to%20BA/intro-to-ba/main.py).

Sub-modules (such as `etl.py`, `splitting.py`, `tuner.py`) contain no standalone entrypoints. Instead, `main.py` ingests data via `preprocessing.pipeline.load_dataset()`, dynamically checking if `data/full_bank_data.csv` exists and compiling it from `data/full_dataset/` raw tables if needed.

### Execution Synopsis

```
python main.py [--mode {compare,ablation,tune,train,recall,separability,eda,representation}]
               [--model-type {xgboost,lr,rf}]
               [--config CONFIG_PATH]
               [--n-trials N_TRIALS]
               [--tuning-metric {PR_AUC,ROC_AUC,F1_Score,Recall,F2_Score,Precision}]
               [--target-recall TARGET_RECALL]
               [--threshold THRESHOLD]
               [--sample-weights]
               [--decay-half-life DECAY_HALF_LIFE]
               [--focal-loss]
               [--focal-gamma FOCAL_GAMMA]
               [--focal-alpha FOCAL_ALPHA]
               [--enable-temporal-embeddings]
               [--temporal-dim TEMPORAL_DIM]
               [--temporal-epochs TEMPORAL_EPOCHS]
               [--feature-selection-metric {MI_Score,Pearson,MI_Pearson_Hybrid,RF_Importance}]
               [--hybrid-pearson-weight HYBRID_PEARSON_WEIGHT]
               [--top-k-features TOP_K_FEATURES]
               [--rebuild-dataset]
               [--output-dir OUTPUT_DIR]
               [--report-name REPORT_FILENAME]
```

---

### Run Modes Breakdown

#### 1. Model Comparison (`--mode compare`)
Trains Logistic Regression, Random Forest, and XGBoost side-by-side on the identical feature pipeline. Evaluates performance on the blind holdout test set, exports metrics to CSV, creates a side-by-side metric comparison bar chart, exports Decile Lift tables for each classifier, and writes an executive Markdown report.

```bash
# Default mode (compares LR vs RF vs XGBoost)
python main.py --mode compare
```
*Outputs generated in:* `results/comparison/`
- `model_comparison_report.md`
- `model_comparison_metrics.csv`
- `model_comparison_chart.png` (300 DPI bar chart)
- `feature_importance_{model}.csv` and `.png`
- `{model}_decile_lift_table.csv` and `{model}_baseline_lift_comparison.csv`
- `lift_and_gains_curve.png` (Champion model Gains & Decile Lift chart)

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
- `ablation_{model_type}_report.md` (Includes ablation metrics table and best-configuration Lift analysis)
- `{model_type}_ablation_metrics.csv`
- `{model_type}_feature_importance.png`
- `lift_and_gains_curve.png`
- `{model_type}_decile_lift_table.csv`
- `{model_type}_baseline_lift_comparison.csv`

---

#### 3. Hyperparameter Optimization (`--mode tune`)
Launches an **Optuna** Bayesian optimization study using the Tree-structured Parzen Estimator (TPE) algorithm with real-time colored console logging. Evaluates parameter candidates using **Stratified K-Fold Cross-Validation** strictly on the training partition. Early-stops unpromising trials via `MedianPruner`.
Supports multiple business objectives: `PR_AUC` (recommended default), `ROC_AUC`, `F1_Score`, `Recall`, and `F2_Score`. When tuning for `Recall` or `F2_Score`, Optuna dynamically explores `scale_pos_weight` (1.5 to 4.5) to directly penalize False Negatives.

```bash
# Tune XGBoost for PR-AUC with 30 trials (default settings from config)
python main.py --mode tune --model-type xgboost

# Tune XGBoost for Recall (prioritizing detection of churners)
python main.py --mode tune --model-type xgboost --n-trials 30 --tuning-metric Recall

# Tune XGBoost for F2-Score (weighting Recall twice as heavily as Precision)
python main.py --mode tune --model-type xgboost --n-trials 30 --tuning-metric F2_Score

# Tune Random Forest for F1-Score with 50 trials
python main.py --mode tune --model-type rf --n-trials 50 --tuning-metric F1_Score

# Tune Logistic Regression for ROC-AUC with 40 trials
python main.py --mode tune --model-type lr --n-trials 40 --tuning-metric ROC_AUC
```
*Outputs generated in:* `results/{model_type}/`
- `optuna_tuning_{model_type}_report.md` (Executive tuning report with best hyperparameters and Decile Lift breakdown)
- `{model_type}_tuning_trials.csv` (Full history of all trials, parameters, and CV metrics)
- `{model_type}_tuning_history.png` (Dual-panel optimization history & trial objective plot)
- `{model_type}_feature_importance_tuned.png`
- `lift_and_gains_curve.png`
- `{model_type}_decile_lift_table.csv`
- `{model_type}_baseline_lift_comparison.csv`

---

#### 4. Single Model Training (`--mode train`)
Trains a single model according to parameters in `config/config.yaml`, evaluates on the blind test partition, saves the serialized model artifact (`.json` for XGBoost, `.pkl` for scikit-learn models), exports feature importance, calculates Decile Lift, and produces an executive evaluation report.

```bash
# Train and save XGBoost
python main.py --mode train --model-type xgboost

# Train and save Random Forest
python main.py --mode train --model-type rf

# Train and save Logistic Regression
python main.py --mode train --model-type lr
```
*Outputs generated in:*
- `results/{model_type}/train_{model_type}_report.md` (Includes Confusion Matrix, Feature Importance, and Decile Lift table)
- `results/{model_type}/{model_type}_train_metrics.csv`
- `results/{model_type}/{model_type}_feature_importance.png` and `.csv`
- `results/{model_type}/lift_and_gains_curve.png`
- `results/{model_type}/{model_type}_decile_lift_table.csv`
- `results/{model_type}/{model_type}_baseline_lift_comparison.csv`
- `models/saved/{model_type}_churn.{json|pkl}`

---

#### 5. Recall Optimization & Baseline Lift Benchmarking (`--mode recall`)
Solves the **Accuracy Paradox (~86%) & Recall Dilemma (~50%)** inherent in imbalanced retail banking churn datasets (~20.4% base churn rate). Instead of relying on the default symmetric threshold ($p = 0.50$), this mode scans the validation partition without data leakage to discover operational decision thresholds tailored to business retention budgets:
- **Default (0.50):** Precision ~72%, Recall ~53%, Accuracy ~86.2%.
- **Optimal F1:** Threshold ~0.49, balancing Precision and Recall.
- **Optimal F2 (High Recall):** Threshold ~0.48, prioritizing customer retention coverage.
- **Target Recall ($\ge 75\%$ or $\ge 80\%$):** Threshold ~0.25–0.48, achieving **Recall $\ge 78\%$** while preserving **Precision Lift $\ge 2.0\text{x}$** over random contact.

Benchmarks against two baseline predictors:
1. **Random Guess Baseline (Stratified Prior):** Predicts churn by chance prevalence ($\approx 20.40\%$, Lift = 1.00x).
2. **Majority Class Baseline:** Predicts all customers stay (Accuracy 79.60%, Recall = 0.00%).

Computes a **10-Decile Lift Table** and **Cumulative Gains Curve** illustrating that **Top 10% (Decile 1) achieves 3.77x Lift** (77.00% churn rate) and **Top 20% (D1 + D2) captures ~60% of all churners** with **3.0x Cumulative Lift**.

```bash
# Optimize threshold to achieve >= 75% Recall (default target)
python main.py --mode recall --model-type xgboost --target-recall 0.75

# Target 80% Recall using Random Forest
python main.py --mode recall --model-type rf --target-recall 0.80

# Evaluate at a custom manual probability threshold (e.g. 0.35)
python main.py --mode recall --model-type xgboost --threshold 0.35
```
*Outputs generated in:* `results/{model_type}/`
- `recall_optimization_{model_type}_report.md` (In-depth analysis of threshold economics and retention campaign strategies)
- `lift_and_gains_curve.png` (300 DPI dual-panel figure: Cumulative Gains Curve + Decile Lift Bar Chart)
- `{model_type}_decile_lift_table.csv` (10-decile risk segmentation metrics D1–D10)
- `{model_type}_baseline_lift_comparison.csv` (Full comparison against Random Guess & Majority Class baselines)

---

#### 6. Customer Separability & Dimensionality Reduction (`--mode separability`)
Explores the high-dimensional feature geometry of churned ($y=1$) vs retained ($y=0$) customers using linear projections, non-linear manifolds, and statistical distributions:
- **PCA Projection (`pca_separability_2d.png`):** 2D linear projection, class centroids, 95% covariance confidence ellipses, scree plot of cumulative explained variance, and top feature loadings on $PC_1$ and $PC_2$.
- **t-SNE Manifold (`tsne_separability_2d.png`):** 2D non-linear manifold projection, density contour overlays, and localized churn rate danger zones (hexbin heatmap).
- **Categorical Churn Rate Bars (`categorical_churn_bars.png`):** Multi-panel bar charts showing churn rate (%) across categories with customer counts and population baseline comparisons.
- **Continuous Distribution Boxplots (`continuous_boxplots.png`):** Multi-panel boxplots with median notches, mean markers, and annotated Cohen's $d$ effect sizes and Mann-Whitney U test p-values.
- **Continuous KDE Distributions (`continuous_kde_distributions.png`):** Overlaid Kernel Density Estimation curves revealing distribution shifts and bimodality.
- **Separability Metrics Table (`feature_separability_metrics.csv`):** Information Value (IV), Chi-square, Kolmogorov-Smirnov (KS) statistic, and Cohen's $d$ ranking across all features.
- **Executive Report (`separability_analysis_report.md`):** Complete business diagnostic synthesis.

```bash
# Run full separability diagnostics suite via main CLI
python main.py --mode separability

# Or run standalone module
python -m visualization.separability --output-dir results/visualizations --sample-size 3000
```

---

#### 7. Temporal Behavioral Sequence Representation & Embeddings (`--mode representation`)
Extracts rich longitudinal customer behavioral dynamics directly from raw event streams (`data/full_dataset/` with ~500k transactions, 1.3M logins, 13k complaints, and 17k accounts) modularized under [`preprocessing/temporal/`](file:///d:/Introduction%20to%20BA/intro-to-ba/preprocessing/temporal/):

1. **Modular Sub-Extractors & Multi-Period Coordinator (`extractor.py` & `extractors/`):**
   Aggregates raw event streams across 6 monthly snapshots ($T=6$, April–September 2026) along **37 longitudinal behavioral channels** ($D=37$), constructing a standardized 3D tensor $(N, T=6, D=37)$ with bounded outlier clamping $[-5.0, 5.0]$:
   * **`TransactionSnapshotExtractor` (11 channels):** Inward/outward volume, net flow, transaction count, cancellation rate, ticket size, external transfer volume/ratio, ATM cash withdrawals, fee deductions, branch visits.
   * **`LoginSnapshotExtractor` (15 channels):** Login count, active days, daily session intensity, habit breadth, inactivity drought, weekend share, circadian night share, business hours share, intra-month decay, desktop share, device diversity, Shannon device entropy, failed logins, fail rate, lockout burst alarm.
   * **`ComplaintSnapshotExtractor` (5 channels):** Complaint count, average resolution delay, service SLA breach count, cumulative unresolved stock, escalation velocity ($\Delta$).
   * **`AccountSnapshotExtractor` (4 channels):** Active account stock, premature savings account liquidation, everyday account closure, manual renewal ratio.
   * **`CrossChannelSnapshotExtractor` (2 channels):** Login-to-transaction ratio (passive anxiety monitoring), failed login to complaint coupling.
   *(Legacy 10-channel extraction path is cleanly preserved via sub-extractor delegation).*
2. **Pluggable Manifold Representation Learning Strategies (`model.py` & `strategies/`):**
   * **Unsupervised Baseline (S0):** Bidirectional GRU autoencoder with reconstruction MSE loss ($\mathcal{L}_{\text{recon}}$), ensuring strict zero target leakage.
   * **Asymmetric ArcFace (S1):** Angular margin penalty ($m_{\text{churn}} > 0, m_{\text{loyal}} = 0$) forcing high-risk churn vectors into a concentrated hyperspherical cone.
   * **Hyperbolic Poincaré Ball (S2):** Conformal Riemannian metric modeling tree-like hierarchy from core loyal banking to peripheral churn.
   * **Optimal Transport (S3):** Wasserstein distance drift regularization ($\vec{v} = z_{t=6} - z_{t=1}$) capturing trajectory acceleration.
   * **Neyman-Pearson Constrained Optimization (S4):** Primal-dual Lagrangian minimizing churn risk subject to an upper bound on False Alarm Rate ($\text{FPR} \le \alpha$).
3. **Feature Space Augmentation & High-Performance Caching (`manager.py`):**
   Concatenates latent vectors $[z_0, \dots, z_{15}]$ and manifold metrics (`temporal_drift_velocity`, angular margins) to tabular customer records. Automatically caches embeddings to `data/processed/temporal_embeddings.csv` and serializes model weights to `models/saved/temporal/` for sub-second downstream inference.
4. **Hybrid Multi-Metric Feature Selection (`feature_selection.py`):**
   Offers Pearson correlation (`Pearson`), Mutual Information (`MI_Score`), and a weighted hybrid score (`MI_Pearson_Hybrid`):
   $$\text{Score}_{\text{hybrid}} = w_p \cdot \text{Pearson}_{\text{norm}} + (1 - w_p) \cdot \text{MI}_{\text{norm}}$$
   preserving both non-linear dependencies (e.g. U-shaped `products_number`) and monotonic trends (e.g. `age`, `recency_days`).

*Outputs generated in:* `results/temporal/`
- `temporal_embedding_comparison.csv` (Downstream performance table comparing baseline tabular vs. temporal augmented models)

```bash
# Run temporal representation learning benchmark suite via main CLI
python main.py --mode representation --temporal-epochs 25

# Train single model augmented with 16D Bi-GRU temporal embeddings
python main.py --mode train --model-type xgboost --enable-temporal-embeddings

# Run Optuna tuning with hybrid Pearson/MI feature selection and temporal embeddings
python main.py --mode tune --model-type xgboost --enable-temporal-embeddings --feature-selection-metric MI_Pearson_Hybrid --hybrid-pearson-weight 0.50 --top-k-features 50
```

---

### Command-Line Arguments Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mode` | string | `compare` | Operational mode: `compare`, `ablation`, `tune`, `train`, `recall`, `separability`, `eda`, or `representation`. |
| `--model-type` | string | `xgboost` | Target classifier: `xgboost`, `lr`, or `rf`. (Ignored in `compare` mode). |
| `--config` | string | `config/config.yaml` | Path to custom YAML configuration file. |
| `--n-trials` | integer | `null` (uses config) | Number of Optuna optimization trials (in `tune` mode). |
| `--tuning-metric` | string | `null` (uses config) | Target metric for Optuna: `PR_AUC`, `ROC_AUC`, `F1_Score`, `Recall`, `F2_Score`, or `Precision`. |
| `--target-recall` | float | `0.75` | Target minimum recall constraint for threshold optimization in `recall` mode. |
| `--threshold` | float | `null` | Optional manual decision threshold override in `recall` or `train` mode (e.g. `0.35`). |
| `--sample-weights` | flag | `False` | Enables composite sample weighting: exponential recency decay + financial balance value weighting + churn multiplier. Supported across all models (`xgboost`, `lr`, `rf`). |
| `--focal-loss` | flag | `False` | Enables Focal Loss objective to dampen gradient of easy negative samples and focus learning on hard decision boundaries. Supported on `xgboost` and `lr`. |
| `--focal-gamma` | float | `2.0` | Focusing parameter $\gamma$ for Focal Loss (reduces easy sample gradients when $\gamma > 0$). |
| `--focal-alpha` | float | `0.25` | Positive class balancing factor $\alpha$ for Focal Loss (churn label weight). |
| `--decay-half-life`| float | `60.0` | Half-life $t_{1/2}$ in days for exponential recency decay weighting. |
| `--enable-temporal-embeddings` | flag | `False` | Enables 6-month longitudinal event sequence extraction from raw logs and Bi-GRU Autoencoder embeddings ($z$). |
| `--temporal-dim` | integer | `16` | Latent embedding dimension $z$ for temporal behavioral representation. |
| `--temporal-epochs` | integer | `25` | Number of training epochs for Bi-GRU Sequence Autoencoder. |
| `--feature-selection-metric` | string | `null` (uses config) | Feature selection metric: `MI_Score`, `Pearson`, `MI_Pearson_Hybrid`, or `RF_Importance`. |
| `--hybrid-pearson-weight` | float | `0.50` | Weight $w_p \in [0, 1]$ of Pearson correlation vs MI in `MI_Pearson_Hybrid`. |
| `--top-k-features` | integer | `null` (uses config) | Number of top features to select after multi-metric ranking. |
| `--rebuild-dataset`| flag | `False` | Forces multi-table ETL rebuild from `data/full_dataset/`. |
| `--output-dir` | string | `null` (auto-routed) | Custom directory for outputs; defaults to `results/{comparison\|lr\|rf\|xgboost\|temporal}`. |
| `--report-name` | string | `null` (auto-templated)| Custom report filename; defaults to `{mode}_{model}_report.md`. |

---

### 6. Asymmetric Loss Interventions & Sample Weighting (Shifting the Pareto Frontier)

Adjusting decision thresholds $\tau$ or changing Optuna objective metrics traverses the existing ROC/PR curve, trading False Positives (FP) for False Negatives (FN). To shift the trade-off frontier outward, the system introduces two mathematical loss-level interventions:

#### Phương án 1: Trọng số mẫu kết hợp (Composite Sample Weighting)
*Supported across all 3 models: XGBoost, Random Forest, Logistic Regression.*

Computes individualized observation weights $w_i$:
$$w_i = w_{\text{recency}}(i) \cdot w_{\text{value}}(i) \cdot w_{\text{churn}}(i)$$
1. **Exponential Recency Decay:** Prioritizes customers with recent activity signals while older observations decay:
   $$w_{\text{recency}}(i) = \exp\left(-\frac{\ln(2)}{t_{1/2}} \cdot \text{recency\_days}_i\right)$$
2. **Financial Value Weighting:** Heavily penalizes churning high-balance customers who inflict significant CLV loss:
   $$w_{\text{value}}(i) = 1.0 + \alpha \cdot \ln\left(1 + \frac{\max(\text{balance}_i, 0)}{\text{median}(\text{balance}^+)}\right)$$
3. **Churn Multiplier:** Scales minority positive class penalization during tree splits or loss calculation.
4. **SMOTE Weight Expansion:** Extrapolates synthetic minority instances with the mean weight of real positive instances, then normalizes so $\text{mean}(w) = 1.0$ (preserving gradient scale).

```bash
# Train XGBoost with sample weights
python main.py --mode train --model-type xgboost --sample-weights

# Train Random Forest with sample weights
python main.py --mode train --model-type rf --sample-weights

# Compare all 3 models side-by-side with sample weights
python main.py --mode compare --sample-weights
```

#### Phương án 2: Hàm mất mát Focal Loss (Custom Gradient Objective)
*Supported on XGBoost (custom objective) and Logistic Regression (`FocalLossLogisticRegression` via L-BFGS-B).*

Standard Cross-Entropy Loss treats all samples symmetrically. Under extreme imbalance (~80% non-churn vs ~20% churn), gradient descent is dominated by easy, well-classified active customers. **Focal Loss** dynamically modulates gradient magnitude:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \ln(p_t)$$
Where $p_t$ is model confidence on the true class. When $\gamma = 2.0$, an easy sample classified with $p_t = 0.9$ receives a modulating factor $(1 - 0.9)^2 = 0.01$ (a **100x reduction in gradient influence**), forcing the booster or linear solver to focus strictly on borderline cases and hard churners.

```bash
# Train XGBoost with Focal Loss
python main.py --mode train --model-type xgboost --focal-loss --focal-gamma 2.0 --focal-alpha 0.25

# Train Logistic Regression with Focal Loss
python main.py --mode train --model-type lr --focal-loss

# Run feature ablation suite including both Sample Weighting and Focal Loss
python main.py --mode ablation --model-type xgboost
```

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

  # Feature Selection (MI, Pearson, MI_Pearson_Hybrid, RF_Importance)
  feature_selection:
    top_k_features: null                  # Set an integer (e.g., 50) or null to keep all
    feature_selection_metric: "MI_Score"  # "MI_Score" | "Pearson" | "MI_Pearson_Hybrid" | "RF_Importance"
    hybrid_pearson_weight: 0.50           # Weight of Pearson vs MI (0.0 to 1.0)
    compute_feature_importance: false

  # Temporal Multi-Snapshot Sequence Embeddings (Self-Supervised Bi-GRU)
  temporal_embeddings:
    enabled: false                        # Set to true or use --enable-temporal-embeddings
    raw_data_dir: "data/full_dataset"
    latent_dim: 16                        # Dimensionality of latent embeddings z (temporal_z0...z15)
    hidden_dim: 64                        # Bi-GRU hidden state size
    num_layers: 2                         # Number of recurrent layers
    epochs: 25                            # Training epochs for sequence autoencoder
    batch_size: 256
    learning_rate: 0.001
    dropout: 0.20
    checkpoint_dir: "models/saved/temporal"
    cache_path: "data/processed/temporal_embeddings.csv"
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
│   ├── model_comparison_report.md          # Executive comparison report
│   ├── model_comparison_metrics.csv        # Metrics across LR, RF, XGBoost
│   ├── model_comparison_chart.png          # 300 DPI comparative bar chart
│   ├── feature_importance_*.png            # Top features for each model
│   ├── champion_lift_and_gains_curve.png   # Champion model Gains & Lift chart
│   └── {model}_decile_lift_table.csv       # Decile lift table per model
├── xgboost/
│   ├── train_xgboost_report.md             # Single model training report with Lift section
│   ├── ablation_xgboost_report.md          # 8-scenario ablation analysis with Lift section
│   ├── optuna_tuning_xgboost_report.md     # Tuning report with best params & Lift section
│   ├── recall_optimization_xgboost_report.md # Dedicated high-recall & lift report
│   ├── lift_and_gains_curve.png            # 300 DPI dual-panel Gains + Decile Lift chart
│   ├── xgboost_decile_lift_table.csv       # 10-decile risk distribution (D1–D10)
│   ├── xgboost_baseline_lift_comparison.csv # Performance vs Random & Majority baselines
│   ├── xgboost_train_metrics.csv           # Holdout test metrics
│   ├── xgboost_feature_importance.png      # Top features by Gain
│   ├── xgboost_tuning_trials.csv           # Optimization trial history
│   └── xgboost_tuning_history.png          # Dual-panel Optuna optimization plot
├── rf/
│   └── ...                                 # Same artifacts for Random Forest
└── lr/
    └── ...                                 # Same artifacts for Logistic Regression
```

### Visual Artifacts Highlights
- **Cumulative Gains & Decile Lift Charts (`visualization/plots.py`):** 300 DPI dual-panel figures displaying cumulative churners captured against a 45° random line (Panel 1) and relative lift multipliers across risk deciles D1–D10 (Panel 2).
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
| **Top-Decile Lift (D1)** | $\ge 3.0\text{x}$ | **Targeting Efficiency:** Measures concentration of actual churners in the top 10% riskiest segment vs. random guessing (achieves **3.77x**). |
| **Cumulative Lift (Top 20%)** | $\ge 2.5\text{x}$ | **Retention Campaign ROI:** Confirms that contacting the top 20% highest-risk customers captures $\approx 60\%$ of all impending churners (achieves **3.0x**). |
| **Log-Loss** | Lower is better | Measures the calibration of predicted churn probabilities, critical for tiered customer intervention. |

---

## Continuous Deployment (CI/CD) & Public Hosting Guide

Anyone can access the live dashboard via a public web link. The platform is designed with a **Unified Single-Port Architecture**: when built, FastAPI automatically serves the compiled React 19 single-page application (`frontend/dist`) alongside the REST API, meaning **only one cloud service is needed** and there are **zero CORS issues**.

### 1. Free Cloud Hosting Options

| Platform | Free Plan Specs | Best For | URL Format |
|---|---|---|---|
| **Render.com** | 512 MB RAM, 0.1 CPU, Free Web Service | Quickest setup, native git push integration | `https://babank-analytics.onrender.com` |
| **Hugging Face Spaces** | **16 GB RAM**, 2 vCPUs, Free Docker Space | Heavy ML / deep learning models (PyTorch + RF) | `https://huggingface.co/spaces/<user>/<repo>` |
| **Railway.app** | $5 monthly credit, fast builds | Developer control & instant build | `https://babank.up.railway.app` |

---

### 2. Deploying on Render (5-Minute Guide)

1. Sign up / Log in to [Render.com](https://render.com).
2. Click **New +** $\rightarrow$ **Web Service**.
3. Connect your GitHub repository (`intro-to-ba`).
4. Configure the service settings:
   - **Name:** `babank-retention-platform` (or any custom name)
   - **Environment:** `Python` (or `Docker` using the included `Dockerfile`)
   - **Region:** Singapore / Frankfurt / Oregon (nearest to users)
   - **Branch:** `main`
   - **Build Command:**
     ```bash
     pip install -r requirements.txt && (cd frontend && npm install && npm run build) && python api.py prepare --download && python api.py official-data --download && python api.py segmentation
     ```
   - **Start Command:**
     ```bash
     uvicorn api:app --host 0.0.0.0 --port $PORT
     ```
5. Click **Create Web Service**. Render will build the React frontend into `frontend/dist`, download official data, export model checkpoints, and start the app.
6. Once deployed, Render provides a public URL (e.g. `https://babank-retention-platform.onrender.com`) that anyone can open in their browser.

---

### 3. Automated Model Updates via GitHub Actions

Whenever you train a new model, tune hyperparameters, or push new files in `models/saved/`, GitHub Actions automatically triggers testing and redeploys the service so your public link is always up-to-date.

Workflow configuration is located in [`.github/workflows/deploy.yml`](file:///.github/workflows/deploy.yml):

```
Push new model / commit to main
               │
               ▼
┌─────────────────────────────────┐
│  GitHub Actions: deploy.yml     │
│  1. Run model tests (unittest)  │
│  2. Build React bundle (Vite)   │
└─────────────────────────────────┘
               │ (Pass)
               ▼
┌─────────────────────────────────┐
│  Automated Cloud Deployment     │
│  - Render Deploy Hook Triggered │
│  - Or Render Git Auto-Deploy    │
└─────────────────────────────────┘
               │
               ▼
🚀 Public URL Updated with New Model!
```

#### Setting Up the GitHub Actions Deploy Hook:
1. In your Render Dashboard, open your Web Service $\rightarrow$ **Settings**.
2. Scroll to **Deploy Hook** and copy the URL (format: `https://api.render.com/deploy/srv-xxxx?key=yyyy`).
3. In your GitHub repository, go to **Settings** $\rightarrow$ **Secrets and variables** $\rightarrow$ **Actions**.
4. Click **New repository secret**:
   - **Name:** `RENDER_DEPLOY_HOOK_URL`
   - **Secret:** (paste your Render Deploy Hook URL)
5. From now on, whenever you push to `main` (including updating `models/saved/xgboost_churn.json` or `models/saved/rf_churn.pkl`), GitHub Actions will run tests and automatically refresh your live public web application!

---

## System Telemetry & Monitoring Architecture

The live monitoring dashboard (`/monitoring`) reports real hardware metrics and dataset facts via `psutil` and FastAPI middleware.

### Why CPU Utilization Previously Showed 0%

In Python's `psutil` library, `Process.cpu_percent(interval=None)` calculates CPU utilization by comparing:
$$\frac{\Delta \text{Process CPU Time}}{\Delta \text{System Wall-Clock Time}}$$
between **two consecutive calls on the exact same `psutil.Process` instance**.

- When `proc = psutil.Process()` was instantiated inside the request handler, every incoming HTTP request created a brand new `Process` instance.
- In `psutil`, the **first call on any new instance always returns 0.0% by design** because there is no prior baseline to compare against.
- As a result, every poll was an initial baseline call, perpetually displaying `0%`.

### The Resolution

1. **Persistent Singleton Process (`_PROCESS`):** A module-level persistent `_PROCESS = psutil.Process()` is instantiated once at server startup and primed immediately to establish the baseline.
2. **Dual Process & System Telemetry:** Subsequent requests calculate the real delta of CPU activity utilized by the Python process (normalized across logical CPU cores) and compare it against the active operating system CPU load (`psutil.cpu_percent(interval=None)`).
3. **Active Spike Capture:** While idle between requests, the server uses negligible CPU (~0.5%-2%), but during model inference (e.g. batch scoring 10,000 customers with XGBoost), the metric accurately reflects the real CPU core activity spike.

---

*Module 5: Model Development & Evaluation — Business Analytics Capstone Project.*