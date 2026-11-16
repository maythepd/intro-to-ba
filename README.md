# BaBank Analytics

## Group project context

The upstream project covers Churn Prediction and Segmentation Modules 1 2 and 3, including business objectives, KPI definitions, synthetic data generation, exploratory analysis and data cleaning. The group dataset is hosted on Kaggle. See [the synthetic data dictionary](synth_data_dict.md) and the [Module 1 document](https://docs.google.com/document/d/1t4vT18zHOeCOTXPCxd7UgwmqREK4O0zQJOycYO0oZ-M/edit?usp=sharing).

Bank-staff analytics interface for customer churn risk, with an XGBoost API. The `demo` branch is isolated from the group's `main` branch.

## Official dataset

The project's **source of truth** is [Full Dataset Intro2BA on Kaggle](https://www.kaggle.com/datasets/maythepd/full-dataset-intro2ba), as referenced in the current `main` README and `synth_data_dict.md`. It contains six related behavioral tables (`accounts`, `login`, `transactions`, `savings_cycles`, `complaints`, `customer_context`) and uses a **2026-10-01 exclusive snapshot boundary**. The original Bank Customer Churn Prediction table provides customer profiles and labels.

The version in `data/cleaned_bank_data.csv` is **legacy experimental data from `model/xgboost`**. It must not be used as the official dataset for the live demo.

Download and aggregate the official tables into one model-ready customer feature table:

```bash
pip install -r requirements.txt
python -m api.official_data --download
```

The output is `data/official/customer_features.csv`; local materialized Kaggle data is generated at runtime and must not be committed. If the Kaggle API is inaccessible or a file's schema differs from the documented dictionary, the preparation task should fail rather than silently use artificial records. The aggregation rules in `api/official_data.py` are **derived features**, not original columns in Kaggle. Review their appropriateness before presentation.

## Temporary XGBoost model

```bash
python -m api.export_bundle
uvicorn api.main:app --reload
```

This trains a **temporary experimental model** using the official feature table (when available) and saves `artifacts/churn_bundle.joblib` containing the fitted preprocessing and trained XGBoost model. It is *not* the final group checkpoint. `/health` indicates if a fitted bundle is loaded; `/api/predict` returns 503 without it. Later replace the bundle via `MODEL_BUNDLE_PATH` without changing the API schema, provided the saved pipeline and expected features are compatible.

Training currently uses the existing classification trainer's holdout for early stopping, so report its metrics as illustrative development results only, **not independent generalization performance**. Before final submission, train with separate train/validation/test partitions.

## Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Set `VITE_API_BASE_URL=http://localhost:8000` or your hosted API URL. Customer listing and lookup now depend on the official-data API. If the API is unavailable, no synthetic customer records are silently substituted. Some dashboard charts are explicitly labelled as *Illustrative* until actual measurements are provided.

## Deployment

A Render API Blueprint is in `render.yaml`; it downloads Kaggle data and trains the temporary model at build time. This requires outbound access to Kaggle and sufficient build resources. Deployment has **not yet been verified**. Set `CORS_ORIGINS` to the frontend origin. Frontend can be deployed as a Vite static site, with SPA rewrite fallback.

API docs after startup: `http://localhost:8000/docs`. Segmentation still requires the team's clustering model; `/api/segments` returns 503 until then.

Do not present the study data, synthetic extension, or temporary trained model as live bank operational information.

## Module 5 model code & evaluation results (merged into `demo`)

The upstream `maythepd/intro-to-ba` branch `model/xgboost` has been brought into
[`model_development/`](model_development/), preserving its own `main.py`,
`config/`, `models/` (Logistic Regression, Random Forest, XGBoost, tuning/recall utilities and saved experimental weights),
`preprocessing/`, `visualization/`, and `results/` (CSV tables, Markdown reports, and PNG plots).
This keeps the existing `demo` API/frontend and its official-data workflow intact rather than overwriting them
with the model-development branch's divergent preprocessing and data assumptions.

To reproduce experiments after obtaining the required source data:

```bash
cd model_development
pip install -r requirements.txt
python main.py --mode compare
python main.py --mode train --model-type xgboost
```

See [the original model-development guide](model_development/README.md) for the full CLI,
expected input data (`data/full_bank_data.csv` or raw tables in `data/full_dataset/`), and experiment variants.
The input dataset required by that pipeline is **not tracked** in this integration; it must be provided separately.

The **Models** page shows the committed [comparison metrics](model_development/results/comparison/model_comparison_metrics.csv)
for all three classifiers; `GET /api/metrics` reads this CSV instead of returning the previous hard-coded values.
For XGBoost the reported ROC-AUC is 0.8372, PR-AUC 0.6060, F1 0.3389, recall 1.0000,
and precision 0.2040. Those figures describe the research study, **not** current deployed predictions.
The committed comparison report's stated test count should be reconciled with the CSV's `X_test_Shape`
before any final presentation.

**Model compatibility:** files in `model_development/models/saved/` are archival checkpoints and must
not be loaded directly as `artifacts/churn_bundle.joblib`; the demo endpoint requires a fitted
preprocessing pipeline plus its compatible feature order. The `api/export_bundle.py` workflow remains
separate until an explicitly validated model export is built. This merge does not claim that the
experimental model has been deployed or that the end-to-end API has been tested on the new data.

## KMeans + hierarchical customer segmentation

Research sources have been preserved in `clustering_research/kmeans/` (source branch `feature/module-5b-kmeans`) and `clustering_research/hierarchical/` (source branch `feature/hierarchical-cluster`). These research trees are references: they depend on their original feature/ETL context and are not imported directly into the production API.

The integrated, tested serving adapter lives in `api/segmentation_service.py` and uses ONLY `data/official/customer_features.csv` prepared by the official six-table Kaggle workflow. It refits both models on that snapshot rather than representing incompatible archived files as deployed models. It uses the same approved original and behavioral feature columns with 50/50 group-balanced preprocessing, without using churn as a clustering input.

```bash
pip install -r requirements.txt
python -m api.official_data --download
python -m api.segmentation_service  # saves artifacts/segmentation/cluster_bundle.joblib
uvicorn api.main:app --reload
cd frontend && npm install && npm run dev
```

The API reports **503** if the clustering bundle is missing or its SHA-256 does not match the official feature table; no fabricated fallback is used. `/health` exposes `segmentation_loaded`. The included Render blueprint runs the segmentation export at build time.

Endpoints:
- `GET /api/segments?algorithm=kmeans` or `algorithm=hierarchical`: cluster counts, profiles, descriptive churn, evaluation sample silhouette, method and provenance.
- `GET /api/segments/customers/{customer_id}`: both assignments for a known customer in the fitted official snapshot.
- `POST /api/segments/predict?algorithm=kmeans|hierarchical`: accepts the same validated `CustomerInput` JSON used for churn prediction, and assigns a new customer to a cluster.

**Important hierarchical limitation:** the original upstream Ward checkpoint has no customer-ID-to-leaf mapping and cannot directly predict new customers. For this serving adapter we fit a fresh Ward hierarchy on up to 1,500 customers drawn deterministically from the official snapshot, retain exact Ward labels for the fitted subset, and assign all other customers (and new incoming records) to the nearest cluster centroid in the same fitted feature space. Those assignments are **approximations**, explicitly identified in responses and UI, not native hierarchical predictions or the original upstream cluster IDs. Upstream k=4 hierarchical business labels and metrics must not be mixed with this new fit.

The UI `/segments` switches between KMeans and Ward, visualizes actual server-returned cluster sizes and profiles, and supports customer-ID lookup. Customer detail pages show both assignments. Models are experimental descriptive clustering of a synthetic-extended dataset, **not operational customer segmentation**. No cloud deployment has been verified.

Tests:
```bash
python -m unittest discover -s tests -p 'test_segmentation_api.py' -v
cd frontend && npm run build
```

## Batch churn scoring and live-inference risk distribution

The dashboard **Customers Requiring Attention** tile and **Risk Level Distribution** now use model inference on all rows of the official snapshot, not illustrative percentages. The historical EDA KPIs and the separately labeled **Churn Risk Trend** remain historical/illustrative; batch results are not a measured time series.

Workflow: download/materialize official Kaggle data, export the fitted temporary XGBoost bundle, start FastAPI, then click **Run batch scoring** on the Dashboard or call:

```bash
curl -X POST 'http://localhost:8000/api/predict/batch'
curl 'http://localhost:8000/api/dashboard/risk-summary'
# Optionally retrieve every customer score; large response for full snapshot:
curl -X POST 'http://localhost:8000/api/predict/batch?include_predictions=true'
# Force a new complete batch calculation:
curl -X POST 'http://localhost:8000/api/predict/batch?force=true'
```

- `POST /api/predict/batch` scores **every** loaded customer, in bounded chunks (default 1,000); by default it returns a summary to avoid transferring 10,000 individual records. Use `include_predictions=true` to return the customer ID, churn probability, predicted churn and risk level for every row. `force=true` recalculates; otherwise later requests reuse the completed in-process cache.
- `GET /api/dashboard/risk-summary` returns the most recent successful batch-scoring counts and timestamp. It returns HTTP **503** before the first completed run, or when the model or official data is unavailable. The UI shows a **Run batch scoring** button rather than fabricating values.
- Risk thresholds are **Low** p < 0.30, **Medium** 0.30 ≤ p < 0.70, **High** p ≥ 0.70. **Customers Requiring Attention = High count**, not Medium+High. These thresholds match single-customer inference and are heuristic business cutoffs, not externally validated intervention rules.
- Both single and batch inference share `api/scoring.py`; the fitted feature list and order are preserved. True `churn` labels are never used as model features, and unexpected model outputs fail closed. If any batch fails, the previously completed snapshot remains cached; there is no partially updated distribution.
- Batch results are stored **only in the API process's memory**, not persisted to the repository or a database. They reset upon API restart and are not shared across multiple Uvicorn workers. For scale and authenticated production use, add a scheduled worker, persistent versioned scoring table, authentication/authorization and request throttling. The current public demo is a **synthetic-extended study dataset** and the trained bundle is experimental; these are actual computations on that model, **not** bank production measurements or independently validated performance.
