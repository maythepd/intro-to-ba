# BaBank Analytics

## Group project context

The upstream project covers Churn Prediction and Segmentation Modules 1 2 and 3, including business objectives, KPI definitions, synthetic data generation, exploratory analysis and data cleaning. The group dataset is hosted on Kaggle. See [the synthetic data dictionary](synth_data_dict.md) and the [Module 1 document](https://docs.google.com/document/d/1t4vT18zHOeCOTXPCxd7UgwmqREK4O0zQJOycYO0oZ-M/edit?usp=sharing).

Bank-staff analytics interface for customer churn risk, with an XGBoost API. The `demo` branch is isolated from the group's `main` branch.

## Official dataset

The project's **source of truth** is [Full Dataset Intro2BA on Kaggle](https://www.kaggle.com/datasets/maythepd/full-dataset-intro2ba), as referenced in the current `main` README and `synth_data_dict.md`. It contains six related behavioral tables (`accounts`, `login`, `transactions`, `savings_cycles`, `complaints`, `customer_context`) and uses a **2026-10-01 exclusive snapshot boundary**. The original Bank Customer Churn Prediction table provides customer profiles and labels.

The version in `data/cleaned_bank_data.csv` is **legacy experimental data from `model/xgboost`**. It must not be used as the official dataset for the live demo.

Prepare the Module 5 customer table and the simpler segmentation snapshot:

```bash
pip install -r requirements.txt
python -m api.prepare_module5 --download
python -m api.official_data --download
```

Module 5 uses its original `BankDataETL` and pinned Kaggle release 3 to produce `data/official/module5_customer_features.csv`. The second command produces `data/official/customer_features.csv` for segmentation. These tables use different aggregation formulas and must not be substituted for each other. Generated data and fitted preprocessing remain untracked. Missing/incompatible sources fail the build.

## Module 5 XGBoost — default serving model

```bash
python -m api.prepare_module5 --download
uvicorn api.main:app --reload
```

The API loads the committed `model_development/models/saved/xgboost_churn.json` directly. Preparation fits only the Module 5 preprocessing on the original training split and writes `artifacts/module5/preprocessor.joblib` and `manifest.json`. It does not retrain XGBoost. The manifest binds checkpoint, preprocessing, config, dataset, feature order and report hashes; startup rejects mismatches. The legacy churn bundle and `MODEL_BUNDLE_PATH` are no longer used.

Before publishing preprocessing, the saved checkpoint must reproduce the committed Module 5 test metrics to four decimal places: ROC-AUC 0.8372, PR-AUC 0.6060, F1 0.3389, precision 0.2040, recall 1.0000 and accuracy 0.2040. These are held-out study results, not ongoing deployment performance. `/health` exposes `model_source=module5_checkpoint`, `feature_count=133`, `checkpoint_sha256` and `study_reproduced=true`. Missing preparation returns 503 without a legacy fallback.

Staff lookup resolves all raw Module 5 aggregates from the customer ID. The API supports a new customer only when the complete `raw_features` input is supplied; it returns 422 for missing aggregates. Transaction/login/complaint summaries use Module 5 formulas, including fractional recency and `No_Channel`. See [the serving guide](docs/module-5-serving.md).

## Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Set `VITE_API_BASE_URL=http://localhost:8000` or your hosted API URL. Customer listing and lookup now depend on the official-data API. If the API is unavailable, no synthetic customer records are silently substituted. Some dashboard charts are explicitly labelled as *Illustrative* until actual measurements are provided.

## Deployment

A Render API Blueprint is in `render.yaml`; it prepares and validates Module 5 preprocessing, materializes the segmentation snapshot, and exports clustering artifacts at build time. Set `CORS_ORIGINS` to the frontend origin. For services configured manually in the Render dashboard, copy the updated `buildCommand` there as well. The API and frontend both need redeployment for this switch; confirm the Module 5 identity fields in `/health` after deployment. Frontend uses a Vite static site with SPA rewrite fallback.

API docs after startup: `http://localhost:8000/docs`. Run `python -m api.segmentation_service` to prepare clustering; `/api/segments` returns 503 without its matching artifacts.

The application uses a synthetic-extended study snapshot, not live bank operational information.

## Module 5 model code & evaluation results (merged into `demo`)

The upstream `maythepd/intro-to-ba` branch `model/xgboost` has been brought into
[`model_development/`](model_development/), preserving its own `main.py`,
`config/`, `models/` (Logistic Regression, Random Forest, XGBoost, tuning/recall utilities and saved experimental weights),
`preprocessing/`, `visualization/`, and `results/` (CSV tables, Markdown reports, and PNG plots).
The serving adapter in `api/module5.py` uses that branch's preprocessing and saved XGBoost checkpoint.
Segmentation continues to use its own documented snapshot and adapter.

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

**Model compatibility:** serving uses the XGBoost JSON plus separately fitted Module 5 preprocessing,
not the LR/RF pickle files or a legacy churn bundle. A build succeeds only when the exact 133-feature
schema and the committed study metrics are reproduced. The actual serving identity is exposed by `/health`.

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

Workflow: run `python -m api.prepare_module5 --download`, start FastAPI, then click **Run batch scoring** on the Dashboard or call:

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
- The dashboard's batch cache resets on restart. Module 7 also records deduplicated, versioned monitoring observations in PostgreSQL; repeated scoring of the same snapshot does not become new incoming data.

## Module 7 monitoring

The `/monitoring` dashboard now uses measured reference/current distributions, labeled window metrics and persistent alert history. It distinguishes incoming API requests, fixed snapshot batches and controlled study replay. Configure `DATABASE_URL` and `MONITORING_ADMIN_TOKEN`, run `python -m monitoring.migrate`, and generate the course demo with `python -m monitoring.simulate`.

See [the complete monitoring guide](docs/module-7-monitoring.md) for rules, delayed-label import, PostgreSQL/RLS setup, API/report downloads, scheduler installation, retention and retraining actions. The [submission notebook](notebooks/module7_monitoring.ipynb) explains the controlled scenarios. Monitoring does not fill missing labels or turn a simulated time series into a live bank measurement.
