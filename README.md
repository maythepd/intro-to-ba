# BaBank Analytics

## Group project context

The upstream project covers Churn Prediction and Segmentation Modules 1 2 and 3, including business objectives, KPI definitions, synthetic data generation, exploratory analysis and data cleaning. The group dataset is hosted on Kaggle. See [the synthetic data dictionary](synth_data_dict.md) and the [Module 1 document](https://docs.google.com/document/d/1t4vT18zHOeCOTXPCxd7UgwmqREK4O0zQJOycYO0oZ-M/edit?usp=sharing).

Bank-staff analytics interface for customer churn risk, with an XGBoost API. The `demo` branch is isolated from the group's `main` branch.

Five-batch synthetic monitoring is implemented: independent optional outcomes, versioned replay, fixed-reference drift, transactional reports, authenticated admin API and dashboard batch history. See [the implementation, execution plan and reproducible commands](docs/synthetic-demo-monitoring.md). Synthetic performance is illustrative and does not measure real-customer accuracy.

The [five-model study replay](docs/multi-model-monitoring.md) adds frozen LR/RF challengers, KMeans and a Ward-centroid surrogate on shared cohorts, with 25 durable reports and a comparison mode on the monitoring page. XGBoost remains the serving model.

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

## Dashboard user guide (English)

Open [BaBank Dashboard](https://babank-dashboard.onrender.com/). The backend API documentation is available at [FastAPI Swagger UI](https://babank.onrender.com/docs). This deployment is a study/demo, not a live bank customer system. An API online indicator confirms connectivity, not the availability of every report.

### Navigation and common tasks

| Sidebar page | How to use it |
|---|---|
| Overview | Review the customer retention overview. Click **Run batch scoring** if risk counts are unavailable. Inspect the resulting risk distribution and high-risk customer count. The Churn Risk Trend chart is illustrative. |
| Customers | Search by customer ID, country, channel or device. Click a customer row to inspect the profile, behavioral information and available model results. The global search also opens the customer directory. |
| Churn Prediction | Enter a known customer ID and use snapshot lookup to populate the required fields. Review the form and submit it for XGBoost scoring. For a new customer, the API requires all raw Module 5 aggregates; an incomplete request is rejected. Read the churn probability and risk level in Model Results, then open View Retention Recommendations if needed. |
| Segmentation | Switch between KMeans and Hierarchical. Inspect cluster sizes and profiles, then use Find a customer's clusters to look up a known ID in both models. Ward assignments for new points use an approximate nearest-centroid rule. Cluster numbers are not risk rankings. |
| Retention | Review suggested interventions based on risk signals. These are decision-support suggestions, not causal evidence or automatically executed campaigns. |
| Monitoring | Compare all five models or inspect the serving champion, as described below. |
| Models | Compare the committed Module 5 holdout metrics for Logistic Regression, Random Forest and XGBoost. These are study test metrics rather than current monitoring performance. |

Risk levels are Low (p < 0.30), Medium (0.30 <= p < 0.70), and High (p >= 0.70). Customers Requiring Attention counts High only. These business cutoffs differ from the classification decision threshold of 0.5.

### Monitoring: Multi-model study replay

1. Open **Monitoring**, select **Multi-model study replay**, and choose **All five models · comparison** (or a specific algorithm).
2. Select a **Campaign**, preferably a complete campaign with simulated labels if classification performance is required. A complete campaign contains 25 model–batch reports; pending or unavailable results are not zero.
3. Select **All five batches** or an individual batch. The synthetic cohorts are normal_01, normal_02, normal_03, engagement_shift and friction_shift, each containing 2,000 distinct customer IDs. The three normal batches establish the baseline. Window dates are simulated.
4. In **Classification monitoring**, choose a Chart metric (ROC-AUC, average precision, F1, recall, accuracy, balanced accuracy, log loss or Brier). Compare XGBoost (green), Logistic Regression (yellow) and Random Forest (orange) across batches. Review the table, confusion matrix, predicted-positive rate, churn prevalence, baseline AP and AP drop. The probability histograms compare each model against its original training reference.
5. In **Clustering monitoring**, select silhouette, DBI, CHI, p95 distance or outlier share. Compare KMeans and Hierarchical (Ward + nearest-centroid surrogate), inspect cluster shares and choose an algorithm and cluster in Cluster profile drift. Silhouette and CHI are higher-is-better; DBI and distance are lower-is-better. Cluster share changes describe population composition, not individual migration.
6. In **Input data drift**, select the classifier domain profiles or segmentation inputs, then a feature. Inspect PSI (numeric), Jensen–Shannon distance (categorical), missingness, new categories, reference/current distributions and critical share. These are shared input references, not a ranking of the models.
7. Use Campaign JSON, Campaign HTML or Campaign CSV to export the full campaign or selected task as indicated by the UI.

ROC-AUC, AP, F1, recall and accuracy generally favor higher values; log loss and Brier favor lower values. Read AP together with prevalence. A high recall can coexist with poor precision when a classifier predicts every customer as positive. Compare absolute performance as well as changes relative to each model's own baseline.

### Monitoring: Serving champion

1. Select **Serving champion** to inspect the deployed XGBoost model.
2. In **Dataset**, choose Incoming API requests (live), Fixed study snapshot, Synthetic Demo · 5 batches, or Held-out study / stress tests.
3. For Synthetic Demo, choose a persisted **Replay / outcome mode** (simulated labels or drift only when available). For Held-out study / stress tests, choose **Stress-test scenario**: Stable population, Behavioral drift or Performance deterioration. These are separate test scenarios, not labels for the five synthetic batches.
4. Select **History range** and **Measurement window**. The Model version field identifies the current serving checkpoint. Click Refresh to reload measurements.
5. Review Reference, Selected-window samples, Labeled coverage and Monitoring state. Inspect feature distributions, performance, prediction drift, segment information and alerts where available.
6. Download the selected window's HTML report or JSON for review.

The live source represents actual API requests, which may themselves contain demo data. It is not proof of real customer traffic. Live performance requires observed outcomes after the configured 90-day horizon. The current policy uses 24-hour windows, at least 500 drift samples and two consecutive qualifying windows for relevant alerts. Review required is a signal to investigate, not a command to automatically retrain.

### Suggested demonstration sequence

Start at Overview and run batch scoring, inspect a known customer, predict churn, view their segmentation and retention recommendations, then open Monitoring to compare the five models across normal and stress batches. End with Serving champion and Models to explain the difference between deployed monitoring, synthetic robustness and original test metrics.

### Troubleshooting and interpretation

- API offline or unavailable: verify the configured backend URL, allow a sleeping Render service to start, and retry. Do not interpret unavailable results as zero.
- No campaign or completed monitoring window: the operator must prepare and run the corresponding replay/job; the UI does not manufacture reports.
- Missing performance: choose a labeled campaign or wait for eligible live outcomes. Drift-only data still supports input drift and clustering quality.
- Synthetic labels are independently simulated and do not establish real-world accuracy. A challenger is not automatically promoted, and no model is automatically retrained.
- For operator setup, migrations, replay and reproducibility, see [multi-model monitoring](docs/multi-model-monitoring.md), [synthetic demo monitoring](docs/synthetic-demo-monitoring.md), and [Module 7 monitoring](docs/module-7-monitoring.md). Never place database credentials or admin tokens in frontend configuration.

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

**Model compatibility:** serving uses the XGBoost JSON plus separately fitted Module 5 preprocessing.
The study replay separately accepts only the two hash-approved LR/RF checkpoints after held-out reproduction. A build succeeds only when the exact 133-feature
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


## Synthetic Demo Data

The optional five-batch synthetic customer generator from upstream `main` is available on this branch. Generate and validate it locally using the source relational tables and fitted artifacts:

```bash
python -m demo_data.generate --multiplier 10 --max-samples 10000 --seed 42
python -m demo_data.validate
```

Set `DEMO_DATA_DIR=data/demo` to enable `/api/demo/*` on the FastAPI server. The normal dashboard and `/api/predict/batch` continue using the official snapshot by default. See [demo data guide](docs/demo_data.md). Install development extras from `requirements-demo.txt` for the upstream preprocessing test suite.
