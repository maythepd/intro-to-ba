# VioletBank Analytics

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

Set `VITE_API_BASE_URL=http://localhost:8000` or your hosted API URL. Customer listing and lookup now depend on the official-data API. If the API is unavailable, no synthetic customer records are silently substituted. Some dashboard charts are explicitly labelled as *Minh họa* until actual measurements are provided.

## Deployment

A Render API Blueprint is in `render.yaml`; it downloads Kaggle data and trains the temporary model at build time. This requires outbound access to Kaggle and sufficient build resources. Deployment has **not yet been verified**. Set `CORS_ORIGINS` to the frontend origin. Frontend can be deployed as a Vite static site, with SPA rewrite fallback.

API docs after startup: `http://localhost:8000/docs`. Segmentation still requires the team's clustering model; `/api/segments` returns 503 until then.

Do not present the study data, synthetic extension, or temporary trained model as live bank operational information.
