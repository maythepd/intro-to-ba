# VioletBank Analytics

Customer retention dashboard and XGBoost churn-risk API for a digital banking analytics project.

## Structure

- `frontend/`: React, Vite, Tailwind CSS, staff dashboard
- `api/`: FastAPI inference and dataset lookup
- `config/`, `preprocessing/`, `models/`: model training code from the classification branch
- `data/cleaned_bank_data.csv`: integrated study dataset

## Run locally

Python 3.11 or 3.12 is recommended.

```bash
pip install -r requirements.txt
python -m api.export_bundle
uvicorn api.main:app --reload
```

The model bundle includes the fitted transformer and classifier. A model file is deliberately not committed. If it is missing, `/api/predict` returns 503 instead of fabricated values.

In a separate terminal:

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

`VITE_API_BASE_URL=http://localhost:8000` connects the frontend to FastAPI. The default interface without an API only shows clearly labelled demonstration records and diagrams.

## HTTP endpoints

- `GET /health`
- `GET /api/customers?search=...&limit=20`
- `GET /api/customers/{customer_id}`
- `POST /api/predict`
- `GET /api/metrics`
- `GET /api/monitoring`
- `GET /api/segments` (503 while clustering model is unavailable)

Open `http://localhost:8000/docs` for the API schema.

## Deployment

`render.yaml` defines the FastAPI Render service and builds the model bundle when deployed. Configure `CORS_ORIGINS` to exactly match the frontend domain. Deploy `frontend` as a static site and define `VITE_API_BASE_URL` at build time. For client-side routing, enable SPA fallback to `index.html` (the `frontend/public/_redirects` file works on hosts that support that syntax).

## Evaluation and limitations

The displayed baseline churn rate and XGBoost metrics are from the group's study. Model performance was measured with a synthetic behavior extension and the current classification code uses its holdout for early stopping; this is not a valid unbiased production evaluation. Clustering and measured data drift are not yet implemented. A separate validation split and independent test holdout are recommended before a final report.

Only use datasets with appropriate permission for any public demo. The demo records are synthetic and labelled.
