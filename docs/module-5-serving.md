# Serving the Module 5 checkpoint

The default classifier is `model_development/models/saved/xgboost_churn.json`. The old `artifacts/churn_bundle.joblib` is never read and `api/export_bundle.py` has been removed.

## Prepare and start

```bash
pip install -r requirements.txt
python -m api.prepare_module5 --download
# Optional clustering, using its separate snapshot:
python -m api.official_data --download
python -m api.segmentation_service
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Offline alternatives are `python -m api.prepare_module5 --source /path/to/raw/tables` or `--data /path/to/full_bank_data.csv`. The raw directory must contain `kaggle_bank_churn.csv`, `customer_context.csv`, `accounts.csv`, `savings_cycles.csv`, `complaints.csv`, `login.csv` and `transactions.csv`. Download pins `maythepd/full-dataset-intro2ba/versions/3`.

Preparation runs the original Module 5 ETL and config, recreates its stratified training/validation/test split, and fits preprocessing only on training customers. It loads the existing JSON checkpoint without fitting a classifier or applying SMOTE to inference data. It requires the original 133-feature order and reproduces all six published metrics within the CSV's four-decimal precision before publishing outputs. This reconstructs preprocessing because the original fitted preprocessor was not committed with the checkpoint.

Generated outputs:

- `data/official/module5_customer_features.csv`: complete raw customer aggregates.
- `artifacts/module5/preprocessor.joblib`: fitted Module 5 transforms only.
- `artifacts/module5/manifest.json`: feature schema, hashes, split sizes and reproduced study metrics.

The saved checkpoint is unchanged. Generated files stay ignored by Git. `MODULE5_CHECKPOINT_PATH`, `MODULE5_PREPROCESSOR_PATH`, `MODULE5_MANIFEST_PATH` and `CUSTOMERS_CSV_PATH` control server paths; all must identify the same prepared export. Default file names are shown in the root `.env.example`. Set environment variables in the server environment; the API does not automatically load a dotenv file.

## API contract

`GET /health` reports `model_source=module5_checkpoint`, `feature_count=133`, the checkpoint SHA-256 and `study_reproduced=true` when ready. Missing Module 5 artifacts leave prediction unavailable with 503. Altered data, preprocessing, checkpoint, report or config hashes fail startup with a preparation instruction.

`GET /api/customers` and `GET /api/customers/{id}` expose staff-facing profiles derived from the full Module 5 raw table. `POST /api/predict` takes the existing profile schema and resolves the remaining raw activity aggregates using `customer_id`. Editable profile fields override the corresponding raw fields for what-if comparisons. Monthly transactions, transaction trend, monthly app logins and the unresolved-complaint flag are derived by Module 5 and are read-only in the form. `recency_days` permits fractional nonnegative values and channels include `No_Channel`.

For a customer outside the snapshot, supply `raw_features`, an object containing every raw input named by the manifest's `input_columns` that is not already provided as a top-level profile field. The response is 422 when required raw aggregates are absent or invalid. Neither `customer_id`, `churn` nor arbitrary transformed features may be injected through `raw_features`. No absent feature is silently invented as zero. Numeric nulls are handled by the fitted training imputer; the input key must still be present.

`POST /api/predict/batch` preserves all raw aggregates and shares the same scoring adapter as single inference. It does not refit preprocessing or use churn labels. `/api/metrics` continues to return the historical three-model study and adds serving source, checkpoint hash and reproduction status. Study results remain distinct from monitoring on future labeled requests.

Segmentation remains separate: `SEGMENTATION_CUSTOMERS_CSV_PATH` defaults to `data/official/customer_features.csv`. Its hash checks and clustering artifacts continue to use that snapshot, not the Module 5 raw table.

## Render deployment

The updated blueprint build command is:

```bash
pip install -r requirements.txt && python -m api.prepare_module5 --download && python -m api.official_data --download && python -m api.segmentation_service
```

For a manually configured Render service, update the dashboard build command to this value; editing `render.yaml` alone does not prove an existing manually managed service uses it. Remove any custom `CUSTOMERS_CSV_PATH` pointing at the old simplified snapshot. Legacy `MODEL_BUNDLE_PATH` is ignored. Redeploy the API and rebuild/redeploy the frontend, then check the health identity, customer lookup, single prediction and batch scoring. A frontend build alone cannot change the deployed classifier.

## Validation

```bash
python -m unittest discover -s tests -v
cd frontend
npm run build
```

The real-checkpoint tests require prepared artifacts; `module5-serving-ci.yml` prepares them before running the suite. Tests cover study reproduction, exact checkpoint identity, direct-versus-API scoring, full-batch equivalence, missing raw data, changed hashes, no legacy fallback, target exclusion and no-activity customers.
