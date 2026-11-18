# Module 7 — Model monitoring

## Components

The classifier remains the committed Module 5 checkpoint. Preparation now also exports a compact training reference and the frozen test cohort for replay. Monitoring does not train or replace a model.

- `monitoring/baseline.py`: fixed quantile/category distributions from the original training split, before resampling, plus probability distributions and independent test metrics.
- `monitoring/store.py`: PostgreSQL or local SQLite; immutable prediction/outcome IDs, window reports and alerts. PostgreSQL uses private schema `bank_monitoring`, RLS and the limited `bank_monitoring_writer` runtime role. The web client receives aggregate reports through FastAPI, not direct table access.
- `monitoring/drift.py`: numeric PSI and base-2 Jensen–Shannon **distance**, including unseen categories and explicit missingness.
- `monitoring/performance.py`: metrics with mature, joined outcomes; Average Precision is the PR metric.
- `monitoring/runner.py`: fixed UTC windows, sample requirements, adjacent-window trigger rules, idempotent persistence and explicit retention cleanup.
- `monitoring/service.py`: request instrumentation, model/reference registry, source separation and refresh of original windows when delayed labels arrive.
- `monitoring/simulate.py`: repeatable, clearly labeled study replay on held-out customers.

## Configuration and startup

Activate the repo virtual environment and install `requirements.txt`. Local commands read `.env`; existing process environment variables take priority.

```bash
python -m api.prepare_module5 --download
python -m monitoring.migrate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Set these values on the API server:

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Actual PostgreSQL connection string, preferably Supabase Session pooler on IPv4 |
| `MIGRATION_DATABASE_URL` | Optional owner connection used only for schema migration |
| `MONITORING_ADMIN_TOKEN` | Random server-side token for jobs, label import and replay |
| `MONITORING_DATABASE_ROLE` | Defaults to `bank_monitoring_writer`; selected transaction-locally for runtime queries |
| `ENABLE_MONITORING_DEMO` | `true` to permit the authenticated replay endpoint; default `false` |
| `SUPABASE_URL` | Project API URL for identification; **not** a PostgreSQL connection string |

No database URL or admin token belongs in a `VITE_*` variable. `.env`, prepared artifacts, local databases and generated reports remain ignored by Git. The tracked `.env.example` contains no credential values.

Without `DATABASE_URL`, local development uses `artifacts/monitoring/monitoring.sqlite3`. On Render, missing database configuration leaves monitoring unavailable rather than claiming ephemeral SQLite is durable. Prediction remains available if telemetry storage fails; the response reports `monitoring_recorded=false`, and monitoring shows its collection error and original report timestamps. A service restart is needed after changing its environment.

The migration creates only the monitoring schema/tables and limited non-login runtime role. For production, prefer a separate login with membership of that role and use an owner URL only for migration. Transaction-local role selection and disabled psycopg prepared statements support pooled connections. Connection/query waits are bounded; database errors never echo connection strings to the public API.

## Identity and source isolation

Model registry identity includes checkpoint, preprocessing and reference hashes. If available, the separate frozen KMeans membership map contributes its own version. A changed version starts a separate history series. Historical models and reports remain selectable.

The three source scopes are:

1. `live`: incoming prediction API requests over the study model, not operational bank traffic.
2. `snapshot_batch`: one fixed study snapshot; cache hits and forced rescoring do not multiply observations.
3. `study_replay`: a named stress scenario, isolated from the first two sources.

Every successful request logs selected features, probability, model identity, timestamp, outcome horizon and model runtime. Customer IDs are hashed for cohort deduplication. Validation/inference failures log only a bounded error category, without request bodies. Repeated successful lookups of a customer count as separate requests for runtime quality, but only the last successful prediction enters that window's drift/performance cohort.

Clustering monitoring compares proportions of known customers under the same frozen KMeans assignment map. Unknown/mismatched memberships are excluded and coverage is shown. The classifier and segmentation snapshots use separate aggregation formulas. Hierarchical assignments remain approximations outside their original Ward-fit sample and are not presented as classification metrics.

## Outcomes and performance

Single prediction responses include `prediction_id`, `prediction_timestamp` and `outcome_horizon_end`. Import a ground-truth label only after the full configured horizon (default 90 days):

```json
{
  "items": [{
    "event_id": "PREDICTION_ID_FROM_THE_API",
    "label": 1,
    "horizon_end": "EXACT_HORIZON_FROM_THE_API",
    "observed_at": "A_TIME_AT_OR_AFTER_THE_HORIZON_WITH_TIMEZONE",
    "source": "observed"
  }]
}
```

Send this to `POST /api/monitoring/labels` with `Authorization: Bearer <MONITORING_ADMIN_TOKEN>`. Labels must refer to existing successful predictions, match their horizon and not be observed in the future. Repeating the same outcome is idempotent; incompatible replacement returns 409. Simulated labels cannot be attached to live or snapshot-batch events.

An import refreshes the original prediction windows and any existing adjacent window whose alert streak may change. This permits quality tracking when labels arrive months after prediction. No historical churn label is silently substituted for a missing future outcome.

The report distinguishes `awaiting_outcome_horizon`, `awaiting_labels`, `insufficient_labels`, `single_class` and measured results. AUC/AP are `null` for a single-class sample. Every report shows label coverage and class counts; the UI does not turn missing metrics into zeros.

## Rules

All thresholds and feature lists are in `config/monitoring.yaml`; each saved report retains the full policy and its hash. Initial thresholds are project choices, not universal banking standards.

| Signal | Threshold / evidence | Result |
| --- | --- | --- |
| Feature comparison | At least 500 valid, distinct subjects | Otherwise `insufficient_data` |
| Numeric PSI | Warning 0.10; significant 0.25 | Per-feature diagnostic |
| Category JS distance | Warning 0.10; significant 0.20 | Per-feature diagnostic |
| Sustained drift | At least 30% significant features, or a configured critical behavioral feature; 2 adjacent non-overlapping eligible windows | `review_required` |
| Quality degradation | Recall or Average Precision drops at least 0.05 absolute versus the matching model's baseline; 2 adjacent eligible windows | `retraining_required` |
| Quality eligibility | 500 mature labels, at least 50 positives and 50 negatives, coverage at least 80% | Insufficient evidence cannot trigger retraining |
| Data errors | More than 5% invalid requests or missingness above its training baseline, on at least 100 requests | `data_quality_alert` |

A single performance-drop candidate is shown as a warning while awaiting confirmation. Gaps, insufficient samples, changed policy/version and different sources/scenarios break a streak. Prediction/segment distribution changes are diagnostics; they do not establish performance loss by themselves.

After an alert: inspect data/labels -> build a candidate -> evaluate on independent data -> compare with the current version -> responsible person approves deployment or rollback. The monitoring job does not automatically replace a model.

## API and reports

- `GET /api/monitoring?source=live&days=30`: storage status, current/historical models, latest report and history.
- `GET /api/monitoring?source=study_replay&scenario=behavior_drift`: isolated replay results.
- `GET /api/monitoring/runs/{id}`: complete aggregate evidence and provenance.
- `GET /api/monitoring/runs/{id}/report?format=html` or `json`: download the same measurements displayed by the dashboard.
- `POST /api/monitoring/runs`: authenticated operation; default last two closed windows. Supports explicit timezone-aware `start`/`end`, model and source.
- `POST /api/monitoring/labels`: authenticated immutable outcome import and affected-window refresh.
- `POST /api/monitoring/demo`: authenticated replay; also requires `ENABLE_MONITORING_DEMO=true`.

Reports never include individual feature records. HTML escapes category/scenario strings and disables scripts through CSP. Report timestamps remain visible when storage is degraded or history is stale.

## Replay and submission evidence

```bash
python -m monitoring.simulate
python -m monitoring.runner --source live --days 2
```

Replay uses the frozen Module 5 **test** cohort, not training rows. Each scenario has two disjoint 500-customer windows. The same held-out customers are reused between scenarios; timestamps are synthetic and anchored to model registration for repeatability. Changing a non-default seed creates a separate scenario scope.

- `stable`: original study features and labels.
- `behavior_drift`: increased recency and reduced transaction/login activity, with original study labels. Feature perturbations are artificial stress inputs.
- `performance_drop`: original features with deliberately adversarial, balanced synthetic labels; metrics are calculated from those labels and predictions rather than hardcoded.

On the verified study replay: stable AP is approximately 0.6073/0.6063; behavioral drift changes 33.3% of monitored features and reaches `review_required`; the adversarial scenario produces AP approximately 0.3079 in both windows and reaches `retraining_required`.

Use `notebooks/module7_monitoring.ipynb` for the explanatory notebook. Download HTML/JSON from the UI or use the aggregate reports under `artifacts/monitoring/reports` after export. These results demonstrate that monitoring rules function; they are not evidence of actual post-deployment bank performance.

## Deployment, scheduling and verification

`render.yaml` prepares baseline/replay artifacts during the build and runs migrations before Uvicorn at startup. Manually managed Render services need the updated commands and environment values copied into their service settings. Deploy both API and frontend; the public frontend needs only `VITE_API_BASE_URL`.

The scheduler calls the deployed API for `live` and `snapshot_batch`, daily at 01:20 UTC (08:20 Vietnam/Thailand time). GitHub schedules execute from the repository's **default branch**. The user-approved installation places only `.github/workflows/monitoring-run.yml` on the fork's `main`; application code remains on `demo`. Set repository variable `MONITORING_API_URL` and encrypted secret `MONITORING_ADMIN_TOKEN`. The optional setup command is:

```bash
pip install -r requirements-ops.txt
python tools/configure_monitoring_github.py --install-workflow-on-main
```

The GitHub job retries cold starts, reports failure rather than fabricating reports, and prevents overlapping scheduled jobs. Manual dispatch mode `demo` generates isolated replay only when enabled by the API operator. GitHub may delay scheduled jobs; the dashboard shows measurement times and stale status.

Acceptance checks: health reports matching Module 5 identity and ready PostgreSQL monitoring; replay yields the three expected scenarios; admin endpoints reject missing/wrong tokens; outcome import updates historical quality; cached/forced batch scoring deduplicates events; restart/redeploy preserves model history/reports; no database/admin credential is sent to the browser. Desktop/mobile visual inspection and the public Render redeploy need to be recorded separately from local API/build tests.

Raw event retention defaults to 180 days. Cleanup is an **explicit operator command**, `python -m monitoring.runner --days 2 --prune`, and removes old prediction/outcome rows while keeping aggregate report/model history. Review retention before executing it; it is not run during tests or installation.
