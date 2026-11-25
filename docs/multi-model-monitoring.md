# Five-model monitoring study replay

The core A–I implementation from the supplied `BaBank_Multi_Model_Monitoring_Implementation_Guide.md` is deployed; browser visual verification remains blocked by the unavailable browser host. Three frozen classifiers and two fixed segmentation adapters evaluate the same 10,000 synthetic IDs, in five cohorts of 2,000. This is a simulated study, not an estimate of real-customer accuracy. XGBoost remains the serving champion at threshold 0.5.

## Audit and implementation decisions

The local `demo` branch was fast-forwarded from `80a5a8f` to remote `50d5647`. An unfinished `feature/five-model-monitoring` branch already contained an offline evaluator. Its [failed Actions run](https://github.com/nthne/Intro-to-Business-Analysis/actions/runs/45576172372) could evaluate RF/XGB and clustering, but LR stopped at `No module named preprocessing.splitting`. Importing the research models package eagerly imported tuning utilities with legacy absolute imports. Commit `1a28d02` makes those optional research utilities lazy; the approved checkpoint loader remaps only the legacy package paths. The [follow-up run passed](https://github.com/nthne/Intro-to-Business-Analysis/actions/runs/45577645803).

The guide's separation of tasks, frozen preprocessing, independent outcomes, common cohorts, and original train references is sound. Five implementation details make the proposed design concrete:

- LR/RF pickle files are accepted only at two committed SHA256 values. Preparation reproduces all eight original held-out metrics within 0.00005 and the four confusion counts exactly, using the original split and ordered 133 features. The checkpoint's sklearn version differs from the pinned runtime; compatibility warnings are preserved in the manifest. Passing this gate validates these files and environment, not arbitrary cross-version pickle compatibility.
- LR/RF use the champion's frozen feature transformation and separate train probability references. RF inference uses one worker for deterministic summation. No demo fit, threshold selection, or automatic model promotion occurs.
- Classifier input drift is calculated once per batch against the original pre-SMOTE train reference. Segmentation has its own input reference and encoded space. Probability drift, assignment drift and cluster profiles remain model specific.
- Ward assignment is explicitly a nearest fixed Ward-centroid surrogate for both reference and current data. Silhouette/DBI/CHI use the same seeded sample of 600 per batch for both clusterers; fixed-center distances and outlier shares use all 2,000 rows. Small/degenerate groups remain N/A with a status.
- The suggested logical registry/link tables are represented by an immutable campaign manifest plus five additive tables: `monitoring_campaigns`, `cluster_model_registry`, `campaign_reports`, `cluster_assignments`, `input_drift_reports`. Existing model/run/alert tables and schema version 1 remain compatible. Model–batch transactions, PostgreSQL advisory locks, unique keys and downstream streak refresh support retry/resume without truncating history.

The campaign digest binds data, independent label policy/seed/hash, frozen model/reference identities, experiment version/config and UTC anchor. Labeled and drift-only campaigns have different identities. Three classifiers share the same outcome for each ID; 30,000 outcome rows represent 10,000 business outcomes mirrored to three prediction streams. Public routes expose aggregates, never assignment rows or customer features.

## Reproduce locally

Use Python 3.11 with `requirements-serving.txt`; prepare the original Module 5 data/preprocessor, official segmentation snapshot and bundle first, as described in the existing serving guide. On this workstation use `.venv-synthetic\Scripts\python.exe` in place of `python`. Generated files remain ignored.

```powershell
python -m api.prepare_challengers --models lr rf --verify-study
python -m demo_data.generate --multiplier 10 --max-samples 10000 --seed 42
python -m demo_data.validate
python -m demo_data.labels --seed 4242

# Dry-run creates exports only; it never opens a database.
python -m monitoring.multi_model_replay --all --labels data/demo/synthetic_outcomes.csv --dry-run --export-dir artifacts/multi-model-dry

# Select a separate local database before any persistent experiment.
$env:DATABASE_URL = 'sqlite:///artifacts/monitoring/multi-model-local.sqlite3'
python -m monitoring.multi_model_replay --all --labels data/demo/synthetic_outcomes.csv --export-dir artifacts/multi-model-labeled
python -m monitoring.multi_model_replay --all --export-dir artifacts/multi-model-drift-only
python -m monitoring.multi_model_replay --campaign-id <stored-campaign-id> --resume --export-dir artifacts/multi-model-resumed
```

`--algorithm` and `--batch` optionally restrict the unit to persist. Preflight validates all artifacts, labels, IDs and numerical outputs before the first write. A failed unit rolls back; previously completed units remain. Resume reads the original anchor/mode and rejects changed provenance. Three pooled normal cohorts (6,000 labels) form each classifier's synthetic baseline; normal reports are baseline-building and do not self-trigger performance alerts. Clustering quality baseline is the median of three normal metrics under identical sampling. Review thresholds in `config/multi_model_monitoring.yaml` are illustrative.

Exports include `campaign_manifest.json`, validation/input drift/classifier/clustering/distribution/profile/robustness CSVs, `verification.json` and 25 JSON + 25 HTML reports. Classifier CSVs include confusion counts, prevalence, predicted-positive rate, calibration diagnostics, normal baselines and absolute drops. No individual prediction export is published.

## Storage, API and dashboard

The additive PostgreSQL migration runs through `python -m monitoring.migrate`, using the existing operator credential before the runtime connection switches to `bank_monitoring_writer`. All eleven tables enable RLS; the private schema is not a public REST surface. The same storage code supports SQLite locally. Rollback restores the previous app deployment and disables replay if necessary; it retains new tables and all historical reports.

Aggregate reads:

```text
GET /api/monitoring/campaigns
GET /api/monitoring/campaigns/{id}
GET /api/monitoring/campaigns/{id}/classification?batch=engagement_shift
GET /api/monitoring/campaigns/{id}/clustering
GET /api/monitoring/campaigns/{id}/input-drift
GET /api/monitoring/campaigns/{id}/export?format=json|csv|html&task=classification|clustering
```

`POST /api/monitoring/multi-model-replay` requires the existing backend admin token and enabled demo mode. Its JSON fields are `algorithm`, `batch`, `with_labels`, optional timezone-aware `anchor`, and optional stored `campaign_id` for resume. No arbitrary file/model URL is accepted. Operators should invoke one model–batch at a time on small hosting plans. A bounded five-minute preflight cache avoids repeated scoring. Builds prepare artifacts but never run replay.

At `/monitoring`, the default view is **Multi-model study replay**. The **Model** dropdown at the top offers all five models together or XGBoost, LR, RF, K-Means and the Ward surrogate individually. Selecting a model filters report rows, chart series, probability distributions and cluster shares/profiles; the matching classification/clustering tab and input reference are selected automatically. Campaign and batch selections stay in place while switching models. **Serving champion** remains available for incoming XGBoost requests; choosing another model switches to its study replay.

Select a measured campaign, batch, chart metric and comparison tab. The complete classification comparison shows three curves, 15 rows, confusion counts, normal AP/drop, prevalence/all-positive flags and probability distributions; individual classifiers show five rows. Clustering shows two curves, distances/outliers, cluster shares and profile drift, or five rows for the selected clusterer. Input drift is shared across the three classifiers or the two clusterers according to their feature space. Missing units and null metrics stay pending/N/A. Campaign JSON/CSV/HTML downloads include the full task/campaign and require no admin secret. Existing serving, customer lookup, segments and single-model replay routes retain their behavior.

## Validation evidence

Local full suite: **81 passed**, including disposable PostgreSQL migration, runtime writer/RLS, atomicity/resume, identity/cohort/label alignment, deterministic no-leakage/no-fit checks, metric parity, degenerate clustering and API authorization. TypeScript + Vite production build passed. The existing large-bundle warning remains. Browser visual verification is a separate gate and is not inferred from the build.

Measured local preflight plus aggregate export: 12.50 seconds wall time, 12.03 process CPU seconds, 305,516,544 bytes (291.4 MiB) peak working set, on Windows/Python 3.11.15, all 10,000 rows and silhouette sample 600. This same-process measurement excludes database writes and is not a Render memory guarantee. The five-model CI records `/usr/bin/time -v` for the full persistent replay as an artifact. Replay loads challengers sequentially and retains only arrays/references after inference.

The [final five-model CI](https://github.com/nthne/Intro-to-Business-Analysis/actions/runs/45602156901) passed 81 tests, including the forced out-of-order two-worker PostgreSQL case. Linux full replay measurement: 14.40 seconds wall time, 15.17 user + 0.71 system CPU seconds, 399,056 KiB (389.7 MiB) peak RSS, exit 0. The [demo branch CI](https://github.com/nthne/Intro-to-Business-Analysis/actions/runs/45602407226) also passed. These resource figures describe the CI runner, not the Render instance.

The complete local labeled campaign `470f82326dd854c3cf4ec8758ddaa44b56d781d8c30b742ae93b512193750ab8` produced 30,000 events, 30,000 event-linked outcomes, 20,000 assignments, 15 classifier reports, 10 clustering reports and 10 shared input reports. Repeating the same identity inserted zero events/outcomes/assignments. A separate drift-only campaign has zero outcomes and N/A classifier performance. Results and repeat receipts are in ignored `artifacts/multi-model-final` and `artifacts/multi-model-final-repeat`.

## Interpretation and recommendation

The original held-out study and synthetic replay answer different questions. XGBoost has the highest original study ranking AUC/AP, while the saved 0.5 threshold predicts all customers positive. This is visible in confusion counts and predicted-positive rate. RF has substantially better original thresholded F1/accuracy. Under the independent illustrative outcome policy, LR has the highest mean stress AUC (0.6608) and AP (0.5009), versus RF (0.5873 / 0.4381) and XGBoost (0.5743 / 0.4292). This supports further real-label/shadow validation of LR; it does not justify replacing the serving champion.

Ward-surrogate has the higher stress mean silhouette (0.1717 vs KMeans 0.0961); KMeans has the lower mean stress DBI (1.7284 vs 1.8797). These objectives disagree, and their cluster IDs have different meanings. Inspect fixed distances, outlier shares, cluster sizes and profile drift before changing centers. No combined ranking across classifiers and clusterers is meaningful.

Positive baseline-minus-current AUC/AP/F1 drops mean deterioration; negative drops mean improvement relative to the simulated normal baseline. AP is interpreted with prevalence. More persistent input drift with stable metrics calls for feature/business investigation; persistent metric deterioration calls for label/pipeline review and real-data validation before recalibration/retraining. Changing cluster shares alone does not establish a need to refit.

Optional research extensions (bootstrap intervals, business-cost assumptions, segment-level classifier slices and multiple registered generation/label seeds) are deferred. The current five batches cover only the specified engagement/friction perturbations and a single simulated label policy. No real-world generalization claim or promotion is made.

## Measured study and replay tables


Original held-out study (1,000 test rows; threshold 0.5):


| Model | AUC | AP | F1 | Precision | Recall | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| LR | 0.5985 | 0.2590 | 0.3570 | 0.2307 | 0.7892 | 0.4200 |
| RF | 0.8213 | 0.5760 | 0.5469 | 0.5833 | 0.5147 | 0.8260 |
| XGBOOST | 0.8372 | 0.6060 | 0.3389 | 0.2040 | 1.0000 | 0.2040 |


Synthetic classifier batches (N=2,000 and coverage=100% each; normal AP baseline pools 6,000 rows):


| Model | Batch | AUC | AP | Normal AP | AP drop | F1 | Recall | Prevalence | Predicted + | TN/FP/FN/TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | normal_01 | 0.5842 | 0.3364 | 0.3439 | N/A | 0.4381 | 1.0000 | 0.2805 | 1.0000 | 0/1439/0/561 |
| lr | normal_01 | 0.6643 | 0.3844 | 0.3941 | N/A | 0.4966 | 0.8503 | 0.2805 | 0.6800 | 556/883/84/477 |
| rf | normal_01 | 0.5963 | 0.3478 | 0.3560 | N/A | 0.3052 | 0.2549 | 0.2805 | 0.1880 | 1206/233/418/143 |
| xgboost | normal_02 | 0.5946 | 0.3477 | 0.3439 | N/A | 0.4320 | 1.0000 | 0.2755 | 1.0000 | 0/1449/0/551 |
| lr | normal_02 | 0.6565 | 0.3735 | 0.3941 | N/A | 0.4768 | 0.8494 | 0.2755 | 0.7060 | 505/944/83/468 |
| rf | normal_02 | 0.6102 | 0.3548 | 0.3560 | N/A | 0.3128 | 0.2632 | 0.2755 | 0.1880 | 1218/231/406/145 |
| xgboost | normal_03 | 0.6034 | 0.3531 | 0.3439 | N/A | 0.4454 | 1.0000 | 0.2865 | 1.0000 | 0/1427/0/573 |
| lr | normal_03 | 0.6813 | 0.4317 | 0.3941 | N/A | 0.4917 | 0.8307 | 0.2865 | 0.6815 | 540/887/97/476 |
| rf | normal_03 | 0.6141 | 0.3694 | 0.3560 | N/A | 0.2974 | 0.2339 | 0.2865 | 0.1640 | 1233/194/439/134 |
| xgboost | engagement_shift | 0.5598 | 0.4128 | 0.3439 | -0.0690 | 0.5401 | 1.0000 | 0.3700 | 1.0000 | 0/1260/0/740 |
| lr | engagement_shift | 0.6531 | 0.4878 | 0.3941 | -0.0937 | 0.5667 | 0.8784 | 0.3700 | 0.7770 | 356/904/90/650 |
| rf | engagement_shift | 0.5802 | 0.4291 | 0.3560 | -0.0731 | 0.3064 | 0.2284 | 0.3700 | 0.1815 | 1066/194/571/169 |
| xgboost | friction_shift | 0.5889 | 0.4456 | 0.3439 | -0.1017 | 0.5565 | 1.0000 | 0.3855 | 1.0000 | 0/1229/0/771 |
| lr | friction_shift | 0.6686 | 0.5140 | 0.3941 | -0.1200 | 0.5934 | 0.8223 | 0.3855 | 0.6830 | 497/732/137/634 |
| rf | friction_shift | 0.5944 | 0.4470 | 0.3560 | -0.0911 | 0.2883 | 0.2075 | 0.3855 | 0.1695 | 1050/179/611/160 |


Fixed segmentation (N=2,000; intrinsic eval N=600; normal baseline is median of three normal batches):


| Model | Batch | Silhouette | Normal silhouette | Silhouette drop | DBI | CHI | p95 distance | Outlier share | Share JS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kmeans | normal_01 | 0.1575 | 0.1575 | 0.0000 | 1.8986 | 102.4532 | 1.0923 | 0.0505 | 0.0071 |
| hierarchical | normal_01 | 0.1480 | 0.1480 | 0.0000 | 1.8986 | 98.3721 | 1.1158 | 0.0505 | 0.0070 |
| kmeans | normal_02 | 0.1608 | 0.1575 | -0.0033 | 1.8958 | 98.3574 | 1.0786 | 0.0555 | 0.0216 |
| hierarchical | normal_02 | 0.1523 | 0.1480 | -0.0043 | 1.8857 | 94.7183 | 1.1102 | 0.0510 | 0.0199 |
| kmeans | normal_03 | 0.1500 | 0.1575 | 0.0075 | 1.9349 | 95.6019 | 1.0994 | 0.0495 | 0.0167 |
| hierarchical | normal_03 | 0.1410 | 0.1480 | 0.0070 | 1.9344 | 92.5324 | 1.1362 | 0.0530 | 0.0173 |
| kmeans | engagement_shift | 0.0085 | 0.1575 | 0.1491 | 1.5834 | 43.1427 | 1.1604 | 0.0840 | 0.5780 |
| hierarchical | engagement_shift | 0.1603 | 0.1480 | -0.0123 | 1.9119 | 80.7066 | 1.1631 | 0.0800 | 0.5139 |
| kmeans | friction_shift | 0.1838 | 0.1575 | -0.0263 | 1.8734 | 97.8966 | 1.3367 | 0.0975 | 0.5639 |
| hierarchical | friction_shift | 0.1831 | 0.1480 | -0.0351 | 1.8475 | 97.6438 | 1.3124 | 0.0880 | 0.5668 |

## Deployment receipts

Verified on 2026-11-25 (Asia/Bangkok). Local and remote `demo` were fast-forwarded to application commit `7ccfca5c58a9461af4a96220cd690abf5e780b33`. Both existing Render services remain linked to `https://github.com/nthne/Intro-to-Business-Analysis`, branch `demo`; no upstream/main push or new repository relink was performed in this rollout. Auto-deploy was temporarily disabled for backend-first deployment and restored to its original `commit` setting after the documentation push.

| Service | Observed deployment | Application commit | Result |
| --- | --- | --- | --- |
| [Backend](https://babank.onrender.com/health) | `dep-dc3c5mm8bjmc73f908o0` | `7ccfca5` | Live, persistent storage ready, 133 features, study reproduced |
| [Frontend monitoring](https://babank-dashboard.onrender.com/monitoring) | `dep-dc3c7o1ksfps739mh9c0` | `7ccfca5` | Live, SPA/assets/CORS and credential scan passed |

Deployed labeled campaign: `5acb77b2976e9d3e500e3dbdd2b28a098bd3b46df352f48af43bbbc3252c0011`, anchor `2026-08-20T00:00:00+00:00`, generation seed 42, outcome policy `synthetic_outcome_v1`, label seed 4242. Data SHA256 remains `8a54c89258e7742ee060d100d38c39e4df41085fd21309f2ed5f61270d70d3cd`; label SHA256 remains `e169dd745190baf0e7117798f964dba036c7dc9a2fa488332a77326fcf5de950`.

SQL verification scoped to this campaign confirmed **30,000 prediction events, 30,000 outcomes, 10,000 distinct subjects, 3 classifiers, 20,000 assignments, 25 reports and 10 input drift reports**, with zero inconsistent labels between classifiers. All eleven tables have RLS enabled; `anon` and `authenticated` have no schema USAGE, while `bank_monitoring_writer` does. Global table totals also contain earlier experiments and explicit live control predictions; they are not new customer counts.

Authenticated rerun of LR/normal_01 using the stored campaign ID inserted **0 events, 0 outcomes, 0 assignments**, retaining complete status. JSON, CSV and HTML exports returned 200. All fifteen preexisting run records were compared with the pre-migration backup and remained unchanged, including the five reports under `synthetic_demo_6bdec224083a`.

After an actual backend process restart (new Uvicorn startup observed at 18:24 UTC+7), all 25 report IDs, model/batch links and campaign row counts remained identical. The serving model ID is still `00621aa7242712b7f388cf8cf0126614fb0d791d13d3fdc74ab71ee80778bcb2`; checkpoint SHA256 remains `a80aba0ed70c16b5a1ac1dccef824501ab1a7a8c9e997e421fae203f689db28f`. The same customer input produced exactly the same probability, predicted class and risk level before rollout, after rollout and after restart. Measured deployed stress metrics agree with the tables above.

Operational receipts and aggregate exports are saved locally under ignored `artifacts/deployed-multi-model/`: pre-migration Render/schema/run backup, control predictions, 25 replay receipts, no-duplicate receipt, downloaded CSV/JSON/HTML, frontend verification and post-restart SQL/API snapshots. Local and deployed campaign identities bind their respective prepared artifact hashes; use the ID returned by the target environment for resume.

Rollback deployments remain `dep-dc35t7k0916s73fqu4a0` / `6000e21` for the backend and `dep-dc37bmhksfps7398sslg` / `7d62198` for the frontend. App rollback retains the additive schema and historical data.

Remaining verification: desktop/mobile visual layout and interactive browser checks. The browser tool failed to open the requested monitoring URL with `browser access host unavailable`; no visual pass is claimed. The checked production HTTP/asset/API behavior and successful TypeScript/Vite build do not replace that gate.

## Completion checklist

- [x] A–F: audit, approved checkpoint reproduction, frozen references/assignments, aligned replay, exports, atomic/resumable storage, PostgreSQL/RLS and authenticated API.
- [x] G: comparison UI, measured charts/tables/distributions, valid N/A states and CSV/JSON/HTML downloads; production build/HTTP checks.
- [ ] G: browser desktop/mobile visual and interactive verification (tool unavailable).
- [x] H: controlled backend/frontend deployment, champion regression checks, 25 durable reports, rerun and actual restart verification.
- [x] I: separate study/synthetic results, task-specific robustness analysis and evidence-limited recommendation.
