# Module 5B: comparative K-Means customer segmentation

## Findings from the real dataset

Combined segmentation reorganizes the original balance-based segments into
activity-based groups. It increases churn separation at selected k=2 but is
weaker than original-only at common k=4. It exceeds five permuted controls;
removing transaction/savings dependencies nearly eliminates its churn association.
The results do not establish consistent improvement over original segmentation
or real-world behavioral fidelity.

Actual population: **10,000 customers**, with **7,999 train / 1,001 validation /
1,000 independent test**. The existing splitter's rounding produces this slight
80/10/10 deviation. Test churn prevalence is **20.3%**. No fixture metrics below.

| Experiment | Selected k | Validation silhouette | Validation DB | Seed stability ARI | Test churn variance fraction |
|---|---:|---:|---:|---:|---:|
| A: original | 2 | 0.1407 | 2.4514 | 1.0000 | 0.01208 |
| B: behavioral | 2 | 0.5118 | 1.1689 | 1.0000 | 0.01109 |
| C: combined | 2 | 0.1385 | 2.3912 | 1.0000 | 0.03055 |
| D: five-seed mean | 2–3 | 0.1426 | 2.3942 | 0.9975 | 0.00280 |

Silhouette, DB and inertia are **within-space** diagnostics: their cross-space
values cannot establish better segmentation. Stability measures initialization
stability, not resampled-population stability. D averages individually selected
models; use common k for a controlled churn comparison.

At **common k=4**, test churn variance fractions are **A=0.05288, B=0.03359,
C=0.04028**; D ranges **0.00384–0.00973**. C exceeds every D control but
underperforms A. The corrected one-sided permutation tail is **1/6=0.1667**,
too coarse for strong inference. Some common-k clusters contain fewer than 1%
of test customers. The variance fraction is customer-weighted variance of
cluster churn rates divided by overall binary churn variance. It is descriptive,
depends on k, and is neither predictive accuracy nor a causal retention effect.

## Personas and transitions

Names use training features; churn is attached after all models are frozen.

| Persona | Test count | Churn | 95% Wilson interval | Lift |
|---|---:|---:|---:|---:|
| A0: Lower-balance customers | 366 | 14.48% | 11.24–18.46% | 0.71 |
| A1: Higher-balance customers | 634 | 23.66% | 20.52–27.12% | 1.17 |
| B0: Observed transactors | 919 | 19.04% | 16.63–21.71% | 0.94 |
| B1: Less-engaged transactors | 81 | 34.57% | 25.12–45.41% | 1.70 |
| C0: Less-engaged transactors | 487 | 27.52% | 23.74–31.65% | 1.36 |
| C1: Active transactors | 513 | 13.45% | 10.77–16.68% | 0.66 |

- A0/A1 training mean balances are approximately **1,930/120,000 CU** and
  product counts **1.83/1.37**. Review account utility, fees and onboarding for
  A0; review savings/service needs for A1. Balance alone is not lifetime value.
- B0/B1 have observed initiated-activity prevalence of **100%/29%** in training.
  B1's mean recency is around **142 days**, with higher observed failure rates.
  Investigate barriers and failed transactions before targeted reactivation.
- C0/C1 mean transaction recency is **143/12.4 days** and initiated monthly
  actions **1.85/9.43**. Original activity flags are positive for **3.7%/100%**.
  Check inactivity/onboarding barriers in C0; maintain reliable payment service
  and relevant account support in C1. Retention actions require prospective tests.

Behavioral personas are absent from A, which cannot describe transaction/login
or complaint patterns. Digital Power User or High Friction names are not forced
when selected profiles do not support them. `personas.csv` contains full positive
and negative deviations, test churn and specific proposed actions.

Held-out A/B agreement: ARI=-0.00151, NMI=0.000026; A/C: ARI=-0.000077,
NMI=0.000424; B/C: ARI=0.03460, NMI=0.12741. Each original group splits almost
evenly across C: largest combined subgroups are **50.3%/52.2%**. This is substantial
reorganization, rather than simple preservation. Low agreement alone is not harm.
The underlying customers and original attribute distribution are unchanged;
only cluster composition changes. Display-label matching is learned from training
overlaps, and ARI/NMI are invariant to arbitrary cluster labels.

## Audit and reuse

Initial checkout: clean **main**, HEAD
**`d9ed523ff0c9323a2dba3fe00c34291b99e6daec`**. Implementation uses
`feature/comparative-kmeans`; no main commit or merge. Per-command Git
`safe.directory` overrides handle the Administrators-owned checkout without
changing global trust settings.

No local source data existed. The tracked `origin/data/module-1-2-3:README.md`
identified the [group's Kaggle release](https://www.kaggle.com/datasets/maythepd/full-dataset-intro2ba).
It supplied seven raw tables and a generation manifest: 10,000 original customers,
10,000 context rows, 17,343 accounts, 495,665 transactions, 1,309,432 login attempts,
13,128 complaints and 5,280 savings cycles. Files are in ignored
`data/full_dataset/`; the archive is `data/source_dataset.zip`.
`source_download.json` records the URL and archive hash; run manifests hash every
raw CSV. The real release has unique primary keys and no records requiring
observation-window exclusion. Expected missing trends are audited separately.

Verified ETL output has **70 columns**, unlike the README's historical 82-column
description. `docs/new_feats.md` proposes RFM quantile/CLV features; they are not
implemented in the existing domain orchestrator. This module uses continuous RFM
quantities without importing proposed risk flags or fictional implementations.

Reused `BankDataETL` loading/aggregation through a small `_load_tables` hook;
`SegmentationETL` validates raw inputs before those joins. Reused stateless
`BehavioralMomentumEngineer` trends, correcting initiated-only frequency and
undefined trends. Reused `DataSplitter` ratios, seed and customer isolation;
it now accepts missing targets when `stratify=False`. Supervised stratified
behavior is unchanged. Segmentation never uses target encoding, supervised
selection/risk interactions, SMOTE, temporal embeddings or sample weighting.
Existing classifier outputs are not changed by segmentation mode; tracked model
artifacts touched by regression tests were restored.

## Exact features and provenance

**A: ten original Kaggle inputs:** `credit_score`, `country`, `gender`, `age`,
`tenure`, `balance`, `products_number`, `credit_card`, `active_member`,
`estimated_salary`.

**B: 24 synthetic-derived inputs:**

- Recency: `recency_days`, `login_recency_days`, `complaint_recency_days`.
- Frequency: `monthly_transactions`, `monthly_logins`, `app_logins_monthly`.
- Monetary: `trans_amount_3m`, `trans_amount_prior_3m`, `total_outflow`,
  `rfm_monetary_monthly`.
- Trends: `trans_trend_ratio`, `trans_count_trend_ratio`, `login_trend_ratio`,
  `is_spending_collapsing`, `is_login_collapsing`.
- Experience: `customer_service_complaints`, `unresolved_complaints_count`,
  `login_failure_rate`, `trans_failure_rate`.
- Savings/history: `num_savings_accounts`, `num_savings_cycles`,
  `prior_history_observed`, `has_initiated_activity`, `has_successful_login`.

Raw `customer_initiated_tx_count` derives monthly frequency but is not duplicated
as a clustering input. `total_outflow` means successful **initiated external**
outflow in this route. RFM uses recency, initiated monthly frequency and initiated
monthly monetary flow; no global quantile thresholds are learned.

**C:** A+B for identical customer rows. **D:** unchanged A plus complete B rows
permuted separately within train, validation and test, over five seeds. Synthetic
within-row relationships and missingness patterns are preserved.

**Mixed/unverified exclusions:** account balance components constrained to original
balance; generated relationship tenure/new-customer flags consistent with source
tenure; original/synthetic wealth-risk interactions; proposed CLV/RFM risk scores.
`customer_id` is joining/reporting only. `churn` is evaluation only. Unreviewed
columns cannot enter clustering. `feature_provenance.csv` records source tables
and dependencies. Synthetic-derived means table lineage, not independence.

## Observation, fitting and model selection

Window: **2026-04-01 inclusive to 2026-10-01 exclusive**, restricted further by
customer exposure and account lifecycles. Future complaint resolutions are
censored to unresolved. Savings cycles must overlap exposure and begin before
snapshot. Internal transfers count once and add no external monetary flow.
Passive receipts, fees, interest and system automatic payments are excluded from
initiated activity; only successful logins enter engagement frequency.

No prior exposure is unobserved history; zero prior activity also leaves trend
denominators undefined. Ratios stay missing with imputation indicators plus
explicit history/activity flags. No-event recency is right-censored at exposure,
not lifetime recency. Failure rates without attempts stay missing. Exact days
divided by 30.4375 handle partial exposure months. Guards reject duplicate/null
keys, unknown foreign keys/categories, invalid lifecycle dates, nonpositive
event amounts and malformed successful internal transfer pairs.

Configured 80/10/10 ratios and seed 42 are reused with **stratification disabled**
to keep churn out of splitting as well as fitting. Train-only median imputation,
missingness indicators, signed log1p on skewed numerics, StandardScaler and OHE
handle missing values and unseen categories. Training-constant dimensions are
removed without labels. Each encoded group is multiplied by
`sqrt(weight / sum(training column variances))`: default 0.5/0.5 gives equal
expected squared-distance contributions, including categorical width.

Explore k=2…10 with `KMeans(n_init=10)` at seeds 42, 137, 2026. Select the
smallest k within 0.02 of the best mean validation silhouette among candidates
with at least 2% training cluster share and 0.8 validation seed-stability ARI.
Fallback to valid silhouette candidates is flagged when constraints exclude all.
Inertia/elbow, DB and size balance remain review diagnostics. The fixed first
seed supplies the saved model. k=2 gives stable large, interpretable balance or
activity groups; churn does not select finer groups. Common k=4 is prespecified.
All primary, control, weight/dependency sensitivity and common-k fits finish
before churn evaluation. Test data never fits preprocessing or centers, or tunes
k/weights. D reuses C's fitted preprocessing and group weights exactly.

## Synthetic-generation caveats and sensitivity

The manifest reports generator **4.3-recurring-payment-contracts**, seed 42,
and **uses_churn_in_generation=false**. The generator notebook/source is absent,
so this declaration cannot be verified against executable source. The manifest
documents uncalibrated simulation rules, activity conditioned on `active_member`,
original balances as hard endpoints, salary-based proposed funding and dates
consistent with source tenure.

Training Spearman correlations with original `active_member`: Q3 spending
**0.9178**, count trend **0.9041**, transaction recency **-0.8640**. No inactive
customer has positive initiated Q3 external spending, versus **99.6%** of active
customers. These are strong generation-imposed dependencies. Excluding account
balance columns does not eliminate indirect transaction/funding dependencies.

| Prespecified sensitivity | Selected k | Test churn variance fraction |
|---|---:|---:|
| C: 25% original / 75% synthetic weight | 2 | 0.01109 |
| C: 75% original / 25% synthetic weight | 2 | 0.03084 |
| B: reduced dependencies | 4 | 0.00793 |
| C: reduced dependencies | 2 | 0.00050 |

Reduced features: `monthly_logins`, `app_logins_monthly`,
`customer_service_complaints`, `unresolved_complaints_count`, `login_failure_rate`,
`has_successful_login`. This removes transactions, savings, monetary features
and explicit prior-history flags, but does not prove remaining independence.
These diagnostics have no matched sensitivity-specific control ensembles.

Kaggle churn's next-quarter interpretation is a project modeling assumption,
not observed future churn. Wilson intervals condition on frozen assignments
and omit model uncertainty/multiple-comparison correction. Added dimensions,
correlated features and simulated missingness can alter geometry. No real
behavioral reference exists: the control tests association beyond shuffled rows,
not realism. More control seeds and external validation are needed for stronger
inference. No optional classifier augmentation was attempted.

## Reproduce and inspect outputs

From the repository root in PowerShell; the task-created `.venv` is ready:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m segmentation.download_data
.\.venv\Scripts\python.exe main.py --mode segmentation
.\.venv\Scripts\python.exe -m pytest tests/test_segmentation.py -q
.\.venv\Scripts\python.exe -X utf8 test_pipeline.py
```

If Kaggle later requires authentication, manually obtain that release and place
the seven CSVs in `data/full_dataset/`. No test data substitutes for raw sources.
Clustering settings live in independent `segmentation:` configuration; `dataset:`
provides raw path and split ratios/seed. `--output-dir` overrides results. Existing
modes and default `compare` remain unchanged. `environment.txt` records installed
versions; `run_manifest.json` records Git HEAD/dirty state, settings, partitions,
input features, library versions and source hashes. Cross-version bit-for-bit
reproduction is not guaranteed. Machine-wide Python packages were not changed.

Under ignored **`results/segmentation/`**:

- `comparative_summary.csv`, `comparison.csv`, `common_k_comparison.csv`: A–D
  summary, individual controls/sensitivities, inertia, silhouette, DB, balance,
  initialization stability and churn separation.
- `cluster_assignments.csv`, `permutation_donors.csv`: one assignment per
  customer, partitions, every selected cluster, aligned A→C transitions and
  reconstructable control donor mappings.
- `cluster_sizes.csv`, `cluster_profiles.csv`, `personas.csv`,
  `churn_by_cluster.csv`: counts/shares, training profiles, retention actions
  and test churn with Wilson intervals.
- `k_selection_*.csv/.png`, `cluster_sizes.png`, `churn_by_cluster.png`,
  `contingency_*.csv`, `transitions_*.csv`, `transition_*.png`,
  `transition_summary.csv`: k diagnostics and cluster comparisons.
- `original_distribution_profiles.csv`, `original_distribution_comparison.png`,
  `original_feature_distributions.png`: original-attribute means, quartiles,
  proportions, standardized differences, numeric KS and distribution figures.
  Standardization uses training dispersion; zero dispersion is undefined.
- `synthetic_distributions.csv`, `synthetic_original_correlations.csv`,
  `activity_conditioning.csv`, `feature_types.csv`, `feature_provenance.csv`,
  `permutation_results.csv`, `permutation_control.png`, `report.md` and
  `run_manifest.json`: audits, controls and full generated interpretation.

Twenty pipelines (12 selected/sensitivity, eight common-k) and encoded feature
names are saved under ignored **`models/saved/segmentation/`**. Example reload:

```python
import joblib
from segmentation.data import load_segmentation_data
from segmentation.features import build_features, ORIGINAL_FEATURES, SYNTHETIC_FEATURES
base, _ = load_segmentation_data("data/full_dataset", verbose=False)
rows = build_features(base).sort_values("customer_id").reset_index(drop=True)
model = joblib.load("models/saved/segmentation/C.joblib")
labels = model.predict(rows[ORIGINAL_FEATURES + SYNTHETIC_FEATURES])
```

Load only trusted joblib artifacts. D requires its recorded synthetic donor
rows, not unpermuted inputs. A production inference flow would need a new dated
snapshot adapter; this module intentionally enforces the project snapshot.

## Validation

**13 segmentation tests pass**, including full A–D fixture execution, exact
population/assignment uniqueness, no forbidden features/supervised steps,
group energy, missing/new customers, time/savings overlap, invalid transfers/FKs,
partition-isolated whole-row permutation, deterministic fitting, reload, and
unchanged centers after changing only test values and every churn label.
One-class churn reports and rendered figures are tested.

**Existing `test_pipeline.py` passed**: full pipeline and group ablations, raw
Kaggle baseline, Pearson/MI/hybrid selection, plotting, stratified CV, Optuna smoke,
processed-data export and temporal representation checks. Its initial non-UTF-8
attempt failed on pre-existing temporal progress characters in Windows cp1252;
`-X utf8` resolved the environment issue. Regression-generated tracked classifier
models were restored. The supervised processed CSVs requested during this task
are available under ignored `data/processed/`.

The real segmentation CLI completed all experiments. Independent artifact
verification reproduced all 10,000 selected/sensitivity assignments, including
reconstructed D permutations, with preprocessors fitted on 7,999 customers.

References: [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html),
[clustering evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation),
[OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
