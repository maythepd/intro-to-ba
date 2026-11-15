# Module 5B: Customer Segmentation

## 1. Experimental settings

| Experiment | Type and inputs | Purpose |
|---|---|---|
| **A — Original-only** | 10 original Kaggle features: age, balance, credit score, product count, etc. | Establish the baseline customer segments. |
| **B — Synthetic-only** | 24 simulated behavioral features: transactions, logins, complaints and savings. | Identify segments described by behavioral information alone. |
| **C — Combined** | All 34 features from A+B, with balanced group contributions. | Assess whether behavioral information adds useful distinctions to the baseline. |
| **D — Permutation control** | Original features unchanged; complete behavioral rows shuffled across customers within each train/validation/test partition. | Test whether C benefits from customer–behavior associations beyond simply adding dimensions. |

Compare **C vs A** for the effect of adding behavior, and **C vs D** for the
information in the original customer–behavior pairing. All experiments use the
same customers and partitions; the control does not establish synthetic-data realism.

| Setting | Value |
|---|---|
| Data | 10,000 customers; train/validation/test = 7,999/1,001/1,000; split seed 42, no churn stratification |
| Observation window | 2026-04-01 inclusive to 2026-10-01 exclusive |
| Preprocessing | Train-fitted imputation, log transform for skewed numerics, StandardScaler and one-hot encoding |
| K-Means | k=2–10; n_init=10; max_iter=300; seeds 42, 137, 2026; silhouette sample capped at 1,000 |
| Group weights | C/D: original 50%, behavioral 50% of expected squared distance; sensitivity: 25%/75% and 75%/25% |
| Permutation control | Whole behavioral rows shuffled within each partition; seeds 101, 211, 307, 401, 503 |

Choose the smallest k within **0.02** of the best mean validation silhouette,
subject to mean minimum training cluster share ≥ **2%** and seed-stability ARI ≥ **0.8**.
If no candidate meets both constraints, fall back to valid silhouettes and flag it in the CSV.
Fit preprocessing/centers only on train; use churn only after all models are frozen.
Feature definitions: [features.py](../segmentation/features.py).

## 2. Clustering metrics

Inertia is from the saved training model (seed 42); silhouette and Davies–Bouldin
are validation means over three seeds. Stability is pairwise validation ARI across
initializations. D reports the mean over five permutation runs.

| Experiment | Selected k | Train inertia ↓ | Validation silhouette ↑ | Validation DB ↓ | Stability ARI ↑ |
|---|---:|---:|---:|---:|---:|
| A | 2 | 6973.56 | 0.1407 | 2.4514 | 1.0000 |
| B | 2 | 6297.41 | 0.5118 | 1.1689 | 1.0000 |
| C | 2 | 6982.48 | 0.1385 | 2.3912 | 1.0000 |
| D (5-seed mean) | 2–3 | 6722.18 | 0.1426 | 2.3942 | 0.9975 |

These geometry metrics describe each feature space; their values do not establish
that B or C is better than A across different feature spaces.

## 3. Held-out comparison

Test churn prevalence is **20.3%**. Churn separation below is the customer-weighted
variance of cluster churn rates divided by overall binary churn variance (higher
means stronger descriptive association). Common k controls for cluster count.

| Experiment | Test churn separation: selected k | Test churn separation: common k=4 |
|---|---:|---:|
| A | 0.01208 | 0.05288 |
| B | 0.01109 | 0.03359 |
| C | 0.03055 | 0.04028 |
| D (5-seed mean) | 0.00280 | 0.00606 (0.00384–0.00973) |

At selected k=2, C's less-engaged/active groups have churn rates **27.52%/13.45%**.
Held-out cluster agreement (ARI/NMI account for arbitrary cluster labels):

| Pair | ARI | NMI |
|---|---:|---:|
| A vs B | -0.00151 | 0.00003 |
| A vs C | -0.00008 | 0.00042 |
| B vs C | 0.03460 | 0.12741 |

## 4. Conclusion

C improves churn separation at selected k=2 but underperforms A at common k=4.
It exceeds all five permutation controls, with a coarse corrected tail estimate
of **0.1667**. Removing transaction/savings features reduces C's separation to
**0.00050**: original activity flags used by the simulator may explain the benefit.
**There is no consistent improvement over original-only segmentation.**
The generator source and real behavioral reference are unavailable; future-quarter
churn is a project assumption, so real-world fidelity is not established.

## Run and outputs

```powershell
.\.venv\Scripts\python.exe main.py --mode segmentation
.\.venv\Scripts\python.exe -m pytest -q
```

Detailed CSVs/charts: `results/segmentation/`; saved pipelines: `models/saved/segmentation/`.
See the [grouped output guide](segmentation_outputs.md) for the consolidated files.
These generated files stay local. **16 tests and the existing regression suite passed**;
reloaded models reproduced their saved assignments.
