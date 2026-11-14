# Module 5B: Customer Segmentation

## Method

Analyze **10,000 customers** from the [group's Kaggle dataset](https://www.kaggle.com/datasets/maythepd/full-dataset-intro2ba),
using 7,999 train, 1,001 validation and 1,000 independent test customers.
Behavioral data covers 1 April to 1 October 2026 (exclusive).

Preprocessing and K-Means fit only on train. Validation selects k from 2–10;
churn is used only after clustering. The combined experiment balances the two
feature groups. Feature definitions are in [features.py](../segmentation/features.py).

## Main results

All churn rates below are from the independent test set; overall churn is **20.3%**.

| Experiment | Features | Selected k | Segments and churn rates |
|---|---|---:|---|
| A | 10 original Kaggle features | 2 | Lower balance: **14.5%**; higher balance: **23.7%** |
| B | 24 synthetic behavioral features | 2 | Observed transactors: **19.0%**; less engaged: **34.6%** |
| C | Original + behavioral features | 2 | Less engaged: **27.5%**; active: **13.5%** |
| D | Original + permuted behavioral rows | 2–3 | Five control seeds; rows shuffled within each partition |

A mainly separates balance levels; C separates activity levels and substantially
reorganizes the original groups. Less-engaged customers warrant checking
onboarding barriers and failed transactions before targeted reactivation.

## Conclusion and limitations

- C separates churn more strongly than A at selected k=2. At common **k=4**,
  A performs better: churn variance fractions **0.0529 vs 0.0403**.
- C exceeds all five permutation controls, but five seeds give limited evidence
  (corrected permutation tail **0.1667**).
- Removing transaction/savings features reduces C's churn variance fraction to
  **0.0005**. The simulator conditions behavior on original activity flags, so
  generation rules may explain the apparent benefit.

**Synthetic data does not consistently improve segmentation over original data.**
The generator source and real behavioral reference data are unavailable; this
analysis cannot establish real-world fidelity. Future-quarter churn is a project
assumption rather than independently observed future behavior.

## Run and outputs

From the repository root, using the prepared environment:

```powershell
.\.venv\Scripts\python.exe -m segmentation.download_data
.\.venv\Scripts\python.exe main.py --mode segmentation
.\.venv\Scripts\python.exe -m pytest -q
```

Results, assignments, personas and charts: `results/segmentation/`.
Saved pipelines: `models/saved/segmentation/`. These generated files stay local;
settings are in `config/config.yaml` and detailed metrics remain in the CSVs.

Validation: **13 tests passed**, the existing regression suite passed, and saved
models reproduced the recorded assignments after reload.
