# Module 5B: comparative customer segmentation

## Initial audit and implementation plan

Initial state: clean `main`, HEAD `d9ed523ff0c9323a2dba3fe00c34291b99e6daec`.
Implementation branch: `feature/comparative-kmeans`. No direct main commits or merges.

The README describes an 82-column cached dataset, but no `data/` directory is
available locally. No generator notebook or generation manifest is tracked.
The ten raw baseline names in `preprocessing/pipeline.py` match the task.
`docs/new_feats.md` proposes RFM/CLV features; they are not implemented in the
current domain orchestrator. Supervised target encoding, MI selection, risk
interactions, temporal representations and SMOTE must stay out of this route.

Reuse `BankDataETL` for table loading/aggregation, `BehavioralMomentumEngineer`
for label-independent exposure-adjusted trends, and `DataSplitter` for customer
partitions. The existing ETL does not filter events by date; add a guarded
segmentation subclass while preserving the supervised ETL defaults.

Ordered slices and acceptance checks:

1. Guard raw schema, dates, joins and feature provenance. Prove one customer row,
   internal-transfer accounting, no system engagement and new-customer handling.
2. Fit train-only unsupervised preprocessing and deterministic KMeans. Prove
   group weighting, train isolation, whole-row partition permutation and reload.
3. Add A–D runner, validation k selection, common-k and weight/dependency
   sensitivity, held-out evaluation, profiles, reports and plots. Exercise all
   outputs with small automated fixtures; never present fixture metrics as results.
4. Add CLI/config without changing existing modes. Run existing checks when
   dependencies/data permit, inspect the diff, document real execution status.

Raw inputs are required to make empirical claims. Running ETL cannot create
the original Kaggle table or reconstruct absent synthetic histories.
