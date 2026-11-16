# Hierarchical clustering

Run from the project directory:

```powershell
py main.py --mode train --model-type hierarchical
py main.py --mode ablation --model-type hierarchical
```

Paths in YAML resolve against the project directory. ETL rebuilds data/full_bank_data.csv when absent or requested with --rebuild-dataset.

## Scaling
Default: log1p for recency_days, initiated_transactions_monthly, trans_amount_3m, successful_logins_monthly, login_recency_days, trans_daily_trend, login_daily_trend, customer_service_complaints, total_accounts. No log for products_number, unresolved_complaint_ratio, trans_failure_rate, savings_share.

After median imputation and optional Pearson filtering, ALL retained features use StandardScaler: subtract population mean and divide by population standard deviation. This puts columns on comparable variance scales for Euclidean/Ward distances. Original and scaled duplicates are never combined in the clustering matrix.

Set clustering.scaler to robust to use RobustScaler (median/IQR) on ALL retained columns instead. This is an alternative experiment, not a second scaler applied after StandardScaler. Sparse complaint features can have IQR zero, so RobustScaler is not automatically better. Every run exports feature_processing.csv including log, imputation, scaler and removal status for each column.

## Features and interpretation
RFM: days since customer-initiated transaction; initiated transactions per exposure month; customer-initiated external outflow amount in Q3 (Monetary is an outflow proxy, not customer profit).
Digital: successful logins per exposure month; successful-login recency.
Momentum: Q3/Q2 daily outflow and successful-login rates, denominator smoothing 0.01, capped at 10. Missing Q2 exposure stays missing and is median-imputed; this does not mean unchanged activity.
Friction: complaint count, unresolved complaint fraction, transaction failure fraction (attempts follow shared ETL definition).
Portfolio: original product count, savings share of balance, currently open account count. Closed savings remain available in the source profiles but are not a default distance feature.

Correlation filtering uses absolute Pearson correlation after log/imputation; later columns are removed at >= 0.9. Column order follows the configured feature groups, prioritizing RFM. The rule is deterministic and never uses churn. Constant/all-missing columns are removed with a recorded reason.

This is descriptive segmentation of the full snapshot, so preprocessing is fit on that population. No target encoding, supervised selection, SMOTE, sample weights or supervised neural representations are used. Do not feed full-population cluster IDs into supervised test evaluation; that requires a separate train-only design and assignment rule for new customers.

Train mode explores k=2..8; profiles use configured n_clusters=4, not automatically the highest score. Review metric table, sizes, subsample stability and profiles before changing k. The three 80% subset refits measure ARI for every candidate k against the full solution restricted to those customers. They do not establish out-of-time validity.

Ablation compares full, RFM, and full minus digital/momentum/friction/portfolio at EVERY k in k_candidates with the same linkage and customer population. Each experiment recomputes its unsupervised preprocessing. Feature spaces differ, so internal scores alone cannot prove one representation is better. Churn rates are descriptive and never determine k/features.

Outputs: results/hierarchical/train or ablation, per-experiment processing table, scaled input, original customer records with cluster_id, median profiles in original units, cluster sizes, churn summaries, truncated dendrogram, relative-profile heatmap, linkage.npy and preprocessing.pkl. Candidate metrics are saved at mode level. Persona names should be assigned after reviewing original-unit profiles, not preassigned as risk labels.


## Business tables and candidate k
Both train and ablation now evaluate every k_candidates value (plus n_clusters). Each experiment builds one hierarchy and cuts it at all k values. Subsample stability is also reported for each k.

- Mode-level hierarchical_business_metrics.csv: all experiments, all k, all clusters.
- experiment/k_2/ ... k_8/business_metrics.csv: one table for each cut.
- experiment/k_*/cluster_profiles.csv and customer_clusters.csv: original-unit medians and customer-ID assignments per cut.
- experiment/k_*/report.md: business table and profile for that cut.
- experiment/business_metrics.csv: convenience copy for configured n_clusters.

Business fields: customers; customer_share; churned_customers; churn_rate; population_churn_rate; churn_lift = cluster churn rate / population rate; churn_share = cluster churn count / all churn count; total_balance; balance_share = cluster balance / population balance; mean_balance; median_balance; churned_customer_balance. Rates/shares are fractions, lift is a multiple, balances are CU. When the overall churn count or balance is zero, the corresponding undefined ratios are blank (NaN), not infinity.

These are descriptive metrics after segmentation, never input features or the objective for choosing k. Deposit balance and churned-customer balance are not CLV, profit, expected loss or retention cost. No financial cost assumptions are invented.


## C?ch ??c b?ng business
- customer_share: c?m chi?m bao nhi?u ph?n kh?ch h?ng. 0.10 ngh?a l? 10%.
- churn_rate: trong c?m c? bao nhi?u ph?n kh?ch r?i b?.
- churn_lift: churn_rate c?a c?m chia churn_rate to?n b?. 2 ngh?a l? g?p 2 l?n m?c chung; kh?ng c? ngh?a gi? ch?n th?nh c?ng g?p 2 l?n.
- churn_share: c?m ch?a bao nhi?u ph?n t?ng kh?ch r?i b?. C?m ch? c? 10% kh?ch nh?ng ch?a 25% churn l? nh?m t?p trung nhi?u ng??i r?i b?.
- total_balance, balance_share: t?ng v? t? tr?ng s? d? ti?n g?i t?i snapshot.
- churned_customer_balance: s? d? t?i snapshot c?a nh?ng kh?ch sau ?? c? churn = 1. ??y kh?ng ph?i s? ti?n ng?n h?ng ch?c ch?n m?t.
- mean_balance, median_balance: s? d? trung b?nh v? trung v? trong c?m.

C?c t? tr?ng c?ng b?ng 1 n?u m?u s? t?ng l?n h?n 0. Churn lift kh?ng c?n c?ng b?ng 1. Cluster ID ch? l? m? nh?m, kh?ng ph?i th? t? r?i ro.
