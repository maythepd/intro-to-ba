# Hierarchical Clustering

Final configuration: **StandardScaler + no_friction + Ward linkage + 4 clusters**.

## Results

All results below use 4 clusters. Higher stability ARI means more consistent groups when customers are sampled again.

| Configuration | Silhouette | Stability ARI | Lowest churn | Highest churn | Gap (percentage points) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Standard / no_friction (selected)** | **0.1429** | **0.5898** | **13.80%** | **25.55%** | **11.75** |
| Robust / no_friction | 0.1248 | 0.3931 | 14.54% | 27.16% | 12.62 |
| Standard / no_digital | 0.1958 | 0.7357 | 14.54% | 24.96% | 10.42 |
| Standard / full | 0.1607 | 0.6514 | 14.49% | 24.45% | 9.96 |
| Robust / RFM after filtering (R and F only) | 0.4501 | 0.6031 | 13.89% | 26.01% | 12.13 |

### Selected customer groups

| Cluster | Customer group | Customers | Churn rate |
| --- | --- | ---: | ---: |
| 0 | Active transactions and app use | 3,232 | 13.80% |
| 1 | Frequent app use, but no recent transactions | 3,018 | 25.55% |
| 2 | Low transaction and app activity | 2,521 | 20.39% |
| 3 | Multiple accounts and savings, with declining transaction activity | 1,229 | 24.86% |

## Why this configuration?

- It balances churn differences, group stability, and clear customer profiles.
- Robust / no_friction has a slightly larger churn gap, but its groups are less stable.
- Standard / no_digital is more stable, but has a smaller churn gap and does not include app-use frequency or login recency.
- Cluster 1 is useful for retention analysis: it contains 30.18% of customers and 37.85% of all churners.
- Churn was not used to create clusters. It was used to compare the results afterwards. These findings still need testing on new data; the groups overlap.

## Run

Run from the shared project root after adding these files. The existing project must provide the ETL module and data in `data/full_dataset/`, or a compatible `data/full_bank_data.csv`.

```powershell
# Train the selected configuration
py run_hierarchical.py --mode train --config config/hierarchical_final.yaml --output-dir results/hierarchical/final

# StandardScaler ablation: 6 feature sets, k = 2 to 8
py run_hierarchical.py --mode ablation --config config/hierarchical_standard_ablation.yaml --output-dir results/hierarchical/standard

# RobustScaler ablation
py run_hierarchical.py --mode ablation --config config/hierarchical_robust_ablation.yaml --output-dir results/hierarchical/robust
```

Results are saved under `results/hierarchical/`. The selected fitted hierarchy and preprocessing parameters are in `models/saved/hierarchical_standard_no_friction_k4/`. Raw customer data is not included. Assignment of new customers to clusters is not implemented.
