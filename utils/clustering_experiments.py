"""Shared runner for descriptive hierarchical segmentation and feature ablation."""
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score
from preprocessing.etl import BankDataETL
from preprocessing.hierarchical_clustering_pipeline import HierarchicalClusteringPipeline, GROUPS
from models.hierarchical import HierarchicalModel
from visualization.clustering_plots import plot_clustering

ROOT = Path(__file__).resolve().parents[1]

def project_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p

def markdown_table(value):
    table = value.to_frame() if isinstance(value, pd.Series) else value
    if table.index.name is not None:
        table = table.reset_index()
    header = "| " + " | ".join(map(str, table.columns)) + " |"
    rule = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join([header, rule] + rows)

def business_metrics(df, labels):
    """Post-clustering descriptions only; never used to select features or k."""
    customers = df.assign(cluster_id=np.asarray(labels))
    grouped = customers.groupby("cluster_id")
    table = grouped.size().to_frame("customers")
    table["customer_share"] = table["customers"] / len(customers)
    if "churn" in customers:
        if customers["churn"].isna().any() or not customers["churn"].isin([0, 1]).all():
            raise ValueError("Business metrics require binary nonmissing churn")
        total_churn = customers["churn"].sum()
        base_rate = customers["churn"].mean()
        table["churned_customers"] = grouped["churn"].sum()
        table["churn_rate"] = table["churned_customers"] / table["customers"]
        table["population_churn_rate"] = base_rate
        table["churn_lift"] = table["churn_rate"] / base_rate if base_rate > 0 else np.nan
        table["churn_share"] = table["churned_customers"] / total_churn if total_churn > 0 else np.nan
    if "balance" in customers:
        if customers["balance"].isna().any() or (customers["balance"] < 0).any():
            raise ValueError("Business metrics require nonnegative nonmissing balance")
        total_balance = customers["balance"].sum()
        table["total_balance"] = grouped["balance"].sum()
        table["balance_share"] = table["total_balance"] / total_balance if total_balance > 0 else np.nan
        table["mean_balance"] = grouped["balance"].mean()
        table["median_balance"] = grouped["balance"].median()
        if "churn" in customers:
            table["churned_customer_balance"] = customers.assign(
                churned_balance=customers["balance"] * customers["churn"]
            ).groupby("cluster_id")["churned_balance"].sum()
    return table


def run_hierarchical(args):
    if args.mode not in ("train", "ablation"):
        raise ValueError("hierarchical supports train and ablation")
    raw = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    cfg = raw.get("clustering", {})
    dataset = raw["dataset"]
    path = project_path(dataset["data_path"])
    if args.rebuild_dataset or dataset.get("rebuild_full_dataset", False) or not path.exists():
        etl = BankDataETL(raw_data_dir=project_path(dataset["raw_full_dataset_dir"]))
        df = etl.process()
        etl.save(df, path)
    else:
        df = pd.read_csv(path)
    output = project_path(args.output_dir or "results/hierarchical") / args.mode
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_used.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    full = list(GROUPS) if args.mode == "ablation" else cfg.get("feature_groups", list(GROUPS))
    if not full or any(group not in GROUPS for group in full):
        raise ValueError("feature_groups must contain known feature groups")
    experiments = {cfg.get("experiment_name", "full") if args.mode == "train" else "full": full}
    if args.mode == "ablation":
        experiments.update({"rfm_baseline": ["rfm"], **{f"no_{g}": [v for v in full if v != g] for g in full if g != "rfm"}})
    results = []
    business_tables = []
    fixed_k = int(cfg.get("n_clusters", 4))
    for name, groups in experiments.items():
        started = time.perf_counter()
        dest = output / name
        dest.mkdir(parents=True, exist_ok=True)
        prep = HierarchicalClusteringPipeline(cfg)
        x = prep.prepare(df, groups)
        model = HierarchicalModel(cfg.get("linkage", "ward")).fit(x)
        ks = sorted(set([fixed_k] + list(cfg.get("k_candidates", [2,3,4,5,6,7,8]))))
        if any(not isinstance(k, int) or not 2 <= k < len(x) for k in ks):
            raise ValueError("Each k must be an integer between 2 and n_customers-1")
        full_labels = {k: model.labels(k) for k in ks}
        stability = {k: [] for k in ks}
        # One hierarchy per subset supports all candidate cuts without refitting for every k.
        for seed in range(int(cfg.get("stability_repeats", 3))):
            rng = np.random.default_rng(42 + seed)
            size = min(len(df), max(max(ks)+1, int(len(df)*0.8)))
            idx = np.sort(rng.choice(len(df), size, replace=False))
            subx = HierarchicalClusteringPipeline(cfg).prepare(df.iloc[idx], groups)
            submodel = HierarchicalModel(cfg.get("linkage", "ward")).fit(subx)
            for k in ks:
                stability[k].append(adjusted_rand_score(full_labels[k][idx], submodel.labels(k)))
        selected = [c for g in groups for c in GROUPS[g]]
        for k in ks:
            labels = full_labels[k]
            results.append({"experiment": name, "k": k, "n_features": x.shape[1],
                            **model.evaluate(x, labels, int(cfg.get("silhouette_sample_size", 2000))),
                            "subsample_stability_ARI": float(np.mean(stability[k])) if stability[k] else None,
                            "elapsed_seconds": time.perf_counter()-started})
            kdest = dest / f"k_{k}"
            kdest.mkdir(parents=True, exist_ok=True)
            business = business_metrics(df, labels)
            business.to_csv(kdest / "business_metrics.csv")
            business_tables.append(business.reset_index().assign(experiment=name, k=k))
            df[["customer_id"]].assign(cluster_id=labels).to_csv(kdest / "customer_clusters.csv", index=False)
            profiles_k = prep.profile_features[selected].assign(cluster_id=labels).groupby("cluster_id").median()
            profiles_k.to_csv(kdest / "cluster_profiles.csv")
            sections = [f"# {name}: k={k}", "## Business metrics", markdown_table(business),
                        "Shares and rates are fractions (0.25 = 25%); lift is a multiple (2 = twice the population churn rate).",
                        "Balance is snapshot deposit balance in CU, not CLV, profit, or an estimated loss. Churned-customer balance is descriptive, not a loss estimate.",
                        "## Profiles in original units", markdown_table(profiles_k)]
            (kdest / "report.md").write_text("\n\n".join(sections), encoding="utf-8")
        labels = full_labels[fixed_k]
        prep.feature_report.to_csv(dest / "feature_processing.csv", index=False)
        scaler_center = prep.scaler.mean_ if hasattr(prep.scaler, "mean_") else prep.scaler.center_
        pd.DataFrame({"feature": x.columns, "center_after_log_imputation": scaler_center,
                      "scale": prep.scaler.scale_, "scaler": type(prep.scaler).__name__}).to_csv(dest / "scaler_parameters.csv", index=False)
        x.assign(customer_id=df["customer_id"].values).to_csv(dest / "clustering_input.csv", index=False)
        customers = df.copy()
        customers["cluster_id"] = labels
        customers.to_csv(dest / "customer_clusters.csv", index=False)
        selected = [c for g in groups for c in GROUPS[g]]
        profiles = prep.profile_features[selected].assign(cluster_id=labels).groupby("cluster_id").median()
        profiles.to_csv(dest / "cluster_profiles.csv")
        sizes = customers.groupby("cluster_id").size()
        sizes.to_csv(dest / "cluster_sizes.csv", header=["customers"])
        churn = None
        if "churn" in customers:
            churn_table = customers.groupby("cluster_id")["churn"].agg(["count", "sum", "mean"])
            churn_table.to_csv(dest / "churn_by_cluster.csv")
            churn = churn_table["mean"]
        business = business_metrics(df, labels)
        business.to_csv(dest / "business_metrics.csv")
        plot_clustering(model, profiles, sizes, churn, dest)
        np.save(dest / "linkage.npy", model.linkage_)
        import copy
        saved_prep = copy.copy(prep)
        saved_prep.feature_columns_ = list(x.columns)
        if hasattr(saved_prep, "profile_features"):
            del saved_prep.profile_features
        with (dest / "preprocessing.pkl").open("wb") as f:
            pickle.dump(saved_prep, f)
        report = [f"# Hierarchical: {name}", f"Customers: {len(df)}; clusters: {fixed_k}; linkage: {model.method}.",
                  "Descriptive segmentation of the full snapshot. No churn/target encoding/SMOTE used for distances.",
                  "The configured k is used for profiles; candidate metrics do not automatically select k.",
                  "Silhouette uses the same fixed row sample; DB/CH use all rows. Compare metrics across feature spaces cautiously.",
                  "Trend without prior exposure is median-imputed, not interpreted as unchanged activity.",
                  "Cluster profiles are medians in original units; assign persona names after reviewing these values.",
                  "## Feature processing", markdown_table(prep.feature_report),
                  "## Profiles", markdown_table(profiles), "## Sizes", markdown_table(sizes),
                  "## Business metrics", markdown_table(business),
                  "Shares/rates are fractions; balance is CU, not CLV or loss. See k_*/report.md for each candidate k."]
        (dest / "report.md").write_text("\n\n".join(report), encoding="utf-8")
        print(f"[OK] {name}: {len(df)} customers, {x.shape[1]} features; scaler={cfg.get('scaler', 'standard')}")
    metrics = pd.DataFrame(results)
    metrics.to_csv(output / "hierarchical_metrics.csv", index=False)
    all_business = pd.concat(business_tables, ignore_index=True)
    all_business.to_csv(output / "hierarchical_business_metrics.csv", index=False)
    (output / f"{args.mode}_hierarchical_report.md").write_text(
        "# Hierarchical " + args.mode + "\n\n" + markdown_table(metrics)
        + "\n\nEvery experiment explores every candidate k. Main profiles use configured k, never a churn-based choice. Review sizes, stability and original-unit profiles. "
        + "Internal metrics across different feature spaces are not directly comparable evidence of business value. "
        + "See hierarchical_business_metrics.csv for all business tables, and experiment/k_*/report.md for each cut. Per-experiment feature_processing.csv documents all transformations and scalers.", encoding="utf-8")
    summary_path = output / f"{args.mode}_hierarchical_report.md"
    with summary_path.open("a", encoding="utf-8") as report:
        report.write("\n\n## Business metrics at configured k\n\n" + markdown_table(all_business[all_business["k"] == fixed_k]))
        report.write("\n\nShares/rates are fractions (0.25 = 25%); lift is a multiple; balances are CU, not CLV or loss.")
    print(f"[OK] Results: {output}")
