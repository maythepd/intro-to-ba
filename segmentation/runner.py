"""A–D orchestration: freeze all unsupervised fits before reading evaluation labels."""
from dataclasses import replace
from pathlib import Path
import importlib.metadata
import json
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from preprocessing.splitting import DataSplitter
from .config import load_config
from .data import load_segmentation_data
from .features import (ORIGINAL_FEATURES, SYNTHETIC_FEATURES, REDUCED_FEATURES,
                       build_features, feature_sets, provenance)
from .kmeans import select_model, fit_at_k, permute_synthetic, quality
from .evaluation import (churn_summary, risk_separation, agreement, align_clusters,
                         distribution_profiles, personas, synthetic_diagnostics)


def run_experiments(frame, cfg, dataset, audit=None, make_plots=True):
    frame = frame.sort_values("customer_id").reset_index(drop=True)
    if frame.customer_id.isna().any() or frame.customer_id.duplicated().any():
        raise ValueError("Exactly one analytical record per customer is required")
    out, saved = Path(cfg.output_dir), Path(cfg.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved.mkdir(parents=True, exist_ok=True)
    # Reuse ratios and seed. Label-independent split satisfies the strict churn-only-after-fit rule.
    split = DataSplitter(test_size=dataset.get("test_size", .1), val_size=dataset.get("val_size", .1),
                         random_state=dataset.get("random_state", 42), stratify=False).train_val_test_split(frame.drop(columns="churn"))
    if not split.has_val:
        raise ValueError("Segmentation requires a separate validation partition")
    partition = pd.Series("train", index=frame.index)
    partition.loc[split.val_df.index] = "validation"
    partition.loc[split.test_df.index] = "test"
    tr, va, te = (partition.eq(name) for name in ("train", "validation", "test"))
    spaces = feature_sets(frame)
    models, selections, donor_records = {}, {}, []

    def fit(name, matrix, settings=cfg, frozen=None):
        print(f"[SEG] Selecting {name}: {matrix.shape[1]} input features", flush=True)
        with threadpool_limits(limits=settings.threads):
            model, scores = select_model(matrix.loc[tr], matrix.loc[va], settings, frozen)
        models[name], spaces[name], selections[name] = model, matrix, scores
        print(f"[SEG] {name}: selected k={model.named_steps['kmeans'].n_clusters}", flush=True)

    for name in ("A", "B", "C"):
        fit(name, spaces[name])
    for seed in cfg.permutation_seeds:
        permuted, donors = permute_synthetic(spaces["C"], partition, seed)
        name = f"D_{seed}"
        fit(name, permuted, frozen=models["C"].named_steps["preprocess"])
        donor_records.append(pd.DataFrame({"experiment": name, "customer_id": frame.customer_id,
            "donor_customer_id": frame.loc[donors.to_numpy(), "customer_id"].to_numpy(), "partition": partition}))
    for weight in cfg.sensitivity_weights:
        fit(f"C_weight_{weight:g}", spaces["C"], replace(cfg, original_weight=weight))
    fit("B_reduced_dependency", frame[REDUCED_FEATURES])
    fit("C_reduced_dependency", frame[ORIGINAL_FEATURES + REDUCED_FEATURES])

    main_names = ["A", "B", "C"] + [f"D_{s}" for s in cfg.permutation_seeds]
    common_k = min(cfg.common_k, *(int(selections[n].k.max()) for n in main_names))
    if common_k < 2:
        raise ValueError("Common k must be at least two")
    common_models = {n: fit_at_k(spaces[n].loc[tr], models[n].named_steps["preprocess"], common_k, cfg) for n in main_names}

    # All fits and sensitivity choices above are now frozen. Churn is used only below.
    assignments = frame[["customer_id"]].assign(partition=partition)
    predictions, common_predictions, comparison, churn_tables, size_tables = {}, {}, [], [], []
    for name, model in models.items():
        predictions[name] = model.predict(spaces[name])
        assignments[f"cluster_{name}"] = predictions[name]
        for part in ("train", "validation", "test"):
            labels = predictions[name][partition.eq(part)]
            counts = np.bincount(labels, minlength=model.named_steps["kmeans"].n_clusters)
            size_tables.append(pd.DataFrame({"experiment": name, "partition": part,
                "cluster": np.arange(len(counts)), "customer_count": counts,
                "customer_share": counts / len(labels)}))
        joblib.dump(model, saved / f"{name}.joblib")
        selected = selections[name].loc[selections[name].selected].iloc[0]
        with threadpool_limits(limits=cfg.threads):
            test_metrics = quality(model.named_steps["preprocess"].transform(spaces[name].loc[te]),
                                   predictions[name][te], model.named_steps["kmeans"], cfg)
        comparison.append({"experiment": name, "selected_k": model.named_steps["kmeans"].n_clusters,
            "input_features": len(spaces[name].columns), "encoded_dimensions": len(model.named_steps["preprocess"].feature_names_),
            "train_inertia": model.named_steps["kmeans"].inertia_,
            "validation_silhouette": selected.silhouette, "validation_davies_bouldin": selected.davies_bouldin,
            "stability_ari": selected.stability_ari, "stability_std": selected.stability_std,
            "test_risk_separation": risk_separation(predictions[name][te], frame.loc[te, "churn"]),
            **{f"test_{key}": value for key, value in test_metrics.items()}})
        churn_tables.append(churn_summary(predictions[name][te], frame.loc[te, "churn"]).assign(experiment=name))
        selections[name].to_csv(out / f"k_selection_{name}.csv", index=False)
        pd.Series(model.named_steps["preprocess"].feature_names_, name="encoded_feature").to_csv(saved / f"{name}_features.csv", index=False)

    common_rows = []
    for name, model in common_models.items():
        labels = model.predict(spaces[name].loc[te])
        common_predictions[name] = labels
        joblib.dump(model, saved / f"{name}_common_k.joblib")
        with threadpool_limits(limits=cfg.threads):
            metrics = quality(model.named_steps["preprocess"].transform(spaces[name].loc[te]), labels, model.named_steps["kmeans"], cfg)
        common_rows.append({"experiment": name, "k": common_k, **metrics,
            "risk_separation": risk_separation(labels, frame.loc[te, "churn"]),
            "ari_vs_A": agreement(common_predictions["A"], labels)["ari"]})
        churn_summary(labels, frame.loc[te, "churn"]).to_csv(out / f"churn_common_k_{name}.csv", index=False)

    comparison = pd.DataFrame(comparison)
    common = pd.DataFrame(common_rows)
    churn = pd.concat(churn_tables, ignore_index=True)
    agreement_rows, contingencies = [], {}
    for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
        labels_l, labels_r = predictions[left][te], predictions[right][te]
        agreement_rows.append({"pair": f"{left}_vs_{right}", **agreement(labels_l, labels_r)})
        table = pd.crosstab(pd.Series(labels_l, name=left), pd.Series(labels_r, name=right))
        contingencies[f"{left}_vs_{right}"] = table
        table.to_csv(out / f"contingency_{left}_vs_{right}.csv")
        table.div(table.sum(axis=1), axis=0).to_csv(out / f"transitions_{left}_vs_{right}.csv")
    aligned, mapping = align_clusters(predictions["A"][tr], predictions["C"][tr], predictions["C"])
    assignments["C_aligned_to_A"] = aligned
    assignments["A_to_C_changed_after_alignment"] = aligned != np.array([f"A_{a}" for a in predictions["A"]])
    transitions = contingencies["A_vs_C"]
    shares = transitions.div(transitions.sum(axis=1), axis=0)
    transition_summary = pd.DataFrame({"original_cluster": shares.index, "customer_count": transitions.sum(axis=1),
        "largest_combined_subgroup_share": shares.max(axis=1),
        "transition_entropy": -(shares * np.log(shares.where(shares > 0))).sum(axis=1)})
    transition_summary.to_csv(out / "transition_summary.csv", index=False)

    profiles, persona_tables, original_profiles = [], [], []
    for name in ("A", "B", "C"):
        columns = list(spaces[name].columns)
        profile = distribution_profiles(frame.loc[tr], predictions[name][tr], frame.loc[tr], columns)
        profiles.append(profile.assign(experiment=name))
        persona_tables.append(personas(profile, name))
        original_profiles.append(distribution_profiles(frame.loc[te], predictions[name][te], frame.loc[tr], ORIGINAL_FEATURES).assign(experiment=name))
    persona_table = pd.concat(persona_tables, ignore_index=True).merge(churn, on=["experiment", "cluster"], how="left")
    pd.concat(profiles, ignore_index=True).to_csv(out / "cluster_profiles.csv", index=False)
    original_profiles = pd.concat(original_profiles, ignore_index=True)
    original_profiles.to_csv(out / "original_distribution_profiles.csv", index=False)
    persona_table.to_csv(out / "personas.csv", index=False)

    description, correlations = synthetic_diagnostics(frame.loc[tr])
    description.to_csv(out / "synthetic_distributions.csv")
    correlations.to_csv(out / "synthetic_original_correlations.csv", index=False)
    # Independently verify the manifest's activity conditioning on this downloaded release.
    dependence = frame.loc[tr].groupby("active_member").agg(customers=("customer_id", "size"),
        q3_spend_positive=("trans_amount_3m", lambda s: s.gt(0).mean()),
        monthly_actions=("monthly_transactions", "mean"))
    dependence.to_csv(out / "activity_conditioning.csv")
    frame[ORIGINAL_FEATURES + SYNTHETIC_FEATURES].dtypes.astype(str).to_csv(out / "feature_types.csv", header=["dtype"])
    assignments.to_csv(out / "cluster_assignments.csv", index=False)
    pd.concat(donor_records, ignore_index=True).to_csv(out / "permutation_donors.csv", index=False)
    comparison.to_csv(out / "comparison.csv", index=False)
    summary = comparison.loc[comparison.experiment.isin(["A", "B", "C"])].copy()
    d_rows = comparison.loc[comparison.experiment.str.startswith("D_")]
    d_mean = d_rows.drop(columns="experiment").mean().to_dict()
    d_mean.update(experiment="D (mean across seeds)", selected_k=f"{d_rows.selected_k.min()}–{d_rows.selected_k.max()}")
    pd.concat([summary, pd.DataFrame([d_mean])], ignore_index=True).to_csv(out / "comparative_summary.csv", index=False)
    pd.concat(size_tables, ignore_index=True).to_csv(out / "cluster_sizes.csv", index=False)
    common.to_csv(out / "common_k_comparison.csv", index=False)
    common.loc[common.experiment.str.startswith("D_")].to_csv(out / "permutation_results.csv", index=False)
    churn.to_csv(out / "churn_by_cluster.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(out / "agreement.csv", index=False)
    pd.DataFrame(provenance()).T.rename_axis("feature").to_csv(out / "feature_provenance.csv")
    manifests = {name: {"input_features": list(spaces[name]), "selected_k": models[name].named_steps["kmeans"].n_clusters,
                       "original_weight": models[name].named_steps["preprocess"].original_weight} for name in models}
    try:
        commit = subprocess.run(["git", "-c", f"safe.directory={Path.cwd().as_posix()}", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-c", f"safe.directory={Path.cwd().as_posix()}", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
        dirty = None
    manifest = {"config": cfg.to_dict(), "dataset_config": dataset, "partitions": partition.value_counts().to_dict(),
        "split_stratified": False, "git_head": commit, "git_worktree_dirty": dirty, "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "joblib")},
        "experiments": manifests, "common_k": common_k, "alignment_learned_on": "train", "C_to_A_mapping": mapping,
        "audit": audit or {}}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    from .reporting import write_report, plot_results
    write_report(out, comparison, common, persona_table, pd.DataFrame(agreement_rows), transition_summary, manifest)
    if make_plots:
        plot_results(out, selections, assignments, churn, contingencies, original_profiles, common, frame)
    print(f"[SEG] Results: {out.resolve()}", flush=True)
    return comparison


def run_from_config(config_path, output_dir=None):
    cfg, dataset = load_config(config_path)
    if output_dir:
        cfg.output_dir = output_dir
    repo = Path(__file__).resolve().parent.parent
    raw = Path(dataset.get("raw_full_dataset_dir", "data/full_dataset"))
    if not raw.is_absolute():
        raw = repo / raw
    frame, audit = load_segmentation_data(raw)
    features = build_features(frame)
    return run_experiments(features, cfg, dataset, audit)
