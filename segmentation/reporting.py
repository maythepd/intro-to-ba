"""Reproducible tables/figures and a cautious, data-driven business report."""
from pathlib import Path
import numpy as np
import pandas as pd


def write_report(out, comparison, common, personas, agreements, transitions, manifest):
    compact = comparison[["experiment", "selected_k", "validation_silhouette", "validation_davies_bouldin",
                          "stability_ari", "test_risk_separation"]]
    controls = common.loc[common.experiment.str.startswith("D_")]
    c = common.loc[common.experiment.eq("C"), "risk_separation"].iloc[0]
    a = common.loc[common.experiment.eq("A"), "risk_separation"].iloc[0]
    null = controls.risk_separation.dropna()
    tail = (1 + null.ge(c).sum()) / (1 + len(null)) if np.isfinite(c) and len(null) else np.nan
    declaration = manifest.get("audit", {}).get("generation_manifest")
    selected_a = comparison.loc[comparison.experiment.eq("A"), "test_risk_separation"].iloc[0]
    selected_c = comparison.loc[comparison.experiment.eq("C"), "test_risk_separation"].iloc[0]
    fixture = manifest.get("audit", {}).get("fixture_only", False)
    lines = ["# Comparative K-Means customer segmentation", "",
        "AUTOMATED TEST FIXTURE ONLY. These metrics are not experimental results." if fixture else
        "Real-data run on the group's published source tables; exact hashes and library versions are in run_manifest.json.", "",
        "## Protocol", "",
        f"Customer partitions: {manifest['partitions']}. The configured ratios and seed are reused with stratification disabled "
        "so labels do not enter splitting or fitting. All preprocessing and KMeans centers are fit on training customers only. "
        "Validation chooses k without churn. All primary, permutation and sensitivity fits finish before churn analysis.", "",
        "For each k, evaluate multiple deterministic KMeans seeds with explicit n_init. Among candidates meeting the training "
        "minimum cluster share and validation stability floor, select the smallest k within the configured tolerance of the best "
        "mean validation silhouette. If constraints exclude every candidate, use valid silhouette candidates and flag the fallback "
        "in k_selection_*.csv. Davies–Bouldin breaks same-k ties; inertia/elbow, size entropy and stability are diagnostic. "
        "The serialized model uses the prespecified first seed, never a seed chosen using test results.", "",
        "Numerics use train-median imputation (including missingness indicators), signed log1p on monetary/count/recency/trend "
        "fields and StandardScaler. Categorical inputs use train-fitted one-hot encoding. Training-constant dimensions are removed. "
        "Each encoded group is weighted by sqrt(group weight / training variance energy). Equal weights give equal expected squared "
        "distance contributions. D reuses C's fitted preprocessing and permutes entire synthetic rows within each partition.", "",
        "## Selected models", "", compact.to_markdown(index=False, floatfmt=".4f"), "",
        "Silhouette, Davies–Bouldin and inertia are within-space diagnostics. Values from different feature spaces cannot establish "
        "that one experiment is better. Cross-space conclusions require the held-out associations, control and profile evidence.", "",
        "## Held-out churn association and permutation baseline", "",
        f"At common k={manifest['common_k']}, original-only between-cluster churn variance fraction is {a:.5f}; "
        f"combined is {c:.5f} (difference {c-a:+.5f}). The {len(null)} permuted controls have mean {null.mean():.5f} "
        f"and range [{null.min():.5f}, {null.max():.5f}]. The one-sided descriptive permutation tail estimate is {tail:.4f} "
        "with the +1 correction. This small control ensemble has coarse resolution; do not call this definitive statistical evidence.", "",
        ("Churn separation and its control comparison are unavailable when the held-out label has no variation."
         if not np.isfinite(tail) else
         "Combined segmentation shows greater descriptive churn separation than every permuted control in this run."
         if c > null.max() else "Combined separation does not exceed every permuted control; the run does not establish a robust advantage over added dimensions."), "",
        "This fraction is Var(cluster churn rate)/Var(binary churn), weighted by held-out cluster counts. It is descriptive and "
        "can increase with k; compare the common-k table. Churn rate confidence intervals are 95% Wilson intervals conditional on "
        "the frozen assignments, without multiplicity or fitted-model uncertainty correction. There is no causal retention estimate.", "",
        common.to_markdown(index=False, floatfmt=".4f"), "",
        f"At individually selected k, the A and C churn variance fractions are {selected_a:.5f} and {selected_c:.5f}. "
        + ("The direction reverses at common k: combined segmentation does not consistently improve over original-only."
           if np.isfinite(c) and np.sign(selected_c-selected_a) != np.sign(c-a) else
           "Assess the selected-k and common-k results together; these are post-fit descriptions, not a basis for tuning."), "",
        "## Agreement and transitions", "", agreements.to_markdown(index=False, floatfmt=".4f"), "",
        "ARI/NMI ignore arbitrary label permutations. Lower agreement can indicate useful subdivision or distortion, so consult "
        "profiles. Display matching uses a Hungarian assignment learned from training overlaps. Extra combined clusters keep new "
        "labels. A customer's matched label changing is a descriptive transition, not a change in their attributes.", "",
        transitions.to_markdown(index=False, floatfmt=".4f"), "",
        "The original customer population and attribute values are identical across A–C. Original distribution profiles describe "
        "changes in cluster composition: numerical mean differences are divided by the training population SD; category-proportion "
        "differences use sqrt(p_train*(1-p_train)). Numeric KS statistics compare each held-out cluster with the rest. Small clusters "
        "and non-Gaussian distributions require care; inspect medians, quartiles and proportions as well.", "",
        "## Personas", "",
        "Persona names and positive/negative deviations come from training profiles, not churn. Churn associations below are "
        "from independent test customers. Proposed actions need customer validation and retention experiments.", "",
        personas[["experiment", "cluster", "persona", "customer_count", "churn_rate", "ci_low", "ci_high", "lift"]].to_markdown(index=False, floatfmt=".4f"), ""]
    for _, row in personas.iterrows():
        lines += [f"- **{row.persona}**: {row.training_evidence}. Action: {row.retention_action}"]
    a_names = {n.split(" (")[0] for n in personas.loc[personas.experiment.eq("A"), "persona"]}
    introduced = sorted({n.split(" (")[0] for n in personas.loc[personas.experiment.isin(["B", "C"]), "persona"]} - a_names)
    lines += ["", "Behavioral persona labels absent in A: " + (", ".join(introduced) or "none under the evidence thresholds") + ". "
              "A cannot describe transaction, login or complaint patterns because those features are absent; new labels do not by themselves prove added real-world information.", "",
        "## Synthetic generation and sensitivity", "",
        "Generation manifest present: " + str(bool(declaration)) + ". " +
        (f"Manifest declares uses_churn_in_generation={declaration.get('uses_churn_in_generation', 'unspecified')} and "
         f"generator_version={declaration.get('generator_version', 'unspecified')}. " if declaration else "") +
        "The generator source notebook is unavailable, so the declaration cannot be independently verified against executable source.", "",
        "The published manifest conditions activity on original active_member, keeps original snapshot balances as hard endpoints, "
        "uses estimated salary for proposed funding and makes simulated dates consistent with tenure. Account balance components "
        "are excluded. Even transaction features are therefore synthetic-derived rather than independent new measurements. "
        "activity_conditioning.csv and synthetic_original_correlations.csv quantify observable dependence on the training partition.", "",
        "B_reduced_dependency/C_reduced_dependency remove transactions, monetary amounts, savings and explicit history indicators, "
        "retaining only login frequency/app usage, complaints, unresolved complaints, login failures and presence of a successful login. "
        "These remaining fields are still simulated and may retain indirect dependence. C_weight_* varies original-group distance "
        "weight; the rest belongs to synthetic features. These analyses were specified before inspecting churn results. "
        "They are not tuned on test churn, and there is no corresponding control ensemble for each sensitivity variant.", "",
        "RFM uses observed recency, initiated monthly frequency and monthly external initiated outflow. No supervised RFM scores or "
        "proposed CLV rules are imported. No prior-quarter exposure or zero prior activity leaves trend ratios missing; missing "
        "indicators and history/activity flags distinguish it from measured inactivity. System postings and incoming passive receipts "
        "do not count as initiated actions. Internal transfers count once and do not contribute external monetary flow.", "",
        "The Kaggle churn label's next-quarter interpretation is a project modeling assumption, not independently observed future "
        "churn. No real behavioral reference exists, so this analysis cannot establish fidelity to real banking behavior. The control "
        "tests association beyond shuffled rows, not realism. Source simulation, geometry, group redundancy and seed sensitivity "
        "remain plausible explanations for apparent gains.", "",
        "## Files and reproduction", "",
        "See docs/segmentation.md for setup and commands. CSVs include assignments/donors, selected and common-k metrics, "
        "profiles/personas, held-out churn, transitions, distributions, correlations and provenance. Models and feature names are "
        "serialized separately under the configured model directory. Reload with joblib.load and supply the recorded input columns. "
        "For D, supply the permuted behavioral rows recorded by permutation_donors.csv. Load only trusted joblib artifacts.", "",
        "References: [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), "
        "[clustering metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation), "
        "[OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).", ""]
    (Path(out) / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_results(out, selections, assignments, churn, contingencies, original_profiles, common, frame=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})
    colors = ["#164b65", "#158f87", "#db9653", "#8b6da8", "#6394a5", "#a94e57"]
    out = Path(out)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, bbox_inches="tight")
        plt.close(fig)

    for name, scores in selections.items():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, metric, title in zip(axes.flat, ["train_inertia", "silhouette", "davies_bouldin", "stability_ari"],
                                      ["Training inertia", "Validation silhouette", "Validation Davies–Bouldin", "Validation stability (ARI)"]):
            ax.plot(scores.k, scores[metric], "o-", color=colors[0])
            ax.axvline(scores.loc[scores.selected, "k"].iloc[0], color=colors[2], linestyle="--")
            ax.set(xlabel="k", title=title, xticks=scores.k)
        fig.suptitle(f"{name}: unsupervised k selection")
        save(fig, f"k_selection_{name}.png")
    main_names = ["A", "B", "C"] + [next(n for n in selections if n.startswith("D_"))]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    test = assignments.loc[assignments.partition.eq("test")]
    for ax, name in zip(axes, main_names):
        counts = test[f"cluster_{name}"].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts / counts.sum() * 100, color=colors[:len(counts)] if len(counts) <= len(colors) else colors[0])
        ax.set(title=name, xlabel="Cluster", ylabel="Held-out customers (%)")
    save(fig, "cluster_sizes.png")
    for pair, table in contingencies.items():
        values = table.div(table.sum(axis=1), axis=0).to_numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        plot = ax.imshow(values, cmap="Blues", vmin=0, vmax=1)
        for (r, c), value in np.ndenumerate(values):
            ax.text(c, r, f"{table.iloc[r, c]}\n{value:.0%}", ha="center", va="center", color="white" if value > .5 else "#17364b")
        ax.set(xticks=range(table.shape[1]), xticklabels=table.columns, yticks=range(table.shape[0]), yticklabels=table.index,
               xlabel=pair.split("_vs_")[1], ylabel=pair.split("_vs_")[0], title=f"{pair}: held-out transitions")
        fig.colorbar(plot, ax=ax, label="Share of row cluster")
        save(fig, f"transition_{pair}.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axes, ["A", "B", "C"]):
        rows = churn.loc[churn.experiment.eq(name)]
        errors = np.vstack([(rows.churn_rate-rows.ci_low).clip(lower=0), (rows.ci_high-rows.churn_rate).clip(lower=0)])
        ax.bar(rows.cluster.astype(str), rows.churn_rate * 100, color=colors[0], yerr=errors * 100, capsize=4)
        ax.axhline(rows.overall_prevalence.iloc[0] * 100, color=colors[2], linestyle="--")
        ax.set(title=name, ylabel="Held-out churn (%)", xlabel="Cluster")
    save(fig, "churn_by_cluster.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name in zip(axes, ["A", "C"]):
        data = original_profiles.loc[original_profiles.experiment.eq(name) & original_profiles.level.fillna("").eq("")]
        matrix = data.pivot(index="feature", columns="cluster", values="standardized_difference")
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set(title=f"{name}: original attribute composition", yticks=range(len(matrix)), yticklabels=matrix.index,
               xticks=range(matrix.shape[1]), xticklabels=matrix.columns, xlabel="Cluster")
    fig.colorbar(im, ax=axes[-1], label="Mean difference / training SD")
    save(fig, "original_distribution_comparison.png")
    if frame is not None:
        features = ["age", "credit_score", "balance", "products_number", "active_member", "country"]
        fig, axes = plt.subplots(2, len(features), figsize=(19, 7))
        for row, name in enumerate(["A", "C"]):
            labels = test[f"cluster_{name}"].to_numpy()
            values = frame.loc[assignments.partition.eq("test")]
            for ax, feature in zip(axes[row], features):
                for cluster in sorted(np.unique(labels)):
                    selected = values.loc[labels == cluster, feature].dropna()
                    color = colors[int(cluster) % len(colors)]
                    if feature in ("active_member", "products_number", "country"):
                        proportions = selected.value_counts(normalize=True).reindex(sorted(values[feature].dropna().unique()), fill_value=0)
                        ax.plot(proportions.index.astype(str), proportions, "o-", color=color, label=f"Cluster {cluster}")
                    else:
                        bins = np.histogram_bin_edges(values[feature].dropna(), bins=20)
                        ax.hist(selected, bins=bins, density=True, histtype="step", linewidth=1.5, color=color, label=f"Cluster {cluster}")
                ax.set(title=f"{name}: {feature}", ylabel="Within-cluster density / proportion")
                ax.tick_params(axis="x", labelsize=8)
            axes[row, 0].legend(fontsize=8)
        fig.suptitle("Same held-out population: original vs combined cluster composition")
        save(fig, "original_feature_distributions.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    null = common.loc[common.experiment.str.startswith("D_"), "risk_separation"].dropna()
    if len(null):
        ax.hist(null, color=colors[0], alpha=.75, bins=max(3, len(null)//2))
    else:
        ax.text(.5, .5, "Unavailable: no variation in held-out churn", ha="center", transform=ax.transAxes)
    for name, color in [("A", colors[2]), ("C", colors[1])]:
        value = common.loc[common.experiment.eq(name), "risk_separation"].iloc[0]
        if np.isfinite(value):
            ax.axvline(value, color=color, linewidth=2, label=name)
    ax.set(xlabel="Common-k held-out between-cluster churn variance fraction", ylabel="Permutation seeds", title="Association control; not evidence of realism")
    if len(null):
        ax.legend()
    save(fig, "permutation_control.png")
