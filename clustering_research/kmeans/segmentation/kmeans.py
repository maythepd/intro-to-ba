"""Training and model selection never receive churn labels or test features."""
from itertools import combinations
import copy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .features import SYNTHETIC_FEATURES
from .preprocessing import GroupPreprocessor


def permute_synthetic(frame, partitions, seed):
    """Permute complete behavioral rows within partitions; return donor row indices."""
    if not frame.index.is_unique or not partitions.index.equals(frame.index) or partitions.isna().any():
        raise ValueError("Permutation requires unique aligned rows and complete partitions")
    rng = np.random.default_rng(seed)
    out = frame.copy()
    donors = pd.Series(index=frame.index, dtype=frame.index.dtype)
    cols = [c for c in SYNTHETIC_FEATURES if c in frame]
    for part in partitions.unique():
        indices = frame.index[partitions.eq(part)]
        shuffled = rng.permutation(indices)
        out.loc[indices, cols] = frame.loc[shuffled, cols].to_numpy()
        donors.loc[indices] = shuffled
    return out, donors


def quality(values, labels, model, cfg):
    n = len(values)
    counts = np.bincount(labels, minlength=model.n_clusters)
    sizes = counts / n
    valid = 1 < len(np.unique(labels)) < n
    sil = np.nan
    if valid:
        try:
            sil = silhouette_score(values, labels, sample_size=min(n, cfg.silhouette_sample), random_state=cfg.seeds[0])
        except ValueError:  # A sample may omit a very small cluster.
            pass
    distances = values - model.cluster_centers_[labels]
    return {"inertia": float(np.square(distances).sum()), "inertia_per_customer": float(np.square(distances).sum() / n),
            "silhouette": float(sil), "davies_bouldin": float(davies_bouldin_score(values, labels)) if valid else np.nan,
            "min_cluster_share": float(sizes.min()), "max_cluster_share": float(sizes.max()),
            "size_entropy": float(-np.sum(sizes[sizes > 0] * np.log(sizes[sizes > 0])) / np.log(model.n_clusters))}


def fit_centers(train_values, k, cfg, seed):
    with threadpool_limits(limits=cfg.threads):
        return KMeans(n_clusters=k, random_state=seed, n_init=cfg.n_init, max_iter=cfg.max_iter).fit(train_values)


def select_model(train, validation, cfg, frozen_preprocessor=None):
    """Select smallest k near the best validation silhouette, with stability/size guards."""
    if len(validation) < 3:
        raise ValueError("Segmentation requires at least three validation customers")
    prep = copy.deepcopy(frozen_preprocessor) if frozen_preprocessor is not None else GroupPreprocessor(cfg.original_weight).fit(train)
    with threadpool_limits(limits=cfg.threads):
        tr, va = prep.transform(train), prep.transform(validation)
    max_k = min(len(train) - 1, len(validation) - 1, len(np.unique(tr, axis=0)))
    candidates = [k for k in cfg.k_values if k <= max_k]
    if not candidates:
        raise ValueError("Not enough distinct training/validation customers for requested k")
    records, models = [], {}
    for k in candidates:
        assignments, seed_metrics = [], []
        for seed in cfg.seeds:
            model = fit_centers(tr, k, cfg, seed)
            predictions = model.predict(va)
            assignments.append(predictions)
            with threadpool_limits(limits=cfg.threads):
                metrics = quality(va, predictions, model, cfg)
            metrics["train_inertia"] = model.inertia_
            metrics["train_min_share"] = np.bincount(model.labels_, minlength=k).min() / len(tr)
            seed_metrics.append(metrics)
            if seed == cfg.seeds[0]:
                models[k] = model
        stability = [adjusted_rand_score(a, b) for a, b in combinations(assignments, 2)]
        record = {"k": k, "stability_ari": np.mean(stability), "stability_std": np.std(stability)}
        for key in seed_metrics[0]:
            values = np.asarray([m[key] for m in seed_metrics])
            record[key] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
        record["silhouette_std"] = float(np.nanstd([m["silhouette"] for m in seed_metrics]))
        records.append(record)
    scores = pd.DataFrame(records)
    feasible = scores.loc[scores.silhouette.notna() & scores.train_min_share.ge(cfg.min_cluster_share)
                          & scores.stability_ari.ge(cfg.stability_floor)]
    fallback = feasible.empty
    if fallback:
        feasible = scores.loc[scores.silhouette.notna()]
    if feasible.empty:
        raise ValueError("No k has a valid validation silhouette")
    near_best = feasible.loc[feasible.silhouette >= feasible.silhouette.max() - cfg.silhouette_tolerance]
    selected_k = int(near_best.sort_values(["k", "davies_bouldin"]).iloc[0].k)
    scores["selected"] = scores.k.eq(selected_k)
    scores["selection_fallback"] = fallback
    pipeline = Pipeline([("preprocess", prep), ("kmeans", models[selected_k])])
    return pipeline, scores


def fit_at_k(train, prep, k, cfg):
    model = fit_centers(prep.transform(train), k, cfg, cfg.seeds[0])
    return Pipeline([("preprocess", copy.deepcopy(prep)), ("kmeans", model)])
