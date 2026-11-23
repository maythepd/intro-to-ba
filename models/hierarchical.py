"""Agglomerative hierarchy with an explicit reusable linkage matrix."""
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class HierarchicalModel:
    def __init__(self, method="ward"):
        if method not in ("ward", "average", "complete"):
            raise ValueError("Supported linkage: ward, average, complete")
        self.method = method

    def fit(self, x):
        self.linkage_ = linkage(x.to_numpy(), method=self.method, metric="euclidean")
        return self

    def labels(self, k):
        return cut_tree(self.linkage_, n_clusters=k).ravel()

    def evaluate(self, x, labels, sample_size=2000, seed=42):
        n_clusters = len(np.unique(labels))
        if not 1 < n_clusters < len(x):
            raise ValueError("Need 2..n-1 nonempty clusters")
        # Fixed row sample across configurations; all clustering itself uses the full population.
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(x), min(sample_size, len(x)), replace=False))
        sx, sl = x.iloc[indices], labels[indices]
        sil = silhouette_score(sx, sl) if 1 < len(np.unique(sl)) < len(sl) else float("nan")
        counts = np.bincount(labels)
        return {"silhouette": sil, "davies_bouldin": davies_bouldin_score(x, labels),
                "calinski_harabasz": calinski_harabasz_score(x, labels),
                "min_cluster_size": int(counts.min()), "max_cluster_share": float(counts.max()/len(x)),
                "silhouette_sample_size": len(indices)}
