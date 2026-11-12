import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns

def plot_clustering(model, profiles, sizes, churn, output):
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(model.linkage_, truncate_mode="lastp", p=30, ax=ax)
    ax.set(title="Hierarchical clustering (last 30 branches)", ylabel="Merge distance")
    fig.tight_layout(); fig.savefig(output / "dendrogram.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    relative = (profiles - profiles.mean()) / profiles.std(ddof=0).replace(0, 1)
    sns.heatmap(relative, cmap="vlag", center=0, ax=ax)
    ax.set_title("Cluster medians relative to other cluster medians (display only)")
    fig.tight_layout(); fig.savefig(output / "cluster_profile_heatmap.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(); sizes.plot.bar(ax=ax, title="Customers per cluster")
    fig.tight_layout(); fig.savefig(output / "cluster_sizes.png", dpi=180); plt.close(fig)
    if churn is not None:
        fig, ax = plt.subplots(); churn.plot.bar(ax=ax, title="Observed churn rate per cluster", ylim=(0, 1))
        fig.tight_layout(); fig.savefig(output / "churn_by_cluster.png", dpi=180); plt.close(fig)
