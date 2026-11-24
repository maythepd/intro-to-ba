"""Segmentation settings are independent of supervised feature settings."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import yaml


@dataclass
class SegmentationConfig:
    k_values: list = field(default_factory=lambda: list(range(2, 11)))
    seeds: list = field(default_factory=lambda: [42, 137, 2026])
    permutation_seeds: list = field(default_factory=lambda: [101, 211, 307, 401, 503])
    n_init: int = 10
    max_iter: int = 300
    silhouette_sample: int = 1000
    min_cluster_share: float = .02
    stability_floor: float = .80
    silhouette_tolerance: float = .02
    original_weight: float = .5
    sensitivity_weights: list = field(default_factory=lambda: [.25, .75])
    common_k: int = 4
    output_dir: str = "results/segmentation"
    model_dir: str = "models/saved/segmentation"
    threads: int = 4

    def __post_init__(self):
        if not self.k_values or any(not isinstance(k, int) or k < 2 for k in self.k_values):
            raise ValueError("k_values must contain integers >= 2")
        self.k_values = sorted(set(self.k_values))
        if len(set(self.seeds)) < 2 or len(set(self.permutation_seeds)) < 2:
            raise ValueError("At least two distinct stability and permutation seeds are required")
        if not 0 < self.original_weight < 1 or any(not 0 < w < 1 for w in self.sensitivity_weights):
            raise ValueError("Original group weights must be between zero and one")
        if min(self.n_init, self.max_iter, self.threads) < 1 or self.silhouette_sample < 3:
            raise ValueError("Invalid KMeans/metric resource settings")
        if not 0 <= self.min_cluster_share < .5 or not 0 <= self.stability_floor <= 1 or self.silhouette_tolerance < 0:
            raise ValueError("Invalid selection constraints")

    def to_dict(self):
        return asdict(self)


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SegmentationConfig(**raw.get("segmentation", {})), raw.get("dataset", {})
