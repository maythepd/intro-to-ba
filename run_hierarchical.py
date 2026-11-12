"""Hierarchical-only entrypoint; does not modify the classifier CLI."""
import argparse
from utils.clustering_experiments import run_hierarchical

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer hierarchical segmentation")
    parser.add_argument("--mode", choices=["train", "ablation"], default="train")
    parser.add_argument("--config", default="config/hierarchical_final.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rebuild-dataset", action="store_true")
    run_hierarchical(parser.parse_args())
