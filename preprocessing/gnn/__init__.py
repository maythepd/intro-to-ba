"""
Graph Neural Network Representation Package for Bank Customer Networks.
Extracts topological neighborhood representations via Self-Supervised Graph Attention Networks (GAT).
Zero target leakage (purely unsupervised).
"""

from .graph_builder import CustomerGraphBuilder
from .model import GraphAttentionLayer, SelfSupervisedGNNAutoencoder, GNNTrainer
from .manager import GNNEmbeddingManager

__all__ = [
    "CustomerGraphBuilder",
    "GraphAttentionLayer",
    "SelfSupervisedGNNAutoencoder",
    "GNNTrainer",
    "GNNEmbeddingManager",
]
