"""
Graph Topology Builder for Bank Customer Similarity Network.
Constructs k-Nearest Neighbor (k-NN) graphs based on normalized customer feature vectors
with strict inductive split isolation (no test-to-train or cross-validation leakage).
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


class CustomerGraphBuilder:
    """
    Constructs k-NN topological graphs from customer representations.
    Supports:
      1. Transductive Train Graph: k-NN connections within the training pool.
      2. Inductive Test Graph: Connects unseen evaluation nodes strictly to
         reference training nodes (zero transductive leakage).
    """

    def __init__(self, k_neighbors: int = 5, chunk_size: int = 2000):
        self.k_neighbors = int(k_neighbors)
        self.chunk_size = int(chunk_size)

    def build_train_graph(
        self,
        features: Union[np.ndarray, torch.Tensor],
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds a k-NN graph over training nodes.
        Self-loops are excluded from the k nearest neighbors.

        Returns:
          edge_index: Tensor of shape (2, E), where E = N * k.
                      edge_index[0] = source nodes, edge_index[1] = destination nodes.
          features_tensor: Normalized node feature tensor (N, D).
        """
        if isinstance(features, np.ndarray):
            X = torch.from_numpy(features).float().to(device)
        else:
            X = features.float().to(device)

        N = X.size(0)
        k = min(self.k_neighbors, max(1, N - 1))
        X_norm = F.normalize(X, p=2, dim=-1)

        src_list = []
        dst_list = []

        for i in range(0, N, self.chunk_size):
            end_i = min(i + self.chunk_size, N)
            chunk = X_norm[i:end_i]  # (chunk_size, D)
            sim = torch.matmul(chunk, X_norm.t())  # (chunk_size, N)

            # Mask self-similarity
            diag_idx = torch.arange(i, end_i, device=device)
            sim.scatter_(1, diag_idx.unsqueeze(1), -1e9)

            # Top-k most similar neighbors for each node in chunk
            _, topk_idx = torch.topk(sim, k=k, dim=1)

            src = torch.arange(i, end_i, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
            dst = topk_idx.reshape(-1)

            src_list.append(src)
            dst_list.append(dst)

        src_all = torch.cat(src_list)
        dst_all = torch.cat(dst_list)
        edge_index = torch.stack([src_all, dst_all], dim=0)

        return edge_index, X_norm

    def build_inductive_eval_graph(
        self,
        eval_features: Union[np.ndarray, torch.Tensor],
        train_features: Union[np.ndarray, torch.Tensor],
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds inductive bipartite k-NN edges from evaluation nodes (Val or Test)
        pointing strictly into training nodes.
        Zero cross-test connection, zero target leakage.

        Returns:
          edge_index: (2, N_eval * k)
          combined_features: (N_train + N_eval, D)
        """
        if isinstance(eval_features, np.ndarray):
            X_eval = torch.from_numpy(eval_features).float().to(device)
        else:
            X_eval = eval_features.float().to(device)

        if isinstance(train_features, np.ndarray):
            X_train = torch.from_numpy(train_features).float().to(device)
        else:
            X_train = train_features.float().to(device)

        N_train = X_train.size(0)
        N_eval = X_eval.size(0)
        k = min(self.k_neighbors, max(1, N_train))

        X_train_norm = F.normalize(X_train, p=2, dim=-1)
        X_eval_norm = F.normalize(X_eval, p=2, dim=-1)

        src_list = []
        dst_list = []

        for j in range(0, N_eval, self.chunk_size):
            end_j = min(j + self.chunk_size, N_eval)
            chunk = X_eval_norm[j:end_j]
            sim = torch.matmul(chunk, X_train_norm.t())  # (chunk_size, N_train)

            _, topk_idx = torch.topk(sim, k=k, dim=1)

            # Node indices in the combined graph: eval nodes start at N_train + j
            eval_indices = torch.arange(N_train + j, N_train + end_j, device=device)
            dst = eval_indices.unsqueeze(1).expand(-1, k).reshape(-1)
            src = topk_idx.reshape(-1)  # Source is train node

            src_list.append(src)
            dst_list.append(dst)

        src_all = torch.cat(src_list)
        dst_all = torch.cat(dst_list)
        edge_index = torch.stack([src_all, dst_all], dim=0)

        combined_features = torch.cat([X_train_norm, X_eval_norm], dim=0)
        return edge_index, combined_features
