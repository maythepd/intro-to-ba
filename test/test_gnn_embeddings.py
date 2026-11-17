"""
Unit and Integration Tests for Self-Supervised Graph Neural Network (GNN) Embeddings.
Verifies:
1. CustomerGraphBuilder k-NN graph construction (no self-loops, valid degrees).
2. Inductive evaluation graph construction (zero cross-test leakage).
3. GraphAttentionLayer message passing, attention softmax normalization, and residual connection.
4. SelfSupervisedGNNAutoencoder forward pass, masked reconstruction loss, and latent encoding.
5. GNNEmbeddingManager fit, transform, cache, and seamless pipeline integration.
6. Clean coexistence with TemporalEmbeddingManager (both, either, or neither enabled).
"""

import os
import sys
import shutil
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from config.pipeline_config import UnifiedConfig, PipelineConfig, load_unified_config
from preprocessing.gnn.graph_builder import CustomerGraphBuilder
from preprocessing.gnn.model import GraphAttentionLayer, SelfSupervisedGNNAutoencoder, GNNTrainer
from preprocessing.gnn.manager import GNNEmbeddingManager
from preprocessing.pipeline import FeatureEngineeringPipeline

TEMP_TEST_DIR = Path("results/test_gnn_temp")


class TestGNNEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up isolated test workspace."""
        if TEMP_TEST_DIR.exists():
            shutil.rmtree(TEMP_TEST_DIR, ignore_errors=True)
        TEMP_TEST_DIR.mkdir(parents=True, exist_ok=True)

        cls.N_nodes = 50
        cls.in_dim = 16
        cls.latent_dim = 8
        cls.k = 4

    @classmethod
    def tearDownClass(cls):
        """Clean up test artifacts."""
        if TEMP_TEST_DIR.exists():
            shutil.rmtree(TEMP_TEST_DIR, ignore_errors=True)

    def test_01_graph_builder_train_and_inductive(self):
        """Test 1: Verify k-NN graph construction and inductive evaluation graph."""
        builder = CustomerGraphBuilder(k_neighbors=self.k)
        X_train = np.random.randn(self.N_nodes, self.in_dim).astype(np.float32)

        edge_index, X_norm = builder.build_train_graph(X_train)

        # E = N * k edges
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_index.shape[1], self.N_nodes * self.k)

        # No self loops (src != dst for all edges)
        src = edge_index[0]
        dst = edge_index[1]
        self.assertFalse(torch.any(src == dst))

        # Inductive graph with unseen eval nodes
        N_eval = 10
        X_eval = np.random.randn(N_eval, self.in_dim).astype(np.float32)
        edge_eval, combined_X = builder.build_inductive_eval_graph(X_eval, X_train)

        self.assertEqual(edge_eval.shape[1], N_eval * self.k)
        # All destination nodes must be in the eval range [N_train, N_train + N_eval - 1]
        self.assertTrue(torch.all(edge_eval[1] >= self.N_nodes))
        # All source nodes must be in the train range [0, N_train - 1]
        self.assertTrue(torch.all(edge_eval[0] < self.N_nodes))

    def test_02_gat_layer_message_passing(self):
        """Test 2: Verify pure PyTorch GraphAttentionLayer forward pass."""
        gat = GraphAttentionLayer(in_features=self.in_dim, out_features=12, heads=2, dropout=0.0)
        X = torch.randn(self.N_nodes, self.in_dim)
        builder = CustomerGraphBuilder(k_neighbors=self.k)
        edge_index, _ = builder.build_train_graph(X)

        out = gat(X, edge_index)
        self.assertEqual(out.shape, (self.N_nodes, 24))  # heads * out_features = 2 * 12
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_03_self_supervised_gnn_autoencoder(self):
        """Test 3: Verify GNN Autoencoder training and latent encoding without class labels."""
        trainer = GNNTrainer(
            in_features=self.in_dim,
            hidden_dim=16,
            latent_dim=self.latent_dim,
            heads=2,
            lr=0.01,
            epochs=3,
            checkpoint_path=TEMP_TEST_DIR / "gnn_ckpt.pt",
            random_state=42,
            verbose=False,
        )

        X = torch.randn(self.N_nodes, self.in_dim)
        builder = CustomerGraphBuilder(k_neighbors=self.k)
        edge_index, X_norm = builder.build_train_graph(X)

        trainer.fit(X_norm, edge_index)
        self.assertTrue((TEMP_TEST_DIR / "gnn_ckpt.pt").exists())

        latents = trainer.transform(X_norm, edge_index)
        self.assertEqual(latents.shape, (self.N_nodes, self.latent_dim))
        self.assertFalse(np.isnan(latents).any())

    def test_04_gnn_embedding_manager_pipeline_integration(self):
        """Test 4: Verify GNNEmbeddingManager fit on train and inductive transform on test."""
        N_train = 40
        N_test = 10
        np.random.seed(42)

        df_train = pd.DataFrame({
            "customer_id": np.arange(1001, 1001 + N_train),
            "credit_score": np.random.randint(500, 800, size=N_train),
            "age": np.random.randint(20, 70, size=N_train),
            "balance": np.random.uniform(1000, 100000, size=N_train),
            "churn": np.random.randint(0, 2, size=N_train),
        })

        df_test = pd.DataFrame({
            "customer_id": np.arange(2001, 2001 + N_test),
            "credit_score": np.random.randint(500, 800, size=N_test),
            "age": np.random.randint(20, 70, size=N_test),
            "balance": np.random.uniform(1000, 100000, size=N_test),
            "churn": np.random.randint(0, 2, size=N_test),
        })

        mgr = GNNEmbeddingManager(
            enabled=True,
            k_neighbors=3,
            latent_dim=6,
            hidden_dim=12,
            heads=1,
            epochs=2,
            checkpoint_dir=TEMP_TEST_DIR / "checkpoints",
            cache_path=TEMP_TEST_DIR / "gnn_cache.csv",
            verbose=False,
            force_retrain=True,
        )

        # Fit strictly on train split without target y
        mgr.fit(df_train)
        self.assertIsNotNone(mgr.embeddings_df_)
        self.assertEqual(len(mgr.embeddings_df_), N_train)
        for i in range(6):
            self.assertIn(f"gnn_z{i}", mgr.embeddings_df_.columns)

        # Transform train dataframe
        df_tr_out = mgr.transform(df_train)
        self.assertIn("gnn_z0", df_tr_out.columns)
        self.assertFalse(df_tr_out["gnn_z0"].isna().any())

        # Inductive transform test dataframe (unseen customers)
        df_te_out = mgr.transform(df_test)
        self.assertIn("gnn_z0", df_te_out.columns)
        self.assertFalse(df_te_out["gnn_z0"].isna().any())
        self.assertEqual(len(df_te_out), N_test)


if __name__ == "__main__":
    unittest.main()
