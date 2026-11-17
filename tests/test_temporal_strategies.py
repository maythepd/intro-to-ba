"""
Unit and Integration Tests for Pluggable Temporal Representation Learning Strategies.
Verifies:
1. Strategy Registry & Factory (get_temporal_strategy) for all 5 strategies.
2. Metric loss computation and gradient backpropagation for each strategy.
3. Manifold feature generation consistency and schema match with get_feature_names().
4. End-to-end integration with TemporalAutoencoderTrainer and TemporalEmbeddingManager.
5. Lossless configuration loading and strategy switching from config.yaml.
6. Complete cleanup of all temporary test artifacts.
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

from config.pipeline_config import UnifiedConfig, PipelineConfig
from preprocessing.temporal.strategies import (
    get_temporal_strategy,
    BaseTemporalStrategy,
    UnsupervisedStrategy,
    AsymmetricArcFaceStrategy,
    HyperbolicPoincareStrategy,
    OptimalTransportStrategy,
    NeymanPearsonStrategy,
    STRATEGY_REGISTRY,
)
from preprocessing.temporal.model import (
    BiGRUTemporalAutoencoder,
    TemporalAttentionAutoencoder,
    TemporalAutoencoderTrainer,
)
from preprocessing.temporal.manager import TemporalEmbeddingManager

TEMP_TEST_DIR = Path("results/test_strategies_temp")


class TestTemporalStrategies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up isolated test workspace."""
        if TEMP_TEST_DIR.exists():
            shutil.rmtree(TEMP_TEST_DIR, ignore_errors=True)
        TEMP_TEST_DIR.mkdir(parents=True, exist_ok=True)

        cls.batch_size = 16
        cls.seq_len = 6
        cls.in_features = 10
        cls.latent_dim = 16

        # Synthetic tensors
        torch.manual_seed(42)
        np.random.seed(42)
        cls.dummy_bx = torch.randn(cls.batch_size, cls.seq_len, cls.in_features)
        cls.dummy_recon = torch.randn(cls.batch_size, cls.seq_len, cls.in_features)
        cls.dummy_z = torch.randn(cls.batch_size, cls.latent_dim)
        cls.dummy_z_t1 = torch.randn(cls.batch_size, cls.latent_dim)
        cls.dummy_z_t6 = torch.randn(cls.batch_size, cls.latent_dim)
        cls.dummy_by = torch.randint(0, 2, (cls.batch_size,)).long()

    @classmethod
    def tearDownClass(cls):
        """Clean up all artifacts produced during testing."""
        if TEMP_TEST_DIR.exists():
            shutil.rmtree(TEMP_TEST_DIR, ignore_errors=True)
            print(f"\n[OK] Cleaned up temporary strategy test artifacts at: {TEMP_TEST_DIR}")

    def test_01_strategy_registry_and_factory(self):
        """Test 1: Verify all 5 strategies are registered and instantiated via factory."""
        expected_keys = [
            "unsupervised",
            "asymmetric_arcface",
            "hyperbolic_poincare",
            "optimal_transport",
            "neyman_pearson",
            "temporal_attention",
        ]
        for key in expected_keys:
            self.assertIn(key, STRATEGY_REGISTRY)
            strategy = get_temporal_strategy(key)
            self.assertIsInstance(strategy, BaseTemporalStrategy)
            self.assertEqual(strategy.name, key)

        with self.assertRaises(ValueError):
            get_temporal_strategy("non_existent_strategy_xyz")

    def test_02_strategy_loss_computation(self):
        """Test 2: Verify compute_loss returns valid loss tensor and metric dictionary for each strategy."""
        device = torch.device("cpu")

        for key in STRATEGY_REGISTRY.keys():
            strategy = get_temporal_strategy(key)
            head = strategy.build_loss_head(latent_dim=self.latent_dim, device=device)

            loss, metrics = strategy.compute_loss(
                recon=self.dummy_recon,
                bx=self.dummy_bx,
                z=self.dummy_z,
                by=self.dummy_by,
                head=head,
            )

            self.assertIsInstance(loss, torch.Tensor)
            self.assertFalse(torch.isnan(loss).item())
            self.assertGreater(loss.item(), 0.0)
            self.assertIn("recon_mse", metrics)
            self.assertIn("total_loss", metrics)

            # Critical test: when recon is near-perfect (recon == bx), total_loss must STILL be >= 0.0
            # (No strategy is permitted to yield negative loss under any convergence stage)
            perfect_loss, _ = strategy.compute_loss(
                recon=self.dummy_bx,
                bx=self.dummy_bx,
                z=self.dummy_z,
                by=self.dummy_by,
                head=head,
            )
            self.assertGreaterEqual(
                perfect_loss.item(),
                0.0,
                f"Strategy {key} yielded negative loss ({perfect_loss.item()}) on perfect reconstruction!",
            )

    def test_03_strategy_features_schema(self):
        """Test 3: Verify compute_features outputs match declared feature names without NaNs."""
        device = torch.device("cpu")

        for key in STRATEGY_REGISTRY.keys():
            strategy = get_temporal_strategy(key)
            head = strategy.build_loss_head(latent_dim=self.latent_dim, device=device)

            latents, geom = strategy.compute_features(
                z=self.dummy_z,
                bx=self.dummy_bx,
                z_t1=self.dummy_z_t1,
                z_t6=self.dummy_z_t6,
                head=head,
            )

            self.assertEqual(latents.shape, (self.batch_size, self.latent_dim))
            self.assertFalse(np.isnan(latents).any())

            declared_names = strategy.get_feature_names(self.latent_dim)
            expected_geom = [c for c in declared_names if not c.startswith("temporal_z")]

            for g_col in expected_geom:
                self.assertIn(g_col, geom, f"Strategy {key} missing declared feature {g_col}")
                self.assertEqual(geom[g_col].shape[0], self.batch_size)
                self.assertFalse(np.isnan(geom[g_col]).any(), f"NaN in feature {g_col} for strategy {key}")

    def test_04_trainer_end_to_end(self):
        """Test 4: Verify TemporalAutoencoderTrainer trains and transforms on small synthetic tensor."""
        ckpt_path = TEMP_TEST_DIR / "test_trainer_ckpt.pt"
        trainer = TemporalAutoencoderTrainer(
            in_features=self.in_features,
            seq_len=self.seq_len,
            hidden_dim=32,
            num_layers=1,
            latent_dim=self.latent_dim,
            epochs=2,
            batch_size=8,
            checkpoint_path=ckpt_path,
            verbose=False,
            force_retrain=True,
            manifold_strategy="asymmetric_arcface",
            lambda_angular=0.30,
            margin_churn=0.20,
            margin_loyal=0.00,
            angular_scale=4.0,
        )

        synthetic_data = np.random.randn(32, self.seq_len, self.in_features).astype(np.float32)
        synthetic_labels = np.random.randint(0, 2, 32).astype(np.int64)

        trainer.fit(synthetic_data, labels=synthetic_labels)
        self.assertTrue(ckpt_path.exists())

        latents, geom = trainer.transform_with_geometry(synthetic_data)
        self.assertEqual(latents.shape, (32, self.latent_dim))
        self.assertIn("temporal_drift_velocity", geom)
        self.assertIn("temporal_angular_churn_dist", geom)

    def test_05_config_yaml_integration(self):
        """Test 5: Verify config.yaml loads new strategy fields accurately into PipelineConfig."""
        cfg = UnifiedConfig.from_yaml("config/config.yaml")
        feat = cfg.features

        self.assertTrue(feat.enable_temporal_embeddings)
        self.assertIn(
            feat.temporal_manifold_strategy,
            ["unsupervised", "asymmetric_arcface", "hyperbolic_poincare", "optimal_transport", "neyman_pearson", "temporal_attention"],
        )
        self.assertEqual(feat.temporal_latent_dim, 16)
        self.assertGreater(feat.temporal_angular_scale, 0.0)
        self.assertGreater(feat.temporal_curvature, 0.0)
        self.assertGreater(feat.temporal_alpha_fpr, 0.0)

    def test_06_checkpoint_mismatch_and_architecture_routing(self):
        """Test 6: Verify architecture routing strictly decouples BiGRU vs TemporalAttention,
        and incompatible checkpoints trigger clean retraining without PyTorch state_dict crashes."""
        manifold_strategies = [
            "unsupervised",
            "asymmetric_arcface",
            "hyperbolic_poincare",
            "optimal_transport",
            "neyman_pearson",
        ]

        dummy_x = np.random.randn(8, self.seq_len, self.in_features).astype(np.float32)

        # 1. Verify routing for all 4 manifold modules + unsupervised: MUST be BiGRUTemporalAutoencoder
        for strat in manifold_strategies:
            trainer = TemporalAutoencoderTrainer(
                in_features=self.in_features,
                seq_len=self.seq_len,
                hidden_dim=16,
                num_layers=1,
                latent_dim=self.latent_dim,
                epochs=1,
                batch_size=8,
                verbose=False,
                manifold_strategy=strat,
                strategy_kwargs={"use_temporal_attention": True},  # Simulated config leak
            )
            trainer.fit(dummy_x)
            self.assertIsInstance(
                trainer.model,
                BiGRUTemporalAutoencoder,
                f"Strategy {strat} should route to BiGRUTemporalAutoencoder even if use_temporal_attention is True",
            )

        # 2. Verify routing for temporal_attention: MUST be TemporalAttentionAutoencoder
        attn_trainer = TemporalAutoencoderTrainer(
            in_features=self.in_features,
            seq_len=self.seq_len,
            hidden_dim=16,
            num_layers=1,
            latent_dim=self.latent_dim,
            epochs=1,
            batch_size=8,
            verbose=False,
            manifold_strategy="temporal_attention",
        )
        attn_trainer.fit(dummy_x)
        self.assertIsInstance(attn_trainer.model, TemporalAttentionAutoencoder)

        # 3. Save a TemporalAttentionAutoencoder checkpoint to disk
        attn_ckpt_path = TEMP_TEST_DIR / "mismatch_attn_ckpt.pt"
        attn_trainer_saving = TemporalAutoencoderTrainer(
            in_features=self.in_features,
            seq_len=self.seq_len,
            hidden_dim=16,
            num_layers=1,
            latent_dim=self.latent_dim,
            epochs=1,
            batch_size=8,
            checkpoint_path=attn_ckpt_path,
            verbose=False,
            manifold_strategy="temporal_attention",
        )
        attn_trainer_saving.fit(dummy_x)
        self.assertTrue(attn_ckpt_path.exists())

        # 4. Attempt to load the TemporalAttentionAutoencoder checkpoint into BiGRUTemporalAutoencoder
        # Must gracefully detect mismatch, not raise Missing key(s) or Unexpected key(s) RuntimeError,
        # and retrain BiGRUTemporalAutoencoder cleanly
        bigru_loading_trainer = TemporalAutoencoderTrainer(
            in_features=self.in_features,
            seq_len=self.seq_len,
            hidden_dim=16,
            num_layers=1,
            latent_dim=self.latent_dim,
            epochs=1,
            batch_size=8,
            checkpoint_path=attn_ckpt_path,
            verbose=False,
            force_retrain=False,
            manifold_strategy="unsupervised",
        )
        # Should not raise RuntimeError
        bigru_loading_trainer.fit(dummy_x)
        self.assertIsInstance(bigru_loading_trainer.model, BiGRUTemporalAutoencoder)


if __name__ == "__main__":
    unittest.main(verbosity=2)

