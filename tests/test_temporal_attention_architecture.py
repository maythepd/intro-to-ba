"""
Comprehensive Unit Tests for Temporal Attention Deep Learning Architecture:
1. Pillar 1: Feature Attention Layer & Temporal Attention Pooling
2. Pillar 2: Multi-Scale 1D Convolutional Block & Residual Skip
3. Pillar 3: Scheduled Gaussian Noise & Regularization Ops
4. Pillar 4: Huber Directional Correlation Hybrid Loss
5. End-to-End TemporalAttentionAutoencoder Forward & Backward Passes
6. Numerical Diagnostics: Vanishing & Exploding Gradient Checks
7. Outlier & Representation Bound Diagnostics
8. Seamless Toggle & Strategy Integration Verification
"""

import math
import shutil
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from preprocessing.temporal.modules import (
    FeatureAttentionLayer,
    TemporalAttentionPooling,
    MultiScaleTemporalConvBlock,
    ScheduledGaussianNoise,
    MishActivation,
    HuberDirectionalCorrelationLoss,
    TemporalAttentionAutoencoder,
)
from preprocessing.temporal.strategies import get_temporal_strategy, TemporalAttentionStrategy
from preprocessing.temporal.model import TemporalAutoencoderTrainer

TEMP_TEST_DIR = Path("results/test_dl_attention_temp")


class TestTemporalAttentionArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TEMP_TEST_DIR.exists():
            shutil.rmtree(TEMP_TEST_DIR, ignore_errors=True)
        TEMP_TEST_DIR.mkdir(parents=True, exist_ok=True)

        cls.batch_size = 16
        cls.seq_len = 6
        cls.in_features = 37
        cls.hidden_dim = 32
        cls.latent_dim = 16

        torch.manual_seed(42)
        np.random.seed(42)

        cls.dummy_input = torch.randn(cls.batch_size, cls.seq_len, cls.in_features)
        cls.dummy_recon = torch.randn(cls.batch_size, cls.seq_len, cls.in_features)

    @classmethod
    def tearDownClass(cls):
        if TEMP_TEST_DIR.exists():
            shutil.rmtree(TEMP_TEST_DIR, ignore_errors=True)
            print(f"\n[OK] Cleaned up temporary test artifacts at: {TEMP_TEST_DIR}")

    def test_01_feature_attention_layer(self):
        """Test Pillar 1: Feature Attention dynamically gates D=37 channels with valid soft weights."""
        feat_attn = FeatureAttentionLayer(in_features=self.in_features, reduction_ratio=2)
        gated_x, gate_weights = feat_attn(self.dummy_input)

        # Output dimensions
        self.assertEqual(gated_x.shape, (self.batch_size, self.seq_len, self.in_features))
        self.assertEqual(gate_weights.shape, (self.batch_size, self.seq_len, self.in_features))

        # Soft gating range: sigmoid output strictly in (0, 1)
        self.assertTrue((gate_weights > 0.0).all().item())
        self.assertTrue((gate_weights < 1.0).all().item())

        # No NaNs or Infs
        self.assertFalse(torch.isnan(gated_x).any().item())
        self.assertFalse(torch.isinf(gated_x).any().item())
        print("[PASS] Test 1: FeatureAttentionLayer verified (valid shape, soft gating in (0, 1), zero NaNs).")

    def test_02_temporal_attention_pooling(self):
        """Test Pillar 1: Temporal Attention learns valid probability distribution and context vector."""
        h_seq = torch.randn(self.batch_size, self.seq_len, self.hidden_dim * 2)
        temp_attn = TemporalAttentionPooling(hidden_dim=self.hidden_dim * 2)
        context, weights, entropy = temp_attn(h_seq)

        # Dimensions
        self.assertEqual(context.shape, (self.batch_size, self.hidden_dim * 2))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len))
        self.assertEqual(entropy.shape, (self.batch_size,))

        # Probabilities sum to 1.0 across time steps
        sums = torch.sum(weights, dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

        # Entropy is non-negative and finite: 0 <= H <= ln(T)
        max_possible_entropy = math.log(self.seq_len)
        self.assertTrue((entropy >= 0.0).all().item())
        self.assertTrue((entropy <= max_possible_entropy + 1e-4).all().item())
        print("[PASS] Test 2: TemporalAttentionPooling verified (weights sum to 1.0, entropy bounded).")

    def test_03_multiscale_conv_block(self):
        """Test Pillar 2: Multi-Scale 1D Convolutions with kernels k=2,3 and learnable skip connection."""
        conv_block = MultiScaleTemporalConvBlock(
            in_features=self.in_features,
            out_channels=self.hidden_dim,
            kernel_sizes=[2, 3],
        )
        conv_out = conv_block(self.dummy_input)

        # Dimension match
        self.assertEqual(conv_out.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        self.assertFalse(torch.isnan(conv_out).any().item())

        # Learnable skip scale parameter
        self.assertIsInstance(conv_block.alpha_skip, nn.Parameter)
        self.assertEqual(conv_block.alpha_skip.numel(), 1)
        print("[PASS] Test 3: MultiScaleTemporalConvBlock verified (receptive fields k=2,3, valid output).")

    def test_04_scheduled_gaussian_noise(self):
        """Test Pillar 3: Scheduled Gaussian noise injects decaying noise during training, inactive in eval."""
        noise_module = ScheduledGaussianNoise(initial_std=0.05, decay_rate=0.90, min_std=0.001)

        # In train mode: noise injected (output differs from input)
        noise_module.train()
        noisy = noise_module(self.dummy_input)
        self.assertFalse(torch.equal(noisy, self.dummy_input))

        # In eval mode: no noise (output identical to input)
        noise_module.eval()
        clean = noise_module(self.dummy_input)
        self.assertTrue(torch.equal(clean, self.dummy_input))

        # Schedule decay
        prev_std = noise_module.current_std
        new_std = noise_module.step_epoch()
        self.assertAlmostEqual(new_std, prev_std * 0.90, places=4)
        print("[PASS] Test 4: ScheduledGaussianNoise verified (active in train, inactive in eval, smooth decay).")

    def test_05_huber_directional_correlation_loss(self):
        """Test Pillar 4: Hybrid trend-aware loss computes Huber, Direction, and Correlation penalties."""
        criterion = HuberDirectionalCorrelationLoss(
            delta=1.0,
            lambda_direction=0.10,
            lambda_correlation=0.05,
        )
        loss, metrics = criterion(self.dummy_recon, self.dummy_input)

        # Loss must be a scalar positive tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss).item())
        self.assertFalse(torch.isinf(loss).item())
        self.assertGreater(loss.item(), 0.0)

        # Verify all metric components exist and are positive
        for k in ["huber_loss", "direction_loss", "correlation_loss", "recon_mse", "total_loss"]:
            self.assertIn(k, metrics)
            self.assertGreaterEqual(metrics[k], 0.0)

        print("[PASS] Test 5: HuberDirectionalCorrelationLoss verified (all 3 loss components well-defined).")

    def test_06_temporal_attention_autoencoder_e2e(self):
        """Test 6: Full TemporalAttentionAutoencoder forward, encode, endpoints and geometry pass."""
        model = TemporalAttentionAutoencoder(
            in_features=self.in_features,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            latent_dim=self.latent_dim,
            dropout=0.10,
            conv_kernels=[2, 3],
            noise_std=0.02,
            use_feature_attention=True,
            use_multiscale_conv=True,
        )

        # 1. Forward pass
        recon, z = model(self.dummy_input)
        self.assertEqual(recon.shape, (self.batch_size, self.seq_len, self.in_features))
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))
        self.assertFalse(torch.isnan(recon).any().item())
        self.assertFalse(torch.isnan(z).any().item())

        # 2. Encode endpoints
        z_enc, z_t1, z_t6 = model.encode_endpoints(self.dummy_input)
        self.assertEqual(z_enc.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(z_t1.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(z_t6.shape, (self.batch_size, self.latent_dim))

        # 3. Encode with geometry
        z_geom, geom_dict = model.encode_with_geometry(self.dummy_input)
        self.assertIn("temporal_drift_velocity", geom_dict)
        self.assertIn("temporal_attention_entropy", geom_dict)
        self.assertIn("temporal_feature_gate_intensity", geom_dict)
        print("[PASS] Test 6: TemporalAttentionAutoencoder e2e execution verified (recon, z, endpoints, geom).")

    def test_07_numerical_gradient_diagnostics(self):
        """Test 7: Backward pass gradient checks with both dynamic and zero-variance constant channels (real data simulation)."""
        model = TemporalAttentionAutoencoder(
            in_features=self.in_features,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            latent_dim=self.latent_dim,
        )
        criterion = HuberDirectionalCorrelationLoss(delta=1.0, lambda_direction=0.10, lambda_correlation=0.05)

        # Simulate real data: set several channels to constant zeros (e.g. inactive accounts, zero complaints)
        input_with_constants = self.dummy_input.clone()
        input_with_constants[:, :, :5] = 0.0  # Constant zero channels across all 6 timesteps
        input_with_constants[:10, :, 5:10] = 1.5  # Constant positive channels for subset of customers

        recon, z = model(input_with_constants)
        loss, _ = criterion(recon, input_with_constants)

        loss.backward()

        grad_norms = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has None gradient!")
                self.assertFalse(torch.isnan(param.grad).any().item(), f"NaN gradient in {name}!")
                self.assertFalse(torch.isinf(param.grad).any().item(), f"Inf gradient in {name}!")

                norm = torch.norm(param.grad, p=2).item()
                grad_norms.append((name, norm))
                # Exploding gradient check: norm must not exceed 20.0
                self.assertLess(norm, 20.0, f"Exploding gradient in {name}: norm={norm:.4f}")

        # Check that gradients actively flowed to early layers (no total vanishing gradient)
        non_zero_grads = [norm for _, norm in grad_norms if norm > 1e-7]
        self.assertGreaterEqual(
            len(non_zero_grads) / len(grad_norms),
            0.90,
            "More than 10% of layers experienced vanishing gradients (< 1e-7)!",
        )

        max_grad = max(n for _, n in grad_norms)
        min_grad = min(n for _, n in grad_norms)
        print(f"[PASS] Test 7: Gradient diagnostics passed! (Max grad norm={max_grad:.4f}, Min non-zero={min_grad:.7f}, Zero NaNs/Infs).")

    def test_08_latent_representation_bounds_and_outliers(self):
        """Test 8: Latent representation diagnostic — bounds, variance, and zero outlier collapse."""
        model = TemporalAttentionAutoencoder(
            in_features=self.in_features,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )
        model.eval()

        with torch.no_grad():
            z = model.encode(self.dummy_input)

        z_np = z.cpu().numpy()

        # Check bounds: Mish and LayerNorm should produce bounded, well-scaled coordinates
        self.assertTrue((np.abs(z_np) < 15.0).all(), "Latent z values exceeded safe numeric threshold (+-15.0)!")

        # Variance check: representations must not collapse to a single constant point
        variances = np.var(z_np, axis=0)
        self.assertTrue((variances > 1e-5).all(), "Latent dimension collapsed to zero variance!")

        print(f"[PASS] Test 8: Representation bounds verified (mean norm={np.mean(np.linalg.norm(z_np, axis=1)):.2f}, all 16 dims active).")

    def test_09_trainer_and_strategy_toggle(self):
        """Test 9: Verify TemporalAutoencoderTrainer trains and toggles between unsupervised and temporal_attention."""
        ckpt_path = TEMP_TEST_DIR / "test_attention_ckpt.pt"

        # 1. Train temporal_attention strategy
        trainer_attn = TemporalAutoencoderTrainer(
            in_features=self.in_features,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            latent_dim=self.latent_dim,
            epochs=2,
            batch_size=8,
            checkpoint_path=ckpt_path,
            verbose=False,
            force_retrain=True,
            manifold_strategy="temporal_attention",
        )
        tensor_np = self.dummy_input.numpy()
        trainer_attn.fit(tensor_np)

        latents_attn, geom_attn = trainer_attn.transform_with_geometry(tensor_np)
        self.assertEqual(latents_attn.shape, (self.batch_size, self.latent_dim))
        self.assertIn("temporal_drift_velocity", geom_attn)

        # 2. Toggle to baseline unsupervised
        trainer_unsup = TemporalAutoencoderTrainer(
            in_features=self.in_features,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            latent_dim=self.latent_dim,
            epochs=2,
            batch_size=8,
            checkpoint_path=None,
            verbose=False,
            force_retrain=True,
            manifold_strategy="unsupervised",
        )
        trainer_unsup.fit(tensor_np)
        latents_unsup, geom_unsup = trainer_unsup.transform_with_geometry(tensor_np)
        self.assertEqual(latents_unsup.shape, (self.batch_size, self.latent_dim))
        self.assertIn("temporal_drift_velocity", geom_unsup)

        print("[PASS] Test 9: Seamless strategy toggle verified (temporal_attention <-> unsupervised without conflict).")


if __name__ == "__main__":
    unittest.main()
