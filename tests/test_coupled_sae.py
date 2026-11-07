#!/usr/bin/env python
"""Tests for coupled sparse autoencoder models.

Tests cover:
- CoupledSAEBase encoding behavior
- CoupledSAE single-subject model
- MultiSubjectCoupledSAE multi-subject model
- TV regularization
- Weight normalization
"""
import numpy as np
import pytest
import scipy.sparse as sp_sparse
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparsebrain.models.coupled_sae import (
    CoupledSAEBase,
    CoupledSAE,
    MultiSubjectCoupledSAE,
    sparse_scipy_to_torch,
)


# =============================================================================
# Tests for sparse_scipy_to_torch
# =============================================================================

class TestSparseSciPyToTorch:
    """Tests for sparse matrix conversion."""

    def test_converts_to_sparse_tensor(self):
        """Test that output is a sparse tensor."""
        E = sp_sparse.random(10, 20, density=0.3, format='csr')
        E_torch = sparse_scipy_to_torch(E)

        assert E_torch.is_sparse

    def test_preserves_shape(self):
        """Test that shape is preserved."""
        E = sp_sparse.random(15, 25, density=0.2, format='csr')
        E_torch = sparse_scipy_to_torch(E)

        assert E_torch.shape == (15, 25)

    def test_preserves_values(self):
        """Test that values are preserved."""
        data = np.array([1.0, 2.0, 3.0])
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2])
        E = sp_sparse.csr_matrix((data, (row, col)), shape=(3, 3))

        E_torch = sparse_scipy_to_torch(E)
        E_dense = E_torch.to_dense().numpy()

        assert np.allclose(E_dense, E.toarray())

    def test_device_placement(self):
        """Test that tensor is on correct device."""
        E = sp_sparse.random(5, 5, density=0.5, format='csr')
        E_torch = sparse_scipy_to_torch(E, device="cpu")

        assert E_torch.device.type == "cpu"


# =============================================================================
# Tests for CoupledSAE
# =============================================================================

class TestCoupledSAE:
    """Tests for single-subject CoupledSAE."""

    @pytest.fixture
    def model(self):
        """Create a basic CoupledSAE model."""
        return CoupledSAE(
            n_sem=300,
            n_voxels=1000,
            n_atoms=50,
            k=5,
            n_delays=4,
            use_delay_scaling=False,
        )

    @pytest.fixture
    def model_with_delays(self):
        """Create a CoupledSAE with delay scaling."""
        return CoupledSAE(
            n_sem=300,
            n_voxels=1000,
            n_atoms=50,
            k=5,
            n_delays=4,
            use_delay_scaling=True,
        )

    def test_encode_shape(self, model):
        """Test that encode produces correct output shape."""
        x = torch.randn(32, 300)
        z = model.encode(x)

        assert z.shape == (32, 50)

    def test_encode_sparse(self, model):
        """Test that encoded output is sparse."""
        x = torch.randn(32, 300)
        z = model.encode(x)

        # At most k=5 non-zeros per sample
        nnz_per_sample = (z != 0).sum(dim=1)
        assert torch.all(nnz_per_sample <= 5)

    def test_encode_nonnegative(self, model):
        """Test that encoded output is non-negative."""
        x = torch.randn(32, 300)
        z = model.encode(x)

        assert torch.all(z >= 0)

    def test_forward_returns_dict(self, model):
        """Test that forward returns expected keys."""
        x = torch.randn(32, 300)
        x_delayed = torch.randn(32, 300 * 4)

        output = model(x, x_delayed)

        assert 's' in output
        assert 's_delayed' in output
        assert 'x_hat' in output
        assert 'y_hat' in output

    def test_forward_shapes(self, model):
        """Test that forward produces correct shapes."""
        x = torch.randn(32, 300)
        x_delayed = torch.randn(32, 300 * 4)

        output = model(x, x_delayed)

        assert output['s'].shape == (32, 50)
        assert output['s_delayed'].shape == (32, 50 * 4)
        assert output['x_hat'].shape == (32, 300)
        assert output['y_hat'].shape == (32, 1000)

    def test_delay_scaling(self, model_with_delays):
        """Test that delay scales are applied."""
        assert model_with_delays.delay_scales is not None
        assert model_with_delays.delay_scales.shape == (4, 50)

    def test_normalize_weights_nonnegative(self, model):
        """Test that weights are non-negative after normalization."""
        # Set some weights negative
        model.decoder_brain.weight.data = torch.randn_like(
            model.decoder_brain.weight
        )

        model.normalize_weights(non_neg_brain=True)

        assert torch.all(model.decoder_brain.weight >= 0)

    def test_normalize_weights_encoder_unit_norm(self, model):
        """Test that encoder rows have unit norm after normalization."""
        model.normalize_weights()

        norms = model.encoder.weight.norm(dim=1)
        expected = torch.ones_like(norms)

        assert torch.allclose(norms, expected, atol=1e-6)

    def test_get_brain_dictionary_shape(self, model):
        """Test that brain dictionary has correct shape."""
        D = model.get_brain_dictionary()

        assert D.shape == (50, 1000)

    def test_get_semantic_dictionary_shape(self, model):
        """Test that semantic dictionary has correct shape."""
        D = model.get_semantic_dictionary()

        assert D.shape == (50, 300)

    def test_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        x = torch.randn(32, 300, requires_grad=True)
        x_delayed = torch.randn(32, 300 * 4, requires_grad=True)

        output = model(x, x_delayed)
        loss = output['y_hat'].sum() + output['x_hat'].sum()
        loss.backward()

        assert model.encoder.weight.grad is not None
        assert model.decoder_brain.weight.grad is not None
        assert model.decoder_sem.weight.grad is not None

    def test_decoder_sem_separate(self, model):
        """Test that decoder_sem is a separate parameter from encoder."""
        assert model.decoder_sem is not None
        assert model.decoder_sem.weight.shape == (300, 50)
        assert not torch.equal(model.encoder.weight, model.decoder_sem.weight.T)

    def test_b_pre_affects_reconstruction(self, model):
        """Test that b_pre shifts semantic reconstruction."""
        x = torch.randn(8, 300)
        x_delayed = torch.randn(8, 300 * 4)
        out1 = model(x, x_delayed)
        model.b_pre.data.fill_(1.0)
        out2 = model(x, x_delayed)
        assert not torch.allclose(out1['x_hat'], out2['x_hat'])

    def test_init_b_pre_from_data_mean(self, model):
        """Test b_pre initialization from mean of data."""
        torch.manual_seed(0)
        x = torch.randn(200, 300) + 2.5
        model.init_b_pre_from_data(x, method="mean")
        assert torch.allclose(model.b_pre.data, x.mean(dim=0), atol=1e-6)

    def test_init_b_pre_from_data_geometric_median(self, model):
        """Test b_pre initialization via geometric median (Weiszfeld)."""
        torch.manual_seed(0)
        x = torch.randn(200, 300) + 1.0
        model.init_b_pre_from_data(x, method="geometric_median", n_iter=50)
        # Geometric median of zero-centered Gaussian + shift should be near the shift
        assert (model.b_pre.data - 1.0).abs().mean() < 0.2

    def test_load_state_dict_compat_old_checkpoint(self):
        """Test that pre-refactor state_dicts load without error."""
        model = CoupledSAE(n_sem=50, n_voxels=100, n_atoms=10, k=3, n_delays=2)
        # Construct a fake old checkpoint (no decoder_sem, no b_pre)
        old_state = {
            "encoder.weight": torch.randn(10, 50),
            "decoder_brain.weight": torch.randn(100, 20),
        }
        result = model.load_state_dict_compat(old_state)
        # decoder_sem should have been populated from encoder.weight.T
        assert torch.allclose(
            model.decoder_sem.weight.data,
            old_state["encoder.weight"].T,
            atol=1e-6,
        )
        # b_pre should be zeros
        assert torch.all(model.b_pre.data == 0)

    # ------------------------------------------------------------------
    # Tests for gauge-fixing flags added after the refactor
    # ------------------------------------------------------------------

    def test_tie_semantic_decoder_no_separate_param(self):
        """tie_semantic_decoder=True removes the separate decoder_sem param."""
        model = CoupledSAE(n_sem=50, n_voxels=100, n_atoms=10, k=3, n_delays=2,
                           tie_semantic_decoder=True)
        assert model.decoder_sem is None
        # Forward pass must still work and produce the right shapes.
        x = torch.randn(4, 50)
        x_del = torch.randn(4, 50 * 2)
        out = model(x, x_del)
        assert out["x_hat"].shape == (4, 50)
        # get_semantic_dictionary falls back to encoder.weight when tied.
        D = model.get_semantic_dictionary()
        assert D.shape == (10, 50)

    def test_semantic_decoder_unit_norm_flag(self):
        """normalize_weights with semantic_decoder_unit_norm projects cols to unit norm."""
        model = CoupledSAE(n_sem=50, n_voxels=100, n_atoms=10, k=3, n_delays=2)
        # Put random non-unit values in decoder_sem.
        model.decoder_sem.weight.data = torch.randn_like(model.decoder_sem.weight)
        model.normalize_weights(
            non_neg_brain=False,
            encoder_unit_norm=False,
            brain_decoder_unit_norm=False,
            semantic_decoder_unit_norm=True,
        )
        col_norms = model.decoder_sem.weight.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-6)

    def test_semantic_decoder_unit_norm_noop_when_tied(self):
        """semantic_decoder_unit_norm is a no-op if decoder_sem is tied (None)."""
        model = CoupledSAE(n_sem=50, n_voxels=100, n_atoms=10, k=3, n_delays=2,
                           tie_semantic_decoder=True)
        # Should not raise.
        model.normalize_weights(semantic_decoder_unit_norm=True)

    def test_brain_decoder_unit_norm_flag(self):
        """brain_decoder_unit_norm projects brain decoder columns to unit norm."""
        model = CoupledSAE(n_sem=50, n_voxels=100, n_atoms=10, k=3, n_delays=2)
        model.decoder_brain.weight.data = torch.abs(
            torch.randn_like(model.decoder_brain.weight))  # keep positive for non_neg
        model.normalize_weights(
            non_neg_brain=False,
            encoder_unit_norm=False,
            brain_decoder_unit_norm=True,
        )
        col_norms = model.decoder_brain.weight.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-6)

    def test_encoder_unit_norm_can_be_disabled(self):
        """encoder_unit_norm=False leaves encoder row norms untouched."""
        model = CoupledSAE(n_sem=50, n_voxels=100, n_atoms=10, k=3, n_delays=2)
        # Scale the encoder weights so no row is unit norm.
        model.encoder.weight.data *= 10.0
        before = model.encoder.weight.norm(dim=1).clone()
        model.normalize_weights(
            non_neg_brain=False,
            encoder_unit_norm=False,
            brain_decoder_unit_norm=False,
        )
        after = model.encoder.weight.norm(dim=1)
        assert torch.allclose(before, after, atol=1e-6)

    def test_temporal_first_forward_shapes(self):
        """temporal_first=True sums delays → 50-dim brain decoder input."""
        model = CoupledSAE(n_sem=50, n_voxels=100, n_atoms=10, k=3, n_delays=4,
                           use_delay_scaling=True, temporal_first=True)
        x = torch.randn(8, 50)
        x_del = torch.randn(8, 50 * 4)
        out = model(x, x_del)
        assert out["s"].shape == (8, 10)
        assert out["s_delayed"].shape == (8, 10)  # summed, not (8, 40)
        assert out["x_hat"].shape == (8, 50)
        assert out["y_hat"].shape == (8, 100)
        D = model.get_brain_dictionary()
        assert D.shape == (10, 100)

    def test_brain_voxel_scale_shape_and_effect(self):
        """brain_voxel_scale=True adds a per-voxel parameter that multiplies y_hat."""
        model = CoupledSAE(n_sem=50, n_voxels=100, n_atoms=10, k=3, n_delays=2,
                           brain_voxel_scale=True)
        assert model.voxel_scale is not None
        assert model.voxel_scale.shape == (100,)
        x = torch.randn(4, 50)
        x_del = torch.randn(4, 50 * 2)
        out_default = model(x, x_del)  # voxel_scale init = 1.0
        # Forward at init should match a model without voxel_scale (up to init
        # noise) — instead, check that doubling voxel_scale doubles y_hat.
        model.voxel_scale.data.fill_(2.0)
        out_scaled = model(x, x_del)
        assert torch.allclose(out_scaled["y_hat"], 2.0 * out_default["y_hat"], atol=1e-5)

    def test_gauge_invariant_l1_is_gauge_invariant(self):
        """Rescaling W_b[:,i] <-> s[:,i] should leave the gauge_l1 penalty fixed."""
        model = CoupledSAE(n_sem=50, n_voxels=100, n_atoms=10, k=3, n_delays=2)
        s_del = torch.randn(8, 10 * 2).abs()  # positive activations
        W = model.decoder_brain.weight.data.clone()
        p_before = model.gauge_invariant_l1(s_del).item()
        # Rescale column 3 by c=5, compensating s_del[:, 3] by 1/5.
        c = 5.0
        model.decoder_brain.weight.data[:, 3] = W[:, 3] * c
        s_rescaled = s_del.clone()
        s_rescaled[:, 3] = s_rescaled[:, 3] / c
        p_after = model.gauge_invariant_l1(s_rescaled).item()
        assert abs(p_before - p_after) < 1e-4, f"{p_before} vs {p_after}"


class TestCoupledSAETV:
    """Tests for TV regularization in CoupledSAE."""

    @pytest.fixture
    def edge_incidence(self):
        """Create a simple edge incidence matrix."""
        # 3 edges for 4 voxels: (0,1), (1,2), (2,3)
        data = [1, -1, 1, -1, 1, -1]
        row = [0, 0, 1, 1, 2, 2]
        col = [0, 1, 1, 2, 2, 3]
        E = sp_sparse.csr_matrix((data, (row, col)), shape=(3, 4))
        return E

    def test_tv_loss_zero_without_init(self):
        """Test that TV loss is zero before init_tv_on_device."""
        model = CoupledSAE(
            n_sem=10, n_voxels=4, n_atoms=3, k=2, n_delays=2,
            tv_weight=1.0,
        )

        tv_loss = model.tv_loss()
        assert tv_loss == 0.0

    def test_tv_loss_zero_when_disabled(self):
        """Test that TV loss is zero when tv_weight=0."""
        model = CoupledSAE(
            n_sem=10, n_voxels=4, n_atoms=3, k=2, n_delays=2,
            tv_weight=0.0,
        )

        tv_loss = model.tv_loss()
        assert tv_loss == 0.0

    def test_tv_loss_nonzero_with_init(self, edge_incidence):
        """Test that TV loss is nonzero after init."""
        model = CoupledSAE(
            n_sem=10, n_voxels=4, n_atoms=3, k=2, n_delays=2,
            edge_incidence=edge_incidence,
            tv_weight=1.0,
        )
        model.init_tv_on_device("cpu")

        tv_loss = model.tv_loss()
        assert tv_loss > 0

    def test_tv_loss_scales_with_weight(self, edge_incidence):
        """Test that TV loss scales with tv_weight."""
        model1 = CoupledSAE(
            n_sem=10, n_voxels=4, n_atoms=3, k=2, n_delays=2,
            edge_incidence=edge_incidence,
            tv_weight=1.0,
        )
        model1.init_tv_on_device("cpu")

        model2 = CoupledSAE(
            n_sem=10, n_voxels=4, n_atoms=3, k=2, n_delays=2,
            edge_incidence=edge_incidence,
            tv_weight=2.0,
        )
        # Copy weights to make comparison fair
        model2.decoder_brain.weight.data = model1.decoder_brain.weight.data.clone()
        model2.init_tv_on_device("cpu")

        tv1 = model1.tv_loss()
        tv2 = model2.tv_loss()

        assert torch.isclose(tv2 / tv1, torch.tensor(2.0), rtol=0.01)


# =============================================================================
# Tests for MultiSubjectCoupledSAE
# =============================================================================

class TestMultiSubjectCoupledSAE:
    """Tests for multi-subject CoupledSAE."""

    @pytest.fixture
    def model(self):
        """Create a basic MultiSubjectCoupledSAE model."""
        return MultiSubjectCoupledSAE(
            n_sem=300,
            n_voxels_per_subject={"UTS01": 800, "UTS02": 900, "UTS03": 1000},
            n_atoms=50,
            k=5,
            n_delays=4,
            use_delay_scaling=True,
            subject_specific_delays=False,
        )

    @pytest.fixture
    def model_subject_delays(self):
        """Create a model with subject-specific delay scales."""
        return MultiSubjectCoupledSAE(
            n_sem=300,
            n_voxels_per_subject={"UTS01": 800, "UTS02": 900},
            n_atoms=50,
            k=5,
            n_delays=4,
            use_delay_scaling=True,
            subject_specific_delays=True,
        )

    def test_subjects_stored(self, model):
        """Test that subjects are stored correctly."""
        assert model.subjects == ["UTS01", "UTS02", "UTS03"]

    def test_separate_brain_decoders(self, model):
        """Test that each subject has its own decoder."""
        assert "UTS01" in model.brain_decoders
        assert "UTS02" in model.brain_decoders
        assert "UTS03" in model.brain_decoders

        # Different output sizes
        assert model.brain_decoders["UTS01"].out_features == 800
        assert model.brain_decoders["UTS02"].out_features == 900
        assert model.brain_decoders["UTS03"].out_features == 1000

    def test_encode_shared(self, model):
        """Test that encoding is shared across subjects."""
        x = torch.randn(32, 300)
        z = model.encode(x)

        assert z.shape == (32, 50)
        assert torch.all(z >= 0)
        assert torch.all((z != 0).sum(dim=1) <= 5)

    def test_forward_single_subject(self, model):
        """Test forward for a specific subject."""
        x = torch.randn(32, 300)
        x_delayed = torch.randn(32, 300 * 4)

        output = model(x, x_delayed, subject="UTS03")

        assert 's' in output
        assert 'x_hat' in output
        assert 'y_hat' in output
        assert output['y_hat'].shape == (32, 1000)

    def test_forward_without_subject(self, model):
        """Test forward without subject (no y_hat)."""
        x = torch.randn(32, 300)
        x_delayed = torch.randn(32, 300 * 4)

        output = model(x, x_delayed)

        assert 's' in output
        assert 'x_hat' in output
        assert 'y_hat' not in output

    def test_forward_all_subjects(self, model):
        """Test forward_all_subjects returns all predictions."""
        x = torch.randn(32, 300)
        x_delayed = torch.randn(32, 300 * 4)

        output = model.forward_all_subjects(x, x_delayed)

        assert 's' in output
        assert 'x_hat' in output
        assert 'y_hat' in output
        assert isinstance(output['y_hat'], dict)

        assert output['y_hat']['UTS01'].shape == (32, 800)
        assert output['y_hat']['UTS02'].shape == (32, 900)
        assert output['y_hat']['UTS03'].shape == (32, 1000)

    def test_shared_delay_scales(self, model):
        """Test that delay scales are shared."""
        assert isinstance(model.delay_scales, nn.Parameter)
        assert model.delay_scales.shape == (4, 50)

    def test_subject_specific_delay_scales(self, model_subject_delays):
        """Test subject-specific delay scales."""
        assert isinstance(model_subject_delays.delay_scales, nn.ParameterDict)
        assert "UTS01" in model_subject_delays.delay_scales
        assert "UTS02" in model_subject_delays.delay_scales

    def test_normalize_weights_all_subjects(self, model):
        """Test that normalization applies encoder rows and brain decoder non-neg."""
        # Set some brain decoder weights negative
        for decoder in model.brain_decoders.values():
            decoder.weight.data = torch.randn_like(decoder.weight)

        model.normalize_weights(non_neg_brain=True)

        # Encoder rows should be unit norm
        norms = model.encoder.weight.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
        # Brain decoders should be non-negative
        for decoder in model.brain_decoders.values():
            assert torch.all(decoder.weight >= 0)

    def test_get_brain_dictionary_per_subject(self, model):
        """Test that brain dictionary is subject-specific."""
        D1 = model.get_brain_dictionary(subject="UTS01")
        D3 = model.get_brain_dictionary(subject="UTS03")

        assert D1.shape == (50, 800)
        assert D3.shape == (50, 1000)

    def test_get_semantic_dictionary_shared(self, model):
        """Test that semantic dictionary is shared."""
        D = model.get_semantic_dictionary()
        assert D.shape == (50, 300)

    def test_tv_loss_zero_by_default(self, model):
        """Test that TV loss is zero when not initialized."""
        tv = model.tv_loss()
        assert tv.item() == 0.0

    def test_tv_loss_nonzero_after_init(self):
        """Test that TV loss is nonzero after initializing edge incidence."""
        model = MultiSubjectCoupledSAE(
            n_sem=50,
            n_voxels_per_subject={"S1": 10, "S2": 10},
            n_atoms=5, k=2, n_delays=2,
        )
        # Build simple edge incidence: 4 edges for 10 voxels (chain)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        rows, cols, data = [], [], []
        for e, (v1, v2) in enumerate(edges):
            rows.extend([e, e])
            cols.extend([v1, v2])
            data.extend([1.0, -1.0])
        E = sp_sparse.csr_matrix((data, (rows, cols)), shape=(len(edges), 10))
        model.init_tv_on_device({"S1": E, "S2": E}, tv_weight=1.0, device="cpu")
        tv = model.tv_loss()
        assert tv.item() > 0

    def test_tv_loss_scales_with_weight(self):
        """Test that TV loss scales linearly with tv_weight."""
        model = MultiSubjectCoupledSAE(
            n_sem=50,
            n_voxels_per_subject={"S1": 10},
            n_atoms=5, k=2, n_delays=2,
        )
        edges = [(0, 1), (1, 2)]
        rows, cols, data = [], [], []
        for e, (v1, v2) in enumerate(edges):
            rows.extend([e, e])
            cols.extend([v1, v2])
            data.extend([1.0, -1.0])
        E = sp_sparse.csr_matrix((data, (rows, cols)), shape=(len(edges), 10))

        model.init_tv_on_device({"S1": E}, tv_weight=1.0, device="cpu")
        tv1 = model.tv_loss()

        model.init_tv_on_device({"S1": E}, tv_weight=2.0, device="cpu")
        tv2 = model.tv_loss()

        assert torch.isclose(tv2 / tv1, torch.tensor(2.0), rtol=0.01)

    def test_tv_loss_gradient_flows(self):
        """Test that TV loss gradients flow to brain decoder."""
        model = MultiSubjectCoupledSAE(
            n_sem=50,
            n_voxels_per_subject={"S1": 10},
            n_atoms=5, k=2, n_delays=2,
        )
        edges = [(0, 1), (1, 2)]
        rows, cols, data = [], [], []
        for e, (v1, v2) in enumerate(edges):
            rows.extend([e, e])
            cols.extend([v1, v2])
            data.extend([1.0, -1.0])
        E = sp_sparse.csr_matrix((data, (rows, cols)), shape=(len(edges), 10))
        model.init_tv_on_device({"S1": E}, tv_weight=1.0, device="cpu")

        tv = model.tv_loss()
        tv.backward()
        assert model.brain_decoders["S1"].weight.grad is not None

    def test_gradient_flow_all_subjects(self, model):
        """Test gradient flow through all decoders."""
        x = torch.randn(32, 300, requires_grad=True)
        x_delayed = torch.randn(32, 300 * 4, requires_grad=True)

        output = model.forward_all_subjects(x, x_delayed)
        loss = sum(y.sum() for y in output['y_hat'].values()) + output['x_hat'].sum()
        loss.backward()

        assert model.encoder.weight.grad is not None
        assert model.decoder_sem.weight.grad is not None
        for subj, decoder in model.brain_decoders.items():
            assert decoder.weight.grad is not None, f"No gradient for {subj}"


# =============================================================================
# Integration tests
# =============================================================================

class TestCoupledSAEIntegration:
    """Integration tests for training loop behavior."""

    def test_training_reduces_loss(self):
        """Test that training reduces reconstruction loss."""
        model = CoupledSAE(
            n_sem=50, n_voxels=100, n_atoms=10, k=3, n_delays=2,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Generate data
        X = torch.randn(100, 50)
        X_delayed = torch.randn(100, 50 * 2)
        Y = torch.randn(100, 100)

        # Initial loss
        output = model(X, X_delayed)
        initial_loss = torch.nn.functional.mse_loss(output['y_hat'], Y).item()

        # Train
        for _ in range(50):
            optimizer.zero_grad()
            output = model(X, X_delayed)
            loss = torch.nn.functional.mse_loss(output['y_hat'], Y)
            loss.backward()
            optimizer.step()
            model.normalize_weights()

        # Final loss
        output = model(X, X_delayed)
        final_loss = torch.nn.functional.mse_loss(output['y_hat'], Y).item()

        assert final_loss < initial_loss

    def test_multisubject_joint_training(self):
        """Test that multi-subject model can train jointly."""
        model = MultiSubjectCoupledSAE(
            n_sem=50,
            n_voxels_per_subject={"S1": 80, "S2": 100},
            n_atoms=10,
            k=3,
            n_delays=2,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Generate data
        X = torch.randn(100, 50)
        X_delayed = torch.randn(100, 50 * 2)
        Y = {"S1": torch.randn(100, 80), "S2": torch.randn(100, 100)}

        # Initial loss
        output = model.forward_all_subjects(X, X_delayed)
        initial_loss = sum(
            torch.nn.functional.mse_loss(output['y_hat'][s], Y[s])
            for s in Y
        ).item()

        # Train
        for _ in range(50):
            optimizer.zero_grad()
            output = model.forward_all_subjects(X, X_delayed)
            loss = sum(
                torch.nn.functional.mse_loss(output['y_hat'][s], Y[s])
                for s in Y
            )
            loss.backward()
            optimizer.step()
            model.normalize_weights()

        # Final loss
        output = model.forward_all_subjects(X, X_delayed)
        final_loss = sum(
            torch.nn.functional.mse_loss(output['y_hat'][s], Y[s])
            for s in Y
        ).item()

        assert final_loss < initial_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
