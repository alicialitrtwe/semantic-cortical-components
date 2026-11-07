#!/usr/bin/env python
"""Tests for training utility functions.

Tests cover:
- set_seed: Reproducibility of random seeds
- apply_delays_np: Temporal delay application
- encode_topk: TopK sparse encoding
"""
import numpy as np
import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparsebrain.training import apply_delays_np, encode_topk, set_seed


# =============================================================================
# Tests for set_seed
# =============================================================================

class TestSetSeed:
    """Tests for set_seed function."""

    def test_numpy_reproducibility(self):
        """Test that numpy random is reproducible after set_seed."""
        set_seed(42)
        a = np.random.randn(10)

        set_seed(42)
        b = np.random.randn(10)

        assert np.allclose(a, b)

    def test_torch_reproducibility(self):
        """Test that torch random is reproducible after set_seed."""
        set_seed(42)
        a = torch.randn(10)

        set_seed(42)
        b = torch.randn(10)

        assert torch.allclose(a, b)

    def test_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        a = np.random.randn(10)

        set_seed(123)
        b = np.random.randn(10)

        assert not np.allclose(a, b)


# =============================================================================
# Tests for apply_delays_np
# =============================================================================

class TestApplyDelaysNp:
    """Tests for apply_delays_np function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        X = np.random.randn(100, 50).astype(np.float32)
        delays = [1, 2, 3, 4]

        result = apply_delays_np(X, delays)

        assert result.shape == (100, 50 * len(delays))

    def test_single_delay(self):
        """Test single delay produces correct shift."""
        X = np.arange(20).reshape(10, 2).astype(float)

        result = apply_delays_np(X, [2])

        # Delay of 2: result[t] should be X[t-2]
        # So result[2] should be X[0]
        assert np.allclose(result[2], X[0])
        # result[3] should be X[1]
        assert np.allclose(result[3], X[1])
        # First 2 rows should be zeros
        assert np.allclose(result[:2], 0)

    def test_multiple_delays(self):
        """Test multiple delays are stacked horizontally."""
        X = np.arange(20).reshape(10, 2).astype(float)

        result = apply_delays_np(X, [1, 2])

        # First half: delay=1
        # Second half: delay=2
        assert result.shape == (10, 4)

        # result[2, :2] should be X[1] (delay=1)
        assert np.allclose(result[2, :2], X[1])
        # result[2, 2:] should be X[0] (delay=2)
        assert np.allclose(result[2, 2:], X[0])

    def test_zero_delay(self):
        """Test that zero delay returns original."""
        X = np.random.randn(10, 5).astype(np.float32)

        result = apply_delays_np(X, [0])

        assert np.allclose(result, X)

    def test_delay_preserves_dtype(self):
        """Test that float32 is preserved."""
        X = np.random.randn(10, 5).astype(np.float32)

        result = apply_delays_np(X, [1, 2, 3])

        assert result.dtype == np.float32

    def test_delay_order_matches_input_order(self):
        """Test that delays are stacked in input order."""
        X = np.arange(10).reshape(5, 2).astype(float)

        # Delays [3, 1] should stack delay=3 first, then delay=1
        result = apply_delays_np(X, [3, 1])

        # At t=3: delay=3 gives X[0], delay=1 gives X[2]
        assert np.allclose(result[3, :2], X[0])  # delay=3
        assert np.allclose(result[3, 2:], X[2])  # delay=1


# =============================================================================
# Tests for encode_topk
# =============================================================================

class TestEncodeTopk:
    """Tests for encode_topk function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        X = np.random.randn(100, 300).astype(np.float32)
        W = np.random.randn(50, 300).astype(np.float32)

        Z = encode_topk(X, W, k=5)

        assert Z.shape == (100, 50)

    def test_at_most_k_nonzeros(self):
        """Test that each row has at most k non-zeros."""
        X = np.random.randn(100, 300).astype(np.float32)
        W = np.random.randn(50, 300).astype(np.float32)
        k = 5

        Z = encode_topk(X, W, k=k)

        nnz_per_sample = (Z != 0).sum(axis=1)
        assert np.all(nnz_per_sample <= k)

    def test_output_nonnegative(self):
        """Test that output is non-negative (ReLU applied)."""
        X = np.random.randn(100, 300).astype(np.float32)
        W = np.random.randn(50, 300).astype(np.float32)

        Z = encode_topk(X, W, k=5)

        assert np.all(Z >= 0)

    def test_top_values_preserved(self):
        """Test that top k values are the largest activations."""
        # Create simple test case
        X = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        # Encoder that produces known activations
        W = np.array([
            [1.0, 0.0, 0.0],  # atom 0: activation = 1
            [2.0, 0.0, 0.0],  # atom 1: activation = 2
            [3.0, 0.0, 0.0],  # atom 2: activation = 3
            [4.0, 0.0, 0.0],  # atom 3: activation = 4
            [5.0, 0.0, 0.0],  # atom 4: activation = 5
        ], dtype=np.float32)

        Z = encode_topk(X, W, k=2)

        # Top 2 should be atoms 3 and 4 (activations 4 and 5)
        assert Z[0, 0] == 0  # atom 0 not in top 2
        assert Z[0, 1] == 0  # atom 1 not in top 2
        assert Z[0, 2] == 0  # atom 2 not in top 2
        assert Z[0, 3] == 4  # atom 3 in top 2
        assert Z[0, 4] == 5  # atom 4 in top 2

    def test_min_activation_threshold(self):
        """Test that min_activation filters out small values."""
        X = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        W = np.array([
            [0.5, 0.0, 0.0],  # atom 0: activation = 0.5
            [1.0, 0.0, 0.0],  # atom 1: activation = 1.0
            [2.0, 0.0, 0.0],  # atom 2: activation = 2.0
        ], dtype=np.float32)

        # Without threshold
        Z_no_thresh = encode_topk(X, W, k=3, min_activation=0.0)
        assert Z_no_thresh[0, 0] == 0.5

        # With threshold > 0.5
        Z_thresh = encode_topk(X, W, k=3, min_activation=0.6)
        assert Z_thresh[0, 0] == 0  # Filtered out
        assert Z_thresh[0, 1] == 1.0  # Kept
        assert Z_thresh[0, 2] == 2.0  # Kept

    def test_min_activation_can_reduce_sparsity(self):
        """Test that min_activation can result in fewer than k active atoms."""
        X = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        W = np.array([
            [0.1, 0.0, 0.0],  # activation = 0.1
            [0.2, 0.0, 0.0],  # activation = 0.2
            [0.3, 0.0, 0.0],  # activation = 0.3
            [1.0, 0.0, 0.0],  # activation = 1.0
            [2.0, 0.0, 0.0],  # activation = 2.0
        ], dtype=np.float32)

        Z = encode_topk(X, W, k=4, min_activation=0.5)

        # Only 2 atoms above threshold, so only 2 active
        nnz = (Z[0] != 0).sum()
        assert nnz == 2

    def test_negative_projections_zeroed(self):
        """Test that negative projections are zeroed by ReLU."""
        X = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        W = np.array([
            [-1.0, 0.0, 0.0],  # atom 0: projection = -1 (ReLU -> 0)
            [1.0, 0.0, 0.0],   # atom 1: projection = 1
        ], dtype=np.float32)

        Z = encode_topk(X, W, k=2)

        assert Z[0, 0] == 0  # Negative projection zeroed
        assert Z[0, 1] == 1  # Positive projection kept

    def test_dtype_preserved(self):
        """Test that float32 dtype is preserved."""
        X = np.random.randn(10, 50).astype(np.float32)
        W = np.random.randn(20, 50).astype(np.float32)

        Z = encode_topk(X, W, k=5)

        assert Z.dtype == np.float32


# =============================================================================
# Integration tests
# =============================================================================

class TestTrainingUtilsIntegration:
    """Integration tests for training utilities."""

    def test_delays_then_encode(self):
        """Test that delayed features can be encoded."""
        set_seed(42)

        # Simulate semantic features over time
        X = np.random.randn(100, 50).astype(np.float32)

        # Apply temporal delays
        X_delayed = apply_delays_np(X, [1, 2, 3, 4])
        assert X_delayed.shape == (100, 200)

        # Create encoder for delayed features
        W = np.random.randn(30, 200).astype(np.float32)

        # Encode
        Z = encode_topk(X_delayed, W, k=5)
        assert Z.shape == (100, 30)
        assert np.all((Z != 0).sum(axis=1) <= 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
