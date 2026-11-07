"""Training utility functions for sparse autoencoders.

This module contains common utilities used across training scripts:
- Temporal delay application
- TopK sparse encoding (for inference/analysis)
- Random seed setting for reproducibility
"""

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for numpy, torch CPU, and torch CUDA (if available).

    Parameters
    ----------
    seed : int
        Random seed value.

    Example
    -------
    >>> set_seed(42)
    >>> np.random.randn(3)  # Reproducible
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_delays_np(X: np.ndarray, delays: list[int]) -> np.ndarray:
    """Apply temporal delays to features.

    Creates a delayed version of input features by shifting along the time axis.
    This is used to model the hemodynamic response function (HRF) in fMRI,
    where brain responses are delayed relative to stimuli.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input features (e.g., semantic embeddings per TR).
    delays : list of int
        Delay values in TRs (e.g., [1, 2, 3, 4, 5]).
        A delay of d means X[t] maps to response at time t+d.

    Returns
    -------
    X_delayed : ndarray, shape (n_samples, n_features * n_delays)
        Horizontally stacked delayed features.
        For delays [1, 2, 3], output is [X_delay1 | X_delay2 | X_delay3].

    Example
    -------
    >>> X = np.arange(20).reshape(10, 2).astype(float)
    >>> X_del = apply_delays_np(X, [1, 2])
    >>> X_del.shape
    (10, 4)
    >>> X_del[2, :2]  # delay=1: X[1]
    array([2., 3.])
    >>> X_del[2, 2:]  # delay=2: X[0]
    array([0., 1.])
    """
    delayed = []
    for d in delays:
        shifted = np.zeros_like(X)
        if d > 0:
            shifted[d:] = X[:-d]
        else:
            shifted = X.copy()
        delayed.append(shifted)
    return np.hstack(delayed)


def encode_topk(
    X: np.ndarray,
    encoder_weight: np.ndarray,
    k: int,
    min_activation: float = 0.0,
    b_pre: np.ndarray = None,
    encoder_bias: np.ndarray = None,
) -> np.ndarray:
    """Compute TopK sparse encoding (numpy, for inference/analysis).

    Applies ReLU followed by TopK sparsity, matching the training procedure.
    Used for computing activations on held-out data or for analysis.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input features (e.g., semantic embeddings).
    encoder_weight : ndarray, shape (n_atoms, n_features)
        Encoder weight matrix from trained model.
    k : int
        Number of atoms to keep active per sample.
    min_activation : float, optional
        Minimum activation threshold. Activations below this are set to zero,
        allowing fewer than k active atoms. Default 0.0 (no threshold).
    b_pre : ndarray, shape (n_features,), optional
        Pre-encoder bias, subtracted before the linear map. Pass the model's
        ``b_pre`` to reproduce its codes.
    encoder_bias : ndarray, shape (n_atoms,), optional
        Per-component encoder bias, added after the linear map. Pass the
        model's ``encoder_bias`` to reproduce its codes.

    Returns
    -------
    Z : ndarray, shape (n_samples, n_atoms)
        Sparse activation matrix with at most k non-zeros per row.

    Example
    -------
    >>> X = np.random.randn(100, 300)
    >>> W = np.random.randn(50, 300)
    >>> Z = encode_topk(X, W, k=5)
    >>> Z.shape
    (100, 50)
    >>> (Z != 0).sum(axis=1).max()  # At most k active
    5
    """
    # Linear projection + ReLU:  z = TopK(ReLU(W (x - b_pre) + b_enc))
    if b_pre is not None:
        X = X - b_pre
    Z = X @ encoder_weight.T  # (n_samples, n_atoms)
    if encoder_bias is not None:
        Z = Z + encoder_bias
    Z = np.maximum(Z, 0)  # ReLU

    # TopK selection
    n_samples, n_atoms = Z.shape
    top_idx = np.argpartition(Z, -k, axis=1)[:, -k:]
    mask = np.zeros_like(Z)
    np.put_along_axis(mask, top_idx, 1.0, axis=1)
    Z = Z * mask

    # Optional minimum activation threshold
    if min_activation > 0:
        Z = Z * (Z > min_activation)

    return Z
