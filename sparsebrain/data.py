"""Data loading utilities for fMRI responses and semantic features.

This module provides functions for loading preprocessed fMRI data and
semantic features, with consistent preprocessing (demeaning, z-scoring).
"""
import os
import warnings
from typing import Optional

import h5py
import numpy as np

from .paths import paths


def demean(x: np.ndarray) -> np.ndarray:
    """Remove column-wise mean from array.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        Input array.

    Returns
    -------
    ndarray, shape (n_samples, n_features)
        Demeaned array with zero mean per column.
    """
    return x - x.mean(axis=0, keepdims=True)


def load_responses(stories: list[str], subject: str) -> np.ndarray:
    """Load preprocessed fMRI responses for given stories.

    Parameters
    ----------
    stories : list of str
        Story names to load (e.g., ['alternateithicatom', 'souls']).
    subject : str
        Subject ID (e.g., 'UTS03').

    Returns
    -------
    ndarray, shape (n_samples, n_voxels)
        Concatenated fMRI responses, dtype float32.

    Raises
    ------
    FileNotFoundError
        If story data file not found.
    """
    subject_dir = paths.subject_data_dir(subject)
    Ys = []
    for story in stories:
        filepath = os.path.join(subject_dir, f"{story}.hf5")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Story data not found: {filepath}")
        with h5py.File(filepath, "r") as hf:
            Ys.append(np.array(hf["data"], dtype=np.float32))
    return np.vstack(Ys)


def load_features(
    stories: list[str],
    embedding: str = "glove",
    subject: Optional[str] = None,
    demean_per_story: bool = True,
) -> np.ndarray:
    """Load pre-computed semantic features for given stories.

    Parameters
    ----------
    stories : list of str
        Story names to load.
    embedding : str
        Embedding type ('glove', 'eng1000', 'glove_factor').
    subject : str, optional
        Subject ID for subject-specific features.
    demean_per_story : bool
        Whether to demean features per story (default True).

    Returns
    -------
    ndarray, shape (n_samples, n_features)
        Concatenated features, dtype float32.

    Raises
    ------
    FileNotFoundError
        If features file not found.
    KeyError
        If story not in features file.
    """
    # Try subject-specific file first, then fall back to default
    feat_path = None
    if subject:
        candidate = paths.features_path(embedding, subject)
        if os.path.exists(candidate):
            feat_path = candidate

    if feat_path is None:
        feat_path = paths.features_path(embedding)

    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"Features file not found: {feat_path}\n"
            f"Run: uv run python scripts/02_save_features.py --embedding {embedding}"
        )

    feat_data = np.load(feat_path)
    Xs = []
    for story in stories:
        if story not in feat_data:
            available = list(feat_data.keys())[:5]
            raise KeyError(
                f"Story '{story}' not in features file. "
                f"Available: {available}..."
            )
        x = feat_data[story]
        if demean_per_story:
            x = demean(x)
        x = np.nan_to_num(x)
        Xs.append(x.astype(np.float32))

    return np.vstack(Xs)


def select_top_voxels(
    Y: np.ndarray, n_voxels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Select top voxels by variance.

    .. deprecated::
        This function selects voxels by variance, which is meaningless on
        z-scored data (all voxels have variance ~ 1). Use
        :func:`select_voxels_by_ridge` instead, which selects voxels based
        on ridge regression test performance.

    Parameters
    ----------
    Y : ndarray, shape (n_samples, n_voxels_total)
        Response matrix.
    n_voxels : int
        Number of voxels to select.

    Returns
    -------
    Y_selected : ndarray, shape (n_samples, n_voxels)
        Selected voxel responses.
    voxel_indices : ndarray, shape (n_voxels,)
        Sorted indices of selected voxels.
    """
    warnings.warn(
        "select_top_voxels selects by variance, which is meaningless on z-scored data. "
        "Use select_voxels_by_ridge() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    variances = np.var(Y, axis=0)
    voxel_indices = np.argsort(variances)[::-1][:n_voxels]
    voxel_indices = np.sort(voxel_indices)  # Keep spatial order
    return Y[:, voxel_indices], voxel_indices


def select_voxels_by_ridge(
    subject: str,
    n_voxels: int,
    embedding: str = "glove",
    ridge_n_stories: int = 10,
) -> np.ndarray:
    """Select top voxels by ridge regression test performance.

    Uses ridge models trained on a subset of stories and tested on
    'wheretheressmoke' to identify voxels that encode language well.

    This is the recommended voxel selection method for SAE training,
    as it identifies voxels that respond to semantic content rather
    than relying on variance (which is ~1 for z-scored data).

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'UTS03').
    n_voxels : int
        Number of voxels to select.
    embedding : str
        Embedding used for ridge model ('glove' or 'eng1000').
    ridge_n_stories : int
        Number of stories used for ridge training (default: 10).
        Using fewer stories is faster and sufficient for voxel selection.

    Returns
    -------
    voxel_indices : ndarray, shape (n_voxels,)
        Indices of top voxels sorted spatially (not by performance).

    Raises
    ------
    FileNotFoundError
        If the ridge model scores file is not found.

    Example
    -------
    >>> indices = select_voxels_by_ridge("UTS03", n_voxels=30000)
    >>> Y_train = load_responses(train_stories, "UTS03")
    >>> Y_train_selected = Y_train[:, indices]
    """
    # Import joblib here to avoid adding dependency if not using this function
    import joblib

    # Construct ridge model path
    if embedding == "glove":
        ridge_dir = f"ridge_{subject}_{ridge_n_stories}stories_glove"
    else:
        ridge_dir = f"ridge_{subject}_{ridge_n_stories}stories"

    ridge_path = os.path.join(paths.models_dir, ridge_dir, "scores.joblib")

    if not os.path.exists(ridge_path):
        raise FileNotFoundError(
            f"Ridge model not found: {ridge_path}\n"
            f"Run: uv run python scripts/03_fit_ridge.py --subject {subject} "
            f"--n_stories {ridge_n_stories} --embedding {embedding}"
        )

    ridge_data = joblib.load(ridge_path)
    corr = np.array(ridge_data["corr_test"])

    # Handle NaN values (set to -1 so they sort to bottom)
    corr = np.nan_to_num(corr, nan=-1.0)

    # Select top voxels by test correlation
    voxel_indices = np.argsort(corr)[::-1][:n_voxels]

    return np.sort(voxel_indices)  # Sort for consistent spatial indexing


def get_all_stories_for_subject(subject: str) -> list[str]:
    """Get all story names available for a subject.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'UTS03').

    Returns
    -------
    list of str
        Sorted list of story names (without .hf5 extension).
    """
    subject_dir = paths.subject_data_dir(subject)
    if not os.path.exists(subject_dir):
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    stories = []
    for f in sorted(os.listdir(subject_dir)):
        if f.endswith(".hf5"):
            stories.append(f.replace(".hf5", ""))
    return stories


def zscore_own_rows(Y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score each voxel over the rows where the subject has data; other rows become 0.

    The training response table is padded with zeros for stories a subject did not hear.
    Z-scoring over the padded table leaves the subject's real rows with variance above 1;
    this re-z-scores them over themselves (z-scoring is invariant to the earlier affine map).
    """
    from scipy.stats import zscore
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(Y, dtype=np.float32)
    out[mask] = np.nan_to_num(zscore(Y[mask], axis=0))
    return out
