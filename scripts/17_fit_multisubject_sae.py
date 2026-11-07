#!/usr/bin/env python
"""17_fit_multisubject_sae.py — Train coupled SAE with shared encoder across multiple subjects.

Architecture:
    X (semantic features) -> shared encoder -> S (sparse, non-neg, top-k)
                                                    |
                    +-------------------------------+-------------------------------+
                    |                               |                               |
                    v                               v                               v
              decoder_sem (tied)           decoder_brain_UTS01           decoder_brain_UTS02 ...
                    |                               |                               |
                    v                               v                               v
                  X_hat                         Y_hat_UTS01                   Y_hat_UTS02 ...

Key insight: The semantic dictionary is universal (language is shared), but each
subject has their own brain anatomy. Shared sparse codes force discovery of
semantic features that explain brain responses across all subjects.

Benefits:
- More robust atoms (must generalize across subjects)
- Automatic cross-subject validation
- Each subject contributes to learning shared semantic structure

Usage:
    uv run python scripts/17_fit_multisubject_sae.py --subjects UTS01,UTS02,UTS03 --n_atoms 50 --k 5
    uv run python scripts/17_fit_multisubject_sae.py --subjects UTS01,UTS02,UTS03 --all_stories --n_atoms 50 --k 5
"""
import hashlib
import json
import os
import sys
import time
from os.path import join

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import zscore as sp_zscore

# Import from sparsebrain package
from sparsebrain import (
    paths,
    load_responses,
    load_features,
    select_top_voxels,
    select_voxels_by_ridge,
    get_all_stories_for_subject,
    STORIES_TRAIN,
    STORIES_TEST,
)
from sparsebrain.models import MultiSubjectCoupledSAE
from sparsebrain.training import apply_delays_np, set_seed


@torch.no_grad()
def project_decoder_gradients(model):
    """Remove radial gradient component from unit-norm brain decoder columns.

    Gao 2024 / Bricken 2023: project Adam's gradient onto the tangent plane
    of the unit sphere so the optimizer doesn't waste updates fighting the
    post-step unit-norm projection. Call AFTER loss.backward(), BEFORE
    optimizer.step().
    """
    decoders = (model.brain_decoders.values()
                if hasattr(model, "brain_decoders")
                else [model.decoder_brain])
    for dec in decoders:
        W = dec.weight
        if W.grad is None:
            continue
        dots = (W.grad * W).sum(dim=0, keepdim=True)
        W.grad.sub_(dots * W)


class MultiSubjectDataset(torch.utils.data.Dataset):
    """Dataset that yields (X, X_delayed, Y) tuples for all subjects.

    If device is specified, tensors are pre-loaded to GPU for faster
    training (avoids CPU→GPU copies each batch).
    """

    def __init__(
        self,
        X: np.ndarray,
        X_delayed: np.ndarray,
        Y_per_subject: dict[str, np.ndarray],
        device: str = "cpu",
        subject_mask: np.ndarray = None,
        need_y: bool = True,
    ):
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.X_delayed = torch.tensor(X_delayed, dtype=torch.float32, device=device)
        # need_y=False (semantic_only): the brain targets are never read by the
        # loss, so skip the per-batch copy of six (batch, n_voxels) blocks.
        self.Y = {
            subj: torch.tensor(Y, dtype=torch.float32, device=device)
            for subj, Y in Y_per_subject.items()
        } if need_y else {}
        self.subjects = list(Y_per_subject.keys())
        self.n_samples = X.shape[0]
        # subject_mask: (n_samples, n_subjects) bool — True if subject has data for this TR
        if subject_mask is not None:
            self.subject_mask = torch.tensor(subject_mask, dtype=torch.bool, device=device)
        else:
            self.subject_mask = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        item = {
            'x': self.X[idx],
            'x_delayed': self.X_delayed[idx],
            'y': {subj: self.Y[subj][idx] for subj in self.subjects},
        }
        if self.subject_mask is not None:
            item['subject_mask'] = self.subject_mask[idx]
        return item

    def __getitems__(self, indices):
        """Batched fetch (used by DataLoader when present): one fancy-index per
        tensor instead of one __getitem__ per row. Same rows, same order as the
        sampler produced; only the copy is batched. Returns the collated dict."""
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.X.device)
        item = {
            'x': self.X[idx],
            'x_delayed': self.X_delayed[idx],
            'y': {subj: self.Y[subj][idx] for subj in self.Y},
        }
        if self.subject_mask is not None:
            item['subject_mask'] = self.subject_mask[idx]
        return item


def collate_multisubject(batch):
    """Custom collate function for multi-subject batches."""
    if isinstance(batch, dict):
        return batch  # already collated by MultiSubjectDataset.__getitems__
    x = torch.stack([b['x'] for b in batch])
    x_delayed = torch.stack([b['x_delayed'] for b in batch])
    y = {
        subj: torch.stack([b['y'][subj] for b in batch])
        for subj in batch[0]['y'].keys()
    }
    result = {'x': x, 'x_delayed': x_delayed, 'y': y}
    if 'subject_mask' in batch[0]:
        result['subject_mask'] = torch.stack([b['subject_mask'] for b in batch])
    return result


def _data_cache_path(args, subjects, delays, train_stories):
    """Build a deterministic cache path from data-affecting config params."""
    key_parts = [
        f"subj={'_'.join(sorted(subjects))}",
        f"nstories={args.n_stories}" if not args.all_stories else "allstories",
        # Hash the actual training story set so subsampled runs never collide
        # with a cached full-data tensor (stability-vs-data runs).
        f"stories={hashlib.sha256(','.join(train_stories).encode()).hexdigest()[:8]}",
        f"nvox={args.n_voxels}",
        f"voxsel={args.voxel_selection}",
        f"emb={args.embedding}",
        f"whiten={args.whiten}",
        f"wcomp={getattr(args, 'whiten_n_components', 300)}" if args.whiten else "",
        f"demean={not args.no_demean}",
        f"trim={getattr(args, 'trim_start', 5)}_{getattr(args, 'trim_end', 5)}",
        f"delays={'_'.join(str(d) for d in delays)}",
    ]
    key_str = "|".join(p for p in key_parts if p)
    h = hashlib.sha256(key_str.encode()).hexdigest()[:12]
    cache_dir = join(PROJECT_DIR, "outputs", "data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return join(cache_dir, f"multisubj_{h}.pt"), key_str


def _save_data_cache(path, key_str, data_dict):
    """Save preprocessed data to a .pt cache file."""
    data_dict["_cache_key"] = key_str
    torch.save(data_dict, path, pickle_protocol=4)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved data cache ({size_mb:.0f} MB): {path}")


def _load_data_cache(path, key_str):
    """Load preprocessed data from cache. Returns None if cache is stale."""
    if not os.path.exists(path):
        return None
    data = torch.load(path, weights_only=False)
    if data.get("_cache_key") != key_str:
        print(f"  Cache key mismatch, regenerating...")
        return None
    return data


def main(args):
    """Multi-subject coupled SAE training. Entry point is train_multisubj_hydra.py.

    `args` is an argparse.Namespace built by the Hydra wrapper from
    ``configs/multisubj.yaml`` and ``configs/experiment/*.yaml``.
    """
    # Fall back to CPU if the requested device is unavailable.
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] {args.device} requested but unavailable; falling back to cpu.")
        args.device = "cpu"

    # Gauge-invariant L1 is only meaningful when the brain decoder is free
    # (columns can grow). With unit-norm columns the penalty degenerates to
    # a plain L1 on activations, which is not what the formulation intends.
    if args.sparsity_mode == "l1" and args.brain_gauge_l1 <= 0:
        print("[warn] sparsity_mode='l1' but brain_gauge_l1=0: no sparsity "
              "pressure at all. Activations will be dense. Set brain_gauge_l1 > 0 "
              "to get Anthropic-style sparse L1 behavior.")

    if args.sparsity_mode == "l1" and args.brain_decoder_unit_norm:
        print("[warn] sparsity_mode='l1' with brain_decoder_unit_norm=True: "
              "gauge-invariant L1 is designed to replace unit-norm, not complement it. "
              "Consider brain_decoder_unit_norm=False for a clean Conerly-style setup.")

    if args.temporal_first and args.no_delay_scaling:
        print("[warn] temporal_first=True AND no_delay_scaling=True: the delay "
              "sum becomes an unweighted average, losing all HRF temporal "
              "structure. Each delay contributes equally. Consider enabling "
              "delay_scaling (remove --no_delay_scaling).")

    if args.brain_gauge_l1 > 0 and args.brain_decoder_unit_norm:
        print("[warn] brain_gauge_l1 > 0 AND brain_decoder_unit_norm = True: "
              "unit-norm columns make the gauge-invariant penalty collapse to "
              "a plain activation L1. Consider setting brain_decoder_unit_norm=False.")

    set_seed(args.seed)

    subjects = args.subjects.split(",")
    delays = [int(d) for d in args.delays.split(",")]

    print("=" * 70)
    print("Multi-Subject Coupled SAE Training")
    print("=" * 70)
    print(f"Subjects: {subjects}")
    print(f"Atoms: {args.n_atoms}, k={args.k}")
    print(f"Delays: {delays}")
    print(f"Alpha (semantic weight): {args.alpha}")
    print(f"Learning rate: {args.lr}, weight decay: {args.weight_decay}")
    print(f"Device: {args.device}, seed: {args.seed}")

    # Determine stories (use union of available stories for heterogeneous data)
    test_stories = STORIES_TEST
    if args.all_stories:
        story_sets = [
            set(get_all_stories_for_subject(s)) - set(test_stories)
            for s in subjects
        ]
        train_stories = sorted(set.union(*story_sets))
        n_stories_str = f"all ({len(train_stories)})"
    else:
        train_stories = STORIES_TRAIN[:args.n_stories]
        n_stories_str = str(args.n_stories)

    # Explicit story list override (stability-vs-data runs). When set,
    # it fully determines the loaded stories; per-subject availability filtering
    # downstream still applies. Includes both train and (held-out) val stories.
    train_list = getattr(args, "train_story_list", None)
    if train_list:
        train_stories = [s.strip() for s in str(train_list).split(",") if s.strip()]
        n_stories_str = f"explicit ({len(train_stories)})"

    print(f"Training stories: {n_stories_str}")

    # -------------------------------------------------------------------------
    # Load data (with caching)
    # -------------------------------------------------------------------------
    cache_path, cache_key = _data_cache_path(args, subjects, delays, train_stories)
    cached = _load_data_cache(cache_path, cache_key)

    if cached is not None:
        t0 = time.time()
        X_train = cached["X_train"]
        X_train_delayed = cached["X_train_delayed"]
        X_test = cached["X_test"]
        X_test_delayed = cached["X_test_delayed"]
        Y_train_per_subject = cached["Y_train_per_subject"]
        Y_test_per_subject = cached["Y_test_per_subject"]
        n_voxels_per_subject = cached["n_voxels_per_subject"]
        voxel_indices_per_subject = cached["voxel_indices_per_subject"]
        story_lengths = cached["story_lengths"]
        story_offsets = np.concatenate([[0], np.cumsum(story_lengths)])
        subject_mask_train = cached.get("subject_mask_train", None)
        print(f"\nLoaded data from cache in {time.time() - t0:.1f}s")
        print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
        for subj in subjects:
            print(f"  Y_train[{subj}]: {Y_train_per_subject[subj].shape}")
    else:
        demean_features = not args.no_demean
        print(f"\nLoading training data (demean={demean_features})...")

        # Load features + delays + brain responses PER STORY, then trim and
        # concatenate. This prevents delayed features from bleeding across story
        # boundaries and drops boundary TRs with incomplete delays.
        trim_s = getattr(args, "trim_start", 5)
        trim_e = getattr(args, "trim_end", 5)
        print(f"  Per-story trim: first {trim_s}, last {trim_e} TRs")

        X_parts, X_del_parts = [], []
        Y_parts_per_subject = {subj: [] for subj in subjects}
        voxel_indices_per_subject = {}
        n_voxels_per_subject = {}

        # First pass: determine voxel selection
        # Use only stories available for each subject
        for subj in subjects:
            subj_stories = [s for s in train_stories if s in set(get_all_stories_for_subject(subj))]
            Y_full = load_responses(subj_stories, subj)
            Y_full = np.nan_to_num(Y_full)
            if args.voxel_selection == "ridge":
                vox_idx = select_voxels_by_ridge(
                    subject=subj, n_voxels=args.n_voxels,
                    embedding="glove", ridge_n_stories=10,
                )
            elif args.voxel_selection == "ridge_gemma2_cv":
                import joblib as _jl
                cv_path = join(PROJECT_DIR, "outputs", "models",
                               f"ridge_{subj}_10stories_gemma2_cv", "scores.joblib")
                cv_scores = _jl.load(cv_path)["mean_cv_corr"]
                vox_idx = np.sort(np.argsort(cv_scores)[-args.n_voxels:])
            else:
                _, vox_idx = select_top_voxels(Y_full, args.n_voxels)
            voxel_indices_per_subject[subj] = vox_idx
            n_voxels_per_subject[subj] = len(vox_idx)
            del Y_full

        # Determine which stories each subject has
        stories_per_subject = {}
        for subj in subjects:
            available = set(get_all_stories_for_subject(subj))
            stories_per_subject[subj] = available

        # Second pass: per-story load, delay, trim, concatenate
        # Build subject mask for heterogeneous story availability
        subject_mask_parts = []
        for s in train_stories:
            # Note: features are looked up for subjects[0]. The stimulus activations
            # are the same for every subject (one file for gemma2), so this is exact
            # here; a per-subject feature file would need a per-subject lookup.
            X_s = load_features(
                [s], embedding=args.embedding, subject=subjects[0],
                demean_per_story=demean_features,
            )
            X_del_s = apply_delays_np(X_s, delays)
            sl = slice(trim_s, -trim_e if trim_e > 0 else None)
            n_trs = X_s[sl].shape[0]
            X_parts.append(X_s[sl])
            X_del_parts.append(X_del_s[sl])

            # Per-subject: load brain responses if available, else pad with zeros
            story_mask = np.zeros((n_trs, len(subjects)), dtype=bool)
            for si, subj in enumerate(subjects):
                if s in stories_per_subject[subj]:
                    try:
                        Y_s = load_responses([s], subj)
                        Y_s = np.nan_to_num(Y_s)[:, voxel_indices_per_subject[subj]]
                        Y_parts_per_subject[subj].append(Y_s[sl])
                        story_mask[:, si] = True
                    except (FileNotFoundError, KeyError):
                        Y_parts_per_subject[subj].append(
                            np.zeros((n_trs, n_voxels_per_subject[subj]), dtype=np.float32))
                else:
                    Y_parts_per_subject[subj].append(
                        np.zeros((n_trs, n_voxels_per_subject[subj]), dtype=np.float32))
            subject_mask_parts.append(story_mask)

        subject_mask_train = np.concatenate(subject_mask_parts, axis=0)

        X_train = np.concatenate(X_parts, axis=0)
        X_train_delayed = np.concatenate(X_del_parts, axis=0)
        story_lengths = np.array([x.shape[0] for x in X_parts])
        story_offsets = np.concatenate([[0], np.cumsum(story_lengths)])
        print(f"  X_train: {X_train.shape} (after per-story trim)")
        print(f"  X_train_delayed: {X_train_delayed.shape}")

        Y_train_per_subject = {}
        for subj in subjects:
            Y = np.concatenate(Y_parts_per_subject[subj], axis=0)
            Y = np.nan_to_num(sp_zscore(Y, axis=0)).astype(np.float32)
            Y_train_per_subject[subj] = Y
            print(f"  Y_train[{subj}]: {Y.shape}")

        n_samples = X_train.shape[0]
        for subj, Y in Y_train_per_subject.items():
            assert Y.shape[0] == n_samples, f"{subj} TR mismatch: {Y.shape[0]} vs {n_samples}"

        # Load test data
        print(f"\nLoading test data ({test_stories[0]})...")
        X_test_parts, X_test_del_parts = [], []
        Y_test_parts = {subj: [] for subj in subjects}
        for s in test_stories:
            # Note: same subjects[0] lookup as in the training loop above.
            X_s = load_features([s], embedding=args.embedding, subject=subjects[0],
                                demean_per_story=demean_features)
            X_del_s = apply_delays_np(X_s, delays)
            sl = slice(trim_s, -trim_e if trim_e > 0 else None)
            X_test_parts.append(X_s[sl])
            X_test_del_parts.append(X_del_s[sl])
            for subj in subjects:
                Y_s = load_responses([s], subj)
                Y_s = np.nan_to_num(Y_s)[:, voxel_indices_per_subject[subj]]
                Y_test_parts[subj].append(Y_s[sl])
        X_test = np.concatenate(X_test_parts, axis=0)
        X_test_delayed = np.concatenate(X_test_del_parts, axis=0)
        Y_test_per_subject = {}
        for subj in subjects:
            Y = np.concatenate(Y_test_parts[subj], axis=0)
            Y = np.nan_to_num(sp_zscore(Y, axis=0)).astype(np.float32)
            Y_test_per_subject[subj] = Y

        print(f"  X_test: {X_test.shape} (after trim)")
        for subj, Y in Y_test_per_subject.items():
            print(f"  Y_test[{subj}]: {Y.shape}")

        # Optional PCA whitening (Shi et al. 2025)
        if getattr(args, "whiten", False):
            n_comp = getattr(args, "whiten_n_components", 300)
            n_sem_orig = X_train.shape[1]
            n_del = len(delays)

            X_mean = X_train.mean(axis=0)
            X_c = X_train - X_mean
            cov = X_c.T @ X_c / (len(X_c) - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = eigenvalues.argsort()[::-1][:n_comp]
            W_pca = eigenvectors[:, idx] / np.sqrt(eigenvalues[idx] + 1e-8)

            explained = eigenvalues[idx].sum() / eigenvalues.sum() * 100
            print(f"\nPCA whitening: {n_sem_orig}d → {n_comp}d "
                  f"({explained:.1f}% variance explained)")

            X_train = X_c @ W_pca
            X_del_r = X_train_delayed.reshape(-1, n_del, n_sem_orig)
            X_train_delayed = np.stack(
                [(X_del_r[:, d, :] - X_mean) @ W_pca for d in range(n_del)],
                axis=1
            ).reshape(-1, n_del * n_comp)

            X_test = (X_test - X_mean) @ W_pca
            X_del_r = X_test_delayed.reshape(-1, n_del, n_sem_orig)
            X_test_delayed = np.stack(
                [(X_del_r[:, d, :] - X_mean) @ W_pca for d in range(n_del)],
                axis=1
            ).reshape(-1, n_del * n_comp)

            print(f"  X_train: {X_train.shape}, X_train_delayed: {X_train_delayed.shape}")
            print(f"  X_test: {X_test.shape}, X_test_delayed: {X_test_delayed.shape}")

        # Save cache for future runs (BEFORE input_rms and feature_scale,
        # which are applied after cache load too). Skippable: the
        # stability-vs-data batch uses unique story sets per run, so caching
        # only bloats disk (multi-GB per tensor) with no reuse.
        if getattr(args, "cache_data", True):
            _save_data_cache(cache_path, cache_key, {
            "X_train": X_train,
            "X_train_delayed": X_train_delayed,
            "X_test": X_test,
            "X_test_delayed": X_test_delayed,
            "Y_train_per_subject": Y_train_per_subject,
            "Y_test_per_subject": Y_test_per_subject,
            "n_voxels_per_subject": n_voxels_per_subject,
            "voxel_indices_per_subject": voxel_indices_per_subject,
            "story_lengths": story_lengths,
            "subject_mask_train": subject_mask_train
                if 'subject_mask_train' in dir() else None,
        })

    n_samples = X_train.shape[0]

    # -------------------------------------------------------------------------
    # Input normalization / scaling (applied after cache, affects both paths)
    # -------------------------------------------------------------------------
    # Per-subject z-scoring over each subject's own time points (robustness
    # check). Y was z-scored over a table zero-padded for stories a subject did
    # not hear, which leaves UTS04-UTS06's real rows with variance ~2.8. Re-z-scoring the
    # real rows over themselves gives exactly the per-subject z-score (z-scoring is
    # invariant to the earlier affine map); padded rows are set to 0 and stay masked.
    # Applied after the data cache, so cached data are unaffected. Default off.
    if getattr(args, "zscore_own_rows", False):
        from sparsebrain.data import zscore_own_rows
        if subject_mask_train is None:
            raise SystemExit("zscore_own_rows needs subject_mask_train")
        for si, subj in enumerate(subjects):
            m = subject_mask_train[:, si].astype(bool)
            Y = Y_train_per_subject[subj]
            before = float(Y[m].var(axis=0).mean())
            Yn = zscore_own_rows(Y, m)
            Y_train_per_subject[subj] = Yn
            print(f"  zscore_own_rows: {subj} real-row variance {before:.2f} -> "
                  f"{float(Yn[m].var(axis=0).mean()):.2f} over {int(m.sum())} TRs")

    do_input_rms = getattr(args, "input_rms_normalize", False)
    if getattr(args, "loss_normalization", "") == "input_rms":
        do_input_rms = True
        args.loss_normalization = "loss"
    if do_input_rms:
        rms = float(np.sqrt(np.mean(X_train ** 2)))
        print(f"\nInput RMS normalization: dividing all features by rms={rms:.2f}")
        X_train = (X_train / rms).astype(np.float32)
        X_train_delayed = (X_train_delayed / rms).astype(np.float32)
        X_test = (X_test / rms).astype(np.float32)
        X_test_delayed = (X_test_delayed / rms).astype(np.float32)

    feature_scale = getattr(args, "feature_scale", 1.0)
    if feature_scale != 1.0:
        print(f"\nFeature scaling: multiplying all features by {feature_scale}")
        X_train = (X_train * feature_scale).astype(np.float32)
        X_train_delayed = (X_train_delayed * feature_scale).astype(np.float32)
        X_test = (X_test * feature_scale).astype(np.float32)
        X_test_delayed = (X_test_delayed * feature_scale).astype(np.float32)

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------
    # MP-SAE requires unit-norm dictionary columns (= encoder norm)
    encoder_mode = getattr(args, "encoder_mode", "linear")
    if encoder_mode == "mp" and args.no_encoder_norm:
        print("  WARNING: MP-SAE requires unit-norm dictionary. Forcing encoder norm ON.")
        args.no_encoder_norm = False

    print("\nCreating model...")
    model = MultiSubjectCoupledSAE(
        n_sem=X_train.shape[1],
        n_voxels_per_subject=n_voxels_per_subject,
        n_atoms=args.n_atoms,
        k=args.k,
        n_delays=len(delays),
        use_delay_scaling=not args.no_delay_scaling,
        subject_specific_delays=args.subject_specific_delays,
        tie_semantic_decoder=args.tie_semantic_decoder,
        brain_voxel_scale=args.brain_voxel_scale,
        temporal_first=args.temporal_first,
        sparsity_mode=args.sparsity_mode,
        encoder_bias=args.encoder_bias,
        encoder_mode=getattr(args, "encoder_mode", "linear"),
        mp_steps=getattr(args, "mp_steps", None),
    )

    # Initialize b_pre. By default: geometric median of training features
    # (Gao 2024). With --freeze_b_pre: zero it out and freeze.
    if args.freeze_b_pre:
        with torch.no_grad():
            model.b_pre.data.zero_()
        model.b_pre.requires_grad_(False)
        print("\nb_pre frozen at 0 (will not be trained).")
    else:
        with torch.no_grad():
            x_sample = torch.as_tensor(X_train[: min(10000, X_train.shape[0])], dtype=torch.float32)
            model.init_b_pre_from_data(x_sample, method="geometric_median", n_iter=50)

    # Load encoder from a trained checkpoint.
    # freeze_from: load and FREEZE encoder (for retrain experiments)
    #   Special value "random": freeze random encoder (baseline control)
    # init_from: load but keep trainable (for fine-tuning)
    train_mode = getattr(args, "train_mode", "coupled")
    freeze_from = getattr(args, "freeze_from", None)
    init_from = getattr(args, "init_from", None)

    # Special case: freeze random encoder (untrained baseline)
    if freeze_from == "random":
        model.encoder.requires_grad_(False)
        model.b_pre.requires_grad_(False)
        if model.encoder_bias is not None:
            model.encoder_bias.requires_grad_(False)
        print("\n  Random encoder FROZEN (untrained baseline).")
        freeze_from = None  # don't try to load a file

    load_from = freeze_from or init_from
    if load_from:
        ckpt = torch.load(load_from, weights_only=False)
        sd = ckpt["state_dict"]
        model.encoder.weight.data = sd["encoder.weight"].clone()
        model.b_pre.data = sd["b_pre"].clone()
        if model.encoder_bias is not None and "encoder_bias" in sd:
            model.encoder_bias.data = sd["encoder_bias"].clone()
        if freeze_from:
            model.encoder.requires_grad_(False)
            model.b_pre.requires_grad_(False)
            if model.encoder_bias is not None:
                model.encoder_bias.requires_grad_(False)
            print(f"\n  Loaded encoder from {freeze_from}, frozen.")
        else:
            print(f"\n  Loaded encoder from {init_from}, trainable (fine-tune).")
        if train_mode in ("brain_only", "coupled"):
            for dec in model.brain_decoders.values():
                nn.init.xavier_uniform_(dec.weight)
        if train_mode in ("semantic_only", "coupled"):
            if model.decoder_sem is not None:
                nn.init.xavier_uniform_(model.decoder_sem.weight)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Shared encoder: {model.n_sem} -> {model.n_atoms}")
    for subj, n_vox in n_voxels_per_subject.items():
        brain_in = model.n_atoms if args.temporal_first else model.n_atoms * len(delays)
        print(f"  Brain decoder [{subj}]: {brain_in} -> {n_vox}")
    print(f"  Total parameters: {n_params:,} ({n_trainable:,} trainable)")
    print(f"  Train mode: {train_mode}")
    if args.subject_specific_delays:
        print(f"  Delay scaling: subject-specific")

    model = model.to(args.device)

    # torch.compile disabled — causes CUDA init errors on shared GPUs

    # TV regularization (spatial smoothness on brain decoder)
    tv_weight = getattr(args, "tv_weight", 0.0)
    if tv_weight > 0:
        import nibabel as nib
        tv_mode = getattr(args, "tv_mode", "3d")
        edge_incidence_per_subject = {}
        for subj in subjects:
            mask_path = join(paths.derivative_dir, subj, "func1pt8mm", "thickmask.nii.gz")
            if not os.path.exists(mask_path):
                mask_path = join(paths.derivative_dir, subj, "func1pt8mm", "brainmask.nii.gz")
            if os.path.exists(mask_path) and tv_mode == "3d":
                mask_data = nib.load(mask_path).get_fdata() > 0
                coords = np.array(np.where(mask_data)).T
                n_mask_voxels = len(coords)
                vox_idx = voxel_indices_per_subject[subj]
                if vox_idx.max() >= n_mask_voxels:
                    print(f"  TV [{subj}]: voxel indices exceed mask size "
                          f"({vox_idx.max()} >= {n_mask_voxels}), skipping")
                    edge_incidence_per_subject[subj] = None
                    continue
                coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}
                edges = []
                for idx, (i, j, k) in enumerate(coords):
                    for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1)]:
                        neighbor = (i+di, j+dj, k+dk)
                        if neighbor in coord_to_idx:
                            edges.append((idx, coord_to_idx[neighbor]))
                n_edges = len(edges)
                rows, cols, data = [], [], []
                for e, (v1, v2) in enumerate(edges):
                    rows.extend([e, e]); cols.extend([v1, v2]); data.extend([1.0, -1.0])
                import scipy.sparse as sp_sparse
                # Subsample to selected voxels
                E_full = sp_sparse.csr_matrix((data, (rows, cols)), shape=(n_edges, len(coords)))
                vox_idx = voxel_indices_per_subject[subj]
                E_sub = E_full[:, vox_idx]
                # Remove edges where both endpoints are zero (not in selection)
                row_nnz = np.array(E_sub.getnnz(axis=1)).flatten()
                keep = row_nnz >= 2
                E_sub = E_sub[keep]
                edge_incidence_per_subject[subj] = E_sub
                print(f"  TV [{subj}]: {E_sub.shape[0]} edges")
            else:
                edge_incidence_per_subject[subj] = None
                print(f"  TV [{subj}]: no mask found, skipping")
        model.init_tv_on_device(edge_incidence_per_subject, tv_weight, args.device)

    # -------------------------------------------------------------------------
    # Train/validation split
    # -------------------------------------------------------------------------
    # Val story selection uses a FIXED seed (val_seed), independent of
    # training seed. This ensures all seeds/configs use the same val stories,
    # so seed variance = pure optimization variance, not val-choice variance.
    val_rng = np.random.default_rng(getattr(args, "val_seed", 0))
    if args.val_mode == "story_level":
        assert args.val_n_stories < len(train_stories), \
            f"val_n_stories ({args.val_n_stories}) must be < n_stories ({len(train_stories)})"
        # Explicit val story list : pin val to a fixed set so it stays
        # identical across data-amount levels. Otherwise fall back to the
        # val_seed-random index choice (default behaviour).
        val_list = getattr(args, "val_story_list", None)
        if val_list:
            val_names = {s.strip() for s in str(val_list).split(",") if s.strip()}
            val_story_idx = sorted(i for i, s in enumerate(train_stories) if s in val_names)
            assert len(val_story_idx) == len(val_names), \
                f"val stories not all present in train_story_list: {val_names}"
        else:
            val_story_idx = sorted(val_rng.choice(
                len(train_stories), size=args.val_n_stories, replace=False
            ).tolist())
        val_mask = np.zeros(n_samples, dtype=bool)
        for si in val_story_idx:
            val_mask[story_offsets[si]:story_offsets[si + 1]] = True
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        print(f"Validation: story_level, held-out stories "
              f"{[train_stories[i] for i in val_story_idx]}")
    elif args.val_mode == "none":
        train_idx = np.arange(n_samples)
        val_idx = np.array([], dtype=int)
        print("Validation: none (all data used for training)")
    else:  # random_tr
        n_val = int(n_samples * args.val_fraction)
        n_train = n_samples - n_val
        indices = val_rng.permutation(n_samples)
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        print(f"Validation: random_tr, val_fraction={args.val_fraction}")

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    X_del_tr, X_del_val = X_train_delayed[train_idx], X_train_delayed[val_idx]
    Y_tr = {subj: Y[train_idx] for subj, Y in Y_train_per_subject.items()}
    Y_val = {subj: Y[val_idx] for subj, Y in Y_train_per_subject.items()}

    # Subject mask for heterogeneous story availability
    if 'subject_mask_train' in dir() and subject_mask_train is not None:
        mask_tr = subject_mask_train[train_idx]
        mask_val = subject_mask_train[val_idx]
        n_present = mask_tr.sum(axis=0)
        for si, subj in enumerate(subjects):
            print(f"  {subj}: {n_present[si]}/{len(train_idx)} TRs with brain data "
                  f"({100*n_present[si]/len(train_idx):.0f}%)")
    else:
        mask_tr = None
        mask_val = None

    print(f"\nTrain/val split: {len(train_idx)} train, {len(val_idx)} val")

    # -------------------------------------------------------------------------
    # Create data loaders
    # -------------------------------------------------------------------------
    # data_device: where the dataset tensors live. Default = args.device (the
    # whole cache on the GPU, as on the A6000 runs). data_device=cpu keeps them
    # in RAM and moves one batch at a time (4 GB cards); the sampler, batches
    # and arithmetic are unchanged.
    data_device = getattr(args, "data_device", None) or args.device
    need_y = train_mode != "semantic_only"
    print(f"Dataset tensors on {data_device}; batches computed on {args.device}"
          f"{'' if need_y else ' (semantic_only: brain targets not loaded)'}")
    train_dataset = MultiSubjectDataset(X_tr, X_del_tr, Y_tr, device=data_device,
                                        subject_mask=mask_tr, need_y=need_y)
    val_dataset = MultiSubjectDataset(X_val, X_del_val, Y_val, device=data_device,
                                      subject_mask=mask_val, need_y=need_y)

    def _batch_to_device(batch):
        """Move one collated batch to args.device (no-op when already there)."""
        if data_device == args.device:
            return batch
        out = {
            'x': batch['x'].to(args.device, non_blocking=True),
            'x_delayed': batch['x_delayed'].to(args.device, non_blocking=True),
            'y': batch['y'],  # moved later, only the subjects the loss reads
        }
        if 'subject_mask' in batch:
            out['subject_mask'] = batch['subject_mask'].to(args.device, non_blocking=True)
        return out

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_multisubject,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_multisubject,
        num_workers=0,
        pin_memory=False,
    )

    # -------------------------------------------------------------------------
    # Training setup
    # -------------------------------------------------------------------------
    # Separate lr for encoder vs decoder when fine-tuning
    encoder_lr = getattr(args, "encoder_lr", None)
    if encoder_lr is not None and encoder_lr != args.lr:
        encoder_params = []
        decoder_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "encoder" in name or "b_pre" in name:
                encoder_params.append(p)
            else:
                decoder_params.append(p)
        param_groups = [
            {"params": decoder_params, "lr": args.lr},
            {"params": encoder_params, "lr": encoder_lr},
        ]
        print(f"  Separate lr: decoder={args.lr}, encoder={encoder_lr}")
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

    # Get git hash early so it's available for W&B and checkpoint.
    import subprocess
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    # Initialize Weights & Biases (optional).
    # Failures in wandb should never kill training — log and continue.
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            config = {**vars(args), "git_hash": git_hash}
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=config,
                name=f"multisubj_{args.name_suffix}" if args.name_suffix else None,
                tags=tags,
                group=args.wandb_group,
                reinit=True,
                settings=wandb.Settings(_disable_stats=False),
            )
        except Exception as e:
            print(f"[wandb] init failed: {e}; continuing without logging.")
            wandb_run = None

    print(f"\nTraining for {args.epochs} epochs "
          f"(warmup={args.warmup_epochs}, patience={args.patience})...")

    history = {
        "train_brain": [], "train_sem": [], "train_total": [], "train_l0": [],
        "val_brain": [], "val_sem": [], "val_total": [],
        "lr": [],
    }
    for subj in subjects:
        history[f"train_brain_{subj}"] = []
        history[f"val_brain_{subj}"] = []

    # Compute normalization constants for the multi-task loss.
    # Three modes:
    #   "none":     raw MSE, no division (recommended with input_rms_normalize)
    #   "loss":     divide by initial loss value (both start at ~1.0)
    #   "gradient": divide by initial encoder gradient norm
    norm_mode = getattr(args, "loss_normalization", "none")
    brain_init = 1.0
    sem_init = 1.0

    if norm_mode == "none":
        print(f"\nNo loss normalization (raw MSE). alpha={args.alpha}")
    else:
        model.eval()
        x_init = torch.as_tensor(X_val, dtype=torch.float32).to(args.device)
        x_del_init = torch.as_tensor(X_del_val, dtype=torch.float32).to(args.device)
        y_init = {s: torch.as_tensor(Y_val[s], dtype=torch.float32).to(args.device)
                  for s in subjects}

        if norm_mode == "gradient":
            model.zero_grad()
            out_init = model.forward_all_subjects(x_init, x_del_init)
            brain_mse = sum(F.mse_loss(out_init['y_hat'][s], y_init[s])
                            for s in subjects) / len(subjects)
            brain_mse.backward(retain_graph=True)
            g_brain = model.encoder.weight.grad.norm().item()
            model.zero_grad()

            sem_mse = F.mse_loss(out_init['x_hat'], x_init)
            sem_mse.backward()
            g_sem = model.encoder.weight.grad.norm().item()
            model.zero_grad()

            brain_init = max(g_brain, 1e-8)
            sem_init = max(g_sem, 1e-8)
            print(f"\nGradient-norm normalization: brain_grad={brain_init:.4f}, "
                  f"sem_grad={sem_init:.4f}, ratio={brain_init/sem_init:.2f}")
        elif norm_mode == "loss":
            with torch.no_grad():
                out_init = model.forward_all_subjects(x_init, x_del_init)
                brain_init = float(sum(F.mse_loss(out_init['y_hat'][s], y_init[s]).item()
                                       for s in subjects) / len(subjects))
                sem_init = float(F.mse_loss(out_init['x_hat'], x_init).item())
            print(f"\nLoss-value normalization: brain={brain_init:.4f}, "
                  f"sem={sem_init:.4f}")

        del x_init, x_del_init, y_init

    gauge_init = 1.0
    if args.brain_gauge_l1 > 0:
        with torch.no_grad():
            x_g = torch.as_tensor(X_val, dtype=torch.float32).to(args.device)
            x_del_g = torch.as_tensor(X_del_val, dtype=torch.float32).to(args.device)
            out_g = model.forward_all_subjects(x_g, x_del_g)
            gauge_init = max(float(model.gauge_invariant_l1(
                out_g['s_delayed']).item()), 1e-8)
            del x_g, x_del_g, out_g

    model.train()

    # Construct output directory early (needed for checkpoint_every)
    n_stories_label = "allstories" if args.all_stories else f"{len(train_stories)}stories"
    emb_suffix = f"_{args.embedding}"
    delay_suffix = "" if args.no_delay_scaling else "_delscale"
    if args.subject_specific_delays:
        delay_suffix += "_subjdel"
    subj_label = "_".join(subjects)
    name_suffix_str = f"_{args.name_suffix}" if args.name_suffix else ""
    model_name = (f"multisubj_sae_{subj_label}_{n_stories_label}_"
                  f"{args.n_atoms}atoms_k{args.k}{emb_suffix}{delay_suffix}{name_suffix_str}")
    out_dir = join(PROJECT_DIR, "outputs", "models", model_name)
    os.makedirs(out_dir, exist_ok=True)

    # Per-subject samplers for round_robin_v2: each subject gets batches
    # from only the TRs where it has data (no masking needed).
    brain_subj_mode = getattr(args, "brain_subj_mode", "all")
    subj_samplers = {}
    if brain_subj_mode == "round_robin_v2" and mask_tr is not None:
        for si, subj in enumerate(subjects):
            valid_idx = np.where(mask_tr[:, si])[0]
            subj_samplers[subj] = valid_idx
            print(f"  round_robin_v2: {subj} has {len(valid_idx)} valid TRs")
    elif brain_subj_mode == "round_robin_v2" and mask_tr is None:
        for subj in subjects:
            subj_samplers[subj] = np.arange(len(X_tr))
            print(f"  round_robin_v2: {subj} uses all {len(X_tr)} TRs (no mask)")

    best_val_loss = float('inf')  # tracks val total loss (lower = better)
    best_epoch = 0
    best_state = None
    best_val_r = -1.0  # tracks best val brain correlation
    best_r_epoch = 0
    best_r_state = None
    patience_counter = 0

    start_time = time.time()
    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        epoch_brain_sum = torch.tensor(0.0, device=args.device)
        epoch_sem_sum = torch.tensor(0.0, device=args.device)
        epoch_total_sum = torch.tensor(0.0, device=args.device)
        epoch_brain_per_subj = {subj: torch.tensor(0.0, device=args.device) for subj in subjects}
        epoch_l0 = 0
        n_batches = 0
        subj_cycle_idx = 0

        # Build batch iterator based on mode
        if brain_subj_mode == "round_robin_v2":
            # Per-subject batching: each step samples from the active
            # subject's valid TRs. No masking needed.
            rr_rng = np.random.default_rng(epoch * 1000 + 42)
            n_batches_per_epoch = max(
                len(idx) for idx in subj_samplers.values()
            ) * len(subjects) // args.batch_size
            batch_iter = []
            for bi in range(n_batches_per_epoch):
                active_subj = subjects[bi % len(subjects)]
                valid_idx = subj_samplers[active_subj]
                sel = rr_rng.choice(len(valid_idx), size=min(args.batch_size, len(valid_idx)), replace=False)
                idx = torch.tensor(valid_idx[sel], dtype=torch.long, device=data_device)
                batch_iter.append((active_subj, idx))
        else:
            batch_iter = None

        for batch_or_info in (batch_iter if batch_iter is not None else train_loader):
            if brain_subj_mode == "round_robin_v2":
                active_subj_v2, idx = batch_or_info
                x = train_dataset.X[idx].to(args.device)
                x_del = train_dataset.X_delayed[idx].to(args.device)
                y = {subj: train_dataset.Y[subj][idx].to(args.device)
                     for subj in subjects if subj in train_dataset.Y}
                batch_mask = None
            else:
                batch = _batch_to_device(batch_or_info)
                x = batch['x']
                x_del = batch['x_delayed']
                y = {subj: batch['y'][subj] for subj in subjects if subj in batch['y']}
                batch_mask = batch.get('subject_mask', None)

            optimizer.zero_grad()

            # --- Compute losses for one forward pass at a given k level ---
            def compute_losses_for_output(output, y, batch_mask, train_mode,
                                          brain_subj_mode, subjects, active_subj_override=None):
                """Compute brain and semantic losses from a forward pass output."""
                if train_mode != "semantic_only":
                    bl = {}
                    total_bl = 0
                    n_active = 0

                    if brain_subj_mode in ("round_robin", "round_robin_v2"):
                        asubj = active_subj_override
                        if brain_subj_mode == "round_robin" and batch_mask is not None:
                            si = subjects.index(asubj)
                            m = batch_mask[:, si]
                            if m.any():
                                bls = F.mse_loss(output['y_hat'][asubj][m], y[asubj][m])
                            else:
                                bls = torch.tensor(0.0, device=output['s'].device, requires_grad=True)
                        else:
                            bls = F.mse_loss(output['y_hat'][asubj], y[asubj])
                        bl[asubj] = bls.item()
                        total_bl = bls
                        n_active = 1
                        for s in subjects:
                            if s != asubj:
                                bl[s] = 0.0
                    else:
                        for si, subj in enumerate(subjects):
                            if batch_mask is not None:
                                m = batch_mask[:, si]
                                if m.any():
                                    bls = F.mse_loss(output['y_hat'][subj][m], y[subj][m])
                                    bl[subj] = bls.item()
                                    total_bl = total_bl + bls
                                    n_active += 1
                                else:
                                    bl[subj] = 0.0
                            else:
                                bls = F.mse_loss(output['y_hat'][subj], y[subj])
                                bl[subj] = bls.item()
                                total_bl = total_bl + bls
                                n_active += 1
                        if n_active > 0:
                            total_bl = total_bl / n_active
                else:
                    total_bl = torch.tensor(0.0, device=output['s'].device)
                    bl = {subj: 0.0 for subj in subjects}

                if train_mode != "brain_only":
                    sl = F.mse_loss(output['x_hat'], x)
                else:
                    sl = torch.tensor(0.0, device=output['s'].device)

                return total_bl, sl, bl

            # Determine active subject for round-robin (once per batch)
            active_subj_rr = None
            if brain_subj_mode in ("round_robin", "round_robin_v2"):
                if brain_subj_mode == "round_robin_v2":
                    active_subj_rr = active_subj_v2
                else:
                    active_subj_rr = subjects[subj_cycle_idx % len(subjects)]
                    subj_cycle_idx += 1

            # Brain targets to the compute device: only the subjects the loss
            # reads (round-robin: the active one). Same loss, 1/6 the memory.
            if train_mode != "semantic_only":
                y = {s: t.to(args.device, non_blocking=True) for s, t in y.items()
                     if active_subj_rr is None or s == active_subj_rr}

            # Matryoshka: compute losses at multiple k levels
            matryoshka_levels = getattr(args, "matryoshka_levels", None)
            if matryoshka_levels:
                total_brain_loss = 0
                sem_loss = 0
                brain_losses = {subj: 0.0 for subj in subjects}
                # Round-robin: only the active subject's prediction enters the loss (and AuxK and the
                # dead-atom counters read only the code), so running only its decoder gives the same
                # gradients; with two k levels the all-subject graph does not fit a 4 GB card.
                mk_subjects = None
                if train_mode == "semantic_only":
                    mk_subjects = []
                elif active_subj_rr is not None and args.brain_gauge_l1 <= 0:
                    mk_subjects = [active_subj_rr]
                for k_prime in matryoshka_levels:
                    output_k = model.forward_all_subjects(x, x_del, k_override=k_prime,
                                                          subjects=mk_subjects)
                    bl_k, sl_k, bld_k = compute_losses_for_output(
                        output_k, y, batch_mask, train_mode,
                        brain_subj_mode, subjects, active_subj_rr)
                    total_brain_loss = total_brain_loss + bl_k
                    sem_loss = sem_loss + sl_k
                    for s in subjects:
                        brain_losses[s] += bld_k[s]
                total_brain_loss = total_brain_loss / len(matryoshka_levels)
                sem_loss = sem_loss / len(matryoshka_levels)
                for s in subjects:
                    brain_losses[s] /= len(matryoshka_levels)
                output = output_k  # use last (full k) for tracking
            else:
                # Only run the brain decoders whose outputs the loss reads:
                # none for semantic_only, the active subject for round-robin.
                # Gradients are identical (the other outputs never enter the
                # loss); it is what lets a batch fit a 4 GB card.
                fwd_subjects = None
                if train_mode == "semantic_only":
                    fwd_subjects = []
                elif active_subj_rr is not None and args.brain_gauge_l1 <= 0:
                    # AuxK reconstructs the semantic residual and the dead-atom counters read
                    # only the code, so neither needs the other subjects' decoders.
                    fwd_subjects = [active_subj_rr]
                output = model.forward_all_subjects(x, x_del, subjects=fwd_subjects)
                total_brain_loss, sem_loss, brain_losses = compute_losses_for_output(
                    output, y, batch_mask, train_mode,
                    brain_subj_mode, subjects, active_subj_rr)

            total_loss = total_brain_loss / brain_init + args.alpha * sem_loss / sem_init
            if args.brain_gauge_l1 > 0 and train_mode != "semantic_only":
                gauge_l1 = model.gauge_invariant_l1(output['s_delayed'])
                total_loss = total_loss + args.brain_gauge_l1 * gauge_l1 / gauge_init
            if tv_weight > 0 and train_mode != "semantic_only":
                total_loss = total_loss + model.tv_loss()

            # AuxK: reconstruct residual from dead atoms (Gao 2024)
            if args.auxk and train_mode != "brain_only" and getattr(args, "encoder_mode", "linear") != "mp" and hasattr(model, '_atom_inactive_steps'):
                dead_mask = model._atom_inactive_steps > args.auxk_dead_threshold
                if dead_mask.any():
                    k_aux = min(args.k * 4, int(dead_mask.sum()))
                    if k_aux > 0:
                        # Re-encode but detach from main graph to avoid
                        # in-place modification conflicts.
                        with torch.no_grad():
                            z_pre = model.encoder(x - model.b_pre)
                            if model.encoder_bias is not None:
                                z_pre = z_pre + model.encoder_bias
                            z_pre = z_pre.clone()
                        z_pre.requires_grad_(True)
                        z_dead = F.relu(z_pre)
                        # Zero non-dead atoms (use multiplication, not in-place)
                        z_dead = z_dead * dead_mask.float().unsqueeze(0)
                        _, top_dead = torch.topk(z_dead, k_aux, dim=-1)
                        mask_dead = torch.zeros_like(z_dead)
                        mask_dead.scatter_(-1, top_dead, 1.0)
                        z_dead = z_dead * mask_dead
                        residual = x - output['x_hat'].detach()
                        if model.decoder_sem is not None:
                            residual_hat = model.decoder_sem(z_dead)
                        else:
                            residual_hat = F.linear(z_dead, model.encoder.weight.T)
                        aux_loss = args.auxk_alpha * F.mse_loss(residual_hat, residual)
                        total_loss = total_loss + aux_loss

            total_loss.backward()

            # Gradient projection: remove radial component before optimizer
            # step so Adam doesn't fight the unit-norm projection (Gao 2024).
            if args.grad_projection and args.brain_decoder_unit_norm:
                project_decoder_gradients(model)

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            # Per-batch weight normalization (matches old script 09 behavior)
            if getattr(args, "normalize_every", "epoch") == "batch":
                model.normalize_weights(
                    non_neg_brain=True,
                    encoder_unit_norm=not args.no_encoder_norm,
                    brain_decoder_unit_norm=args.brain_decoder_unit_norm,
                    semantic_decoder_unit_norm=args.semantic_decoder_unit_norm,
                )

            # Track metrics (accumulate on GPU, sync once per epoch)
            epoch_brain_sum += total_brain_loss.detach()
            epoch_sem_sum += sem_loss.detach()
            epoch_total_sum += total_loss.detach()
            for subj in subjects:
                if isinstance(brain_losses[subj], float):
                    epoch_brain_per_subj[subj] += brain_losses[subj]
                else:
                    epoch_brain_per_subj[subj] += brain_losses[subj]
            s_active = (output["s"] > 0).float()
            epoch_l0 += s_active.sum(dim=-1).mean().item()
            n_batches += 1

            # Update AuxK dead-atom tracking
            if args.auxk:
                with torch.no_grad():
                    active = s_active.any(dim=0)
                    if not hasattr(model, '_atom_inactive_steps'):
                        model._atom_inactive_steps = torch.zeros(
                            model.n_atoms, device=args.device, dtype=torch.long)
                    model._atom_inactive_steps[active] = 0
                    model._atom_inactive_steps[~active] += 1

        # Per-epoch weight normalization (default; skipped if normalize_every=batch)
        if getattr(args, "normalize_every", "epoch") != "batch":
            model.normalize_weights(
            non_neg_brain=True,
            encoder_unit_norm=not args.no_encoder_norm,
            brain_decoder_unit_norm=args.brain_decoder_unit_norm,
            semantic_decoder_unit_norm=args.semantic_decoder_unit_norm,
        )

        scheduler.step()

        # Average metrics (sync from GPU once)
        epoch_losses = {
            "brain": (epoch_brain_sum / n_batches).item(),
            "semantic": (epoch_sem_sum / n_batches).item(),
            "total": (epoch_total_sum / n_batches).item(),
        }
        for subj in subjects:
            if isinstance(epoch_brain_per_subj[subj], torch.Tensor):
                epoch_brain_per_subj[subj] = (epoch_brain_per_subj[subj] / n_batches).item()
            else:
                epoch_brain_per_subj[subj] /= n_batches
        epoch_l0 /= n_batches

        history["train_brain"].append(epoch_losses["brain"])
        history["train_sem"].append(epoch_losses["semantic"])
        history["train_total"].append(epoch_losses["total"])
        history["train_l0"].append(epoch_l0)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        for subj in subjects:
            history[f"train_brain_{subj}"].append(epoch_brain_per_subj[subj])

        # --- Validation ---
        # Free the last training batch (predictions, targets, graph) first:
        # on a 4 GB card they would otherwise sit next to the validation pass.
        output = x = x_del = y = total_loss = batch = batch_mask = None
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        model.eval()
        val_losses = {"brain": 0, "semantic": 0, "total": 0}
        val_brain_per_subj = {subj: 0 for subj in subjects}
        n_val_batches = 0

        val_r_per_subj = {subj: 0.0 for subj in subjects}
        val_r_mean = 0.0

        if len(val_idx) > 0:
            with torch.no_grad():
                for batch in val_loader:
                    batch = _batch_to_device(batch)
                    x = batch['x']
                    x_del = batch['x_delayed']
                    y = batch['y']  # stays where the dataset lives; moved per subject

                    # Encoder / semantic path once; brain decoders one subject at
                    # a time (each prediction is (batch, 30000) floats), so the
                    # six predictions and targets never coexist on the GPU.
                    output = model.forward_all_subjects(x, x_del, subjects=[])

                    def _y_hat(subj, _x=x, _xd=x_del):
                        return model.forward_all_subjects(_x, _xd, subjects=[subj])['y_hat'][subj]

                    if train_mode != "semantic_only":
                        # Validation brain loss, masked per subject like the
                        # training loss (val_masked=true). Without the mask,
                        # subjects lacking a validation story are scored
                        # against zero-padded targets, which dominated early
                        # stopping for brain-only runs (stopped at epoch ~6).
                        val_batch_mask = batch.get('subject_mask', None)
                        use_val_mask = getattr(args, "val_masked", True) and val_batch_mask is not None
                        total_brain_loss = 0
                        n_val_active = 0
                        for si, subj in enumerate(subjects):
                            if use_val_mask:
                                m = val_batch_mask[:, si]
                                if not m.any():
                                    continue
                                brain_loss_subj = F.mse_loss(
                                    _y_hat(subj)[m], y[subj].to(args.device)[m])
                            else:
                                brain_loss_subj = F.mse_loss(
                                    _y_hat(subj), y[subj].to(args.device))
                            val_brain_per_subj[subj] += brain_loss_subj.item()
                            total_brain_loss = total_brain_loss + brain_loss_subj
                            n_val_active += 1
                        if n_val_active > 0:
                            total_brain_loss = total_brain_loss / n_val_active
                        else:
                            total_brain_loss = torch.tensor(0.0)
                    else:
                        total_brain_loss = torch.tensor(0.0)

                    if train_mode != "brain_only":
                        sem_loss = F.mse_loss(output['x_hat'], x)
                    else:
                        sem_loss = torch.tensor(0.0)

                    val_recon = total_brain_loss / brain_init + args.alpha * sem_loss / sem_init

                    val_losses["brain"] += total_brain_loss.item()
                    val_losses["semantic"] += sem_loss.item()
                    val_losses["total"] += val_recon.item()
                    n_val_batches += 1

            for k in val_losses:
                val_losses[k] /= n_val_batches
            for subj in subjects:
                val_brain_per_subj[subj] /= n_val_batches

            # Compute val brain correlation
            if train_mode != "semantic_only":
                with torch.no_grad():
                    X_v = torch.tensor(X_val, dtype=torch.float32).to(args.device)
                    X_del_v = torch.tensor(X_del_val, dtype=torch.float32).to(args.device)
                    for si, subj in enumerate(subjects):
                        # one subject at a time (memory); same values
                        out_v = model.forward_all_subjects(X_v, X_del_v, subjects=[subj])
                        Y_hat_v = out_v["y_hat"][subj].cpu().numpy()
                        if mask_val is not None:
                            smask = mask_val[:, si]
                            if smask.sum() < 3:
                                continue
                            Y_hat_v = Y_hat_v[smask]
                            Y_v_subj = Y_val[subj][smask]
                        else:
                            Y_v_subj = Y_val[subj]
                        Y_hat_z = np.nan_to_num(sp_zscore(Y_hat_v, axis=0))
                        Y_v_z = np.nan_to_num(sp_zscore(Y_v_subj, axis=0))
                        val_r_per_subj[subj] = float(
                            np.nan_to_num((Y_hat_z * Y_v_z).mean(axis=0)).mean()
                        )
                    val_r_mean = float(np.mean(list(val_r_per_subj.values())))

        history["val_brain"].append(val_losses["brain"])
        history["val_sem"].append(val_losses["semantic"])
        history["val_total"].append(val_losses["total"])
        history.setdefault("val_r_mean", []).append(val_r_mean)
        for subj in subjects:
            history[f"val_brain_{subj}"].append(val_brain_per_subj[subj])

        # W&B per-epoch logging
        if wandb_run is not None:
            try:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/brain": epoch_losses["brain"],
                    "train/semantic": epoch_losses["semantic"],
                    "train/total": epoch_losses["total"],
                    "train/l0": epoch_l0,
                    "val/brain": val_losses["brain"],
                    "val/semantic": val_losses["semantic"],
                    "val/total": val_losses["total"],
                    "val/brain_r": val_r_mean,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                for subj in subjects:
                    log_dict[f"train/brain_{subj}"] = epoch_brain_per_subj[subj]
                    log_dict[f"val/brain_{subj}"] = val_brain_per_subj[subj]
                wandb_run.log(log_dict)
            except Exception as e:
                print(f"[wandb] epoch log failed: {e}")

        # Early stopping on val total loss — same formula as training:
        # brain_MSE/brain_init + alpha * sem_MSE/sem_init.
        # Consistent: train on total loss, stop on total loss.
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Also track best val brain_r checkpoint
        if val_r_mean > best_val_r:
            best_val_r = val_r_mean
            best_r_epoch = epoch + 1
            best_r_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            subj_str = ", ".join(f"{s}={val_brain_per_subj[s]:.4f}" for s in subjects)
            print(f"  Epoch {epoch+1:3d}: "
                  f"val_brain={val_losses['brain']:.4f} ({subj_str}), "
                  f"L0={epoch_l0:.1f}, lr={optimizer.param_groups[0]['lr']:.2e}")

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (best: {best_epoch})")
            break

        # Periodic checkpointing with val scores
        checkpoint_every = getattr(args, "checkpoint_every", 0)
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            ckpt_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ckpt_path = join(out_dir, f"model_ep{epoch+1}.pt")
            torch.save({
                "state_dict": ckpt_state,
                "args": vars(args),
                "epoch": epoch + 1,
                "subjects": subjects,
                "voxel_indices": voxel_indices_per_subject,
                "n_voxels_per_subject": n_voxels_per_subject,
            }, ckpt_path)
            val_r_str = ", ".join(f"{s}={val_r_per_subj.get(s, 0):.4f}" for s in subjects)
            print(f"  Checkpoint ep{epoch+1}: val_r={val_r_mean:.4f} ({val_r_str}), "
                  f"val_sem={val_losses.get('semantic', 0):.4f}")

    # Save last-epoch state before restoring best
    last_epoch = epoch + 1
    last_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f}s ({train_time/60:.1f} min)")
    print(f"Best val total loss at epoch {best_epoch} (loss={best_val_loss:.4f})")

    # Restore best model for primary evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(args.device)

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Evaluation")
    print("=" * 70)
    model.eval()

    def evaluate_subject(X_np, X_del_np, Y_np, subj, name):
        """Evaluate model on a single subject's data."""
        with torch.no_grad():
            X_t = torch.tensor(X_np, dtype=torch.float32).to(args.device)
            X_del_t = torch.tensor(X_del_np, dtype=torch.float32).to(args.device)
            output = model(X_t, X_del_t, subject=subj)

            Y_hat = output["y_hat"].cpu().numpy()
            S = output["s"].cpu().numpy()

            # Brain correlation (per voxel)
            Y_hat_z = np.nan_to_num(sp_zscore(Y_hat, axis=0))
            Y_z = np.nan_to_num(sp_zscore(Y_np, axis=0))
            brain_corr = np.nan_to_num((Y_hat_z * Y_z).mean(axis=0))

            print(f"\n{name} [{subj}]:")
            print(f"  mean r = {brain_corr.mean():.4f}, "
                  f"median = {np.median(brain_corr):.4f}, "
                  f"max = {brain_corr.max():.4f}")
            for thresh in [0.1, 0.2, 0.3]:
                n_above = (brain_corr > thresh).sum()
                pct = 100.0 * n_above / len(brain_corr)
                print(f"  r > {thresh}: {n_above:,} voxels ({pct:.1f}%)")

            return brain_corr, S

    # Evaluate on val set (for HP selection — report this, not test)
    print("\n--- VAL brain prediction ---")
    val_results = {}
    for subj in subjects:
        brain_corr, S = evaluate_subject(
            X_val, X_del_val, Y_val[subj], subj, "VAL"
        )
        val_results[subj] = {"brain_corr": brain_corr}
    val_brain_corr = {subj: val_results[subj]["brain_corr"] for subj in subjects}

    # Evaluate on test set for each subject
    print("\n--- TEST brain prediction ---")
    test_results = {}
    for subj in subjects:
        brain_corr, S = evaluate_subject(
            X_test, X_test_delayed, Y_test_per_subject[subj], subj, "TEST"
        )
        test_results[subj] = {"brain_corr": brain_corr, "S": S}

    # Semantic reconstruction
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(args.device)
        X_del_t = torch.tensor(X_test_delayed, dtype=torch.float32).to(args.device)
        output = model(X_t, X_del_t)

        X_hat = output["x_hat"].cpu().numpy()
        X_hat_z = np.nan_to_num(sp_zscore(X_hat, axis=0))
        X_z = np.nan_to_num(sp_zscore(X_test, axis=0))
        sem_corr = np.nan_to_num((X_hat_z * X_z).mean(axis=0))

        S = output["s"].cpu().numpy()

    print(f"\nTEST Semantic reconstruction:")
    print(f"  mean r = {sem_corr.mean():.4f}, median = {np.median(sem_corr):.4f}")

    print(f"\nTEST Sparsity:")
    print(f"  L0 per sample: {(S > 0).sum(axis=1).mean():.2f} "
          f"(target: {args.k})")
    active_atoms = (S.sum(axis=0) > 0).sum()
    print(f"  Active atoms: {active_atoms} / {args.n_atoms}")

    # Per-atom activation statistics
    atom_freq = (S > 0).mean(axis=0)
    atom_mean = S.mean(axis=0)
    print(f"  Atom activation freq: {atom_freq.mean():.3f} "
          f"(range: {atom_freq.min():.3f} - {atom_freq.max():.3f})")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    # out_dir already constructed before training loop

    # Collect per-subject correlations
    test_brain_corr = {subj: test_results[subj]["brain_corr"] for subj in subjects}
    val_brain_corr_final = {subj: val_brain_corr[subj] for subj in subjects}


    checkpoint = {
        "state_dict": model.state_dict(),
        "args": vars(args),
        "git_hash": git_hash,
        "subjects": subjects,
        "train_stories": train_stories,
        "voxel_indices": voxel_indices_per_subject,
        "n_voxels_per_subject": n_voxels_per_subject,
        "history": history,
        "best_epoch": best_epoch,
        "test_brain_corr": test_brain_corr,
        "val_brain_corr": val_brain_corr_final,
        "test_sem_corr": sem_corr,
        "atom_activation_freq": atom_freq,
        "brain_init": brain_init,
        "sem_init": sem_init,
    }
    torch.save(checkpoint, join(out_dir, "model.pt"))
    # Training settings, read by sparsebrain.runs to select runs without relying on folder names.
    with open(join(out_dir, "args.json"), "w") as _f:
        json.dump(vars(args), _f, indent=1, default=str)

    # Final W&B summary + artifact
    if wandb_run is not None:
        try:
            import wandb
            summary = {
                "final/test_r_mean": float(np.mean([test_brain_corr[s].mean()
                                                     for s in subjects])),
                "final/val_r_mean": float(np.mean([val_brain_corr_final[s].mean()
                                                    for s in subjects])),
                "final/test_sem_r_mean": float(sem_corr.mean()),
                "final/best_epoch": best_epoch,
                "final/brain_init": brain_init,
                "final/sem_init": sem_init,
            }
            for s in subjects:
                summary[f"final/test_r_{s}"] = float(test_brain_corr[s].mean())
                summary[f"final/val_r_{s}"] = float(val_brain_corr_final[s].mean())
                summary[f"final/test_r_{s}_gt02_pct"] = float(
                    (test_brain_corr[s] > 0.2).mean() * 100
                )
            wandb_run.summary.update(summary)

            art = wandb.Artifact(
                name=f"model-{wandb_run.id}",
                type="model",
                metadata={"best_epoch": best_epoch, **vars(args)},
            )
            art.add_file(join(out_dir, "model.pt"))
            wandb_run.log_artifact(art)
            wandb_run.finish()
        except Exception as e:
            print(f"[wandb] finalize failed: {e}")
            if wandb_run is not None:
                try:
                    wandb_run.finish(exit_code=1)
                except Exception:
                    pass

    # Save dictionaries. Encoder rows = unit-norm input directions ("what
    # activates atom i"); decoder_sem rows = semantic output shape with
    # magnitude ("what atom i generates"). semantic_dictionary.npy is kept
    # as an alias of decoder_sem_dictionary.npy for scripts 09, 13, 14, 16.
    encoder_dict = model.encoder.weight.detach().cpu().numpy()  # (n_atoms, n_sem)
    decoder_sem_dict = model.get_semantic_dictionary()           # (n_atoms, n_sem)
    np.save(join(out_dir, "encoder_dictionary.npy"), encoder_dict)
    np.save(join(out_dir, "decoder_sem_dictionary.npy"), decoder_sem_dict)
    np.save(join(out_dir, "semantic_dictionary.npy"), decoder_sem_dict)

    # Per-subject brain dictionaries are just brain_decoders.<subj>.weight
    # averaged over delays -- model.pt already stores that weight in full, and
    # downstream scripts (27, 30, 31, 32, paper/figures/fig1_atom_example.py)
    # read it from the state_dict rather than these files. Off by default:
    # ~19 MB/run of output that nothing loads. To rebuild them for an existing
    # run: python scripts/cleanup/regen_brain_dictionary.py <model_dir> --write
    if getattr(args, "save_brain_dicts", False):
        for subj in subjects:
            brain_dict = model.get_brain_dictionary(subj)
            np.save(join(out_dir, f"brain_dictionary_{subj}.npy"), brain_dict)

    # -------------------------------------------------------------------------
    # Also evaluate and save last-epoch model
    # -------------------------------------------------------------------------
    # Always evaluate and save the final-epoch checkpoint (model_last.pt): every result in the
    # paper uses it. model.pt holds the best-validation epoch.
    if last_state is not None:
        print("\n" + "=" * 70)
        print(f"Last-epoch evaluation (epoch {last_epoch})")
        print("=" * 70)
        model.load_state_dict(last_state)
        model = model.to(args.device)
        model.eval()

        print("\n--- TEST brain prediction (last epoch) ---")
        last_test_corr = {}
        for subj in subjects:
            brain_corr, _ = evaluate_subject(
                X_test, X_test_delayed, Y_test_per_subject[subj], subj, "TEST-last"
            )
            last_test_corr[subj] = brain_corr

        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32).to(args.device)
            X_del_t = torch.tensor(X_test_delayed, dtype=torch.float32).to(args.device)
            output = model(X_t, X_del_t)
            X_hat = output["x_hat"].cpu().numpy()
            X_hat_z = np.nan_to_num(sp_zscore(X_hat, axis=0))
            X_z = np.nan_to_num(sp_zscore(X_test, axis=0))
            last_sem_corr = np.nan_to_num((X_hat_z * X_z).mean(axis=0))

        last_test_r = float(np.mean([last_test_corr[s].mean() for s in subjects]))
        last_sem_r = float(last_sem_corr.mean())
        print(f"\nLast-epoch: test_r={last_test_r:.4f}, sem_r={last_sem_r:.4f}")
        print(f"Best-epoch: test_r={float(np.mean([test_brain_corr[s].mean() for s in subjects])):.4f}")

        # Save last-epoch checkpoint
        last_ckpt = dict(checkpoint)
        last_ckpt["state_dict"] = last_state
        last_ckpt["best_epoch"] = last_epoch
        last_ckpt["test_brain_corr"] = last_test_corr
        last_ckpt["test_sem_corr"] = last_sem_corr
        torch.save(last_ckpt, join(out_dir, "model_last.pt"))
        print(f"  Saved last-epoch model to {join(out_dir, 'model_last.pt')}")

        if wandb_run is not None:
            try:
                wandb_run.summary.update({
                    "final/last_test_r_mean": last_test_r,
                    "final/last_sem_r_mean": last_sem_r,
                    "final/last_epoch": last_epoch,
                })
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Evaluate best-val-brain_r checkpoint
    # -------------------------------------------------------------------------
    if best_r_state is not None and best_r_epoch != best_epoch and best_r_epoch != last_epoch:
        print("\n" + "=" * 70)
        print(f"Best-val-brain_r evaluation (epoch {best_r_epoch}, val_r={best_val_r:.4f})")
        print("=" * 70)
        model.load_state_dict(best_r_state)
        model = model.to(args.device)
        model.eval()

        print("\n--- TEST brain prediction (best val brain_r) ---")
        bestr_test_corr = {}
        for subj in subjects:
            brain_corr, _ = evaluate_subject(
                X_test, X_test_delayed, Y_test_per_subject[subj], subj, "TEST-bestR"
            )
            bestr_test_corr[subj] = brain_corr

        bestr_test_r = float(np.mean([bestr_test_corr[s].mean() for s in subjects]))
        print(f"\nBest-val-brain_r: test_r={bestr_test_r:.4f} (epoch {best_r_epoch})")

        # The best-val-brain_r checkpoint is a third model-selection point that
        # no script in this repo loads (~99 MB/run). Its headline numbers are
        # still recorded to W&B below, and the evaluation above still runs, so
        # only the file is skipped. Set save_best_r=true to keep it.
        if getattr(args, "save_best_r", False):
            torch.save({**checkpoint, "state_dict": best_r_state, "best_epoch": best_r_epoch,
                         "test_brain_corr": bestr_test_corr},
                       join(out_dir, "model_best_r.pt"))
            print(f"  Saved to {join(out_dir, 'model_best_r.pt')}")
        else:
            print("  (model_best_r.pt not written; set save_best_r=true to keep it)")

        if wandb_run is not None:
            try:
                wandb_run.summary.update({
                    "final/bestr_test_r_mean": bestr_test_r,
                    "final/bestr_epoch": best_r_epoch,
                })
            except Exception:
                pass

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Best-val-total epoch: {best_epoch}, Best-val-brain_r epoch: {best_r_epoch}, Last epoch: {last_epoch}")
    print(f"Test performance (best-val epoch):")
    for subj in subjects:
        r = test_brain_corr[subj]
        print(f"  {subj}: mean r = {r.mean():.4f}, "
              f"r>0.2: {(r>0.2).sum()} ({100*(r>0.2).mean():.1f}%)")
    print(f"\nModel saved to: {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    raise SystemExit(
        "This script is no longer a direct entry point. "
        "Run `uv run python scripts/train_multisubj_hydra.py` instead. "
        "See configs/multisubj.yaml for options."
    )
