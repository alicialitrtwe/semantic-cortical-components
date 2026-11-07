#!/usr/bin/env python
"""23_fit_ridge_voxsel.py — Fit ridge encoding for voxel selection.

Uses himalaya with GPU backend and leave-one-story-out CV.
Auto-selects KernelRidgeCV (n_samples < n_features) or RidgeCV (otherwise).
After fitting, re-fits per CV fold to compute proper CV correlation scores
on held-out training data for voxel selection.

Adapted from compose/model_fitting/fit_banded_ridge_models.py.

Datasets:
- lebel (default): LeBel et al. deep-fMRI, 25+1 stories, TR=2s
- narratives: Nastase et al. 2021, train=merlin, test=sherlock, TR=1.5s
- hcp: HCP Language task, 8 story blocks, TR=0.72s, leave-one-block-out CV

Usage:
    uv run python scripts/23_fit_ridge_voxsel.py --subject UTS03 --embedding gemma2
    uv run python scripts/23_fit_ridge_voxsel.py --subject UTS03 --embedding gemma2 --n_stories 10

    uv run python scripts/23_fit_ridge_voxsel.py --dataset narratives --subject sub-001 --embedding gemma2

    uv run python scripts/23_fit_ridge_voxsel.py --dataset hcp --subject 100206 --embedding gemma2
"""
import argparse
import os
import sys
from os.path import join

import joblib
import numpy as np
import torch
from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV, KernelRidge
from himalaya.ridge import RidgeCV, Ridge
from sklearn.model_selection import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore as sp_zscore
from voxelwise_tutorials.delayer import Delayer
from voxelwise_tutorials.utils import generate_leave_one_run_out

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from sparsebrain import paths, load_features, load_responses
from sparsebrain.data import get_all_stories_for_subject
from sparsebrain.paths import paths as _paths

import h5py

TEST_STORY = "wheretheressmoke"
ALPHAS = np.logspace(1, 20, 20)
DELAYS = [1, 2, 3, 4]
N_TARGETS_BATCH = 40000

# Narratives-specific constants
NARRATIVES_TRAIN_STORIES = ["merlin"]
NARRATIVES_TEST_STORY = "sherlock"
NARRATIVES_DELAYS = [1, 2, 3, 4, 5]  # 1.5–7.5s HRF window at TR=1.5s
NARRATIVES_CV_FOLDS = 5

# HCP-specific constants (TR=0.72s, 8 story blocks)
HCP_TR = 0.72
HCP_DELAYS = list(range(1, 12))  # 0.72–7.92s HRF window at TR=0.72s
# Train on blocks 0–6 (7 LO-block-out CV folds), test on block 7
# block 7 is the shortest (Story2, 17s) — used as held-out test
HCP_TRAIN_BLOCKS = list(range(7))   # blocks 0-6
HCP_TEST_BLOCK = 7                   # Story2
HCP_ALPHAS = np.logspace(1, 20, 20)  # wide range for kernel ridge; HCP has few TRs so needs heavy regularization


# ---------------------------------------------------------------------------
# Narratives loaders
# ---------------------------------------------------------------------------
def _load_narratives_features(stories: list[str], embedding: str) -> dict:
    feat_path = join(_paths.features_dir, f"stories-narratives-{embedding}.npz")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"Narratives features not found: {feat_path}\n"
            "Run: uv run python scripts/02b_save_features_llm.py --dataset narratives"
        )
    feat_data = np.load(feat_path)
    return {s: feat_data[s].astype(np.float32) for s in stories}


def _load_narratives_responses(stories: list[str], subject: str) -> np.ndarray:
    parts = []
    sub_dir = _paths.narratives_subject_data_dir(subject)
    for story in stories:
        fp = join(sub_dir, f"{story}.hf5")
        with h5py.File(fp, "r") as hf:
            parts.append(hf["data"][:].astype(np.float32))
    return np.concatenate(parts, axis=0)


def run_narratives(subject: str, embedding: str, args):
    """Run himalaya ridge on Narratives dataset.

    Training: merlin + first 2/3 of sherlock.
    Test: last 1/3 of sherlock (temporal split — no future leakage).
    CV: leave-one-story-out on the two training stories (2 folds).
    """
    train_stories = NARRATIVES_TRAIN_STORIES
    test_story = NARRATIVES_TEST_STORY

    print(f"Dataset: narratives  Subject: {subject}  Embedding: {embedding}")
    print(f"Train stories: {train_stories} + sherlock[:2/3]  Test: sherlock[2/3:]")

    print("\nLoading features...")
    feat = _load_narratives_features(train_stories + [test_story], embedding)

    print("\nLoading responses...")
    try:
        Y_merlin = _load_narratives_responses(train_stories, subject)
        Y_sherlock_all = _load_narratives_responses([test_story], subject)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None

    # Clip merlin to feature length
    n_merlin = min(feat[train_stories[0]].shape[0], Y_merlin.shape[0])
    X_merlin = feat[train_stories[0]][:n_merlin]
    Y_merlin = Y_merlin[:n_merlin]

    # Sherlock: split last 1/3 as test, first 2/3 as additional train
    n_sherlock = min(feat[test_story].shape[0], Y_sherlock_all.shape[0])
    n_sherlock_test = n_sherlock // 3
    n_sherlock_train = n_sherlock - n_sherlock_test

    X_sherlock_train = feat[test_story][:n_sherlock_train]
    X_test = feat[test_story][n_sherlock_train:n_sherlock]
    Y_sherlock_train = Y_sherlock_all[:n_sherlock_train]
    Y_test = Y_sherlock_all[n_sherlock_train:n_sherlock]

    # Concatenate training data: merlin + sherlock_train
    X_train = np.concatenate([X_merlin, X_sherlock_train], axis=0)
    Y_train = np.concatenate([Y_merlin, Y_sherlock_train], axis=0)
    n_train = X_train.shape[0]

    print(f"  merlin: {X_merlin.shape[0]} TRs  sherlock_train: {X_sherlock_train.shape[0]} TRs  "
          f"sherlock_test: {X_test.shape[0]} TRs")
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"  Y_train: {Y_train.shape}  Y_test: {Y_test.shape}")

    # Leave-one-story-out CV: merlin and sherlock_train as two separate runs
    run_onsets = np.array([0, n_merlin])
    cv = check_cv(generate_leave_one_run_out(n_train, run_onsets))
    cv_splits = list(cv.split(X_train))
    print(f"\nCV: leave-one-story-out ({len(cv_splits)} folds)")

    if args.device == "cuda" and torch.cuda.is_available():
        backend = set_backend("torch_cuda", on_error="warn")
    else:
        backend = set_backend("torch", on_error="warn")

    n_delayed_features = X_train.shape[1] * len(NARRATIVES_DELAYS)
    use_kernel = n_train < n_delayed_features
    mode = "kernel" if use_kernel else "primal"
    print(f"Mode: {mode} (n_train={n_train}, n_delayed_features={n_delayed_features})")

    solver_params = dict(n_targets_batch=N_TARGETS_BATCH)
    if use_kernel:
        solver_params["diagonalize_method"] = "svd"
    ModelCV = KernelRidgeCV if use_kernel else RidgeCV

    pipeline = make_pipeline(
        StandardScaler(with_mean=False, with_std=False),
        Delayer(delays=NARRATIVES_DELAYS),
        ModelCV(alphas=ALPHAS, cv=cv_splits, solver_params=solver_params),
    )
    print("\nFitting model...")
    pipeline.fit(X_train, Y_train)
    best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)

    Y_pred_test = backend.to_numpy(pipeline.predict(X_test))
    Y_test_z = np.nan_to_num(sp_zscore(Y_test, axis=0))
    Y_pred_z = np.nan_to_num(sp_zscore(Y_pred_test, axis=0))
    corr_test = np.nan_to_num((Y_test_z * Y_pred_z).mean(0))

    print(f"\nTest corr: median={np.median(corr_test):.4f}  "
          f"mean={corr_test.mean():.4f}  "
          f"top-1k={np.sort(corr_test)[-1000:].mean():.4f}")

    out_dir = join(PROJECT_DIR, "outputs", "models",
                   f"ridge_narratives_{subject}_{embedding}_loso")
    if args.suffix:
        out_dir += f"_{args.suffix}"
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(
        {"subject": subject, "embedding": embedding,
         "train_stories": train_stories, "test_story": test_story,
         "delays": NARRATIVES_DELAYS, "alphas": ALPHAS,
         "best_alphas": best_alphas, "corr_test": corr_test},
        join(out_dir, "scores.joblib"),
    )
    print(f"Saved to {out_dir}")
    return corr_test


# ---------------------------------------------------------------------------
# HCP loaders
# ---------------------------------------------------------------------------
def _load_hcp_features(blocks: list[int], embedding: str) -> dict:
    feat_path = join(_paths.features_dir, f"stories-hcp-{embedding}.npz")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"HCP features not found: {feat_path}\n"
            "Run: uv run python scripts/47_hcp_features.py"
        )
    feat_data = np.load(feat_path)
    return {b: feat_data[f"block_{b}"].astype(np.float32) for b in blocks}


def _load_hcp_responses(blocks: list[int], subject: str) -> dict:
    fp = join(_paths.hcp_subject_data_dir(subject), "language_story.hf5")
    with h5py.File(fp, "r") as hf:
        labels = hf["block_labels"][:]
        data = hf["data"][:]
    result = {}
    for b in blocks:
        mask = labels == b
        result[b] = data[mask].astype(np.float32)
    return result


def _fit_hcp_single(feat, resp, train_blocks, backend, n_features,
                    extra_feat=None, extra_resp=None):
    """Fit one ridge model (train on train_blocks, return pipeline + best_alphas).

    extra_feat/resp: additional rows appended after train_blocks (e.g. first 2/3
    of the held-out test block). Treated as a separate run for CV purposes.

    Boundary masking: the first max(HCP_DELAYS)=11 TRs of each non-first run are
    excluded from training because their delay features reach back into the previous
    story, contaminating the signal across block boundaries.
    """
    n_mask = max(HCP_DELAYS)

    block_feats = [feat[b] for b in train_blocks]
    block_resps = [resp[b] for b in train_blocks]
    if extra_feat is not None:
        block_feats.append(extra_feat)
        block_resps.append(extra_resp)

    run_lengths = [f.shape[0] for f in block_feats]

    # Build boundary mask: drop first n_mask TRs of each non-first run
    offsets = np.concatenate([[0], np.cumsum(run_lengths[:-1])])
    boundary_mask = np.ones(sum(run_lengths), dtype=bool)
    for off in offsets[1:]:
        boundary_mask[off:min(off + n_mask, sum(run_lengths))] = False

    X_tr = np.concatenate(block_feats, axis=0)[boundary_mask]
    Y_tr = np.concatenate(block_resps, axis=0)[boundary_mask]

    # CV run lengths after masking (first run unchanged, rest lose n_mask TRs)
    masked_lengths = [run_lengths[0]] + [max(1, rl - n_mask) for rl in run_lengths[1:]]
    masked_onsets = np.concatenate([[0], np.cumsum(masked_lengths[:-1])])
    n_samples = X_tr.shape[0]
    cv = check_cv(generate_leave_one_run_out(n_samples, masked_onsets))
    cv_splits = list(cv.split(X_tr))

    n_delayed = n_features * len(HCP_DELAYS)
    use_kernel = n_samples < n_delayed
    solver_params = dict(n_targets_batch=N_TARGETS_BATCH)
    if use_kernel:
        solver_params["diagonalize_method"] = "svd"
    ModelCV = KernelRidgeCV if use_kernel else RidgeCV

    pipeline = make_pipeline(
        StandardScaler(with_mean=False, with_std=False),
        Delayer(delays=HCP_DELAYS),
        ModelCV(alphas=HCP_ALPHAS, cv=cv_splits, solver_params=solver_params),
    )
    pipeline.fit(X_tr, Y_tr)
    return pipeline, backend.to_numpy(pipeline[-1].best_alphas_)


def run_hcp(subject: str, embedding: str, args):
    """Run himalaya ridge on HCP Language task.

    Default: train on blocks 0-6, test on block 7.
    With --lobo: 8-fold leave-one-block-out — all blocks serve as test once,
    giving ~286 total test TRs for reliable per-voxel correlation estimates.
    """
    lobo = getattr(args, "lobo", False)
    all_blocks = list(range(8))

    print(f"Dataset: hcp  Subject: {subject}  Embedding: {embedding}")
    print(f"Mode: {'8-fold LOBO' if lobo else f'train 0-6 / test 7'}")
    print(f"Delays: {HCP_DELAYS}  Alphas: {len(HCP_ALPHAS)}")

    print("\nLoading features...")
    feat = _load_hcp_features(all_blocks, embedding)
    n_features = feat[0].shape[1]
    print(f"  Feature dim: {n_features}")

    print("\nLoading responses...")
    try:
        resp = _load_hcp_responses(all_blocks, subject)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None

    n_voxels = resp[0].shape[1]

    # Clip each block to min(feat_trs, resp_trs)
    for b in all_blocks:
        n = min(feat[b].shape[0], resp[b].shape[0])
        feat[b] = feat[b][:n]
        resp[b] = resp[b][:n]

    if args.device == "cuda" and torch.cuda.is_available():
        backend = set_backend("torch_cuda", on_error="warn")
    else:
        backend = set_backend("torch", on_error="warn")

    if lobo:
        # 8-fold LOBO: each block serves as test once.
        # First 2/3 of test block → additional training data.
        # Last 1/3 of test block → held-out test (avoids wasting data).
        # Boundary masking inside _fit_hcp_single removes contaminated TRs.
        block_corrs = []
        total_test_trs = 0
        for test_b in all_blocks:
            train_bs = [b for b in all_blocks if b != test_b]

            n_b = feat[test_b].shape[0]
            n_prefix = (n_b * 2) // 3          # first 2/3 → extra train
            n_suffix = n_b - n_prefix           # last 1/3 → test

            extra_feat_b = feat[test_b][:n_prefix]
            extra_resp_b = resp[test_b][:n_prefix]
            feat_test_b  = feat[test_b][n_prefix:]
            resp_test_b  = resp[test_b][n_prefix:]

            print(f"\n  Fold: test=block_{test_b}  train={train_bs}+block_{test_b}[:2/3]"
                  f"  test_TRs={n_suffix}")
            pipeline, _ = _fit_hcp_single(feat, resp, train_bs, backend, n_features,
                                          extra_feat=extra_feat_b, extra_resp=extra_resp_b)

            Y_pred = backend.to_numpy(pipeline.predict(feat_test_b))
            Yt_z = np.nan_to_num(sp_zscore(resp_test_b, axis=0))
            Yp_z = np.nan_to_num(sp_zscore(Y_pred,      axis=0))
            block_corrs.append(np.nan_to_num((Yt_z * Yp_z).mean(0)))
            total_test_trs += n_suffix

        block_corrs = np.array(block_corrs)       # (8, n_voxels)
        corr_test = block_corrs.mean(0)
        best_alphas = np.zeros(n_voxels, dtype=np.float64)
        out_suffix = "lobo"
    else:
        train_blocks = HCP_TRAIN_BLOCKS
        test_block = HCP_TEST_BLOCK
        print(f"Train blocks: {train_blocks}  Test block: {test_block}")
        X_test = feat[test_block]
        Y_test = resp[test_block]
        print(f"  X_train: ({sum(feat[b].shape[0] for b in train_blocks)}, {n_features})")
        print(f"  X_test:  {X_test.shape}   Y_test: {Y_test.shape}")

        pipeline, best_alphas = _fit_hcp_single(feat, resp, train_blocks, backend, n_features)

        Y_pred_test = backend.to_numpy(pipeline.predict(X_test))
        Y_test_z = np.nan_to_num(sp_zscore(Y_test, axis=0))
        Y_pred_z = np.nan_to_num(sp_zscore(Y_pred_test, axis=0))
        corr_test = np.nan_to_num((Y_test_z * Y_pred_z).mean(0))
        out_suffix = "cv"

    n_test_trs = total_test_trs if lobo else Y_test.shape[0]
    print(f"\nTest corr ({n_test_trs} TRs): median={np.median(corr_test):.4f}  "
          f"mean={corr_test.mean():.4f}  "
          f"top-1k={np.sort(corr_test)[-1000:].mean():.4f}")

    out_dir = join(PROJECT_DIR, "outputs", "models",
                   f"ridge_hcp_{subject}_{embedding}_{out_suffix}")
    if args.suffix:
        out_dir += f"_{args.suffix}"
    os.makedirs(out_dir, exist_ok=True)

    save_dict = {
        "subject": subject, "embedding": embedding,
        "lobo": lobo, "delays": HCP_DELAYS, "alphas": HCP_ALPHAS,
        "best_alphas": best_alphas, "corr_test": corr_test,
        "n_voxels": n_voxels, "n_test_trs": n_test_trs,
    }
    if lobo:
        save_dict["block_corrs"] = block_corrs   # (8, n_voxels) per-block r
    joblib.dump(save_dict, join(out_dir, "scores.joblib"))
    print(f"Saved to {out_dir}")
    return corr_test


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="lebel",
                        choices=["lebel", "narratives", "hcp"],
                        help="Dataset (default: lebel)")
    parser.add_argument("--subject", type=str, default=None,
                        help="Subject ID (default: UTS03 for lebel, sub-001 for narratives)")
    parser.add_argument("--embedding", type=str, default="gemma2")
    parser.add_argument("--n_stories", type=int, default=None,
                        help="Number of training stories (default: all; lebel only)")
    parser.add_argument("--exclude_stories", nargs="*", default=None,
                        help="Stories to exclude from training (lebel only)")
    parser.add_argument("--val_stories", nargs="*", default=None,
                        help="Use these stories as single val fold (lebel only)")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Suffix for output directory name")
    parser.add_argument("--refit_on_all", action="store_true",
                        help="With --val_stories: refit the final model on training + validation stories "
                             "(the matched ridge baseline of the paper). Default: refit on training stories only.")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--lobo", action="store_true",
                        help="HCP only: 8-fold leave-one-block-out (all blocks as test)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "narratives":
        subject = args.subject or "sub-001"
        run_narratives(subject, args.embedding, args)
        return

    if args.dataset == "hcp":
        subject = args.subject or "100206"
        run_hcp(subject, args.embedding, args)
        return

    subject = args.subject or "UTS03"
    embedding = args.embedding

    all_stories = get_all_stories_for_subject(subject)
    train_stories = [s for s in all_stories if s != TEST_STORY]
    if args.exclude_stories:
        train_stories = [s for s in train_stories if s not in args.exclude_stories]
    if args.val_stories:
        # Remove val stories from training set
        val_stories = [s for s in args.val_stories if s in train_stories]
        train_stories = [s for s in train_stories if s not in val_stories]
    else:
        val_stories = None
    if args.n_stories is not None:
        train_stories = train_stories[:args.n_stories]

    print(f"Subject: {subject}")
    print(f"Embedding: {embedding}")
    print(f"Training stories: {len(train_stories)}")
    if val_stories:
        print(f"Val stories: {len(val_stories)} — {val_stories}")
    print(f"Test story: {TEST_STORY}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading features...")
    X_parts, run_lengths = [], []
    for story in train_stories:
        X_s = load_features([story], embedding=embedding)
        X_parts.append(X_s)
        run_lengths.append(X_s.shape[0])
    X_train = np.concatenate(X_parts, axis=0)
    X_test = load_features([TEST_STORY], embedding=embedding)
    run_onsets = np.concatenate([[0], np.cumsum(run_lengths)[:-1]])
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")

    # Load val data if specified
    if val_stories:
        X_val = np.concatenate([load_features([s], embedding=embedding)
                                for s in val_stories], axis=0)
        print(f"  X_val:   {X_val.shape}")

    print("\nLoading responses...")
    Y_train = np.nan_to_num(load_responses(train_stories, subject))
    Y_test = np.nan_to_num(load_responses([TEST_STORY], subject))
    n_voxels = Y_train.shape[1]
    n_samples = X_train.shape[0]
    n_delayed_features = X_train.shape[1] * len(DELAYS)
    print(f"  Y_train: {Y_train.shape}")
    print(f"  Y_test:  {Y_test.shape}")

    if val_stories:
        Y_val = np.nan_to_num(load_responses(val_stories, subject))
        print(f"  Y_val:   {Y_val.shape}")

    # ------------------------------------------------------------------
    # CV and backend setup
    # ------------------------------------------------------------------
    if val_stories:
        # Single train/val split: train on train_stories, validate on val_stories
        # Build a combined dataset with val appended, and a single CV split
        X_all = np.concatenate([X_train, X_val], axis=0)
        Y_all = np.concatenate([Y_train, Y_val], axis=0)
        n_train = X_train.shape[0]
        n_val = X_val.shape[0]
        train_idx = np.arange(n_train)
        val_idx = np.arange(n_train, n_train + n_val)
        cv_splits = [(train_idx, val_idx)]
        n_folds = 1
        # Replace X_train/Y_train with combined for fitting
        X_train_orig = X_train
        X_train = X_all
        Y_train = Y_all
        n_samples = X_train.shape[0]
        # Recompute run_lengths for combined
        run_lengths_val = [load_features([s], embedding=embedding).shape[0]
                           for s in val_stories]
        run_lengths = run_lengths + run_lengths_val
        run_onsets = np.concatenate([[0], np.cumsum(run_lengths)[:-1]])
        print(f"\nValidation: single fold (4 val stories, {n_val} TRs)")
    else:
        cv = check_cv(generate_leave_one_run_out(n_samples, run_onsets))
        cv_splits = list(cv.split(X_train))
        n_folds = len(cv_splits)
        print(f"\nCV: leave-one-story-out, {n_folds} folds")

    if args.device == "cuda" and torch.cuda.is_available():
        backend = set_backend("torch_cuda", on_error="warn")
    else:
        backend = set_backend("torch", on_error="warn")

    # Auto-select: kernel ridge when n_samples < n_features
    use_kernel = n_samples < n_delayed_features
    mode = "kernel" if use_kernel else "primal"
    print(f"\nMode: {mode} (n_samples={n_samples}, n_delayed_features={n_delayed_features})")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    solver_params = dict(n_targets_batch=N_TARGETS_BATCH)
    if use_kernel:
        solver_params["diagonalize_method"] = "svd"

    ModelCV = KernelRidgeCV if use_kernel else RidgeCV
    cv_arg = cv_splits if val_stories else cv
    pipeline = make_pipeline(
        StandardScaler(with_mean=False, with_std=False),
        Delayer(delays=DELAYS),
        ModelCV(alphas=ALPHAS, cv=cv_arg, solver_params=solver_params),
    )
    print(f"  Model: {ModelCV.__name__}")
    print(f"  Alphas: {len(ALPHAS)} values from {ALPHAS[0]:.0e} to {ALPHAS[-1]:.0e}")

    print("\nFitting model...")
    pipeline.fit(X_train, Y_train)
    print("Fitting complete.")

    # Extract best_alphas right after fit (needed for two-step refit below)
    best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    print("\nEvaluating on test set...")
    if val_stories and not args.refit_on_all:
        # Two-step refit: alphas selected on val fold, final model on train-only.
        # This ensures ridge and freeze-retrain use the same n training stories.
        preprocess_eval = make_pipeline(
            StandardScaler(with_mean=False, with_std=False),
            Delayer(delays=DELAYS),
        )
        X_tr_d = preprocess_eval.fit_transform(X_train_orig)
        X_te_d = preprocess_eval.transform(X_test)
        Y_tr_orig = Y_train[:n_train]  # original train rows from Y_all
        Y_pred_test = np.zeros((X_test.shape[0], n_voxels))
        for alpha_val in np.unique(best_alphas):
            vox_mask = best_alphas == alpha_val
            if use_kernel:
                K_tr = backend.asarray(X_tr_d @ X_tr_d.T)
                K_te = backend.asarray(X_te_d @ X_tr_d.T)
                m = KernelRidge(alpha=float(alpha_val), kernel="precomputed")
                m.fit(K_tr, backend.asarray(Y_tr_orig[:, vox_mask]))
                Y_pred_test[:, vox_mask] = backend.to_numpy(m.predict(K_te))
            else:
                m = Ridge(alpha=float(alpha_val),
                          solver_params=dict(n_targets_batch=N_TARGETS_BATCH))
                m.fit(backend.asarray(X_tr_d), backend.asarray(Y_tr_orig[:, vox_mask]))
                Y_pred_test[:, vox_mask] = backend.to_numpy(
                    m.predict(backend.asarray(X_te_d)))
    else:
        Y_pred_test = backend.to_numpy(pipeline.predict(X_test))
    Y_test_z = np.nan_to_num(sp_zscore(Y_test, axis=0))
    Y_pred_z = np.nan_to_num(sp_zscore(Y_pred_test, axis=0))
    corr_test = np.nan_to_num((Y_test_z * Y_pred_z).mean(0))

    print(f"  Test corr: mean={corr_test.mean():.4f}, max={corr_test.max():.4f}")
    for thresh in [0.1, 0.2, 0.3]:
        n = (corr_test > thresh).sum()
        print(f"  Voxels with r > {thresh}: {n:,} ({100*n/n_voxels:.1f}%)")

    # ------------------------------------------------------------------
    # Re-fit per CV fold for CV scores
    # ------------------------------------------------------------------
    print("\nComputing CV scores per fold...")

    preprocess = make_pipeline(
        StandardScaler(with_mean=False, with_std=False),
        Delayer(delays=DELAYS),
    )

    cv_corrs = []
    cv_r2s = []
    for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
        x_tr, x_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = Y_train[train_idx], Y_train[val_idx]

        x_tr_d = preprocess.fit_transform(x_tr)
        x_val_d = preprocess.transform(x_val)

        y_pred_val = np.zeros_like(y_val)
        unique_alphas = np.unique(best_alphas)

        for alpha_val in unique_alphas:
            vox_mask = best_alphas == alpha_val
            if not vox_mask.any():
                continue

            if use_kernel:
                K_tr = backend.asarray(x_tr_d @ x_tr_d.T)
                K_val = backend.asarray(x_val_d @ x_tr_d.T)
                model = KernelRidge(alpha=alpha_val, kernel="precomputed")
                model.fit(K_tr, backend.asarray(y_tr[:, vox_mask]))
                y_pred_val[:, vox_mask] = backend.to_numpy(model.predict(K_val))
            else:
                model = Ridge(
                    alpha=alpha_val,
                    solver_params=dict(n_targets_batch=N_TARGETS_BATCH),
                )
                model.fit(backend.asarray(x_tr_d), backend.asarray(y_tr[:, vox_mask]))
                y_pred_val[:, vox_mask] = backend.to_numpy(model.predict(backend.asarray(x_val_d)))

        y_val_z = np.nan_to_num(sp_zscore(y_val, axis=0))
        y_pred_z = np.nan_to_num(sp_zscore(y_pred_val, axis=0))
        fold_corr = np.nan_to_num((y_val_z * y_pred_z).mean(0))
        cv_corrs.append(fold_corr)

        resvar = (y_val - y_pred_val).var(0)
        yvar = y_val.var(0)
        fold_r2 = np.where(yvar > 0, 1 - resvar / yvar, 0.0)
        cv_r2s.append(fold_r2)

        print(f"  Fold {fold_i+1}/{n_folds}: "
              f"val corr mean={fold_corr.mean():.4f}, "
              f"val r² mean={fold_r2.mean():.4f}")

    cv_corrs = np.stack(cv_corrs)
    cv_r2s = np.stack(cv_r2s)
    mean_cv_corr = cv_corrs.mean(0)
    mean_cv_r2 = cv_r2s.mean(0)

    print(f"\nCV summary:")
    print(f"  Mean CV corr: {mean_cv_corr.mean():.4f} (max={mean_cv_corr.max():.4f})")
    print(f"  Mean CV R²:   {mean_cv_r2.mean():.4f} (max={mean_cv_r2.max():.4f})")
    for thresh in [0.1, 0.2, 0.3]:
        n = (mean_cv_corr > thresh).sum()
        print(f"  CV voxels with r > {thresh}: {n:,} ({100*n/n_voxels:.1f}%)")

    n_select = 30000
    if n_voxels >= n_select:
        test_top = set(np.argsort(corr_test)[-n_select:].tolist())
        cv_top = set(np.argsort(mean_cv_corr)[-n_select:].tolist())
        overlap = len(test_top & cv_top)
        print(f"\n  Top {n_select} overlap (test vs CV): {overlap} ({100*overlap/n_select:.1f}%)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    n_stories = len(train_stories)
    dir_name = f"ridge_{subject}_{n_stories}stories_{embedding}_cv"
    if args.suffix:
        dir_name += f"_{args.suffix}"
    out_dir = join(PROJECT_DIR, "outputs", "models", dir_name)
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "subject": subject,
        "embedding": embedding,
        "n_stories": n_stories,
        "train_stories": train_stories,
        "test_story": TEST_STORY,
        "delays": DELAYS,
        "alphas": ALPHAS,
        "best_alphas": best_alphas,
        "corr_test": corr_test,
        "cv_corrs": cv_corrs,
        "cv_r2s": cv_r2s,
        "mean_cv_corr": mean_cv_corr,
        "mean_cv_r2": mean_cv_r2,
    }

    out_fn = join(out_dir, "scores.joblib")
    joblib.dump(results, out_fn)
    print(f"\nSaved to {out_fn}")


if __name__ == "__main__":
    main()
