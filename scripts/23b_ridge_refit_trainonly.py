#!/usr/bin/env python
"""23b_ridge_refit_trainonly.py — Refit ridge on train-only using saved best_alphas.

Loads best_alphas from an existing matched scores.joblib, refits KernelRidge on
train stories only (no val), and evaluates on the test story.

This gives the correct two-step val-matched baseline: alpha selected on val fold,
final model trained on train-only — without repeating the expensive KernelRidgeCV.

Usage:
    uv run python scripts/23b_ridge_refit_trainonly.py --subject UTS01
    uv run python scripts/23b_ridge_refit_trainonly.py --subject UTS01 --device cuda
"""
import argparse
import os
import sys
from os.path import join

import joblib
import numpy as np
from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore as sp_zscore
from voxelwise_tutorials.delayer import Delayer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from sparsebrain import paths, load_features, load_responses
from sparsebrain.data import get_all_stories_for_subject

TEST_STORY = "wheretheressmoke"
DELAYS = [1, 2, 3, 4]
N_TARGETS_BATCH = 40000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True)
    p.add_argument("--embedding", default="gemma2")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--sae_model_dir", default=None,
                   help="Path to SAE model dir (model.pt). When set, also saves "
                        "delay-summed primal weights for the SAE-selected voxels.")
    p.add_argument("--weights_only", action="store_true",
                   help="Skip full-brain scoring; only save SAE-selected voxel weights. "
                        "Requires --sae_model_dir.")
    return p.parse_args()


def main():
    args = parse_args()
    subject = args.subject
    embedding = args.embedding

    import torch
    if args.device == "cuda" and torch.cuda.is_available():
        backend = set_backend("torch_cuda", on_error="warn")
    else:
        backend = set_backend("torch", on_error="warn")

    # Find the existing matched scores.joblib
    model_pattern = f"outputs/models/ridge_{subject}_*stories_{embedding}_cv_matched"
    import glob
    candidates = glob.glob(join(PROJECT_DIR, model_pattern))
    if not candidates:
        raise FileNotFoundError(f"No matched ridge model found for {subject}")
    src_dir = sorted(candidates)[-1]
    src_path = join(src_dir, "scores.joblib")
    print(f"Loading best_alphas from: {os.path.basename(src_dir)}")

    saved = joblib.load(src_path)
    best_alphas = saved["best_alphas"]
    train_stories = saved["train_stories"]
    n_stories = saved["n_stories"]
    print(f"  {len(train_stories)} train stories, {len(best_alphas)} voxels")
    print(f"  Unique alphas: {len(np.unique(best_alphas))}")

    # Load train features/responses (train only, no val)
    print("\nLoading train data...")
    X_parts = [load_features([s], embedding=embedding) for s in train_stories]
    X_train = np.concatenate(X_parts, axis=0)
    Y_train = np.nan_to_num(load_responses(train_stories, subject))
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")

    n_voxels = Y_train.shape[1]

    # Preprocess
    preprocess = make_pipeline(
        StandardScaler(with_mean=False, with_std=False),
        Delayer(delays=DELAYS),
    )
    X_tr_d = preprocess.fit_transform(X_train).astype(np.float32)

    print("\nBuilding train gram matrix K_tr...")
    K_tr = backend.asarray(X_tr_d) @ backend.asarray(X_tr_d).T

    out_dir = join(PROJECT_DIR, "outputs", "models",
                   f"ridge_{subject}_{n_stories}stories_{embedding}_cv_matched2")
    os.makedirs(out_dir, exist_ok=True)

    if not args.weights_only:
        # Load test
        print("Loading test data...")
        X_test = load_features([TEST_STORY], embedding=embedding)
        Y_test = np.nan_to_num(load_responses([TEST_STORY], subject))
        print(f"  X_test: {X_test.shape}, Y_test: {Y_test.shape}")
        X_te_d = preprocess.transform(X_test).astype(np.float32)
        K_te = backend.asarray(X_te_d) @ backend.asarray(X_tr_d).T

        # Refit on train only, per unique alpha
        print("\nRefitting on train-only per unique alpha (all voxels)...")
        Y_pred_test = np.zeros((X_test.shape[0], n_voxels), dtype=np.float32)
        for alpha_val in np.unique(best_alphas):
            vox_mask = best_alphas == alpha_val
            m = KernelRidge(alpha=float(alpha_val), kernel="precomputed")
            m.fit(K_tr, backend.asarray(Y_train[:, vox_mask].astype(np.float32)))
            Y_pred_test[:, vox_mask] = backend.to_numpy(m.predict(K_te))

        # Score
        Y_test_z = np.nan_to_num(sp_zscore(Y_test, axis=0))
        Y_pred_z = np.nan_to_num(sp_zscore(Y_pred_test, axis=0))
        corr_test = np.nan_to_num((Y_test_z * Y_pred_z).mean(0))
        print(f"\nTest corr: mean={corr_test.mean():.4f}, max={corr_test.max():.4f}")
        for thresh in [0.1, 0.2, 0.3]:
            n = (corr_test > thresh).sum()
            print(f"  r > {thresh}: {n:,} ({100*n/n_voxels:.1f}%)")

        results = {**saved, "corr_test": corr_test,
                   "note": "refit on train-only using best_alphas from matched run"}
        joblib.dump(results, join(out_dir, "scores.joblib"))
        print(f"\nSaved to {out_dir}/scores.joblib")

    # Optional: save delay-summed primal weights for SAE atom projection analysis
    if args.sae_model_dir:
        import torch as _torch
        sae_path = args.sae_model_dir
        if not os.path.isabs(sae_path):
            sae_path = join(PROJECT_DIR, sae_path)
        print(f"\nSaving primal weights for SAE projection (--sae_model_dir)...")
        ckpt_sae = _torch.load(join(sae_path, "model.pt"),
                               map_location="cpu", weights_only=False)
        voxel_indices = np.array(ckpt_sae["voxel_indices"][subject])
        n_sel = len(voxel_indices)
        F = X_train.shape[1]                            # 2304 for gemma2
        print(f"  SAE-selected voxels for {subject}: {n_sel}")

        best_alphas_sel = best_alphas[voxel_indices]    # (n_sel,)
        Y_sel = Y_train[:, voxel_indices].astype(np.float32)

        # Sum delay blocks: X_sum[t] = sum_d X_tr_d[t, d*F:(d+1)*F]
        n_train = X_train.shape[0]
        X_sum = np.zeros((n_train, F), dtype=np.float32)
        for d in range(len(DELAYS)):
            X_sum += X_tr_d[:, d * F:(d + 1) * F]

        W_sum = np.zeros((n_sel, F), dtype=np.float32)
        for alpha_val in np.unique(best_alphas_sel):
            vox_mask = best_alphas_sel == alpha_val
            m_sel = KernelRidge(alpha=float(alpha_val), kernel="precomputed")
            m_sel.fit(K_tr, backend.asarray(Y_sel[:, vox_mask]))
            # dual_coef_ may be CPU tensor; bring to numpy then matmul on CPU
            dual = backend.to_numpy(m_sel.dual_coef_).astype(np.float32)  # (n_train, n_vox)
            W_sum[vox_mask] = (X_sum.T @ dual).T                           # (n_vox, F)
            print(f"  alpha={alpha_val:.0f}  {vox_mask.sum()} voxels", flush=True)

        np.save(join(out_dir, "primal_weights_sum.npy"), W_sum)       # (n_sel, F)
        np.save(join(out_dir, "voxel_indices_sel.npy"), voxel_indices) # (n_sel,)
        print(f"  Saved primal_weights_sum.npy {W_sum.shape}")


if __name__ == "__main__":
    main()
