#!/usr/bin/env python
"""24_analyze_stability.py — Cross-seed stability analysis for SAE atoms.

Two complementary measures:
1. Hungarian atom matching: cosine similarity of encoder weights across seeds
2. CKA: representation-level similarity of activation matrices

Usage:
    uv run python scripts/24_analyze_stability.py \
        --model_dirs outputs/models/model_seed0 outputs/models/model_seed1 outputs/models/model_seed2
"""
import os as _os
# Checkpoint per run: model_last.pt (final epoch) unless CKPT is set (e.g. CKPT=model.pt for rows scored at the best epoch).
CKPT = _os.environ.get("CKPT", "model_last.pt")
import argparse
import os
import sys
from itertools import combinations
from os.path import join

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import zscore as sp_zscore

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from sparsebrain.data import load_features, load_responses, get_all_stories_for_subject

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Hungarian atom matching
# -------------------------------------------------------------------------

def cosine_similarity_matrix(W1, W2):
    """Cosine similarity between rows of W1 and W2.

    Parameters
    ----------
    W1, W2 : ndarray, shape (n_atoms, n_features)

    Returns
    -------
    sim : ndarray, shape (n_atoms, n_atoms)
    """
    W1_norm = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-8)
    W2_norm = W2 / (np.linalg.norm(W2, axis=1, keepdims=True) + 1e-8)
    return W1_norm @ W2_norm.T


def hungarian_match(W1, W2):
    """Match atoms between two models using Hungarian algorithm.

    Parameters
    ----------
    W1, W2 : ndarray, shape (n_atoms, n_features)
        Encoder weight matrices.

    Returns
    -------
    matching : list of (i, j, cosine_sim)
        Matched atom pairs and their cosine similarity.
    cost_matrix : ndarray
        Full cosine similarity matrix.
    """
    sim = cosine_similarity_matrix(W1, W2)
    # Hungarian minimizes cost, so negate similarity
    row_ind, col_ind = linear_sum_assignment(-sim)
    matching = [(i, j, sim[i, j]) for i, j in zip(row_ind, col_ind)]
    return matching, sim


# -------------------------------------------------------------------------
# CKA (Centered Kernel Alignment)
# -------------------------------------------------------------------------

def center_kernel(K):
    """Double-center a kernel matrix (efficient, no H matrix)."""
    row_mean = K.mean(axis=0, keepdims=True)
    col_mean = K.mean(axis=1, keepdims=True)
    grand_mean = K.mean()
    return K - row_mean - col_mean + grand_mean


def linear_cka(X1, X2):
    """Compute linear CKA between two activation matrices.

    Parameters
    ----------
    X1, X2 : ndarray, shape (n_samples, n_atoms)

    Returns
    -------
    cka : float
        CKA value in [0, 1].
    """
    K1 = X1 @ X1.T
    K2 = X2 @ X2.T
    K1c = center_kernel(K1)
    K2c = center_kernel(K2)

    hsic_12 = np.sum(K1c * K2c)
    hsic_11 = np.sum(K1c * K1c)
    hsic_22 = np.sum(K2c * K2c)

    if hsic_11 < 1e-10 or hsic_22 < 1e-10:
        return 0.0
    return hsic_12 / np.sqrt(hsic_11 * hsic_22)


# -------------------------------------------------------------------------
# Load model and compute activations
# -------------------------------------------------------------------------

def load_encoder_weights(model_dir):
    """Load encoder weight matrix from checkpoint."""
    ckpt_path = join(model_dir, CKPT)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    W = sd["encoder.weight"].numpy()  # (n_atoms, n_features)
    args = ckpt.get("args", {})
    return W, args, ckpt


def compute_activations(model_dir, X, input_rms=False):
    """Compute sparse activations for given features.

    Parameters
    ----------
    model_dir : str
        Path to model directory.
    X : ndarray, shape (n_samples, n_features)
        Semantic features.
    input_rms : bool
        Whether to apply input RMS normalization.

    Returns
    -------
    S : ndarray, shape (n_samples, n_atoms)
        Sparse activation matrix.
    """
    from sparsebrain.models.coupled_sae import MultiSubjectCoupledSAE

    ckpt_path = join(model_dir, CKPT)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    args = ckpt.get("args", {})

    n_atoms = sd["encoder.weight"].shape[0]
    n_sem = sd["encoder.weight"].shape[1]
    k = args.get("k", 5) if isinstance(args, dict) else getattr(args, "k", 5)
    n_delays = sd["delay_scales"].shape[0]
    has_encoder_bias = "encoder_bias" in sd
    has_decoder_sem = "decoder_sem.weight" in sd

    # Build model (only need encoder, but full model for compatibility)
    subjects = ckpt.get("subjects", ["UTS03"])
    n_vox = ckpt.get("n_voxels_per_subject", {s: 100 for s in subjects})

    model = MultiSubjectCoupledSAE(
        n_sem=n_sem,
        n_voxels_per_subject=n_vox,
        n_atoms=n_atoms,
        k=k,
        n_delays=n_delays,
        tie_semantic_decoder=not has_decoder_sem,
        encoder_bias=has_encoder_bias,
        # use the encoder the model was trained with (matching pursuit for the MP-SAE row); without
        # this the MP-SAE codes were computed with a linear TopK encoder
        **({"encoder_mode": "mp", "mp_steps": (args.get("mp_steps") if isinstance(args, dict) else getattr(args, "mp_steps", None)) or k}
           if ((args.get("encoder_mode") if isinstance(args, dict) else getattr(args, "encoder_mode", "linear")) == "mp") else {}),
    )
    model.load_state_dict(sd, strict=False)
    model.eval()

    X_input = X.copy()
    if input_rms:
        rms = np.sqrt((X_input ** 2).mean())
        X_input = X_input / rms

    with torch.no_grad():
        x_t = torch.tensor(X_input, dtype=torch.float32)
        s = model.encode(x_t).numpy()

    return s


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dirs", nargs="+", required=True,
                        help="Paths to model directories (2 or more)")
    parser.add_argument("--n_stories", type=int, default=10,
                        help="Number of stories for activation computation")
    parser.add_argument("--subject", type=str, default="UTS03",
                        help="Subject for activation computation")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: reports/figures/stability/)")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dirs = args.model_dirs
    n_models = len(model_dirs)
    assert n_models >= 2, "Need at least 2 models for comparison"

    out_dir = args.out_dir or join(PROJECT_DIR, "reports", "figures", "stability")
    os.makedirs(out_dir, exist_ok=True)

    # Short labels
    labels = [os.path.basename(d).split("_s")[-1].split("_")[0]
              if "_s" in os.path.basename(d) else f"m{i}"
              for i, d in enumerate(model_dirs)]
    labels = [f"seed{l}" if l.isdigit() else l for l in labels]

    print(f"Comparing {n_models} models: {labels}")

    # ------------------------------------------------------------------
    # Load encoder weights
    # ------------------------------------------------------------------
    print("\nLoading encoder weights...")
    weights = []
    all_args = []
    for d in model_dirs:
        W, a, _ = load_encoder_weights(d)
        weights.append(W)
        all_args.append(a)
        print(f"  {os.path.basename(d)}: {W.shape}")

    n_atoms = weights[0].shape[0]

    # ------------------------------------------------------------------
    # Hungarian matching
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("HUNGARIAN ATOM MATCHING")
    print("=" * 60)

    pairs = list(combinations(range(n_models), 2))
    all_matchings = {}
    all_sim_matrices = {}

    for i, j in pairs:
        matching, sim = hungarian_match(weights[i], weights[j])
        all_matchings[(i, j)] = matching
        all_sim_matrices[(i, j)] = sim

        sims = [m[2] for m in matching]
        print(f"\n{labels[i]} vs {labels[j]}:")
        print(f"  Mean matched cosine: {np.mean(sims):.4f} ± {np.std(sims):.4f}")
        print(f"  Median: {np.median(sims):.4f}")
        print(f"  Min: {np.min(sims):.4f}, Max: {np.max(sims):.4f}")
        for thresh in [0.5, 0.7, 0.8, 0.9]:
            n_above = sum(1 for s in sims if s > thresh)
            print(f"  Atoms with cosine > {thresh}: {n_above}/{n_atoms} ({100*n_above/n_atoms:.0f}%)")

    # ------------------------------------------------------------------
    # CKA
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CKA (Centered Kernel Alignment)")
    print("=" * 60)

    # Load features for activation computation
    stories = get_all_stories_for_subject(args.subject)[:args.n_stories]
    print(f"\nLoading features ({len(stories)} stories, {args.subject})...")
    X = load_features(stories, embedding="gemma2")
    print(f"  X: {X.shape}")

    # NOTE: input_rms is read from the first model's args and applied to all
    # models in this run. If comparing variants with different input_rms_normalize
    # settings (e.g. mixing old and new configs), activations will be computed
    # inconsistently. All current paper models use input_rms_normalize=True so
    # this is safe in practice, but pass --model_dirs within one variant group.
    input_rms = all_args[0].get("input_rms_normalize", False) \
        if isinstance(all_args[0], dict) else getattr(all_args[0], "input_rms_normalize", False)

    print(f"Computing activations (input_rms={input_rms})...")
    activations = []
    for i, d in enumerate(model_dirs):
        S = compute_activations(d, X, input_rms=input_rms)
        activations.append(S)
        l0 = (S > 0).sum(1).mean()
        n_dead = (S.sum(0) == 0).sum()
        print(f"  {labels[i]}: S shape={S.shape}, mean L0={l0:.1f}, dead atoms={n_dead}/{n_atoms}")

    # Compute CKA for all pairs
    cka_matrix = np.eye(n_models)
    for i, j in pairs:
        cka = linear_cka(activations[i], activations[j])
        cka_matrix[i, j] = cka
        cka_matrix[j, i] = cka
        print(f"\n{labels[i]} vs {labels[j]}: CKA = {cka:.4f}")

    print(f"\nMean pairwise CKA: {np.mean([cka_matrix[i,j] for i,j in pairs]):.4f}")

    # ------------------------------------------------------------------
    # Δr rank consistency
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("IMPORTANCE RANK CONSISTENCY")
    print("=" * 60)

    # For each model, compute per-atom activation frequency as a proxy for importance
    for i, j in pairs:
        matching = all_matchings[(i, j)]
        freq_i = (activations[i] > 0).mean(0)
        freq_j = (activations[j] > 0).mean(0)

        # Rank correlation of matched atoms' frequencies
        matched_freq_i = [freq_i[m[0]] for m in matching]
        matched_freq_j = [freq_j[m[1]] for m in matching]
        from scipy.stats import spearmanr
        rho, pval = spearmanr(matched_freq_i, matched_freq_j)
        print(f"{labels[i]} vs {labels[j]}: frequency rank corr = {rho:.3f} (p={pval:.1e})")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")

    # Plot 1: Cosine similarity matrices
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]
    for ax, (i, j) in zip(axes, pairs):
        sim = all_sim_matrices[(i, j)]
        # Reorder by Hungarian matching
        matching = all_matchings[(i, j)]
        order_j = [m[1] for m in sorted(matching, key=lambda m: m[0])]
        sim_ordered = sim[:, order_j]

        im = ax.imshow(sim_ordered, vmin=-0.2, vmax=1.0, cmap="RdBu_r", aspect="auto")
        ax.set_title(f"{labels[i]} vs {labels[j]}\nmean={np.mean([m[2] for m in matching]):.3f}")
        ax.set_xlabel(f"{labels[j]} atoms (matched)")
        ax.set_ylabel(f"{labels[i]} atoms")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(join(out_dir, "cosine_similarity_matrices.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved cosine_similarity_matrices.png")
    plt.close(fig)

    # Plot 2: Distribution of matched cosine similarities
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, j in pairs:
        sims = [m[2] for m in all_matchings[(i, j)]]
        ax.hist(sims, bins=30, alpha=0.5, label=f"{labels[i]} vs {labels[j]} (mean={np.mean(sims):.3f})")
    ax.set_xlabel("Cosine similarity (matched atoms)")
    ax.set_ylabel("Count")
    ax.set_title("Hungarian-matched atom similarity distribution")
    ax.legend()
    ax.axvline(0.8, color="red", linestyle="--", alpha=0.5, label="threshold=0.8")
    fig.savefig(join(out_dir, "matched_cosine_distribution.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved matched_cosine_distribution.png")
    plt.close(fig)

    # Plot 3: CKA matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cka_matrix, vmin=0.5, vmax=1.0, cmap="YlOrRd")
    for ii in range(n_models):
        for jj in range(n_models):
            ax.text(jj, ii, f"{cka_matrix[ii, jj]:.3f}", ha="center", va="center",
                    fontsize=12, color="white" if cka_matrix[ii, jj] > 0.85 else "black")
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("CKA (Centered Kernel Alignment)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(join(out_dir, "cka_matrix.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved cka_matrix.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    mean_cosine = np.mean([np.mean([m[2] for m in all_matchings[p]]) for p in pairs])
    mean_cka = np.mean([cka_matrix[i, j] for i, j in pairs])
    print(f"Mean matched cosine similarity: {mean_cosine:.4f}")
    print(f"Mean CKA: {mean_cka:.4f}")
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
