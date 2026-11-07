#!/usr/bin/env python
"""27_atom_yeo7_mapping.py — Map each SAE atom's brain decoder to Yeo 2011
7-network parcellation on fsaverage.

For each atom k, subject s, and Yeo network n, computes:

    cos(atom_k, network_n; s) = dot(D_s[mask_s, k], 1_n[mask_s])
                                 / (‖D_s[mask_s, k]‖ · ‖1_n[mask_s]‖)

    mask_s = selected-voxel coverage(s) ∩ cortex (yeo > 0)

Then averages cosine and Pearson r across subjects.

Pipeline (per subject):
    1) D_eff[v, k] = |Σ_d delay_scale[d,k] · W[v,d,k]|  (selected 30k voxels)
    2) Build selected-voxel coverage mask: project binary indicator through mapper
    3) Embed D_eff into full brain-mask volume via voxel_indices[s]
    4) Project to fsaverage via sparse mapper, clip to ≥ 0
    5) Compute cosine and Pearson within mask_s = coverage ∩ cortex
Aggregate across subjects: mean ± SEM.

Inputs:
    --model_dir   path to model dir containing model_last.pt + brain decoders
    --ckpt_name   defaults to model_last.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import join

import h5py
import nibabel as nib
import numpy as np
import scipy.sparse
from scipy.spatial import cKDTree
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

# Pycortex setup (needed for build_hemisphere_masks-style operations later;
# for now we only need the brain mask volume size which we get via h5 metadata)
PYCORTEX_DB = join(PROJECT_DIR, "external", "deep-fMRI-dataset",
                   "data", "ds003020", "derivative", "pycortex-db")
import cortex
cortex.options.config.set("basic", "filestore", PYCORTEX_DB)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

YEO_ANNOT_LH = os.path.join(os.environ.get("FREESURFER_HOME", ""), "subjects", "fsaverage", "label", "lh.Yeo2011_7Networks_N1000.annot")
YEO_ANNOT_RH = os.path.join(os.environ.get("FREESURFER_HOME", ""), "subjects", "fsaverage", "label", "rh.Yeo2011_7Networks_N1000.annot")

FSAV_SPHERE_LH = os.path.join(os.environ.get("FREESURFER_HOME", ""), "subjects", "fsaverage", "surf", "lh.sphere.reg")
FSAV_SPHERE_RH = os.path.join(os.environ.get("FREESURFER_HOME", ""), "subjects", "fsaverage", "surf", "rh.sphere.reg")

# Standard Yeo 2011 7-network names in the order they appear in the annot
# (label index 0 = medial wall / unlabeled; 1..7 = networks)
YEO7_NAMES = [
    "MedialWall",
    "Visual",
    "Somatomotor",
    "DorsalAttn",
    "VentralAttn",
    "Limbic",
    "Frontoparietal",
    "Default",
]


def load_yeo7_labels():
    """Concatenated [LH; RH] vertex label vector for fsaverage. 0..7."""
    labs_l, _, _ = nib.freesurfer.read_annot(YEO_ANNOT_LH)
    labs_r, _, _ = nib.freesurfer.read_annot(YEO_ANNOT_RH)
    return np.concatenate([labs_l, labs_r])  # (327684,)


def load_mapper(subject):
    """voxel→fsaverage sparse matrix (n_fsaverage_vertices, n_brain_voxels)."""
    path = join(PROJECT_DIR, "outputs", "mappers", f"{subject}_voxel_to_fsaverage.hdf")
    with h5py.File(path, "r") as f:
        data = f["data"][:]
        indices = f["indices"][:]
        indptr = f["indptr"][:]
        shape = tuple(f["shape"][:])
        n_voxels_total = int(f.attrs.get("n_voxels", shape[1]))
    M = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
    return M, n_voxels_total


def per_subject_atom_maps(sd, subject, n_atoms, n_delays):
    """D_eff of shape (n_voxels_selected, n_atoms).

    |Σ_d δ_d · W[v, d, k]|: signed sum across delays then magnitude.
    Note: per-voxel scale intentionally skipped (key name varies across
    model versions; omitting keeps both analysis scripts consistent).
    """
    W = sd[f"brain_decoders.{subject}.weight"].numpy()        # (V, n_delays * n_atoms)
    V = W.shape[0]
    Wr = W.reshape(V, n_delays, n_atoms)                      # (V, D, K)
    delay_scales = sd["delay_scales"].numpy()                 # (D, K)
    contrib = Wr * delay_scales[None, :, :]                   # (V, D, K)
    return np.abs(contrib.sum(axis=1))                        # (V, K)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--ckpt_name", default="model_last.pt")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--cache_dir", default=None,
                   help="Where to cache per-subject fsav_D (default: outputs/cache/yeo7_<base>)")
    p.add_argument("--n_perm", type=int, default=1000,
                   help="Number of permutations for null. 0 to skip.")
    p.add_argument("--null_method",
                   choices=["spin", "spin_fullcortex", "label_shuffle"],
                   default="spin",
                   help="Permutation null: 'spin' (within coverage mask, default), "
                        "'spin_fullcortex' (full-cortex spin, no mask restriction), "
                        "or 'label_shuffle' (naive vertex shuffle, legacy).")
    p.add_argument("--fdr_q", type=float, default=0.05,
                   help="BH FDR threshold (default 0.05).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def benjamini_hochberg(pvals, q):
    """Return boolean mask of FDR-significant tests at level q.
    Standard BH procedure on a flat array of p-values."""
    p = np.asarray(pvals).flatten()
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    if not below.any():
        cutoff = -1
    else:
        cutoff = np.where(below)[0].max()
    sig = np.zeros(n, dtype=bool)
    sig[order[: cutoff + 1]] = True
    return sig.reshape(np.shape(pvals))


def bh_qvalues(pvals):
    """Return BH q-values matching the input shape (no thresholding)."""
    p = np.asarray(pvals, dtype=float).flatten()
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    # enforce monotonicity (running min from the right)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(p)
    out[order] = q
    return out.reshape(np.shape(pvals))


def _random_rotation(rng):
    """Uniformly random SO(3) rotation matrix via QR of a Gaussian matrix."""
    M = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(M)
    Q *= np.sign(np.diag(R))    # make QR decomposition unique
    Q[:, 0] *= np.linalg.det(Q) # fold O(3) → SO(3): flip col 0 if det = -1
    return Q


def _load_sphere_coords():
    """Load and unit-normalize LH + RH fsaverage sphere coordinates.

    Returns two (163842, 3) float64 arrays on the unit sphere.
    """
    coords_lh, _ = nib.freesurfer.read_geometry(FSAV_SPHERE_LH)
    coords_rh, _ = nib.freesurfer.read_geometry(FSAV_SPHERE_RH)
    def unit(c):
        return c / np.linalg.norm(c.astype(np.float64), axis=1, keepdims=True).clip(1e-12)
    return unit(coords_lh), unit(coords_rh)


def make_spin_sampler(coords_lh, coords_rh, yeo):
    """Build spin infrastructure once; return a callable(rng) -> (327684,) perm.

    Each call draws one random rotation per hemisphere and returns a vertex
    permutation array such that yeo[perm] gives the spun labels.

    Medial-wall fix: if a cortical vertex's nearest spin-neighbor is medial
    wall (label 0), re-query the cortex-only tree to find the nearest
    non-medial-wall neighbor instead.  This keeps all originally-cortical
    vertices assigned to a valid network label on every permutation.
    """
    n_lh = coords_lh.shape[0]
    yeo_lh = yeo[:n_lh]
    yeo_rh = yeo[n_lh:]

    tree_lh = cKDTree(coords_lh)
    tree_rh = cKDTree(coords_rh)

    cortex_lh = np.where(yeo_lh > 0)[0]   # LH cortical vertex indices
    cortex_rh = np.where(yeo_rh > 0)[0]   # RH cortical vertex indices
    tree_lh_ctx = cKDTree(coords_lh[cortex_lh])
    tree_rh_ctx = cKDTree(coords_rh[cortex_rh])

    def sample(rng):
        rot_lh = coords_lh @ _random_rotation(rng).T
        rot_rh = coords_rh @ _random_rotation(rng).T

        _, idx_lh = tree_lh.query(rot_lh)
        _, idx_rh = tree_rh.query(rot_rh)

        # Remap medial-wall hits to nearest cortical vertex
        bad_lh = yeo_lh[idx_lh] == 0
        if bad_lh.any():
            _, nn = tree_lh_ctx.query(rot_lh[bad_lh])
            idx_lh[bad_lh] = cortex_lh[nn]

        bad_rh = yeo_rh[idx_rh] == 0
        if bad_rh.any():
            _, nn = tree_rh_ctx.query(rot_rh[bad_rh])
            idx_rh[bad_rh] = cortex_rh[nn]

        return np.concatenate([idx_lh, idx_rh + n_lh])

    return sample


def make_spin_sampler_within_mask(union_mask, coords_lh, coords_rh, yeo):
    """Spin within the union of all subjects' analysis masks.

    Restricts the spin to vertices that are actually covered by the selected
    voxels (union_mask = OR of mask_per_subj across subjects).  Network labels
    are only permuted within this footprint, so network vertex counts within
    any subject's mask can never drop to zero due to a spin landing outside
    the coverage region.  No medial-wall fix needed: union_mask is already
    coverage ∩ cortex, so it contains no medial-wall vertices.

    Returns a callable(rng) -> yeo_shuf (327684,) where labels outside
    union_mask are unchanged and labels inside are spin-permuted.
    """
    n_lh = coords_lh.shape[0]
    union_global = np.where(union_mask)[0]   # (U,) global fsav indices
    U = len(union_global)

    lh_in_u = union_global < n_lh           # bool (U,)
    lh_pos = np.where(lh_in_u)[0]           # positions in U for LH verts
    rh_pos = np.where(~lh_in_u)[0]          # positions in U for RH verts

    lh_coords = coords_lh[union_global[lh_pos]]              # (U_lh, 3)
    rh_coords = coords_rh[union_global[rh_pos] - n_lh]       # (U_rh, 3)

    tree_lh = cKDTree(lh_coords)
    tree_rh = cKDTree(rh_coords)

    def sample(rng):
        _, lh_nn = tree_lh.query(lh_coords @ _random_rotation(rng).T)
        _, rh_nn = tree_rh.query(rh_coords @ _random_rotation(rng).T)

        perm_u = np.empty(U, dtype=np.intp)
        perm_u[lh_pos] = lh_pos[lh_nn]
        perm_u[rh_pos] = rh_pos[rh_nn]

        yeo_shuf = yeo.copy()
        yeo_shuf[union_global] = yeo[union_global[perm_u]]
        return yeo_shuf

    return sample


def compute_selected_voxel_mask(mapper, vox_idx, n_full):
    """Boolean (n_fsav,) mask of fsaverage vertices covered by selected voxels.

    Projects a binary indicator of the selected voxels through the mapper.
    Independent of atom weights — same for all atoms within a subject.
    """
    indicator = np.zeros(n_full, dtype=np.float32)
    indicator[vox_idx] = 1.0
    return np.asarray(mapper @ indicator).ravel() > 0


def compute_subject_metrics(fsav_D, mask, yeo, n_networks):
    """Cosine and Pearson r between each atom and each Yeo7 network indicator.

    Both restricted to mask = selected-voxel coverage ∩ cortex.
    Standard cosine: dot(atom, 1_n) / (‖atom‖ · ‖1_n‖) where 1_n is the
    binary indicator for network n within mask.
    """
    D_m = fsav_D[mask]                                    # (M, K)
    yeo_m = yeo[mask]                                     # (M,)
    K = D_m.shape[1]

    atom_norms = np.linalg.norm(D_m, axis=0)              # (K,)
    D_centered = D_m - D_m.mean(axis=0)                   # (M, K)
    atom_norms_c = np.linalg.norm(D_centered, axis=0)     # (K,)

    cos_out = np.zeros((n_networks, K), dtype=np.float64)
    pearson_out = np.zeros((n_networks, K), dtype=np.float64)

    for net_id in range(1, n_networks + 1):
        in_net = (yeo_m == net_id)
        net_size = in_net.sum()
        if net_size == 0:
            continue

        # Cosine: dot(atom, 1_n) / (‖atom‖ · sqrt(net_size))
        net_dot = D_m[in_net].sum(axis=0)                 # (K,)
        cos_out[net_id - 1] = net_dot / np.maximum(
            atom_norms * np.sqrt(net_size), 1e-12)

        # Pearson: mean-centered
        indicator = in_net.astype(np.float64)
        ind_c = indicator - indicator.mean()
        ind_norm_c = np.linalg.norm(ind_c)
        if ind_norm_c < 1e-12:
            continue
        dots = ind_c @ D_centered                         # (K,)
        pearson_out[net_id - 1] = np.where(
            atom_norms_c > 1e-12,
            dots / (ind_norm_c * atom_norms_c),
            0.0,
        )

    return cos_out, pearson_out


def main():
    args = parse_args()
    base = os.path.basename(os.path.normpath(args.model_dir))
    out_dir = args.out_dir or join(PROJECT_DIR, "reports", "figures", f"yeo7_{base}")
    cache_dir = args.cache_dir or join(PROJECT_DIR, "outputs", "cache", f"yeo7_{base}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[load] {join(args.model_dir, args.ckpt_name)}")
    ckpt = torch.load(join(args.model_dir, args.ckpt_name),
                      map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    subjects = ckpt["subjects"]
    voxel_indices = ckpt["voxel_indices"]
    n_atoms, n_delays = sd["encoder.weight"].shape[0], sd["delay_scales"].shape[0]
    print(f"  n_atoms={n_atoms}, n_delays={n_delays}, subjects={subjects}")

    yeo = load_yeo7_labels()                # (327684,) labels 0..7
    n_networks = 7

    yeo_ref_mask = (yeo > 0)     # cortical vertices, excludes medial wall

    # Per-subject: compute true selected-voxel coverage mask, project atom
    # maps, then compute cosine and Pearson within (selected ∩ cortex).
    # No atom weights enter the mask definition.
    fsav_D_per_subj = {}
    mask_per_subj = {}
    per_subj_cos = {}
    per_subj_pearson = {}

    for s in subjects:
        print(f"\n[{s}]")
        D_eff = per_subject_atom_maps(sd, s, n_atoms, n_delays)
        M, n_total = load_mapper(s)
        vi = voxel_indices[s]
        if vi.max() >= n_total:
            print(f"  voxel_indices.max()={vi.max()} >= n_total={n_total} -> SKIPPING")
            continue

        # True coverage: which fsaverage vertices receive any projection from
        # the selected voxels (independent of atom weights)
        selected_mask_s = compute_selected_voxel_mask(M, vi, n_total)
        mask_s = selected_mask_s & yeo_ref_mask
        print(f"  selected-voxel coverage: {selected_mask_s.sum()} fsav vertices; "
              f"analysis mask (∩ cortex): {mask_s.sum()}")

        full_D = np.zeros((n_total, n_atoms), dtype=np.float64)
        full_D[vi] = D_eff
        fsav_D = np.clip(np.asarray(M @ full_D), 0, None)   # (n_fsav, K)
        np.save(join(cache_dir, f"fsav_D_{s}.npy"), fsav_D)

        fsav_D_per_subj[s] = fsav_D
        mask_per_subj[s] = mask_s

        cos_s, pearson_s = compute_subject_metrics(fsav_D, mask_s, yeo, n_networks)
        per_subj_cos[s] = cos_s
        per_subj_pearson[s] = pearson_s

    # Aggregate across subjects
    used_subjects = [s for s in subjects if s in per_subj_cos]
    cos_stack = np.stack([per_subj_cos[s] for s in used_subjects])        # (S, 7, K)
    pearson_stack = np.stack([per_subj_pearson[s] for s in used_subjects]) # (S, 7, K)
    cos_mean = cos_stack.mean(axis=0)        # (n_networks, n_atoms)
    pearson_mean = pearson_stack.mean(axis=0)
    cos_sem = cos_stack.std(axis=0, ddof=1) / np.sqrt(cos_stack.shape[0])

    # Permutation null: shuffle Yeo7 labels within all cortical vertices.
    # Per-subject masks are applied during metric computation; the shared
    # shuffle preserves the spatial covariance structure of the parcellation.
    pvals = np.full((n_networks, n_atoms), np.nan)
    qvals = np.full((n_networks, n_atoms), np.nan)
    null_mean = None
    null_std = None
    if args.n_perm > 0 and len(used_subjects) > 0:
        rng = np.random.default_rng(args.seed)
        null_dist = np.empty((args.n_perm, n_networks, n_atoms), dtype=np.float64)

        if args.null_method == "spin":
            print(f"\n[null] running {args.n_perm} spin permutations "
                  f"(within-mask, Alexander-Bloch et al. 2018) "
                  f"across {len(used_subjects)} subjects ...")
            coords_lh, coords_rh = _load_sphere_coords()
            union_mask = np.zeros(len(yeo), dtype=bool)
            for s in used_subjects:
                union_mask |= mask_per_subj[s]
            print(f"  Union mask: {union_mask.sum()} fsaverage vertices "
                  f"(LH: {(np.where(union_mask)[0] < coords_lh.shape[0]).sum()}, "
                  f"RH: {(np.where(union_mask)[0] >= coords_lh.shape[0]).sum()})")
            spin_sample = make_spin_sampler_within_mask(
                union_mask, coords_lh, coords_rh, yeo)
            for p_idx in range(args.n_perm):
                yeo_shuf = spin_sample(rng)
                perm_cos_per_subj = []
                for s in used_subjects:
                    cos_s_p, _ = compute_subject_metrics(
                        fsav_D_per_subj[s], mask_per_subj[s], yeo_shuf, n_networks)
                    perm_cos_per_subj.append(cos_s_p)
                null_dist[p_idx] = np.mean(perm_cos_per_subj, axis=0)
                if (p_idx + 1) % max(1, args.n_perm // 10) == 0:
                    print(f"  spin {p_idx + 1}/{args.n_perm}")

        elif args.null_method == "spin_fullcortex":
            print(f"\n[null] running {args.n_perm} full-cortex spin permutations "
                  f"(Alexander-Bloch et al. 2018, no mask restriction) "
                  f"across {len(used_subjects)} subjects ...")
            coords_lh, coords_rh = _load_sphere_coords()
            spin_sample = make_spin_sampler(coords_lh, coords_rh, yeo)
            for p_idx in range(args.n_perm):
                spin_perm = spin_sample(rng)
                yeo_shuf = yeo[spin_perm]
                perm_cos_per_subj = []
                for s in used_subjects:
                    cos_s_p, _ = compute_subject_metrics(
                        fsav_D_per_subj[s], mask_per_subj[s], yeo_shuf, n_networks)
                    perm_cos_per_subj.append(cos_s_p)
                null_dist[p_idx] = np.mean(perm_cos_per_subj, axis=0)
                if (p_idx + 1) % max(1, args.n_perm // 10) == 0:
                    print(f"  spin {p_idx + 1}/{args.n_perm}")

        else:  # label_shuffle (legacy)
            print(f"\n[null] running {args.n_perm} label-shuffle permutations "
                  f"across {len(used_subjects)} subjects ...")
            cortex_idx = np.where(yeo > 0)[0]
            cortex_labels = yeo[cortex_idx].copy()
            for p_idx in range(args.n_perm):
                yeo_shuf = yeo.copy()
                yeo_shuf[cortex_idx] = rng.permutation(cortex_labels)
                perm_cos_per_subj = []
                for s in used_subjects:
                    cos_s_p, _ = compute_subject_metrics(
                        fsav_D_per_subj[s], mask_per_subj[s], yeo_shuf, n_networks)
                    perm_cos_per_subj.append(cos_s_p)
                null_dist[p_idx] = np.mean(perm_cos_per_subj, axis=0)
                if (p_idx + 1) % max(1, args.n_perm // 10) == 0:
                    print(f"  perm {p_idx + 1}/{args.n_perm}")
        ge = (null_dist >= cos_mean[None, :, :]).sum(axis=0)
        pvals = (ge + 1) / (args.n_perm + 1)
        qvals = bh_qvalues(pvals)
        null_mean = null_dist.mean(axis=0)
        null_std = null_dist.std(axis=0, ddof=1)
        n_sig = int((qvals < args.fdr_q).sum())
        print(f"  FDR-significant cells (q < {args.fdr_q}): {n_sig} / {n_networks * n_atoms}")
        sig_atoms_per_net = (qvals < args.fdr_q).sum(axis=1)
        for i in range(n_networks):
            print(f"    {YEO7_NAMES[i+1]:<16s}: {sig_atoms_per_net[i]} atoms FDR-significant")

    # ---- Save main CSV: cosine + Pearson r + SEM + FDR ----
    csv_path = join(out_dir, "atom_yeo7_cos.csv")
    has_null = args.n_perm > 0 and not np.all(np.isnan(qvals))
    sig_mask = (qvals < args.fdr_q) if has_null else np.zeros_like(cos_mean, dtype=bool)
    with open(csv_path, "w") as f:
        cols = (["atom"]
                + [f"{YEO7_NAMES[i+1]}_cos" for i in range(n_networks)]
                + [f"{YEO7_NAMES[i+1]}_r" for i in range(n_networks)]
                + [f"{YEO7_NAMES[i+1]}_sem" for i in range(n_networks)]
                + ["dominant_net", "dominant_cos", "dominant_r", "dominant_q",
                   "n_sig_networks"])
        f.write(",".join(cols) + "\n")
        for k in range(n_atoms):
            dom = int(cos_mean[:, k].argmax())
            row = ([str(k)]
                   + [f"{cos_mean[i, k]:.4f}" for i in range(n_networks)]
                   + [f"{pearson_mean[i, k]:.4f}" for i in range(n_networks)]
                   + [f"{cos_sem[i, k]:.4f}" for i in range(n_networks)]
                   + [YEO7_NAMES[dom + 1], f"{cos_mean[dom, k]:.4f}",
                      f"{pearson_mean[dom, k]:.4f}",
                      f"{qvals[dom, k]:.4g}" if has_null else "",
                      str(int(sig_mask[:, k].sum()))])
            f.write(",".join(row) + "\n")
    print(f"\n[save] {csv_path}")

    # ---- Save p-values CSV (per atom × per network) ----
    if has_null:
        pval_path = join(out_dir, "atom_yeo7_pvalues.csv")
        with open(pval_path, "w") as f:
            cols = ["atom"]
            for i in range(n_networks):
                cols += [f"{YEO7_NAMES[i+1]}_p", f"{YEO7_NAMES[i+1]}_q",
                         f"{YEO7_NAMES[i+1]}_null_mean",
                         f"{YEO7_NAMES[i+1]}_null_std", f"{YEO7_NAMES[i+1]}_sig"]
            f.write(",".join(cols) + "\n")
            for k in range(n_atoms):
                row = [str(k)]
                for i in range(n_networks):
                    row += [
                        f"{pvals[i, k]:.4g}", f"{qvals[i, k]:.4g}",
                        f"{null_mean[i, k]:.4f}", f"{null_std[i, k]:.4f}",
                        "1" if sig_mask[i, k] else "0",
                    ]
                f.write(",".join(row) + "\n")
        print(f"[save] {pval_path}")

    # ---- Heatmap (atoms × networks) ----
    fig, ax = plt.subplots(figsize=(10, 12))
    order = np.lexsort((np.arange(n_atoms), -cos_mean.argmax(axis=0)))  # group by dominant net
    im = ax.imshow(cos_mean[:, order].T, aspect="auto", cmap="magma", vmin=0, vmax=0.5)
    ax.set_yticks(range(n_atoms))
    ax.set_yticklabels([f"a{order[i]}" for i in range(n_atoms)], fontsize=7)
    ax.set_xticks(range(n_networks))
    ax.set_xticklabels(YEO7_NAMES[1:], rotation=45, ha="right")
    plt.colorbar(im, ax=ax, label="cosine similarity (atom map vs network indicator)")
    ax.set_title(f"Atom × Yeo7 cosine similarity ({base[:60]}...)")
    fig.tight_layout()
    fig.savefig(join(out_dir, "atom_yeo7_heatmap.png"), dpi=120)
    plt.close(fig)
    print(f"[save] heatmap → {join(out_dir, 'atom_yeo7_heatmap.png')}")

    # ---- Top 5 atoms per network ----
    print("\n=== Top atoms per Yeo7 network (mean cos ± SEM) ===")
    for i in range(n_networks):
        net_name = YEO7_NAMES[i + 1]
        top = np.argsort(-cos_mean[i])[:5]
        line = f"  {net_name:<16s}: " + " | ".join(
            f"a{k:2d}({cos_mean[i, k]:.2f}±{cos_sem[i, k]:.2f})" for k in top
        )
        print(line)

    # ---- JSON summary ----
    summary = {
        "model_dir": args.model_dir,
        "checkpoint": args.ckpt_name,
        "metric": "cosine_similarity_vs_binary_network_indicator",
        "n_atoms": int(n_atoms),
        "n_subjects": int(cos_stack.shape[0]),
        "subjects_used": [s for s in subjects if s in per_subj_cos],
        "yeo7_names": YEO7_NAMES[1:],
        "atom_dominant_network": {
            f"a{k}": YEO7_NAMES[int(cos_mean[:, k].argmax()) + 1]
            for k in range(n_atoms)
        },
        "atom_dominant_cos": {
            f"a{k}": float(cos_mean[:, k].max()) for k in range(n_atoms)
        },
        "top_atoms_per_network": {
            YEO7_NAMES[i + 1]: [int(k) for k in np.argsort(-cos_mean[i])[:5]]
            for i in range(n_networks)
        },
    }
    if has_null:
        summary["null"] = {
            "n_perm": int(args.n_perm),
            "fdr_q_threshold": float(args.fdr_q),
            "n_significant_cells": int(sig_mask.sum()),
            "n_significant_per_network": {
                YEO7_NAMES[i + 1]: int(sig_mask[i].sum()) for i in range(n_networks)
            },
            "n_atoms_with_any_sig": int((sig_mask.sum(axis=0) > 0).sum()),
            "shuffle_scheme": {
                "spin": "spin_within_coverage_mask_alexander_bloch_2018",
                "spin_fullcortex": "spin_fullcortex_alexander_bloch_2018",
                "label_shuffle": "label_shuffle_cortex_only",
            }[args.null_method],
            "tail": "one_sided_right",
        }
    with open(join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] summary → {join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
