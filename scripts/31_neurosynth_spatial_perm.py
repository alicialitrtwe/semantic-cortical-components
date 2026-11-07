#!/usr/bin/env python
"""31_neurosynth_spatial_perm.py — Neurosynth validation: spatial permutation test.

Yeo7-style spatial null: for each (atom, NS-term) cell in the n_atoms × n_terms
brain-correlation matrix, test significance by shuffling NS map vertex positions
within the coverage mask — the same null design as script 27 uses for Yeo7 networks.

  Observed:  r[k,j] = Pearson(atom_k_mean_map, NS_j_map)  within shared mask
  Null:      r_perm[k,j] = Pearson(atom_k_mean_map, NS_j_map[perm_idx])
             one random vertex permutation per iteration, same applied to all terms

FDR (BH) across all n_atoms × n_terms cells.
Per-atom summary: best text-selected term, its p/q, all FDR-significant terms.

Usage:
    uv run python scripts/31_neurosynth_spatial_perm.py \\
        --model_dir outputs/models/best_model \\
        --n_perm 10000
"""
import argparse
import importlib
import json
import os
import sys
from os.path import join

import numpy as np
import nibabel as nib
import torch
from scipy.spatial import cKDTree
from scipy.stats import false_discovery_control

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, join(PROJECT_DIR, "scripts"))

# Reuse loaders from script 30
_ns30 = importlib.import_module("30_neurosynth_validation")
load_terms = _ns30.load_terms
generate_neurosynth_maps = _ns30.generate_neurosynth_maps
get_atom_brain_maps_per_subject = _ns30.get_atom_brain_maps_per_subject
get_atom_descriptions = _ns30.get_atom_descriptions
match_atoms_to_terms = _ns30.match_atoms_to_terms
N_VOXELS_FULL = _ns30.N_VOXELS_FULL
MIN_STUDIES_DEFAULT = _ns30.MIN_STUDIES_DEFAULT


FSAV_SPHERE_LH = os.path.join(os.environ.get("FREESURFER_HOME", ""), "subjects", "fsaverage", "surf", "lh.sphere.reg")
FSAV_SPHERE_RH = os.path.join(os.environ.get("FREESURFER_HOME", ""), "subjects", "fsaverage", "surf", "rh.sphere.reg")
N_LH_FSAV = 163842


def _load_sphere_coords():
    coords_lh, _ = nib.freesurfer.read_geometry(FSAV_SPHERE_LH)
    coords_rh, _ = nib.freesurfer.read_geometry(FSAV_SPHERE_RH)
    def unit(c):
        return c / np.linalg.norm(c.astype(np.float64), axis=1, keepdims=True).clip(1e-12)
    return unit(coords_lh), unit(coords_rh)


def _random_rotation(rng):
    """Uniformly random SO(3) rotation matrix via QR of a Gaussian matrix."""
    M = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(M)
    Q *= np.sign(np.diag(R))
    Q[:, 0] *= np.linalg.det(Q)  # fold O(3) → SO(3)
    return Q


def make_ns_spin_sampler(mask, coords_lh, coords_rh):
    """Build a spin sampler for the Neurosynth within-mask spatial permutation.

    Splits the shared coverage mask into LH and RH vertices, builds one
    kD-tree per hemisphere from the masked vertices' sphere coordinates,
    and returns a callable(rng) -> perm_idx that maps each mask position
    to a spatially-rotated neighbor within the same hemisphere's mask.

    perm_idx is a (V,) array suitable for ns_n[:, perm_idx] column shuffle.
    No medial-wall fix needed: the mask is defined over real cortex only,
    so every vertex's NN stays within cortex.
    """
    mask_global = np.where(mask)[0]   # global fsav indices, sorted ascending
    V = len(mask_global)

    lh_pos = np.where(mask_global < N_LH_FSAV)[0]    # positions in V for LH verts
    rh_pos = np.where(mask_global >= N_LH_FSAV)[0]   # positions in V for RH verts

    lh_coords = coords_lh[mask_global[lh_pos]]              # (V_lh, 3)
    rh_coords = coords_rh[mask_global[rh_pos] - N_LH_FSAV]  # (V_rh, 3)

    tree_lh = cKDTree(lh_coords)
    tree_rh = cKDTree(rh_coords)

    def sample(rng):
        _, lh_nn = tree_lh.query(lh_coords @ _random_rotation(rng).T)
        _, rh_nn = tree_rh.query(rh_coords @ _random_rotation(rng).T)
        perm_idx = np.empty(V, dtype=np.intp)
        perm_idx[lh_pos] = lh_pos[lh_nn]
        perm_idx[rh_pos] = rh_pos[rh_nn]
        return perm_idx

    return sample


# ── Core computation ────────────────────────────────────────────────────────

def build_masked_matrices(atom_fsav_per_subj, ns_maps, available_terms,
                           selected_masks):
    """Build atom mean map and NS map matrices restricted to shared coverage mask.

    Shared mask = union of subject coverage masks ∩ NS union coverage.
    Atom matrix is the mean across subjects.

    Returns:
        atom_mat:  (n_atoms, V) float32 — mean atom maps within mask
        ns_mat:    (n_terms, V) float32 — NS maps within mask
        mask:      (327684,) bool — shared coverage mask
    """
    subjects = list(atom_fsav_per_subj.keys())
    n_fsav = next(iter(atom_fsav_per_subj.values())).shape[0]

    # NS union mask
    ns_union = np.zeros(n_fsav, dtype=bool)
    for t in available_terms:
        ns_union |= (ns_maps[t] > 0)

    # Subject union coverage mask
    subj_union = np.zeros(n_fsav, dtype=bool)
    for s in subjects:
        subj_union |= selected_masks[s]

    mask = ns_union & subj_union
    V = mask.sum()
    print(f"  Shared mask: {V} fsaverage vertices "
          f"(NS union: {ns_union.sum()}, subject union: {subj_union.sum()})")

    # Atom mean map: average across subjects, clip to >= 0
    n_atoms = next(iter(atom_fsav_per_subj.values())).shape[1]
    atom_stack = np.stack(
        [atom_fsav_per_subj[s][mask, :] for s in subjects], axis=0
    )  # (n_subjects, V, n_atoms)
    atom_mat = atom_stack.mean(axis=0).T.astype(np.float32)  # (n_atoms, V)
    atom_mat = np.clip(atom_mat, 0, None)

    # NS maps: positive-clipped, restricted to mask
    ns_mat = np.stack(
        [np.clip(ns_maps[t][mask], 0, None) for t in available_terms], axis=0
    ).astype(np.float32)  # (n_terms, V)

    return atom_mat, ns_mat, mask



def spatial_permutation_test(atom_mat, ns_mat, n_perm, seed=42,
                             spin_sampler=None):
    """Spatial permutation test: shuffle NS map vertex positions within mask.

    For each permutation p:
      - perm_idx = spin-based permutation (default) or random permutation of {0..V-1}
      - null_pearson[k, j] = Pearson(atom_mat[k], ns_mat[j, perm_idx])
    Same perm_idx applied to all NS terms per iteration.

    spin_sampler: callable(rng) -> (V,) perm_idx, or None for random shuffle.

    Returns:
        obs:   (n_atoms, n_terms) float64 — observed Pearson r
        pvals: (n_atoms, n_terms) float64 — empirical p-values
    """
    rng = np.random.default_rng(seed)
    n_atoms, V = atom_mat.shape
    n_terms = ns_mat.shape[0]

    atom_c = atom_mat - atom_mat.mean(1, keepdims=True)
    atom_n = (atom_c / (np.linalg.norm(atom_c, axis=1, keepdims=True) + 1e-12)
              ).astype(np.float32)

    ns_c = ns_mat - ns_mat.mean(1, keepdims=True)
    ns_n = (ns_c / (np.linalg.norm(ns_c, axis=1, keepdims=True) + 1e-12)
            ).astype(np.float32)

    obs = (atom_n @ ns_n.T).astype(np.float64)
    count = np.zeros((n_atoms, n_terms), dtype=np.int32)

    null_type = "spin" if spin_sampler is not None else "random shuffle"
    print(f"  Permuting ({null_type}): {n_perm} iterations × "
          f"{n_atoms} atoms × {n_terms} terms × {V} vertices")

    for p in range(n_perm):
        perm_idx = spin_sampler(rng) if spin_sampler is not None else rng.permutation(V)
        null_p = atom_n @ ns_n[:, perm_idx].T
        count += (null_p >= obs)
        if (p + 1) % 1000 == 0:
            print(f"    {p+1}/{n_perm} done")

    pvals = count.astype(np.float64) / n_perm
    return obs, pvals


def fdr_correct(pvals, n_perm, q_threshold=0.05):
    """BH FDR correction over all cells. Returns q-value matrix."""
    shape = pvals.shape
    flat = pvals.flatten()
    # Clamp to (0, 1] — empirical p from finite perms can be exactly 0
    flat_clamped = np.maximum(flat, 1.0 / (n_perm + 1))
    q_flat = false_discovery_control(flat_clamped, method="bh")
    return q_flat.reshape(shape)


# ── Reporting ───────────────────────────────────────────────────────────────

def report_results(obs, pvals, qvals, available_terms, atom_descriptions,
                   sims, n_perm, q_threshold=0.05):
    """Print per-atom and per-network summary."""
    n_atoms, n_terms = obs.shape

    n_sig = (qvals < q_threshold).sum()
    print(f"\n  FDR results (q < {q_threshold}):")
    print(f"    Significant cells: {n_sig} / {n_atoms * n_terms} "
          f"({100*n_sig/(n_atoms*n_terms):.1f}%)")
    print(f"    Atoms with ≥1 sig term: "
          f"{(qvals < q_threshold).any(axis=1).sum()} / {n_atoms}")
    print(f"    Terms with ≥1 sig atom: "
          f"{(qvals < q_threshold).any(axis=0).sum()} / {n_terms}")

    # Text-best term per atom
    text_best = sims.argmax(axis=1)  # (n_atoms,)

    print(f"\n  Per-atom results (sorted by text-best term p-value):")
    print(f"  {'Atom':>4}  {'Text-best term':20s}  {'r':>6}  {'p':>6}  {'q':>6}  "
          f"{'#sig':>4}  Description")
    print("  " + "-" * 90)

    order = np.argsort(pvals[np.arange(n_atoms), text_best])
    for k in order:
        tidx = text_best[k]
        r_obs = obs[k, tidx]
        p_val = pvals[k, tidx]
        q_val = qvals[k, tidx]
        n_sig_k = (qvals[k] < q_threshold).sum()
        sig_terms = [available_terms[j] for j in range(n_terms)
                     if qvals[k, j] < q_threshold]
        desc = atom_descriptions.get(k, "")[:45]
        sig_str = ",".join(sig_terms[:4]) + ("..." if len(sig_terms) > 4 else "")
        print(f"  a{k:02d}  {available_terms[tidx]:20s}  {r_obs:+.3f}  "
              f"{p_val:.3f}  {q_val:.3f}  {n_sig_k:>4d}  {desc}")
        if sig_terms:
            print(f"         → sig: {sig_str}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--null_method", choices=["spin", "random_shuffle"], default="spin",
                   help="Spatial null: 'spin' (Alexander-Bloch 2018, default) or "
                        "'random_shuffle' (naive vertex permutation, legacy).")
    p.add_argument("--terms", default="expanded",
                   help="Term set: 'original', 'expanded', or path to file")
    p.add_argument("--min_studies", type=int, default=MIN_STUDIES_DEFAULT)
    p.add_argument("--q_threshold", type=float, default=0.05)
    p.add_argument("--force_ns", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = (args.model_dir if os.path.isabs(args.model_dir)
                 else join(PROJECT_DIR, args.model_dir))
    print(f"Model: {os.path.basename(model_dir)}")
    print(f"Term set: {args.terms}  |  n_perm: {args.n_perm}  |  seed: {args.seed}")

    # ── Load terms ──
    print("\n0. Loading Neurosynth terms...")
    terms = load_terms(args.terms, min_studies=args.min_studies)

    # ── Load model ──
    ckpt = torch.load(join(model_dir, "model.pt"),
                      map_location="cpu", weights_only=False)
    last = join(model_dir, "model_last.pt")
    sd = (torch.load(last, map_location="cpu", weights_only=False)["state_dict"]
          if os.path.exists(last) else ckpt["state_dict"])

    n_atoms = sd["encoder.weight"].shape[0]
    n_delays = sd["delay_scales"].shape[0]
    subjects = [s for s in ckpt["subjects"] if s in N_VOXELS_FULL]
    print(f"  {n_atoms} atoms, {n_delays} delays, {len(subjects)} subjects")

    # ── Neurosynth maps ──
    print("\n1. Loading Neurosynth maps...")
    ns_maps = generate_neurosynth_maps(terms, force=args.force_ns)
    available_terms = [t for t in terms if t in ns_maps]
    print(f"  {len(available_terms)} term maps available")

    # ── Atom brain maps ──
    print("\n2. Computing atom brain maps in fsaverage...")
    atom_fsav_per_subj, selected_masks = get_atom_brain_maps_per_subject(
        sd, ckpt, n_atoms, n_delays, subjects)

    # ── Atom descriptions ──
    print("\n3. Loading atom descriptions...")
    model_name = os.path.basename(model_dir)
    atom_descriptions, atom_keywords = get_atom_descriptions(n_atoms, model_name)

    # ── Text similarity ──
    print("\n4. Computing text similarities...")
    sims, _, _ = match_atoms_to_terms(atom_descriptions, atom_keywords,
                                       available_terms)
    print(f"  sims shape: {sims.shape}")

    # ── Build masked matrices ──
    print("\n5. Building shared-mask atom/NS matrices...")
    atom_mat, ns_mat, mask = build_masked_matrices(
        atom_fsav_per_subj, ns_maps, available_terms, selected_masks)
    print(f"  atom_mat: {atom_mat.shape}  ns_mat: {ns_mat.shape}")

    # ── Spin sampler ──
    if args.null_method == "spin":
        print("\n6a. Building spin sampler (Alexander-Bloch et al. 2018)...")
        coords_lh, coords_rh = _load_sphere_coords()
        spin_sampler = make_ns_spin_sampler(mask, coords_lh, coords_rh)
        print(f"    LH mask vertices: {(np.where(mask)[0] < N_LH_FSAV).sum()}, "
              f"RH mask vertices: {(np.where(mask)[0] >= N_LH_FSAV).sum()}")
    else:
        spin_sampler = None

    # ── Spatial permutation test ──
    print(f"\n6. Spatial permutation test ({args.n_perm} perms, {args.null_method})...")
    obs, pvals = spatial_permutation_test(
        atom_mat, ns_mat, n_perm=args.n_perm, seed=args.seed,
        spin_sampler=spin_sampler)

    # ── FDR correction ──
    print("\n7. FDR correction (BH, all cells)...")
    qvals = fdr_correct(pvals, n_perm=args.n_perm, q_threshold=args.q_threshold)

    # ── Report ──
    report_results(obs, pvals, qvals, available_terms, atom_descriptions,
                   sims, args.n_perm, q_threshold=args.q_threshold)

    # ── Save ──
    import pandas as pd
    _csv = os.environ.get("AUTOINTERP_CSV", "autointerp.csv")
    _suffix = ("" if _csv == "autointerp.csv"
               else "_" + os.path.splitext(_csv)[0].removeprefix("autointerp_"))
    out_dir = join(PROJECT_DIR, "reports", "figures",
                   f"ns_spatial_perm_{model_name}{_suffix}")
    os.makedirs(out_dir, exist_ok=True)

    n_terms = len(available_terms)
    n_sig_cells = int((qvals < args.q_threshold).sum())
    n_sig_atoms = int((qvals < args.q_threshold).any(axis=1).sum())
    n_sig_terms = int((qvals < args.q_threshold).any(axis=0).sum())

    # Per-atom summary CSV
    text_best = sims.argmax(axis=1)
    rows = []
    for k in range(n_atoms):
        tidx = text_best[k]
        sig_terms = sorted(
            [available_terms[j] for j in range(n_terms) if qvals[k, j] < args.q_threshold]
        )
        rows.append({
            "atom": k,
            "description": atom_descriptions.get(k, ""),
            "text_best_term": available_terms[tidx],
            "text_sim": float(sims[k, tidx]),
            "obs_r": float(obs[k, tidx]),
            "p_text_best": float(pvals[k, tidx]),
            "q_text_best": float(qvals[k, tidx]),
            "n_sig_terms": len(sig_terms),
            "sig_terms": ";".join(sig_terms),
            "best_r_term": available_terms[int(obs[k].argmax())],
            "best_r": float(obs[k].max()),
            "best_r_p": float(pvals[k, int(obs[k].argmax())]),
            "best_r_q": float(qvals[k, int(obs[k].argmax())]),
        })
    pd.DataFrame(rows).to_csv(join(out_dir, "atom_summary.csv"), index=False)

    # Full matrices
    pd.DataFrame(obs, index=[f"a{k}" for k in range(n_atoms)],
                 columns=available_terms).to_csv(join(out_dir, "pearson_observed.csv"))
    pd.DataFrame(pvals, index=[f"a{k}" for k in range(n_atoms)],
                 columns=available_terms).to_csv(join(out_dir, "pvalues.csv"))
    pd.DataFrame(qvals, index=[f"a{k}" for k in range(n_atoms)],
                 columns=available_terms).to_csv(join(out_dir, "qvalues.csv"))

    summary = {
        "model": model_name,
        "n_atoms": n_atoms,
        "n_terms": n_terms,
        "n_perm": args.n_perm,
        "q_threshold": args.q_threshold,
        "null": ("spin_alexander_bloch_2018_within_mask"
                 if args.null_method == "spin"
                 else "random_vertex_permutation_shared_mask"),
        "n_sig_cells": n_sig_cells,
        "n_sig_atoms": n_sig_atoms,
        "n_sig_terms": n_sig_terms,
        "frac_sig_cells": n_sig_cells / (n_atoms * n_terms),
    }
    with open(join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
