#!/usr/bin/env python
"""30_neurosynth_validation.py — Validate SAE atoms against Neurosynth.

For each atom:
  1. Get semantic content (top keywords from decoder weights)
  2. Match keywords → Neurosynth terms (embedding similarity)
  3. Get Neurosynth meta-analytic brain map for matched term
  4. Get atom brain map (decoder weights → fsaverage, averaged across subjects)
  5. Correlate atom brain map with Neurosynth map
  6. Permutation test: each atom vs random term (tests semantic matching)

Also computes semantic-anatomical RSA (Mantel test).

Usage:
    uv run python scripts/30_neurosynth_validation.py \
        --model_dir outputs/models/best_model
"""
import argparse
import json
import os
import sys
from os.path import join

import h5py
import numpy as np
import scipy.sparse
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ──────────────────────────────────────────────────────────────

MAPPER_DIR = join(PROJECT_DIR, "outputs", "mappers")
NS_DIR = join(PROJECT_DIR, "outputs", "neurosynth_maps")
NS_FSAV_DIR = join(NS_DIR, "fsaverage")

# 19 curated Neurosynth terms from QA encoding paper (Antonello et al.)
TERMS_ORIGINAL = [
    "abstract", "action", "arithmetic", "communication", "judgments",
    "location", "negative", "nouns", "object", "personal",
    "place", "planning", "sensation", "social", "spatial",
    "thoughts", "time", "touch", "visual",
]

# Expanded term set: cognitive, perceptual, emotional, and behavioral constructs.
# All terms present in Neurosynth v7 TF-IDF vocabulary (≥50 studies by default).
# Organized by domain; singular forms preferred over plurals.
TERMS_EXPANDED = [
    # Attention & executive control
    "attention", "inhibition", "monitoring", "conflict", "switching",
    "executive", "planning", "selection", "detection", "salience",
    "orienting", "flexibility", "updating", "interference", "suppression",
    # Memory
    "memory", "encoding", "retrieval", "recall", "recognition",
    "recollection", "familiarity", "episodic", "autobiographical",
    "prospective", "consolidation", "novelty",
    # Language & reading
    "language", "reading", "speech", "comprehension", "naming",
    "phonological", "lexical", "syntactic", "semantic", "verbal",
    "word", "fluency", "sentence", "orthographic",
    # Numerical cognition
    "arithmetic", "numerical",
    # Emotion & affect
    "emotional", "fear", "anxiety", "anger", "disgust", "mood",
    "affective", "valence", "arousal", "threat", "aversive", "pleasant",
    "unpleasant", "stress", "reappraisal", "sad", "happy", "angry",
    "fearful",
    # Motivation & reward
    "reward", "motivation", "incentive", "reinforcement", "anticipation",
    "impulsivity", "punishment", "craving", "gambling", "effort",
    # Social cognition
    "social", "empathy", "mentalizing", "communication", "moral",
    "imitation",
    # Perception — visual
    "visual", "face", "facial", "color", "motion", "object", "scene",
    # Perception — auditory
    "auditory", "music", "voice", "pitch", "sound", "listening",
    "acoustic",
    # Perception — somatosensory & pain
    "pain", "touch", "tactile", "sensation",
    # Perception — other
    "food", "taste", "olfactory", "multisensory", "audiovisual",
    # Motor & action
    "motor", "action", "movement", "execution", "imagery", "coordination",
    "grasping", "reaching", "balance",
    # Spatial cognition
    "spatial", "navigation", "location", "place", "visuospatial",
    # Learning
    "learning", "conditioning", "extinction", "avoidance", "priming",
    "adaptation", "categorization", "discrimination",
    # Decision making & evaluation
    "decision", "choice", "uncertainty", "risk", "prediction",
    "expectation", "judgments",
    # Consciousness & self-related
    "awareness", "sleep", "consciousness",
    # Higher cognition
    "reasoning", "thinking", "abstract", "concrete", "insight",
    "intelligence", "problem", "solving", "inference",
    # Body & interoception
    "body", "interoceptive",
    # Self & personal
    "personal", "identity", "mind",
    # Timing
    "timing",
    # Other
    "subjective", "nouns", "verbs", "thoughts",
]

MIN_STUDIES_DEFAULT = 50


def load_terms(terms_source, min_studies=MIN_STUDIES_DEFAULT):
    """Load and validate Neurosynth terms.

    Args:
        terms_source: "original" (19 from Antonello), "expanded" (curated
            cognitive terms), or a path to a text file with one term per line.
        min_studies: minimum study count in Neurosynth TF-IDF features.

    Returns:
        List of validated term strings.
    """
    if terms_source == "original":
        candidates = TERMS_ORIGINAL
    elif terms_source == "expanded":
        candidates = TERMS_EXPANDED
    else:
        with open(terms_source) as f:
            candidates = [line.strip() for line in f if line.strip()
                          and not line.startswith("#")]

    # Validate against actual Neurosynth vocabulary + study counts
    vocab_path = join(NS_DIR, "neurosynth",
                      "data-neurosynth_version-7_vocab-terms_vocabulary.txt")
    feat_path = join(NS_DIR, "neurosynth",
                     "data-neurosynth_version-7_vocab-terms_source-abstract"
                     "_type-tfidf_features.npz")
    if not os.path.exists(vocab_path):
        print(f"  Warning: vocab file not found, using terms as-is")
        return candidates

    vocab = open(vocab_path).read().splitlines()
    vocab_set = set(vocab)
    feat = scipy.sparse.load_npz(feat_path)
    counts = np.array((feat > 0.001).sum(axis=0)).flatten()
    vocab_counts = {t: int(counts[i]) for i, t in enumerate(vocab)}

    validated = []
    dropped = []
    for t in candidates:
        if t not in vocab_set:
            dropped.append((t, "not in vocabulary"))
        elif vocab_counts[t] < min_studies:
            dropped.append((t, f"only {vocab_counts[t]} studies"))
        else:
            validated.append(t)

    if dropped:
        print(f"  Dropped {len(dropped)} terms:")
        for t, reason in dropped:
            print(f"    {t}: {reason}")

    # Deduplicate (preserves order)
    seen = set()
    unique = []
    for t in validated:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    print(f"  {len(unique)} validated terms (min {min_studies} studies)")
    return unique


N_VOXELS_FULL = {
    "UTS01": 81126, "UTS02": 94251, "UTS03": 95556,
    "UTS04": 109469, "UTS05": 99322, "UTS06": 92198,
}


# ── Helpers ────────────────────────────────────────────────────────────────

def load_mapper(subject):
    """Load voxel→fsaverage sparse mapper."""
    path = join(MAPPER_DIR, f"{subject}_voxel_to_fsaverage.hdf")
    with h5py.File(path, "r") as f:
        return scipy.sparse.csr_matrix(
            (f["data"][:], f["indices"][:], f["indptr"][:]),
            shape=tuple(f["shape"][:]),
        )


def per_subject_atom_maps(sd, subject, n_atoms, n_delays):
    """Extract per-voxel atom magnitude from brain decoder weights.

    Returns (n_voxels_selected, n_atoms) — |Σ_d delay_scale_d · W[v,d,k]|.
    Note: per-voxel scale intentionally skipped (key name varies across
    model versions; omitting keeps both analysis scripts consistent).
    """
    W = sd[f"brain_decoders.{subject}.weight"].numpy()
    delay_scales = sd["delay_scales"].numpy()
    n_vox = W.shape[0]
    W_re = W.reshape(n_vox, n_delays, n_atoms)
    weighted = (W_re * delay_scales[None, :, :]).sum(axis=1)  # (V, A)
    return np.abs(weighted)


def compute_selected_voxel_mask(mapper, vox_idx, n_full):
    """Boolean (n_fsav,) mask of fsaverage vertices covered by selected voxels.

    Projects a binary indicator of the selected voxels through the mapper.
    Independent of atom weights — same for all atoms within a subject.
    """
    indicator = np.zeros(n_full, dtype=np.float32)
    indicator[vox_idx] = 1.0
    return np.asarray(mapper.dot(indicator)).ravel() > 0


def get_atom_brain_maps_per_subject(sd, ckpt, n_atoms, n_delays, subjects):
    """Get per-atom brain maps and selected-voxel coverage masks in fsaverage.

    Returns:
        per_subj: {subject: (327684, n_atoms)} — clipped atom maps
        selected_masks: {subject: (327684,) bool} — selected-voxel coverage
    """
    per_subj = {}
    selected_masks = {}
    for subj in subjects:
        D_subj = per_subject_atom_maps(sd, subj, n_atoms, n_delays)
        vox_idx = ckpt["voxel_indices"][subj]
        n_full = N_VOXELS_FULL[subj]
        mapper = load_mapper(subj)
        selected_masks[subj] = compute_selected_voxel_mask(mapper, vox_idx, n_full)
        D_full = np.zeros((n_full, n_atoms), dtype=np.float32)
        D_full[vox_idx] = D_subj
        D_fsav = np.asarray(mapper.dot(D_full))
        D_fsav = np.clip(D_fsav, 0, None)
        per_subj[subj] = D_fsav
    return per_subj, selected_masks


def get_atom_descriptions(n_atoms, model_name):
    """Get autointerp descriptions and keywords for each atom.

    Returns (descriptions, keywords) where:
      descriptions: {atom_idx: str} — one-sentence autointerp description
      keywords: {atom_idx: [str]} — top keywords from context files
    """
    import csv

    descriptions = {}
    keywords = {}

    # Try autointerp CSV (v1)
    autointerp_path = join(
        PROJECT_DIR, "outputs", "cache", "coherence", "autointerp",
        model_name, os.environ.get("AUTOINTERP_CSV", "autointerp.csv"),
    )
    if os.path.exists(autointerp_path):
        for row in csv.DictReader(open(autointerp_path)):
            atom_idx = int(row["atom"])
            descriptions[atom_idx] = row["description"]
            # Keywords from label field
            kw = row.get("label", "").replace("_", " ").split()
            keywords[atom_idx] = kw
        print(f"  Loaded {len(descriptions)} autointerp descriptions")
    else:
        print(f"  Warning: autointerp.csv not found at {autointerp_path}")

    # Also try atom_contexts/ for keywords
    figs_dir = join(PROJECT_DIR, "reports", "figures", model_name)
    ctx_dir = join(figs_dir, "atom_contexts")
    if os.path.isdir(ctx_dir):
        import re
        for fn in sorted(os.listdir(ctx_dir)):
            if not fn.endswith(".txt"):
                continue
            with open(join(ctx_dir, fn)) as f:
                header = f.readline()
            m = re.match(r"=== Atom (\d+)", header)
            if not m:
                continue
            atom_idx = int(m.group(1))
            for line in open(join(ctx_dir, fn)):
                if line.startswith("Keywords:"):
                    kw = [w.strip() for w in line.split(":", 1)[1].split(",")]
                    keywords[atom_idx] = kw[:20]
                    break
        print(f"  Loaded keywords for {len(keywords)} atoms from atom_contexts/")

    # Fill missing
    for k in range(n_atoms):
        if k not in descriptions:
            descriptions[k] = f"atom {k}"
        if k not in keywords:
            keywords[k] = [f"atom_{k}"]

    return descriptions, keywords


# ── Part A: Generate Neurosynth term maps ──────────────────────────────────

def generate_neurosynth_maps(terms, force=False):
    """Generate meta-analytic z-maps for each term using nimare."""
    os.makedirs(NS_FSAV_DIR, exist_ok=True)

    # Check if already cached
    all_cached = all(
        os.path.exists(join(NS_FSAV_DIR, f"{t}_fsaverage.npy"))
        for t in terms
    )
    if all_cached and not force:
        print("  All Neurosynth fsaverage maps cached, loading...")
        return {t: np.load(join(NS_FSAV_DIR, f"{t}_fsaverage.npy"))
                for t in terms}

    from nimare import extract, meta
    import nibabel as nib

    print("  Loading Neurosynth v7 dataset...")
    files = extract.fetch_neurosynth(data_dir=NS_DIR, version="7",
                                      overwrite=False)
    # fetch_neurosynth returns a list; first element is a Studyset
    dset = files[0]
    print(f"  {len(dset.ids)} studies loaded")

    ns_maps = {}
    for term in terms:
        nii_path = join(NS_DIR, f"{term}_z.nii.gz")
        fsav_path = join(NS_FSAV_DIR, f"{term}_fsaverage.npy")

        if os.path.exists(fsav_path) and not force:
            ns_maps[term] = np.load(fsav_path)
            print(f"  {term}: loaded from cache")
            continue

        # Generate meta-analytic map
        col = f"terms_abstract_tfidf__{term}"
        ann_df = dset.annotations_df
        if col not in ann_df.columns:
            print(f"  {term}: NOT in Neurosynth vocabulary, skipping")
            continue

        # Get proper study IDs (not integer indices)
        mask = ann_df[col] > 0.001
        study_ids = ann_df.loc[mask, "id"].values
        print(f"  {term}: {len(study_ids)} studies", end="")

        if len(study_ids) < 10:
            print(" — too few, skipping")
            continue

        dset_term = dset.slice(study_ids)
        est = meta.cbma.MKDADensity()
        results = est.fit(dset_term)
        z_img = results.get_map("z")

        # Save NIfTI
        nib.save(z_img, nii_path)

        # Project to fsaverage via FreeSurfer mri_vol2surf
        fsav_map = project_mni_to_fsaverage(nii_path)
        np.save(fsav_path, fsav_map)
        ns_maps[term] = fsav_map
        print(f" → fsaverage ({(fsav_map != 0).sum()} nonzero vertices)")

    return ns_maps


def project_mni_to_fsaverage(nii_path):
    """Project MNI volumetric map to fsaverage surface.

    Uses FreeSurfer mri_vol2surf for both hemispheres.
    """
    import subprocess
    import tempfile

    fs_subjects = os.environ.get(
        "SUBJECTS_DIR",
        join(PROJECT_DIR, "external", "deep-fMRI-dataset", "data",
             "ds003020", "derivative", "freesurfer_subjdir"),
    )

    fsav_verts = []
    for hemi in ["lh", "rh"]:
        with tempfile.NamedTemporaryFile(suffix=".mgz", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            os.path.join(os.environ["FREESURFER_HOME"], "bin", "mri_vol2surf"),
            "--src", nii_path,
            "--out", tmp_path,
            "--regheader", "fsaverage",
            "--hemi", hemi,
            "--projfrac", "0.5",
            "--surf", "white",
        ]
        env = os.environ.copy()
        env["SUBJECTS_DIR"] = fs_subjects

        try:
            subprocess.run(cmd, capture_output=True, check=True, env=env)
            import nibabel as nib
            data = nib.load(tmp_path).get_fdata().flatten()
            fsav_verts.append(data)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\n  Warning: mri_vol2surf failed ({e}), using pycortex fallback")
            return project_mni_to_fsaverage_pycortex(nii_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return np.concatenate(fsav_verts)


def project_mni_to_fsaverage_pycortex(nii_path):
    """Fallback: project MNI → fsaverage using pycortex."""
    import cortex
    import nibabel as nib

    nii = nib.load(nii_path)
    mni_vol = cortex.Volume(nii.get_fdata(), "fsaverage", "atlas_2mm")
    fs_mapper = cortex.get_mapper("fsaverage", "atlas_2mm")
    fs_surf = fs_mapper(mni_vol)
    return np.hstack([fs_surf.left, fs_surf.right])


# ── Part D: Match atoms → Neurosynth terms ─────────────────────────────────

def match_atoms_to_terms(atom_descriptions, atom_keywords, terms):
    """Match atoms ↔ Neurosynth terms via embedding similarity.

    Uses autointerp descriptions (preferred) or keywords as fallback.
    Returns:
      sims: (n_atoms, n_terms) cosine similarity matrix
      best_atom_per_term: {term_idx: atom_idx} — top-1 atom for each term
      best_term_per_atom: (n_atoms,) — best term for each atom
    """
    from sentence_transformers import SentenceTransformer

    print("  Loading sentence-transformer...")
    embedder = SentenceTransformer("all-mpnet-base-v2")

    # Embed atom descriptions (preferred) or keywords
    n_atoms = len(atom_descriptions)
    atom_strings = []
    for k in range(n_atoms):
        desc = atom_descriptions.get(k, "")
        kw = " ".join(atom_keywords.get(k, [])[:10])
        # Use description if available, else keywords
        atom_strings.append(desc if len(desc) > 10 else kw)

    atom_embs = embedder.encode(atom_strings, show_progress_bar=False)
    term_embs = embedder.encode(terms, show_progress_bar=False)

    # Normalize
    atom_embs = atom_embs / (np.linalg.norm(atom_embs, axis=1, keepdims=True) + 1e-8)
    term_embs = term_embs / (np.linalg.norm(term_embs, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity
    sims = atom_embs @ term_embs.T  # (n_atoms, n_terms)

    # Best atom per term (top-1 match for each Neurosynth term)
    best_atom_per_term = sims.argmax(axis=0)  # (n_terms,)

    # Best term per atom (for reference)
    best_term_per_atom = sims.argmax(axis=1)  # (n_atoms,)

    return sims, best_atom_per_term, best_term_per_atom


# ── Part E+F: Correlate and permutation test ─��─────────────────────────────

def _atom_term_metrics(a, b):
    """Cosine and Pearson r between pre-masked vectors a and b."""
    if len(a) < 100:
        return np.nan, np.nan
    return cosine_sim(a, b), float(np.corrcoef(a, b)[0, 1])


def correlate_atom_neurosynth(atom_fsav_per_subj, selected_masks, ns_union_mask,
                               ns_maps, terms, sims):
    """Compute full (n_atoms × n_terms) brain-cosine/Pearson matrix, then
    derive per-term and per-atom summaries using brain-cosine argmax.

    Mask per subject: selected-voxel coverage ∩ Neurosynth union coverage.
    No atom-weight masking — mask is the same for all atoms within a subject.

    Returns (term_results, atom_results, cosine_matrix, pearson_matrix, terms)
    so both permutation tests can reuse the precomputed matrices.
    """
    subjects = list(atom_fsav_per_subj.keys())
    n_atoms = next(iter(atom_fsav_per_subj.values())).shape[1]
    n_terms = len(terms)

    mask_per_subj = {s: selected_masks[s] & ns_union_mask for s in subjects}
    ns_pos = {t: np.clip(ns_maps[t], 0, None) for t in terms if t in ns_maps}

    # Full (n_atoms, n_terms) brain-cosine and Pearson matrices
    cosine_matrix = np.full((n_atoms, n_terms), np.nan)
    pearson_matrix = np.full((n_atoms, n_terms), np.nan)
    for tidx, term in enumerate(terms):
        if term not in ns_maps:
            continue
        ns = ns_pos[term]
        ns_masked = {s: ns[mask_per_subj[s]] for s in subjects}
        for k in range(n_atoms):
            subj_cos, subj_r = [], []
            for s in subjects:
                c, r = _atom_term_metrics(
                    atom_fsav_per_subj[s][mask_per_subj[s], k], ns_masked[s])
                if not np.isnan(c):
                    subj_cos.append(c); subj_r.append(r)
            if subj_cos:
                cosine_matrix[k, tidx] = float(np.mean(subj_cos))
                pearson_matrix[k, tidx] = float(np.mean(subj_r))

    # Per-term: best atom per term by brain cosine (not text similarity)
    term_results = []
    for tidx, term in enumerate(terms):
        col = cosine_matrix[:, tidx]
        if np.all(np.isnan(col)):
            continue
        atom_idx = int(np.nanargmax(col))
        term_results.append({
            "term": term, "atom": atom_idx,
            "term_sim": float(sims[atom_idx, tidx]) if sims is not None else np.nan,
            "brain_cosine": float(cosine_matrix[atom_idx, tidx]),
            "brain_corr": float(pearson_matrix[atom_idx, tidx]),
            "n_subjects": len(subjects),
        })

    # Per-atom: best term per atom by brain cosine (not text similarity)
    atom_results = []
    for k in range(n_atoms):
        row = cosine_matrix[k, :]
        if np.all(np.isnan(row)):
            atom_results.append({"atom": k, "term": None,
                                 "brain_corr": np.nan, "brain_cosine": np.nan})
            continue
        tidx = int(np.nanargmax(row))
        atom_results.append({
            "atom": k, "term": terms[tidx],
            "brain_cosine": float(cosine_matrix[k, tidx]),
            "brain_corr": float(pearson_matrix[k, tidx]),
        })

    return term_results, atom_results, cosine_matrix, pearson_matrix, terms


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def permutation_test(cosine_matrix, sims, available_terms, n_perm=1000):
    """Per-atom permutation test: brain-first matching, then test semantic alignment.

    Design (brain-first):
      1. brain_best[k] = argmax_j cosine_matrix[k, j]  (brain picks best Neurosynth term)
      2. Observed: sims[k, brain_best[k]]  (does atom k's description match that term?)
      3. Null (atom-label permutation): sims[π(k), brain_best[k]]  (would another
         atom's description match equally well?)

    This avoids selection bias — observed is text similarity (not brain cosine argmax).
    Null permutes atom labels so the brain-selected term stays fixed.

    Also computes Mantel-style global test: Spearman ρ between flattened sims matrix
    and flattened cosine_matrix; null = row-permutation of sims (atom-label shuffle).
    FDR corrected per-atom (BH).
    """
    n_atoms, n_terms = cosine_matrix.shape
    assert sims.shape == (n_atoms, n_terms), f"sims shape mismatch: {sims.shape}"
    rng = np.random.default_rng(42)

    # Brain-first: find best Neurosynth term per atom by brain cosine
    brain_best = np.full(n_atoms, -1, dtype=int)
    matched_terms = [None] * n_atoms
    for k in range(n_atoms):
        row = cosine_matrix[k, :]
        if np.all(np.isnan(row)):
            continue
        brain_best[k] = int(np.nanargmax(row))
        matched_terms[k] = available_terms[brain_best[k]]

    # Observed: text similarity at brain-selected term
    obs_per_atom = np.full(n_atoms, np.nan)
    for k in range(n_atoms):
        if brain_best[k] >= 0:
            obs_per_atom[k] = sims[k, brain_best[k]]

    # Null: atom-label permutation — sims[π(k), brain_best[k]]
    null_per_atom = np.full((n_atoms, n_perm), np.nan)
    valid_bb = brain_best >= 0
    for p in range(n_perm):
        perm = rng.permutation(n_atoms)
        null_per_atom[valid_bb, p] = sims[perm[valid_bb], brain_best[valid_bb]]

    # Global test: mean text-sim at brain-best term
    valid = ~np.isnan(obs_per_atom)
    obs_mean = float(np.nanmean(obs_per_atom))
    null_means = np.nanmean(null_per_atom[valid], axis=0) if valid.any() else np.array([])
    p_global = float((null_means >= obs_mean).mean()) if len(null_means) > 0 else np.nan

    # Mantel-style test: Spearman ρ(flattened sims, flattened cosine_matrix)
    valid_mask2d = ~np.isnan(cosine_matrix)  # sims has no NaNs
    flat_cos = cosine_matrix[valid_mask2d]
    flat_sims_obs = sims[valid_mask2d]
    if len(flat_cos) > 10:
        rho_obs, _ = spearmanr(flat_sims_obs, flat_cos)
        null_rhos = []
        for p in range(n_perm):
            perm = rng.permutation(n_atoms)
            sims_perm = sims[perm, :]
            flat_sims_p = sims_perm[valid_mask2d]
            rho_p, _ = spearmanr(flat_sims_p, flat_cos)
            null_rhos.append(rho_p)
        null_rhos = np.array(null_rhos)
        p_mantel = float((null_rhos >= rho_obs).mean()) if len(null_rhos) > 0 else np.nan
    else:
        rho_obs = np.nan
        null_rhos = np.array([])
        p_mantel = np.nan

    mantel_results = {
        "rho": float(rho_obs) if not np.isnan(rho_obs) else None,
        "p_value": float(p_mantel) if not np.isnan(p_mantel) else None,
        "null_rho_mean": float(null_rhos.mean()) if len(null_rhos) > 0 else None,
        "null_rho_std": float(null_rhos.std()) if len(null_rhos) > 0 else None,
    }

    # Per-atom test with FDR correction
    atom_p_uncorr = np.full(n_atoms, np.nan)
    for k in range(n_atoms):
        if np.isnan(obs_per_atom[k]):
            continue
        null_k = null_per_atom[k]
        valid_null = ~np.isnan(null_k)
        if valid_null.sum() < 10:
            continue
        atom_p_uncorr[k] = float((null_k[valid_null] >= obs_per_atom[k]).mean())

    from scipy.stats import false_discovery_control
    fdr_valid = ~np.isnan(atom_p_uncorr)
    atom_q = np.full(n_atoms, np.nan)
    if fdr_valid.sum() > 0:
        valid_ps = atom_p_uncorr[fdr_valid].copy()
        valid_ps = np.maximum(valid_ps, 1.0 / max(n_perm, 1))
        atom_q[fdr_valid] = false_discovery_control(valid_ps, method="bh")

    # Assemble per-atom results
    atom_perm_results = []
    for k in range(n_atoms):
        atom_perm_results.append({
            "atom": k,
            "matched_term": matched_terms[k],
            "brain_cosine": float(cosine_matrix[k, brain_best[k]])
                            if brain_best[k] >= 0 else None,
            "obs_text_sim": float(obs_per_atom[k])
                            if not np.isnan(obs_per_atom[k]) else None,
            "null_mean": float(np.nanmean(null_per_atom[k])),
            "null_std": float(np.nanstd(null_per_atom[k])),
            "p_uncorrected": float(atom_p_uncorr[k])
                             if not np.isnan(atom_p_uncorr[k]) else None,
            "q_fdr": float(atom_q[k]) if not np.isnan(atom_q[k]) else None,
        })

    return obs_mean, null_means, p_global, atom_perm_results, mantel_results


def permutation_test_text_first(pearson_matrix, sims, available_terms, n_perm=1000):
    """Per-atom permutation test: text-first matching, then test brain alignment.

    Design (text-first):
      1. text_best[k] = argmax_j sims[k, j]  (text picks best Neurosynth term)
      2. Observed: pearson_matrix[k, text_best[k]]  (brain r of atom k vs text-selected term)
      3. Null (atom-label permutation on brain): pearson_matrix[π(k), text_best[k]]
         (brain r of a different atom's map vs atom k's text-selected term)

    Tests: does atom k's brain map correlate better with its own text-selected term
    than a random atom's brain map would?
    FDR corrected per-atom (BH).
    """
    n_atoms, n_terms = pearson_matrix.shape
    assert sims.shape == (n_atoms, n_terms), f"sims shape mismatch: {sims.shape}"
    rng = np.random.default_rng(42)

    # Text-first: find best Neurosynth term per atom by text similarity
    text_best = np.argmax(sims, axis=1)  # (n_atoms,) — no NaNs in sims
    matched_terms = [available_terms[tidx] for tidx in text_best]

    # Observed: brain Pearson r at text-selected term
    obs_per_atom = np.array([pearson_matrix[k, text_best[k]] for k in range(n_atoms)])

    # Null: atom-label permutation on brain rows — pearson_matrix[π(k), text_best[k]]
    null_per_atom = np.full((n_atoms, n_perm), np.nan)
    for p in range(n_perm):
        perm = rng.permutation(n_atoms)
        null_per_atom[:, p] = pearson_matrix[perm, text_best]

    # Global test
    valid = ~np.isnan(obs_per_atom)
    obs_mean = float(np.nanmean(obs_per_atom))
    null_means = np.nanmean(null_per_atom[valid], axis=0) if valid.any() else np.array([])
    p_global = float((null_means >= obs_mean).mean()) if len(null_means) > 0 else np.nan

    # Per-atom FDR
    atom_p_uncorr = np.full(n_atoms, np.nan)
    for k in range(n_atoms):
        if np.isnan(obs_per_atom[k]):
            continue
        null_k = null_per_atom[k]
        valid_null = ~np.isnan(null_k)
        if valid_null.sum() < 10:
            continue
        atom_p_uncorr[k] = float((null_k[valid_null] >= obs_per_atom[k]).mean())

    from scipy.stats import false_discovery_control
    fdr_valid = ~np.isnan(atom_p_uncorr)
    atom_q = np.full(n_atoms, np.nan)
    if fdr_valid.sum() > 0:
        valid_ps = atom_p_uncorr[fdr_valid].copy()
        valid_ps = np.maximum(valid_ps, 1.0 / max(n_perm, 1))
        atom_q[fdr_valid] = false_discovery_control(valid_ps, method="bh")

    atom_perm_results = []
    for k in range(n_atoms):
        atom_perm_results.append({
            "atom": k,
            "text_term": matched_terms[k],
            "text_sim": float(sims[k, text_best[k]]),
            "obs_brain_r": float(obs_per_atom[k]) if not np.isnan(obs_per_atom[k]) else None,
            "null_mean": float(np.nanmean(null_per_atom[k])),
            "null_std": float(np.nanstd(null_per_atom[k])),
            "p_uncorrected": float(atom_p_uncorr[k]) if not np.isnan(atom_p_uncorr[k]) else None,
            "q_fdr": float(atom_q[k]) if not np.isnan(atom_q[k]) else None,
        })

    return obs_mean, null_means, p_global, atom_perm_results


# ── Part G: Semantic-anatomical RSA ────────────────────────────────────────

def semantic_anatomical_rsa(sd, atom_fsav):
    """Mantel test: semantic similarity predicts anatomical proximity."""
    encoder_w = sd["encoder.weight"].numpy()  # (n_atoms, n_sem)

    # Semantic distance
    sem_dist = pdist(encoder_w, metric="cosine")

    # Anatomical distance (correlation distance of brain maps)
    anat_dist = pdist(atom_fsav.T, metric="correlation")

    # Remove NaN pairs
    valid = ~(np.isnan(sem_dist) | np.isnan(anat_dist))
    r_mantel, _ = spearmanr(sem_dist[valid], anat_dist[valid])

    # Permutation p-value
    rng = np.random.default_rng(42)
    n_perm = 1000
    null_rs = []
    n_atoms = encoder_w.shape[0]
    for _ in range(n_perm):
        perm = rng.permutation(n_atoms)
        perm_anat = pdist(atom_fsav[:, perm].T, metric="correlation")
        valid_p = ~(np.isnan(sem_dist) | np.isnan(perm_anat))
        r_p, _ = spearmanr(sem_dist[valid_p], perm_anat[valid_p])
        null_rs.append(r_p)

    null_rs = np.array(null_rs)
    p_rsa = float((null_rs >= r_mantel).mean())
    return r_mantel, p_rsa, sem_dist, anat_dist


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--force_ns", action="store_true",
                   help="Force regeneration of Neurosynth maps")
    p.add_argument("--terms", type=str, default="expanded",
                   help="Term set: 'original' (19 from Antonello), "
                        "'expanded' (curated cognitive terms, default), "
                        "or path to a text file with one term per line")
    p.add_argument("--min_studies", type=int, default=MIN_STUDIES_DEFAULT,
                   help="Minimum Neurosynth study count per term "
                        f"(default: {MIN_STUDIES_DEFAULT})")
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = (args.model_dir if os.path.isabs(args.model_dir)
                 else join(PROJECT_DIR, args.model_dir))

    print(f"Model: {os.path.basename(model_dir)}")
    print(f"Term set: {args.terms}")

    # Load and validate terms
    print("\n0. Loading Neurosynth terms...")
    terms = load_terms(args.terms, min_studies=args.min_studies)

    # Load model
    ckpt = torch.load(join(model_dir, "model.pt"),
                       map_location="cpu", weights_only=False)
    last = join(model_dir, "model_last.pt")
    if os.path.exists(last):
        sd = torch.load(last, map_location="cpu", weights_only=False)["state_dict"]
    else:
        sd = ckpt["state_dict"]

    n_atoms = sd["encoder.weight"].shape[0]
    n_delays = sd["delay_scales"].shape[0]
    subjects = [s for s in ckpt["subjects"] if s in N_VOXELS_FULL]
    print(f"  {n_atoms} atoms, {n_delays} delays, {len(subjects)} subjects")

    # ── Part A: Neurosynth maps ──
    print("\n1. Generating Neurosynth meta-analytic maps...")
    ns_maps = generate_neurosynth_maps(terms, force=args.force_ns)
    print(f"  {len(ns_maps)} term maps available")

    # ── Part B: Atom brain maps + selected-voxel coverage masks ──
    print("\n2. Computing atom brain maps in fsaverage...")
    atom_fsav_per_subj, selected_masks = get_atom_brain_maps_per_subject(
        sd, ckpt, n_atoms, n_delays, subjects)
    atom_fsav_mean = np.mean(list(atom_fsav_per_subj.values()), axis=0)
    print(f"  Shape: {atom_fsav_mean.shape}")
    for subj, m in selected_masks.items():
        print(f"  {subj} selected-voxel coverage: {m.sum()} fsav vertices")

    # Neurosynth union mask: any vertex with data for at least one term.
    # Analogous to yeo > 0 (any cortical network) in script 27.
    ns_union_mask = np.zeros(atom_fsav_mean.shape[0], dtype=bool)
    for ns_map in ns_maps.values():
        ns_union_mask |= (ns_map > 0)
    print(f"  Neurosynth union coverage: {ns_union_mask.sum()} fsav vertices")

    # ── Part C: Atom descriptions + keywords ──
    print("\n3. Extracting atom descriptions...")
    model_name = os.path.basename(model_dir)
    atom_descriptions, atom_keywords = get_atom_descriptions(n_atoms, model_name)
    for k in [0, 5, 6, 8]:
        desc = atom_descriptions.get(k, "")[:80]
        print(f"  Atom {k}: {desc}")

    # ── Part D: Match atoms ↔ terms ──
    print("\n4. Matching atoms to Neurosynth terms...")
    available_terms = [t for t in terms if t in ns_maps]
    sims, best_atom_per_term, best_term_per_atom = match_atoms_to_terms(
        atom_descriptions, atom_keywords, available_terms)

    print(f"\n  Per-atom best term (top-1 term per atom, {len(available_terms)} terms):")
    for k in range(n_atoms):
        tidx = best_term_per_atom[k]
        term = available_terms[tidx]
        s = sims[k, tidx]
        desc = atom_descriptions.get(k, "")[:50]
        print(f"    Atom {k:2d} → {term:20s} (sim={s:.3f}) | {desc}")

    # ── Part E: Correlate brain maps ──
    print("\n5. Correlating atom brain maps with Neurosynth maps...")
    term_results, atom_results, cosine_matrix, pearson_matrix, available_terms = correlate_atom_neurosynth(
        atom_fsav_per_subj, selected_masks, ns_union_mask,
        ns_maps, available_terms, sims)

    # Per-term results
    term_rs = [r["brain_corr"] for r in term_results]
    term_cs = [r["brain_cosine"] for r in term_results]
    print(f"\n  Per-term (top-1 atom per term, {len(term_results)} terms):")
    print(f"    Mean Pearson r: {np.mean(term_rs):.4f}")
    print(f"    Mean cosine:    {np.mean(term_cs):.4f}")

    print("\n  Top 5 by cosine:")
    for r in sorted(term_results, key=lambda x: x["brain_cosine"], reverse=True)[:5]:
        desc = atom_descriptions.get(r["atom"], "")[:50]
        print(f"    {r['term']:20s} → Atom {r['atom']:2d} cos={r['brain_cosine']:.4f} r={r['brain_corr']:.4f} | {desc}")

    print("\n  Bottom 5:")
    for r in sorted(term_results, key=lambda x: x["brain_cosine"])[:5]:
        desc = atom_descriptions.get(r["atom"], "")[:50]
        print(f"    {r['term']:20s} → Atom {r['atom']:2d} cos={r['brain_cosine']:.4f} r={r['brain_corr']:.4f} | {desc}")

    # Per-atom results
    valid_rs = [r["brain_corr"] for r in atom_results
                if not np.isnan(r["brain_corr"])]
    valid_cs = [r["brain_cosine"] for r in atom_results
                if not np.isnan(r["brain_cosine"])]
    print(f"\n  Per-atom (best term per atom, {len(valid_rs)} atoms):")
    print(f"    Mean Pearson r: {np.mean(valid_rs):.4f}")
    print(f"    Mean cosine:    {np.mean(valid_cs):.4f}")

    # ── Part F: Permutation test (per-atom) ──
    print(f"\n6. Permutation test — brain-first, test semantic alignment ({args.n_perm} perms)...")
    print(f"  Brain picks best Neurosynth term per atom; test: does atom's description "
          f"match that term better than another atom's description?")
    obs_mean, null_means, p_value, atom_perm_results, mantel_results = permutation_test(
        cosine_matrix, sims, available_terms, n_perm=args.n_perm)
    null_m_str = (f"{null_means.mean():.4f} ± {null_means.std():.4f}"
                  if len(null_means) > 0 else "n/a")
    p_str_global = f"{p_value:.4f}" if not np.isnan(p_value) else "n/a"
    print(f"\n  Global test (mean text-sim at brain-best term):")
    print(f"    Observed mean text-sim: {obs_mean:.4f}")
    print(f"    Null mean ± std:        {null_m_str}")
    print(f"    P-value: {p_str_global}")

    rho = mantel_results["rho"]
    p_m = mantel_results["p_value"]
    rho_str = f"{rho:.4f}" if rho is not None else "n/a"
    pm_str = f"{p_m:.4f}" if p_m is not None else "n/a"
    print(f"\n  Mantel test (Spearman ρ, flattened sims vs cosine_matrix):")
    print(f"    ρ = {rho_str},  p = {pm_str}")

    # Per-atom results
    n_sig_fdr = sum(1 for r in atom_perm_results
                    if r["q_fdr"] is not None and r["q_fdr"] < 0.05)
    n_sig_uncorr = sum(1 for r in atom_perm_results
                       if r["p_uncorrected"] is not None
                       and r["p_uncorrected"] < 0.05)
    n_tested = sum(1 for r in atom_perm_results
                   if r["p_uncorrected"] is not None)
    print(f"\n  Per-atom tests ({n_tested} atoms tested):")
    print(f"    Significant (p<0.05 uncorrected): {n_sig_uncorr}/{n_tested}")
    print(f"    Significant (q<0.05 FDR):         {n_sig_fdr}/{n_tested}")

    print(f"\n  Per-atom results (sorted by q_fdr):")
    for r in sorted(atom_perm_results,
                    key=lambda x: x["q_fdr"] if x["q_fdr"] is not None else 999):
        if r["obs_text_sim"] is None:
            continue
        sig = "*" if r["q_fdr"] is not None and r["q_fdr"] < 0.05 else " "
        desc = atom_descriptions.get(r["atom"], "")[:40]
        p_str = f"{r['p_uncorrected']:.4f}" if r["p_uncorrected"] is not None else "n/a"
        q_str = f"{r['q_fdr']:.4f}" if r["q_fdr"] is not None else "n/a"
        null_m = r["null_mean"]
        null_s = r["null_std"]
        null_str = (f"{null_m:.4f}±{null_s:.4f}"
                    if not (np.isnan(null_m) or np.isnan(null_s)) else "n/a")
        bcos = r["brain_cosine"]
        bcos_str = f"{bcos:.4f}" if bcos is not None else "n/a"
        print(f"    {sig} Atom {r['atom']:2d} → {r['matched_term']:20s} "
              f"bcos={bcos_str} tsim={r['obs_text_sim']:.4f} "
              f"null={null_str} p={p_str} q={q_str} | {desc}")

    # ── Part F2: Text-first permutation test ──
    print(f"\n6b. Permutation test — text-first, test brain alignment ({args.n_perm} perms)...")
    print(f"  Text picks best Neurosynth term per atom; test: does atom's brain map "
          f"correlate better with that term than another atom's brain map would?")
    (obs_mean_tf, null_means_tf, p_value_tf,
     atom_perm_tf) = permutation_test_text_first(
        pearson_matrix, sims, available_terms, n_perm=args.n_perm)
    null_m_tf_str = (f"{null_means_tf.mean():.4f} ± {null_means_tf.std():.4f}"
                     if len(null_means_tf) > 0 else "n/a")
    p_tf_str = f"{p_value_tf:.4f}" if not np.isnan(p_value_tf) else "n/a"
    print(f"\n  Global test (mean brain r at text-best term):")
    print(f"    Observed mean brain r: {obs_mean_tf:.4f}")
    print(f"    Null mean ± std:       {null_m_tf_str}")
    print(f"    P-value: {p_tf_str}")

    n_sig_fdr_tf = sum(1 for r in atom_perm_tf
                       if r["q_fdr"] is not None and r["q_fdr"] < 0.05)
    n_sig_uncorr_tf = sum(1 for r in atom_perm_tf
                          if r["p_uncorrected"] is not None
                          and r["p_uncorrected"] < 0.05)
    n_tested_tf = sum(1 for r in atom_perm_tf if r["p_uncorrected"] is not None)
    print(f"\n  Per-atom tests ({n_tested_tf} atoms tested):")
    print(f"    Significant (p<0.05 uncorrected): {n_sig_uncorr_tf}/{n_tested_tf}")
    print(f"    Significant (q<0.05 FDR):         {n_sig_fdr_tf}/{n_tested_tf}")

    print(f"\n  Per-atom results (sorted by q_fdr):")
    for r in sorted(atom_perm_tf,
                    key=lambda x: x["q_fdr"] if x["q_fdr"] is not None else 999):
        if r["obs_brain_r"] is None:
            continue
        sig = "*" if r["q_fdr"] is not None and r["q_fdr"] < 0.05 else " "
        desc = atom_descriptions.get(r["atom"], "")[:40]
        p_str = f"{r['p_uncorrected']:.4f}" if r["p_uncorrected"] is not None else "n/a"
        q_str = f"{r['q_fdr']:.4f}" if r["q_fdr"] is not None else "n/a"
        null_m = r["null_mean"]
        null_s = r["null_std"]
        null_str = (f"{null_m:.4f}±{null_s:.4f}"
                    if not (np.isnan(null_m) or np.isnan(null_s)) else "n/a")
        print(f"    {sig} Atom {r['atom']:2d} → {r['text_term']:20s} "
              f"tsim={r['text_sim']:.3f} br={r['obs_brain_r']:+.4f} "
              f"null={null_str} p={p_str} q={q_str} | {desc}")

    # ── Part G: RSA ──
    print("\n7. Semantic-anatomical RSA (Mantel test)...")
    r_mantel, p_rsa, sem_dist, anat_dist = semantic_anatomical_rsa(sd, atom_fsav_mean)
    print(f"  Mantel r (Spearman): {r_mantel:.4f}")
    print(f"  P-value: {p_rsa:.4f}")

    # ── Save results ──
    out_dir = join(PROJECT_DIR, "reports", "figures", "neurosynth_validation")
    os.makedirs(out_dir, exist_ok=True)

    # Save CSV
    import pandas as pd
    df_terms = pd.DataFrame(term_results)
    df_terms.to_csv(join(out_dir, "term_atom_matching.csv"), index=False)
    df_atoms = pd.DataFrame(atom_results)
    df_atoms["keywords"] = [", ".join(atom_keywords.get(k, [])[:8])
                            for k in range(n_atoms)]
    df_atoms["description"] = [atom_descriptions.get(k, "")
                               for k in range(n_atoms)]
    df_atoms.to_csv(join(out_dir, "atom_term_matching.csv"), index=False)

    # Save per-atom permutation results (brain-first)
    df_perm = pd.DataFrame(atom_perm_results)
    df_perm["description"] = [atom_descriptions.get(k, "")
                              for k in range(n_atoms)]
    df_perm.to_csv(join(out_dir, "atom_permutation_brain_first.csv"), index=False)

    # Save per-atom permutation results (text-first)
    df_perm_tf = pd.DataFrame(atom_perm_tf)
    df_perm_tf["description"] = [atom_descriptions.get(k, "")
                                 for k in range(n_atoms)]
    df_perm_tf.to_csv(join(out_dir, "atom_permutation_text_first.csv"), index=False)

    # Save summary
    summary = {
        "model": os.path.basename(model_dir),
        "n_atoms": n_atoms,
        "n_terms": len(ns_maps),
        "term_set": args.terms,
        "min_studies": args.min_studies,
        "terms_used": available_terms,
        # Brain-first perm test: text-sim at brain-best term
        "brain_first_obs_textsim": float(obs_mean),
        "brain_first_null_textsim_mean": float(null_means.mean()) if len(null_means) > 0 else None,
        "brain_first_p_value": float(p_value) if not np.isnan(p_value) else None,
        "brain_first_n_sig_fdr05": n_sig_fdr,
        "brain_first_n_sig_uncorr05": n_sig_uncorr,
        # Text-first perm test: brain r at text-best term
        "text_first_obs_brain_r": float(obs_mean_tf),
        "text_first_null_brain_r_mean": float(null_means_tf.mean()) if len(null_means_tf) > 0 else None,
        "text_first_p_value": float(p_value_tf) if not np.isnan(p_value_tf) else None,
        "text_first_n_sig_fdr05": n_sig_fdr_tf,
        "text_first_n_sig_uncorr05": n_sig_uncorr_tf,
        # Mantel test: text-sim matrix vs brain-cosine matrix
        "mantel_rho": mantel_results["rho"],
        "mantel_p_value": mantel_results["p_value"],
        # Encoder-RSA (atom-atom semantic vs anatomical)
        "rsa_mantel_r": float(r_mantel),
        "rsa_p_value": float(p_rsa),
    }
    with open(join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plot: per-atom correlations ──
    n_terms_plot = len(term_results)
    fig_h = max(6, n_terms_plot * 0.18 + 1)  # scale height to term count
    fig, axes = plt.subplots(1, 2, figsize=(14, fig_h),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Panel A: per-term bar chart (top-1 atom per Neurosynth term)
    ax = axes[0]
    terms_sorted = sorted(term_results, key=lambda x: x.get("brain_corr", 0))
    ys = range(len(terms_sorted))
    rs = [r["brain_corr"] for r in terms_sorted]
    colors = ["tab:blue" if r > 0 else "tab:red" for r in rs]
    ax.barh(ys, rs, color=colors, alpha=0.7)
    ax.set_yticks(ys)
    fontsize = max(4, min(7, 120 // n_terms_plot))
    ax.set_yticklabels([f"{r['term']}→A{r['atom']}" for r in terms_sorted],
                       fontsize=fontsize)
    ax.set_xlabel("Pearson r (atom map vs Neurosynth z>0 map)")
    ax.set_title(f"Per-term correlation ({n_terms_plot} terms, "
                 f"mean cos={obs_mean:.3f}, p={p_value:.3f})")
    ax.axvline(0, color="gray", linewidth=0.5)

    # Panel B: null distribution
    ax = axes[1]
    ax.hist(null_means, bins=30, alpha=0.7, color="gray", label="Null")
    ax.axvline(obs_mean, color="red", linewidth=2, label=f"Observed ({obs_mean:.3f})")
    ax.set_xlabel("Mean cosine (permuted)")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation test (p={p_value:.4f})")
    ax.legend()

    plt.tight_layout()
    fig.savefig(join(out_dir, "atom_neurosynth_correlations.png"),
                dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_dir}/")

    # ── Plot: RSA ──
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(sem_dist, anat_dist, alpha=0.1, s=2)
    ax2.set_xlabel("Semantic distance (cosine)")
    ax2.set_ylabel("Anatomical distance (correlation)")
    ax2.set_title(f"Semantic–Anatomical RSA\n"
                  f"Mantel r={r_mantel:.3f}, p={p_rsa:.3f}")
    fig2.savefig(join(out_dir, "semantic_anatomical_rsa.png"),
                 dpi=150, bbox_inches="tight")

    print("\nDone.")


if __name__ == "__main__":
    main()
