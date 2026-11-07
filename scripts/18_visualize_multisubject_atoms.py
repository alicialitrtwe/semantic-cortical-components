#!/usr/bin/env python
"""18_visualize_multisubject_atoms.py — Visualize multi-subject SAE atoms.

For top predictive atoms, shows:
1. Semantic interpretation (top activating words from shared encoder)
2. Three flatmaps in a row (one per subject)

Prerequisites:
    Pycortex data must be downloaded for all subjects. If flatmaps fail with
    "Cannot find a valid mask", download the missing data:

    cd <your pycortex filestore>
    git annex get UTS01/transforms/UTS01_auto/*
    git annex get UTS02/transforms/UTS02_auto/*
    git annex get UTS01/surfaces/*
    git annex get UTS02/surfaces/*

Usage:
    uv run python scripts/18_visualize_multisubject_atoms.py \
        --model_dir outputs/models/<run>
"""
import argparse
import json
import os
import sys
from os.path import join

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import zscore as sp_zscore
from gensim.models import KeyedVectors

# Paths
from sparsebrain import paths

DATASET_DIR = paths.dataset_dir
PYCORTEX_DB = paths.pycortex_db
DERIVATIVE_DIR = paths.derivative_dir
REPORTS_DIR = paths.figures_dir
ENCODING_DIR = paths.encoding_dir

# Configure pycortex
os.environ["PYCORTEX_FILESTORE"] = PYCORTEX_DB
import cortex

sys.path.insert(0, ENCODING_DIR)
from ridge_utils.dsutils import make_word_ds
from ridge_utils.stimulus_utils import load_simulated_trfiles, load_textgrids

from sparsebrain import load_features, load_responses, STORIES_TRAIN, STORIES_TEST


def load_word_sequences(stories: list[str]) -> dict:
    """Load TextGrids and build DataSequence per story."""
    grids = load_textgrids(stories, join(DATASET_DIR, "data"))
    with open(join(DERIVATIVE_DIR, "respdict.json")) as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict)
    return make_word_ds(grids, trfiles)


def load_glove_vectors():
    """Load GloVe word vectors."""
    kv_path = join(paths.compose_dir, "data/wordvectors/glove-transformed-v2.kv")
    return KeyedVectors.load(kv_path)


def build_word_embeddings(
    embedding: str,
    stories: list[str],
    subject: str,
) -> dict[str, np.ndarray]:
    """Build word→embedding lookup by averaging per-word feature vectors.

    For LLM embeddings (gemma2, etc.), each TR has a context-dependent
    embedding. We average across all occurrences of each word type to get
    a static word vector suitable for cosine-similarity word lookup.

    For GloVe, returns None (use load_glove_vectors instead).
    """
    if embedding == "glove":
        return None

    word_seqs = load_word_sequences(stories)
    from sparsebrain import load_features
    from sparsebrain.training import apply_delays_np

    word_vecs = {}
    word_counts = {}

    for story in stories:
        if story not in word_seqs:
            continue
        X_s = load_features([story], embedding=embedding, subject=subject)
        chunks = word_seqs[story].chunks()
        n_trs = min(len(chunks), X_s.shape[0])

        for tr in range(n_trs):
            chunk = chunks[tr]
            if len(chunk) == 0:
                continue
            words = [str(w) for w in chunk]
            for w in words:
                w_lower = w.lower().strip()
                if not w_lower:
                    continue
                if w_lower not in word_vecs:
                    word_vecs[w_lower] = np.zeros(X_s.shape[1], dtype=np.float64)
                    word_counts[w_lower] = 0
                word_vecs[w_lower] += X_s[tr].astype(np.float64)
                word_counts[w_lower] += 1

    for w in word_vecs:
        word_vecs[w] /= word_counts[w]
        word_vecs[w] = word_vecs[w].astype(np.float32)

    print(f"  Built {len(word_vecs)} word embeddings from {embedding}")
    return word_vecs


def encode_topk(X: np.ndarray, encoder_weight: np.ndarray, k: int,
                b_pre: np.ndarray = None, encoder_bias: np.ndarray = None) -> np.ndarray:
    """ReLU + TopK encoding with optional b_pre and encoder_bias."""
    x = X
    if b_pre is not None:
        x = x - b_pre
    Z = x @ encoder_weight.T  # (n_samples, n_atoms)
    if encoder_bias is not None:
        Z = Z + encoder_bias
    Z = np.maximum(Z, 0)  # ReLU
    top_idx = np.argpartition(Z, -k, axis=1)[:, -k:]
    mask = np.zeros_like(Z)
    np.put_along_axis(mask, top_idx, 1.0, axis=1)
    return Z * mask


def get_top_words_for_atom(
    atom_idx: int,
    atom_weight: np.ndarray,
    word_vectors,
    n_words: int = 20,
) -> list[str]:
    """Find words whose vectors align best with an atom direction.

    word_vectors: either a gensim KeyedVectors (GloVe) or a dict of
    {word: ndarray} (LLM word embeddings from build_word_embeddings).
    """
    atom_vec = atom_weight[atom_idx]
    atom_vec = atom_vec / (np.linalg.norm(atom_vec) + 1e-8)

    if hasattr(word_vectors, 'key_to_index'):
        words = list(word_vectors.key_to_index.keys())
        vecs = np.array([word_vectors[w] for w in words])
    else:
        words = list(word_vectors.keys())
        vecs = np.array([word_vectors[w] for w in words])

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    vecs_normed = vecs / norms
    scores = vecs_normed @ atom_vec
    top_idx = np.argsort(scores)[-n_words:][::-1]
    return [words[i] for i in top_idx]


def compute_encoder_decoder_alignment(
    encoder_weight: np.ndarray,
    decoder_sem_weight: np.ndarray,
) -> np.ndarray:
    """Cosine similarity between encoder row and decoder_sem row per atom.

    High alignment = atom encodes and decodes the same concept.
    Low alignment = encoder responds to one pattern but reconstructs another.
    """
    n_atoms = encoder_weight.shape[0]
    alignment = np.zeros(n_atoms)
    for i in range(n_atoms):
        e = encoder_weight[i]
        d = decoder_sem_weight[i]
        alignment[i] = np.dot(e, d) / (np.linalg.norm(e) * np.linalg.norm(d) + 1e-8)
    return alignment


def compute_cross_subject_consistency(
    state_dict: dict,
    subjects: list[str],
    n_atoms: int,
    n_delays: int,
) -> np.ndarray:
    """Per-atom cross-subject consistency (mean pairwise rank correlation).

    For each atom, computes Spearman rank correlation of brain decoder
    weight magnitudes across all subject pairs. High consistency means
    the atom maps to similar cortical patterns across subjects.
    """
    from scipy.stats import spearmanr

    weights = {}
    for subj in subjects:
        W = state_dict[f'brain_decoders.{subj}.weight'].numpy()
        W_avg = W.reshape(W.shape[0], n_delays, n_atoms).mean(axis=1)
        weights[subj] = W_avg

    consistency = np.zeros(n_atoms)
    for a in range(n_atoms):
        corrs = []
        for i, s1 in enumerate(subjects):
            for s2 in subjects[i+1:]:
                r, _ = spearmanr(weights[s1][:, a], weights[s2][:, a])
                corrs.append(r if not np.isnan(r) else 0)
        consistency[a] = np.mean(corrs) if corrs else 0
    return consistency


def compute_delay_profile(
    state_dict: dict,
    subjects: list[str],
    n_atoms: int,
    n_delays: int,
) -> np.ndarray:
    """Average delay profile per atom across subjects.

    Incorporates delay_scales if present: the effective contribution of
    atom i at delay d is |delay_scale[d,i]| × mean(|W_brain[:, d*n_atoms+i]|).

    Returns (n_atoms, n_delays).
    """
    delay_scales = state_dict.get('delay_scales', None)
    if delay_scales is not None:
        delay_scales = delay_scales.numpy()  # (n_delays, n_atoms)

    profiles = np.zeros((n_atoms, n_delays))
    for subj in subjects:
        W = state_dict[f'brain_decoders.{subj}.weight'].numpy()
        W_reshaped = W.reshape(W.shape[0], n_delays, n_atoms)
        w_mean = np.abs(W_reshaped).mean(axis=0).T  # (n_atoms, n_delays)
        if delay_scales is not None:
            w_mean = w_mean * np.abs(delay_scales.T)  # (n_atoms, n_delays)
        profiles += w_mean
    profiles /= len(subjects)
    return profiles


def compute_leave_one_out_importance(
    state_dict: dict,
    X_test: np.ndarray,
    X_test_delayed: np.ndarray,
    Y_test_per_subject: dict,
    subjects: list[str],
    n_atoms: int,
    n_delays: int,
    k: int,
    b_pre: np.ndarray = None,
    encoder_bias: np.ndarray = None,
    voxel_scales: dict = None,
    delay_scales: np.ndarray = None,
) -> dict:
    """Leave-one-atom-out importance: Δr and Δr² per atom.

    Encodes test data once, then for each atom zeroes it out from the
    sparse codes and recomputes brain prediction. The drop in performance
    is that atom's marginal contribution.

    Returns dict with keys:
        full_r: (n_atoms_total,) — NOT per atom, just the full model's
                mean voxel r per subject, averaged across subjects
        full_r2: same but r²
        delta_r: (n_atoms,) — drop in mean r from removing each atom
        delta_r2: (n_atoms,) — drop in mean r² from removing each atom
        per_subject: {subj: {full_r, full_r2, delta_r, delta_r2}}
    """
    encoder_weight = state_dict['encoder.weight'].numpy()

    # Encode test data
    S = encode_topk(X_test, encoder_weight, k, b_pre=b_pre, encoder_bias=encoder_bias)

    # Encode delayed test data
    n_samples = X_test_delayed.shape[0]
    n_sem = X_test.shape[1]
    X_del_reshaped = X_test_delayed.reshape(n_samples, n_delays, n_sem)
    S_delayed_list = []
    for d in range(n_delays):
        S_d = encode_topk(X_del_reshaped[:, d, :], encoder_weight, k,
                          b_pre=b_pre, encoder_bias=encoder_bias)
        if delay_scales is not None:
            S_d = S_d * delay_scales[d]
        S_delayed_list.append(S_d)
    S_delayed = np.concatenate(S_delayed_list, axis=-1)  # (n_samples, n_atoms * n_delays)

    def _compute_brain_r(S_del, subj):
        """Compute per-voxel r and r² for a given subject."""
        W = state_dict[f'brain_decoders.{subj}.weight'].numpy()
        Y_hat = S_del @ W.T
        if voxel_scales and subj in voxel_scales:
            Y_hat = Y_hat * voxel_scales[subj]
        Y = Y_test_per_subject[subj]
        Y_hat_z = np.nan_to_num(sp_zscore(Y_hat, axis=0))
        Y_z = np.nan_to_num(sp_zscore(Y, axis=0))
        voxel_r = np.nan_to_num((Y_hat_z * Y_z).mean(axis=0))
        return voxel_r, voxel_r ** 2

    # Full model performance
    per_subject = {}
    full_r_all = []
    full_r2_all = []
    for subj in subjects:
        vr, vr2 = _compute_brain_r(S_delayed, subj)
        per_subject[subj] = {"full_r": float(vr.mean()), "full_r2": float(vr2.mean())}
        full_r_all.append(vr.mean())
        full_r2_all.append(vr2.mean())

    full_r = float(np.mean(full_r_all))
    full_r2 = float(np.mean(full_r2_all))

    # Leave-one-out: zero each atom across all delays
    delta_r = np.zeros(n_atoms)
    delta_r2 = np.zeros(n_atoms)
    for subj in subjects:
        per_subject[subj]["delta_r"] = np.zeros(n_atoms)
        per_subject[subj]["delta_r2"] = np.zeros(n_atoms)

    for atom_idx in range(n_atoms):
        S_del_masked = S_delayed.copy()
        for d in range(n_delays):
            col = d * n_atoms + atom_idx
            S_del_masked[:, col] = 0.0

        for subj in subjects:
            vr, vr2 = _compute_brain_r(S_del_masked, subj)
            dr = per_subject[subj]["full_r"] - float(vr.mean())
            dr2 = per_subject[subj]["full_r2"] - float(vr2.mean())
            per_subject[subj]["delta_r"][atom_idx] = dr
            per_subject[subj]["delta_r2"][atom_idx] = dr2
            delta_r[atom_idx] += dr
            delta_r2[atom_idx] += dr2

    delta_r /= len(subjects)
    delta_r2 /= len(subjects)

    print(f"  Full model: r={full_r:.4f}, r²={full_r2:.4f}")
    print(f"  Top 5 atoms by Δr: {np.argsort(delta_r)[-5:][::-1]} "
          f"(Δr={delta_r[np.argsort(delta_r)[-5:][::-1]]})")

    return {
        "full_r": full_r,
        "full_r2": full_r2,
        "delta_r": delta_r,
        "delta_r2": delta_r2,
        "per_subject": per_subject,
    }


TRIM_OFFSET = 10

STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "were", "are", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "it", "its", "i", "me", "my", "you", "your", "he", "him", "his",
    "she", "her", "we", "us", "our", "they", "them", "their", "this",
    "that", "these", "those", "what", "which", "who", "whom", "whose",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "about", "above", "after", "again", "all", "also", "am",
    "an", "any", "as", "back", "because", "before", "between", "both",
    "each", "even", "every", "few", "get", "go", "got", "here", "how",
    "into", "know", "like", "make", "more", "most", "much", "must",
    "new", "now", "off", "oh", "ok", "old", "one", "only", "other",
    "out", "over", "own", "part", "people", "right", "said", "same",
    "see", "some", "still", "such", "take", "tell", "thing", "think",
    "through", "time", "two", "under", "up", "upon", "us", "use",
    "want", "way", "well", "went", "when", "where", "while",
    "um", "uh", "ah", "oh", "yeah", "okay", "gonna", "gotta", "wanna",
})


def _get_tr_words(wordseq, tr_idx, context_trs=5):
    """Map a feature TR index to words with surrounding context."""
    raw_idx = tr_idx + TRIM_OFFSET
    chunks = wordseq.chunks()
    n_chunks = len(chunks)

    if 0 <= raw_idx < n_chunks:
        target_words = list(chunks[raw_idx])
    else:
        target_words = []

    before_words = []
    for i in range(max(0, raw_idx - context_trs), raw_idx):
        if i < n_chunks:
            before_words.extend(chunks[i])

    after_words = []
    for i in range(raw_idx + 1, min(n_chunks, raw_idx + context_trs + 1)):
        after_words.extend(chunks[i])

    return list(before_words), list(target_words), list(after_words)


def _format_context(before, target, after):
    """Format words with target highlighted in **bold**."""
    parts = []
    if before:
        parts.append("..." + " ".join(before))
    parts.append("**" + " ".join(target) + "**" if target else "**[empty TR]**")
    if after:
        parts.append(" ".join(after) + "...")
    return " ".join(parts)


def _extract_keywords(target_word_lists, n=8):
    """Extract top keywords from target TR word lists, excluding stop words."""
    from collections import Counter
    counter = Counter()
    for words in target_word_lists:
        for w in words:
            w_clean = w.strip().lower()
            if w_clean and w_clean not in STOP_WORDS and len(w_clean) > 1:
                counter[w_clean] += 1
    return counter.most_common(n)


def get_top_activating_contexts(
    atom_idx: int,
    activations: np.ndarray,
    word_seqs: dict,
    stories: list[str],
    story_lengths: list[int],
    n_top: int = 10,
    context_trs: int = 5,
) -> tuple[list[str], list[str]]:
    """Find top activating TRs and extract word contexts.

    Returns (contexts, keywords) where contexts is a list of formatted
    strings with target words **bolded** and activation values shown.
    """
    atom_acts = activations[:, atom_idx]
    top_tr_indices = np.argsort(atom_acts)[-n_top:][::-1]

    # Map global TR → (story, local_tr)
    tr_to_story = []
    for si, story in enumerate(stories):
        if story not in word_seqs:
            continue
        n_trs = story_lengths[si] if si < len(story_lengths) else len(word_seqs[story].chunks())
        for local_tr in range(n_trs):
            tr_to_story.append((story, local_tr))

    contexts_formatted = []
    contexts_split = []
    target_word_lists = []
    for global_tr in top_tr_indices:
        act_val = atom_acts[global_tr]
        if act_val <= 0 or global_tr >= len(tr_to_story):
            continue
        story, local_tr = tr_to_story[global_tr]
        before, target, after = _get_tr_words(word_seqs[story], local_tr, context_trs)

        before_str = ("..." + " ".join(before) + " ") if before else ""
        target_str = " ".join(target) if target else "[empty]"
        after_str = (" " + " ".join(after) + "...") if after else ""

        ctx = _format_context(before, target, after)
        contexts_formatted.append(f'[act={act_val:.1f}] "{story}" TR {local_tr}\n       {ctx}')
        contexts_split.append((before_str, target_str, after_str))
        target_word_lists.append(target)

    keywords = _extract_keywords(target_word_lists)
    keyword_str = [f"{w}" for w, c in keywords]
    return contexts_formatted, contexts_split, keyword_str


def compute_atom_predictiveness(
    state_dict: dict,
    test_corr: dict,
    subjects: list[str],
    n_atoms: int,
    n_delays: int,
) -> np.ndarray:
    """Compute how predictive each atom is (correlation with high-r voxels)."""
    predictiveness = {}
    for subj in subjects:
        W = state_dict[f'brain_decoders.{subj}.weight'].numpy()
        W_avg = W.reshape(W.shape[0], n_delays, n_atoms).mean(axis=1)
        r = test_corr[subj]

        atom_pred = []
        for a in range(n_atoms):
            corr = np.corrcoef(np.abs(W_avg[:, a]), r)[0, 1]
            atom_pred.append(corr if not np.isnan(corr) else 0)
        predictiveness[subj] = np.array(atom_pred)

    # Average across subjects
    return np.mean([predictiveness[s] for s in subjects], axis=0)


def render_flatmap_to_image(
    weights: np.ndarray,
    vox_idx: np.ndarray,
    subject: str,
    target_size: tuple = (550, 280),
    vmax: float = None,
) -> "Image":
    """Render a single flatmap to a PIL Image with consistent size.

    Saves to a temp file and loads back to ensure proper rendering.
    All flatmaps are resized to target_size for consistent grid layout.

    Parameters
    ----------
    vmax : float, optional
        Maximum value for colormap. If None, uses 95th percentile of weights.
        Pass a global vmax to ensure consistent scale across all plots.

    Note: Brain decoder weights are constrained to be non-negative during
    training (clamp_(min=0) after each optimizer step), so all weights >= 0.
    """
    from PIL import Image

    # Create full-brain volume
    _n_voxels_map = {'UTS01': 81126, 'UTS02': 94251, 'UTS03': 95556,
                     'UTS04': 109469, 'UTS05': 99322, 'UTS06': 92198}
    n_voxels_full = _n_voxels_map[subject]
    full_weights = np.full(n_voxels_full, np.nan)

    # Show all selected voxels (no thresholding)
    full_weights[vox_idx] = weights

    # Use provided vmax or compute from this plot
    if vmax is None:
        vmax = np.percentile(np.abs(weights), 95)
        vmax = max(vmax, 0.01)

    # Create pycortex volume
    xfm_name = f"{subject}_auto"
    vol = cortex.Volume(full_weights, subject=subject, xfmname=xfm_name,
                       vmin=0, vmax=vmax, cmap='YlOrRd')

    # Render to temp file (most reliable method)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name

    try:
        fig = cortex.quickshow(vol, with_curvature=True, with_colorbar=False,
                              with_rois=False, with_labels=False)
        fig.savefig(temp_path, dpi=80, bbox_inches='tight', facecolor='white',
                   pad_inches=0.02)
        plt.close(fig)

        # Load and resize to target size
        img = Image.open(temp_path).convert('RGB')
        img = img.resize(target_size, Image.LANCZOS)
    except Exception as e:
        # Create placeholder on error
        img = Image.new('RGB', target_size, color='lightgray')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.text((target_size[0]//2, target_size[1]//2),
                 f"{subject}\nError: {str(e)[:20]}", fill='red', anchor='mm')
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return img


def create_atom_figure(
    atom_idx: int,
    rank: int,
    label: str,
    brain_weights: dict,
    voxel_indices: dict,
    subjects: list[str],
    delta_r: float,
    consistency: float,
    freq: float,
    delay_profile: np.ndarray,
    delay_times: list[float],
    contexts: list[tuple[str, str, str]],
    save_path: str,
):
    """Plot one atom: flatmaps on top, delay profile, contexts with yellow highlight.

    Parameters
    ----------
    contexts : list of (before_str, target_str, after_str)
        Pre-split context strings for highlighting.
    """
    n_subjects = len(subjects)
    n_contexts = min(len(contexts), 15)
    line_height = 0.35
    fig_height = 5.5 + line_height * n_contexts
    fig = plt.figure(figsize=(20, fig_height))

    fig.suptitle(
        f"Atom {atom_idx}  ({label})        "
        f"Δr = {delta_r:.4f}        consistency = {consistency:.3f}        "
        f"freq = {freq:.1%}",
        fontsize=14, fontweight="bold", y=0.99,
    )

    brain_height_frac = 4.5 / fig_height
    flatmap_w = 0.85 / n_subjects
    delay_w = 0.10

    # Render flatmaps
    global_vmax = 0
    for subj in subjects:
        w = brain_weights[subj]
        v = np.percentile(np.abs(w), 95)
        global_vmax = max(global_vmax, v)
    global_vmax = max(global_vmax, 0.01)

    for j, subj in enumerate(subjects):
        _n_voxels_map = {'UTS01': 81126, 'UTS02': 94251, 'UTS03': 95556,
                         'UTS04': 109469, 'UTS05': 99322, 'UTS06': 92198}
        n_voxels_full = _n_voxels_map[subj]
        full_weights = np.full(n_voxels_full, np.nan)
        full_weights[voxel_indices[subj]] = brain_weights[subj]

        xfm_name = f"{subj}_auto"
        vol = cortex.Volume(full_weights, subject=subj, xfmname=xfm_name,
                           vmin=0, vmax=global_vmax, cmap='YlOrRd')

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        try:
            fig_c = cortex.quickshow(vol, with_curvature=True, with_colorbar=False,
                                     with_rois=False, with_labels=False)
            fig_c.savefig(temp_path, dpi=80, bbox_inches='tight', facecolor='white',
                         pad_inches=0.02)
            plt.close(fig_c)
            brain_img = plt.imread(temp_path)
        except Exception:
            brain_img = np.ones((100, 200, 3))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        ax = fig.add_axes([0.02 + j * flatmap_w, 1.0 - brain_height_frac - 0.02,
                          flatmap_w - 0.01, brain_height_frac])
        ax.imshow(brain_img)
        ax.set_title(subj, fontsize=12)
        ax.axis("off")

    # Delay profile (line plot)
    ax_delay = fig.add_axes([0.02 + n_subjects * flatmap_w + 0.02,
                             1.0 - brain_height_frac - 0.02 + brain_height_frac * 0.2,
                             delay_w, brain_height_frac * 0.6])
    ax_delay.plot(delay_times, delay_profile, 'o-', color='#4477AA', linewidth=2, markersize=6)
    ax_delay.set_xlabel("delay (s)", fontsize=9)
    ax_delay.set_ylabel("weight", fontsize=9)
    ax_delay.set_title("HRF", fontsize=10)
    ax_delay.tick_params(labelsize=8)
    ax_delay.grid(True, alpha=0.3)

    # Context sentences with yellow highlight
    text_top_frac = 1.0 - brain_height_frac - 0.04
    ax_text = fig.add_axes([0.02, 0.01, 0.96, text_top_frac - 0.01])
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    ax_text.axis("off")

    fontsize = 9
    y_step = 1.0 / max(n_contexts, 1)

    for i, (before_str, target_str, after_str) in enumerate(contexts[:n_contexts]):
        y = 1.0 - (i + 0.5) * y_step

        max_len = 130
        if len(before_str) + len(target_str) + len(after_str) > max_len:
            avail = max_len - len(target_str)
            half = avail // 2
            if len(before_str) > half:
                before_str = "..." + before_str[-(half - 3):]
            if len(after_str) > half:
                after_str = after_str[:half - 3] + "..."

        x_pos = 0.0
        ax_text.text(x_pos, y, before_str, fontsize=fontsize,
                     fontfamily="monospace", verticalalignment="center",
                     color="#888888")

        x_pos += len(before_str) * 0.0055
        ax_text.text(x_pos, y, target_str, fontsize=fontsize + 1,
                     fontfamily="monospace", verticalalignment="center",
                     fontweight="bold", color="#000000",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFEB3B",
                               edgecolor="#FFC107", alpha=0.7))

        x_pos += len(target_str) * 0.0058
        ax_text.text(x_pos, y, after_str, fontsize=fontsize,
                     fontfamily="monospace", verticalalignment="center",
                     color="#888888")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _dummy_grid(*args, **kwargs):
    """Placeholder — individual atom figures replace the old grid."""
    pass


def create_grid_figure(
    top_atoms: list[int],
    brain_weights_per_atom: dict,
    voxel_indices: dict,
    subjects: list[str],
    top_words_per_atom: dict,
    avg_pred: np.ndarray,
    pred_per_subj_per_atom: dict,
    save_path: str,
    top_words_per_atom_generates: dict | None = None,
    cross_subj_consistency: np.ndarray = None,
    enc_dec_alignment: np.ndarray = None,
    delay_profiles: np.ndarray = None,
    atom_freq: np.ndarray = None,
    delay_times: list[float] = None,
    top_contexts_per_atom: dict | None = None,
    loo_delta_r: np.ndarray = None,
    loo_delta_r2: np.ndarray = None,
):
    """Create a grid figure with atoms as rows and subjects as columns.

    Layout: n atom rows x subject columns + left info column.

    If ``top_words_per_atom_generates`` is provided, the words column is split
    into two sub-columns: "activates" (from encoder) and "generates" (from
    decoder_sem). Otherwise only one column is shown.
    """
    from PIL import Image, ImageDraw, ImageFont

    n_atoms_vis = len(top_atoms)
    n_subjects = len(subjects)
    two_word_views = top_words_per_atom_generates is not None
    has_contexts = top_contexts_per_atom is not None
    has_delays = delay_profiles is not None

    # Dimensions
    cell_w, cell_h = 550, 280
    label_w = 200
    words_w = 200 if two_word_views else 100
    context_w = 300 if has_contexts else 0
    delay_w = 120 if has_delays else 0
    info_w = label_w + words_w + context_w + delay_w
    header_h = 50

    # Total image size
    total_w = info_w + cell_w * n_subjects
    total_h = header_h + cell_h * n_atoms_vis

    # Create blank image
    grid_img = Image.new('RGB', (total_w, total_h), color='white')
    draw = ImageDraw.Draw(grid_img)

    # Try to load a font
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font_large = ImageFont.load_default()
        font_med = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw header row with subject names
    for j, subj in enumerate(subjects):
        x = info_w + j * cell_w + cell_w // 2
        draw.text((x, header_h // 2), subj, fill='black', font=font_large, anchor='mm')

    # Words column headers
    if two_word_views:
        col_w = words_w // 2
        draw.text(
            (label_w + col_w // 2, header_h // 2),
            "activates", fill='#004400', font=font_small, anchor='mm',
        )
        draw.text(
            (label_w + col_w + col_w // 2, header_h // 2),
            "generates", fill='#004400', font=font_small, anchor='mm',
        )
    else:
        draw.text(
            (label_w + words_w // 2, header_h // 2),
            "keywords", fill='#004400', font=font_small, anchor='mm',
        )

    # Compute global vmax across all atoms and subjects for consistent colormap
    all_weights = []
    for atom_idx in top_atoms:
        for subj in subjects:
            all_weights.append(brain_weights_per_atom[atom_idx][subj])
    all_weights = np.concatenate(all_weights)
    global_vmax = np.percentile(np.abs(all_weights), 95)
    global_vmax = max(global_vmax, 0.01)
    print(f"  Global vmax for colormap: {global_vmax:.4f}")

    # Add context and delay column headers
    if has_contexts:
        draw.text(
            (label_w + words_w + context_w // 2, header_h // 2),
            "top contexts", fill='#333333', font=font_small, anchor='mm',
        )
    if has_delays:
        draw.text(
            (label_w + words_w + context_w + delay_w // 2, header_h // 2),
            "delay", fill='#333333', font=font_small, anchor='mm',
        )

    # Draw each atom row
    print(f"  Rendering {n_atoms_vis} atoms x {n_subjects} subjects...")
    for i, atom_idx in enumerate(top_atoms):
        y_base = header_h + i * cell_h

        # === Left column: atom info ===
        pred = avg_pred[atom_idx]
        pred_subj = pred_per_subj_per_atom.get(atom_idx, {})

        draw.text((10, y_base + 15), f"#{i+1} Atom {atom_idx}", fill='black', font=font_large)
        draw.text((10, y_base + 45), f"pred = {pred:.3f}", fill='#444444', font=font_med)

        # Additional metrics
        y_offset = 65
        if cross_subj_consistency is not None:
            cons = cross_subj_consistency[atom_idx]
            color = '#228B22' if cons > 0.3 else '#666666'
            draw.text((10, y_base + y_offset), f"x-subj = {cons:.2f}", fill=color, font=font_small)
            y_offset += 16
        if enc_dec_alignment is not None:
            align = enc_dec_alignment[atom_idx]
            color = '#228B22' if align > 0.5 else '#CC6600' if align > 0.2 else '#CC0000'
            draw.text((10, y_base + y_offset), f"enc-dec = {align:.2f}", fill=color, font=font_small)
            y_offset += 16
        if loo_delta_r is not None:
            dr = loo_delta_r[atom_idx]
            color = '#228B22' if dr > 0.005 else '#666666'
            draw.text((10, y_base + y_offset), f"Δr = {dr:.4f}", fill=color, font=font_small)
            y_offset += 16
        if loo_delta_r2 is not None:
            dr2 = loo_delta_r2[atom_idx]
            draw.text((10, y_base + y_offset), f"Δr² = {dr2:.4f}", fill='#666666', font=font_small)
            y_offset += 16
        if atom_freq is not None:
            freq = atom_freq[atom_idx]
            draw.text((10, y_base + y_offset), f"freq = {freq:.2f}", fill='#666666', font=font_small)
            y_offset += 16

        # Per-subject scores
        for subj in subjects:
            score = pred_subj.get(subj, 0)
            color = '#228B22' if score > 0.2 else '#666666'
            draw.text((10, y_base + y_offset), f"{subj}: {score:.2f}", fill=color, font=font_small)
            y_offset += 16

        # === Words column ===
        if two_word_views:
            col_w = words_w // 2
            for col_i, (words_dict, color) in enumerate([
                (top_words_per_atom, '#006400'),          # activates (encoder)
                (top_words_per_atom_generates, '#4B0082'),  # generates (decoder_sem)
            ]):
                words = words_dict.get(atom_idx, [])[:10]
                x_words = label_w + col_i * col_w + 5
                for w_idx, word in enumerate(words):
                    y_word = y_base + 10 + w_idx * 26
                    if y_word < y_base + cell_h - 15:
                        draw.text((x_words, y_word), word, fill=color, font=font_small)
        else:
            words = top_words_per_atom.get(atom_idx, [])[:10]
            x_words = label_w + 5
            for w_idx, word in enumerate(words):
                y_word = y_base + 10 + w_idx * 26
                if y_word < y_base + cell_h - 15:
                    draw.text((x_words, y_word), word, fill='#006400', font=font_small)

        # === Context column: top activating text snippets ===
        if has_contexts:
            contexts = top_contexts_per_atom.get(atom_idx, [])[:6]
            x_ctx = label_w + words_w + 5
            for c_idx, ctx in enumerate(contexts):
                y_ctx = y_base + 10 + c_idx * 44
                if y_ctx < y_base + cell_h - 15:
                    ctx_trunc = ctx[:50] + "..." if len(ctx) > 50 else ctx
                    draw.text((x_ctx, y_ctx), ctx_trunc, fill='#333333', font=font_small)

        # === Delay profile: mini bar chart ===
        if has_delays:
            profile = delay_profiles[atom_idx]
            x_delay = label_w + words_w + context_w + 10
            bar_area_w = delay_w - 20
            bar_area_h = cell_h - 40
            y_delay_base = y_base + 30
            max_val = max(profile.max(), 1e-8)
            n_del = len(profile)
            bar_w = bar_area_w // n_del - 2
            for d_idx, val in enumerate(profile):
                bar_h = int(bar_area_h * val / max_val)
                x0 = x_delay + d_idx * (bar_w + 2)
                y0 = y_delay_base + bar_area_h - bar_h
                draw.rectangle([x0, y0, x0 + bar_w, y_delay_base + bar_area_h],
                               fill='#4477AA')
                if delay_times is not None and d_idx < len(delay_times):
                    draw.text((x0, y_delay_base + bar_area_h + 2),
                              f"{delay_times[d_idx]}s", fill='#666666', font=font_small)

        # Flatmaps for each subject
        for j, subj in enumerate(subjects):
            weights = brain_weights_per_atom[atom_idx][subj]
            vox_idx = voxel_indices[subj]

            # Render flatmap to PIL Image with global vmax for consistent scale
            flatmap_img = render_flatmap_to_image(
                weights, vox_idx, subj, target_size=(cell_w, cell_h),
                vmax=global_vmax,
            )

            # Paste into grid
            x = info_w + j * cell_w
            y = y_base
            grid_img.paste(flatmap_img, (x, y))

        # Progress
        if (i + 1) % 5 == 0:
            print(f"    {i + 1}/{n_atoms_vis} atoms rendered")

    # Save
    grid_img.save(save_path, quality=95)
    print(f"  Saved grid: {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--n_atoms", type=int, default=20, help="Number of top atoms to visualize")
    parser.add_argument("--y_test_cache", type=str, default=None,
                        help="Training cache .pt (outputs/data_cache/multisubj_*.pt) to take the "
                             "test-story responses from, on a machine without the raw dataset. "
                             "Its Y_test_per_subject is already voxel-selected; the voxel indices "
                             "are checked against the checkpoint. Requires trim_start=trim_end=0.")
    parser.add_argument("--n_words", type=int, default=20, help="Number of top words per atom")
    parser.add_argument("--rank_by", type=str, default="delta_r",
                        choices=["predictiveness", "consistency", "frequency", "delta_r"],
                        help="Metric to rank atoms by")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_dir}")
    ckpt_path = join(args.model_dir, "model_last.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"{ckpt_path} not found: every result uses the final-epoch checkpoint model_last.pt")
    print(f"Loading checkpoint: {os.path.basename(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    state = ckpt['state_dict']
    model_args = ckpt['args']
    subjects = ckpt['subjects']
    test_corr = ckpt['test_brain_corr']
    voxel_indices = ckpt['voxel_indices']
    train_stories = ckpt.get('train_stories', STORIES_TRAIN[:26])

    n_atoms = model_args['n_atoms']
    delays_str = model_args['delays']
    delays = [int(d) for d in delays_str.split(',')]
    n_delays = len(delays)
    k = model_args['k']
    embedding = model_args.get('embedding', 'glove')

    print(f"  Subjects: {subjects}")
    print(f"  Atoms: {n_atoms}, k: {k}")
    print(f"  Embedding: {embedding}")

    # -------------------------------------------------------------------------
    # Compute all atom metrics
    # -------------------------------------------------------------------------

    # 1. Predictiveness (correlation with high-r voxels)
    print("\nComputing atom predictiveness...")
    avg_pred = compute_atom_predictiveness(state, test_corr, subjects, n_atoms, n_delays)

    # 2. Cross-subject consistency
    print("Computing cross-subject consistency...")
    cross_subj = compute_cross_subject_consistency(state, subjects, n_atoms, n_delays)

    # 3. Activation frequency (from checkpoint if available)
    atom_freq = ckpt.get('atom_activation_freq', None)
    if atom_freq is not None:
        print(f"  Atom frequency: mean={atom_freq.mean():.3f}, range=[{atom_freq.min():.3f}, {atom_freq.max():.3f}]")

    # 4. Encoder-decoder alignment
    encoder_weight = state['encoder.weight'].numpy()
    has_decoder_sem = 'decoder_sem.weight' in state
    if has_decoder_sem:
        decoder_sem_weight = state['decoder_sem.weight'].numpy().T
        enc_dec_align = compute_encoder_decoder_alignment(encoder_weight, decoder_sem_weight)
        print(f"  Enc-dec alignment: mean={enc_dec_align.mean():.3f}, range=[{enc_dec_align.min():.3f}, {enc_dec_align.max():.3f}]")
    else:
        decoder_sem_weight = None
        enc_dec_align = None

    # 5. Delay profiles (HRF time course per atom)
    print("Computing delay profiles...")
    delay_profiles = compute_delay_profile(state, subjects, n_atoms, n_delays)
    delay_times = [d * 2 for d in delays]  # delays in seconds (TR=2s)

    # Rank atoms (delta_r computed later, use predictiveness as placeholder)
    rank_by = args.rank_by
    if rank_by == "consistency":
        ranking = cross_subj
    elif rank_by == "frequency":
        ranking = atom_freq if atom_freq is not None else avg_pred
    elif rank_by == "delta_r":
        ranking = None  # computed after LOO
    else:
        ranking = avg_pred

    # -------------------------------------------------------------------------
    # Word interpretation (embedding-aware)
    # -------------------------------------------------------------------------
    print("\nLoading semantic resources...")
    if embedding == 'glove':
        word_vectors = load_glove_vectors()
    else:
        word_vectors = build_word_embeddings(embedding, train_stories, subjects[0])

    # -------------------------------------------------------------------------
    # Pre-compute activations and word sequences (needed for contexts + LOO)
    # -------------------------------------------------------------------------
    print("\nComputing activations...")
    b_pre = state.get('b_pre', None)
    if b_pre is not None:
        b_pre = b_pre.numpy()
    enc_bias = state.get('encoder_bias', None)
    if enc_bias is not None:
        enc_bias = enc_bias.numpy()

    word_seqs = load_word_sequences(train_stories)
    X_all = load_features(train_stories, embedding=embedding, subject=subjects[0])

    # Apply input RMS normalization if the model used it
    do_rms = model_args.get('input_rms_normalize', False) or \
             model_args.get('loss_normalization', '') == 'input_rms'
    if do_rms:
        train_rms = float(np.sqrt(np.mean(X_all ** 2)))
        X_all = (X_all / train_rms).astype(np.float32)
        print(f"  Applied input RMS normalization (rms={train_rms:.2f})")

    activations = encode_topk(X_all, encoder_weight, k, b_pre=b_pre, encoder_bias=enc_bias)

    # -------------------------------------------------------------------------
    # Leave-one-out importance (Δr and Δr²)
    # -------------------------------------------------------------------------
    print("\nComputing leave-one-out importance...")
    from sparsebrain import load_responses
    from sparsebrain.training import apply_delays_np

    trim_s = model_args.get('trim_start', 4)
    trim_e = model_args.get('trim_end', 0)
    sl = slice(trim_s, -trim_e if trim_e > 0 else None)

    X_test_parts, X_test_del_parts = [], []
    Y_test_parts = {subj: [] for subj in subjects}
    for s in STORIES_TEST:
        X_s = load_features([s], embedding=embedding, subject=subjects[0])
        X_del_s = apply_delays_np(X_s, delays)
        X_test_parts.append(X_s[sl])
        X_test_del_parts.append(X_del_s[sl])
        for subj in subjects:
            if args.y_test_cache:
                continue  # taken from the cache below
            Y_s = load_responses([s], subj)
            Y_s = np.nan_to_num(Y_s)[:, voxel_indices[subj]]
            Y_test_parts[subj].append(Y_s[sl])
    if args.y_test_cache:
        assert len(STORIES_TEST) == 1 and trim_s == 0 and trim_e == 0, \
            "--y_test_cache supports one untrimmed test story"
        _cache = torch.load(args.y_test_cache, map_location='cpu', mmap=True, weights_only=False)
        for subj in subjects:
            _vi = np.asarray(_cache['voxel_indices_per_subject'][subj])
            assert np.array_equal(_vi, np.asarray(voxel_indices[subj])), f"voxel indices differ for {subj}"
            Y_test_parts[subj].append(np.nan_to_num(np.asarray(_cache['Y_test_per_subject'][subj], dtype=np.float32)))
        print(f"  Test responses taken from {args.y_test_cache}")
        del _cache

    X_test = np.concatenate(X_test_parts, axis=0)
    X_test_delayed = np.concatenate(X_test_del_parts, axis=0)
    Y_test_per_subject = {}
    for subj in subjects:
        Y = np.concatenate(Y_test_parts[subj], axis=0)
        Y_test_per_subject[subj] = np.nan_to_num(sp_zscore(Y, axis=0)).astype(np.float32)

    if do_rms:
        X_test = (X_test / train_rms).astype(np.float32)
        X_test_delayed = (X_test_delayed / train_rms).astype(np.float32)

    vscales = {}
    for subj in subjects:
        key = f'voxel_scales.{subj}'
        if key in state:
            vscales[subj] = state[key].numpy()

    dscales = state.get('delay_scales', None)
    if dscales is not None:
        dscales = dscales.numpy()

    loo = compute_leave_one_out_importance(
        state, X_test, X_test_delayed, Y_test_per_subject,
        subjects, n_atoms, n_delays, k,
        b_pre=b_pre, encoder_bias=enc_bias,
        voxel_scales=vscales if vscales else None,
        delay_scales=dscales,
    )

    # -------------------------------------------------------------------------
    # Select top atoms (now that all metrics including LOO are computed)
    # -------------------------------------------------------------------------
    if rank_by == "delta_r":
        ranking = loo["delta_r"]
    if ranking is None:
        ranking = avg_pred
    top_atoms = list(np.argsort(ranking)[-args.n_atoms:][::-1])
    print(f"\nTop {args.n_atoms} atoms (by {rank_by}): {top_atoms}")

    # -------------------------------------------------------------------------
    # Word interpretation + contexts (now that top_atoms is selected)
    # -------------------------------------------------------------------------
    # For GloVe: use cosine similarity word lookup (Activates/Generates)
    # For LLM embeddings: use only keyword extraction from top contexts
    use_cosine_words = (embedding == "glove") and (word_vectors is not None)

    top_words_activates = {}
    top_words_generates = {} if has_decoder_sem and use_cosine_words else None
    if use_cosine_words:
        print("Getting top words per atom (GloVe cosine)...")
        for atom_idx in top_atoms:
            top_words_activates[atom_idx] = get_top_words_for_atom(
                atom_idx, encoder_weight, word_vectors, args.n_words
            )
            if has_decoder_sem:
                top_words_generates[atom_idx] = get_top_words_for_atom(
                    atom_idx, decoder_sem_weight, word_vectors, args.n_words
                )

    print("Extracting top activating contexts...")
    feat_story_lengths = []
    for story in train_stories:
        X_s = load_features([story], embedding=embedding, subject=subjects[0])
        feat_story_lengths.append(X_s.shape[0])

    top_contexts = {}
    top_contexts_split = {}
    atom_keywords = {}
    for atom_idx in top_atoms:
        ctxs, ctxs_split, kws = get_top_activating_contexts(
            atom_idx, activations, word_seqs, train_stories,
            story_lengths=feat_story_lengths, n_top=20,
        )
        top_contexts[atom_idx] = ctxs
        top_contexts_split[atom_idx] = ctxs_split
        atom_keywords[atom_idx] = kws

    # -------------------------------------------------------------------------
    # Brain weights per atom
    # -------------------------------------------------------------------------
    print("\nExtracting brain weights...")
    brain_weights_per_atom = {}
    for atom_idx in top_atoms:
        brain_weights_per_atom[atom_idx] = {}
        for subj in subjects:
            W = state[f'brain_decoders.{subj}.weight'].numpy()
            W_avg = W.reshape(W.shape[0], n_delays, n_atoms).mean(axis=1)
            brain_weights_per_atom[atom_idx][subj] = W_avg[:, atom_idx]

    pred_per_subj_per_atom = {}
    for atom_idx in top_atoms:
        pred_per_subj_per_atom[atom_idx] = {
            s: compute_atom_predictiveness(state, test_corr, [s], n_atoms, n_delays)[atom_idx]
            for s in subjects
        }

    # -------------------------------------------------------------------------
    # Create output
    # -------------------------------------------------------------------------
    model_name = os.path.basename(args.model_dir)
    out_dir = join(REPORTS_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Save per-atom context files (like script 11)
    # -------------------------------------------------------------------------
    contexts_dir = join(out_dir, "atom_contexts")
    os.makedirs(contexts_dir, exist_ok=True)

    for rank, atom_idx in enumerate(top_atoms):
        kws = atom_keywords.get(atom_idx, [])
        kw_label = "_".join(kws[:3]) if kws else "unknown"
        kw_label = kw_label.replace("'", "").replace(" ", "_")
        freq = atom_freq[atom_idx] if atom_freq is not None else 0
        mean_act = activations[:, atom_idx][activations[:, atom_idx] > 0].mean() \
            if (activations[:, atom_idx] > 0).any() else 0

        lines = []
        lines.append(f"=== Atom {atom_idx} ({kw_label}) "
                      f"(rank {rank}, Δr={loo['delta_r'][atom_idx]:.4f}, "
                      f"freq={freq:.1%}, mean_act={mean_act:.2f}) ===")
        if kws:
            lines.append(f"Keywords: {', '.join(kws)}")
        lines.append("")
        ctxs = top_contexts.get(atom_idx, [])
        lines.append(f"Top activating contexts ({len(ctxs)} shown):")
        for i, ctx in enumerate(ctxs):
            lines.append(f"    {i+1}. {ctx}")
        lines.append("")

        atom_path = join(contexts_dir, f"rank{rank:02d}_{kw_label}.txt")
        with open(atom_path, "w") as f:
            f.write("\n".join(lines))

    print(f"  Saved {len(top_atoms)} atom context files to {contexts_dir}/")

    # -------------------------------------------------------------------------
    # Create per-atom figures (flatmaps + contexts with yellow highlight)
    # -------------------------------------------------------------------------
    atoms_dir = join(out_dir, "brain_atoms")
    os.makedirs(atoms_dir, exist_ok=True)
    print(f"\nCreating per-atom figures for top {args.n_atoms} atoms...")
    for rank, atom_idx in enumerate(top_atoms):
        kws = atom_keywords.get(atom_idx, [])
        kw_label = "_".join(kws[:3]) if kws else "unknown"
        kw_label = kw_label.replace("'", "").replace(" ", "_")
        freq = atom_freq[atom_idx] if atom_freq is not None else 0

        atom_path = join(atoms_dir, f"rank{rank:02d}_{kw_label}.png")
        create_atom_figure(
            atom_idx=atom_idx,
            rank=rank,
            label=kw_label,
            brain_weights=brain_weights_per_atom[atom_idx],
            voxel_indices=voxel_indices,
            subjects=subjects,
            delta_r=loo["delta_r"][atom_idx],
            consistency=cross_subj[atom_idx],
            freq=freq,
            delay_profile=delay_profiles[atom_idx],
            delay_times=delay_times,
            contexts=top_contexts_split.get(atom_idx, []),
            save_path=atom_path,
        )
        if (rank + 1) % 5 == 0:
            print(f"    {rank + 1}/{args.n_atoms} atoms rendered")

    print(f"  Saved atom figures to {atoms_dir}/")

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------
    summary_path = join(out_dir, "top_atoms_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Top {args.n_atoms} Atoms (ranked by {args.rank_by})\n")
        f.write(f"Full model: r={loo['full_r']:.4f}, r²={loo['full_r2']:.4f}\n")
        f.write("=" * 60 + "\n\n")
        for rank, atom_idx in enumerate(top_atoms):
            kws = atom_keywords.get(atom_idx, [])
            kw_label = "_".join(kws[:3]) if kws else "unknown"
            freq = atom_freq[atom_idx] if atom_freq is not None else 0
            mean_act = activations[:, atom_idx][activations[:, atom_idx] > 0].mean() \
                if (activations[:, atom_idx] > 0).any() else 0

            f.write(f"#{rank+1} Atom {atom_idx} ({kw_label})\n")
            f.write(f"    Δr={loo['delta_r'][atom_idx]:.4f}  "
                    f"Δr²={loo['delta_r2'][atom_idx]:.4f}  "
                    f"pred={avg_pred[atom_idx]:.3f}  "
                    f"x-subj={cross_subj[atom_idx]:.3f}  "
                    f"freq={freq:.1%}  mean_act={mean_act:.2f}\n")
            per_subj = [f"{s}={pred_per_subj_per_atom[atom_idx][s]:.3f}" for s in subjects]
            f.write(f"    Per subject: {', '.join(per_subj)}\n")
            if kws:
                f.write(f"    Keywords: {', '.join(kws)}\n")
            if use_cosine_words and atom_idx in top_words_activates:
                f.write(f"    Activates: {', '.join(top_words_activates[atom_idx][:10])}\n")
                if top_words_generates and atom_idx in top_words_generates:
                    f.write(f"    Generates: {', '.join(top_words_generates[atom_idx][:10])}\n")

            ctxs = top_contexts.get(atom_idx, [])
            f.write(f"\n    Top activating contexts ({len(ctxs)} shown):\n")
            for i, ctx in enumerate(ctxs):
                f.write(f"    {i+1}. {ctx}\n")
            f.write("\n")

    print(f"\nSaved summary to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
