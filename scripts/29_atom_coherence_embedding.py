#!/usr/bin/env python
"""29_atom_coherence_embedding.py — Per-atom semantic coherence + story diversity.

For each atom in a coupled SAE checkpoint, parses its existing
`atom_contexts/rankNN_*.txt` file to get the top-N activating contexts,
embeds them with sentence-transformers, and computes:

  * top_coherence: mean pairwise cosine similarity of the N embeddings
  * random_coherence_mean / std: same metric on N random training-story
    contexts, repeated 10x — gives a chance baseline
  * coherence_z: (top_coherence - random_mean) / random_std
  * n_unique_stories, story_entropy: how diverse the top contexts are
    across the training-story corpus

Story diversity is reported alongside coherence so we can distinguish
"semantically focused cross-story atom" (high coherence + high diversity,
top-right of the scatter) from "story-specific atom" (high coherence + low
diversity).

Default embedder is sentence-transformers/all-mpnet-base-v2 (top of MTEB
sentence-similarity benchmark, free, offline). Use --embedder to swap.

Usage:
    uv run python scripts/29_atom_coherence_embedding.py \\
        --model_dir <path-to-coupled-SAE-checkpoint-dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from os.path import join

import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from sparsebrain.data import load_features

# Pycortex setup needed by ridge_utils on import
import cortex  # noqa: E402
PYCORTEX_DB = os.path.join(PROJECT_DIR, "external", "deep-fMRI-dataset",
                            "data", "ds003020", "derivative", "pycortex-db")
cortex.options.config.set("basic", "filestore", PYCORTEX_DB)

DATASET_DIR = os.path.join(PROJECT_DIR, "external", "deep-fMRI-dataset")
DATA_DIR = os.path.join(DATASET_DIR, "data")
DERIVATIVE_DIR = os.path.join(DATA_DIR, "ds003020", "derivative")
ENCODING_DIR = os.path.join(DATASET_DIR, "encoding")
sys.path.insert(0, ENCODING_DIR)

from ridge_utils.dsutils import make_word_ds  # noqa: E402
from ridge_utils.stimulus_utils import load_simulated_trfiles, load_textgrids  # noqa: E402

# TRIM offset: feature TR j corresponds to raw chunk j + TRIM_OFFSET
# (matches scripts/11_interpret_coupled_atoms.py)
TRIM_OFFSET = 10

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parsing the existing atom_contexts/rankNN_*.txt files
# ---------------------------------------------------------------------------

CTX_HEADER_RE = re.compile(
    r"=== Atom (\d+) \(([^)]+)\) "
    r"\(rank (\d+), [^=]*Δr=([0-9.\-]+), freq=([0-9.]+)%, mean_act=([0-9.]+)\) ==="
)
CTX_LINE_RE = re.compile(r'^\s*\d+\.\s+\[act=([0-9.\-]+)\]\s+"([^"]+)"\s+TR\s+(\d+)\s*$')


def parse_atom_contexts(model_figs_dir):
    """Parse all rankNN_*.txt files in the contexts dir.

    Returns
    -------
    dict[atom_idx] = {
        "label": str, "rank": int, "dr": float, "freq": float, "mean_act": float,
        "kw": str,
        "contexts": list of dicts with keys (story, tr, act, text)
    }
    """
    ctx_dir = join(model_figs_dir, "atom_contexts")
    out = {}
    for path in sorted(glob.glob(join(ctx_dir, "*.txt"))):
        lines = open(path).readlines()
        if not lines:
            continue
        m = CTX_HEADER_RE.search(lines[0])
        if not m:
            print(f"  skip (header parse fail): {os.path.basename(path)}")
            continue
        atom = int(m.group(1))
        kw_line = lines[1] if len(lines) > 1 else ""
        kw = kw_line.replace("Keywords:", "").strip() if kw_line.startswith("Keywords:") else ""

        # Walk the lines: each context spans 2 lines: header + text
        contexts = []
        i = 0
        while i < len(lines):
            mm = CTX_LINE_RE.match(lines[i])
            if mm:
                act = float(mm.group(1))
                story = mm.group(2)
                tr = int(mm.group(3))
                # Next non-empty line is the context body
                if i + 1 < len(lines):
                    text = lines[i + 1].strip()
                    # Strip the leading/trailing "..." and the bold markers
                    text = re.sub(r"^\.\.\.", "", text)
                    text = re.sub(r"\.\.\.$", "", text)
                    text = text.replace("**", "").strip()
                    contexts.append({"story": story, "tr": tr, "act": act, "text": text})
                i += 2
            else:
                i += 1

        out[atom] = {
            "label": m.group(2), "rank": int(m.group(3)), "dr": float(m.group(4)),
            "freq": float(m.group(5)), "mean_act": float(m.group(6)),
            "kw": kw, "contexts": contexts,
        }
    return out


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def load_embedder(name):
    """Load a sentence-transformers model. Returns a SentenceTransformer."""
    from sentence_transformers import SentenceTransformer
    print(f"[embedder] loading {name!r} ...")
    return SentenceTransformer(name, device="cpu")


def embed_texts(embedder, texts, batch_size=32):
    """Return L2-normalized embeddings of shape (N, dim)."""
    embs = embedder.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                            show_progress_bar=False, normalize_embeddings=True)
    return embs


def coherence(embs):
    """Mean pairwise cosine of L2-normalized embeddings (upper triangular)."""
    if len(embs) < 2:
        return np.nan
    sim = embs @ embs.T
    iu = np.triu_indices(len(embs), k=1)
    return float(sim[iu].mean())


# ---------------------------------------------------------------------------
# Story diversity
# ---------------------------------------------------------------------------

def story_diversity(stories, n_total):
    """Returns (n_unique, entropy_bits, entropy_normalized).

    Entropy is normalized to [0, 1] by log2(n_total) (the cap when each context
    is from a different story).
    """
    if not stories:
        return 0, 0.0, 0.0
    from collections import Counter
    cnt = Counter(stories)
    n = len(stories)
    probs = np.array([c / n for c in cnt.values()])
    ent = float(-np.sum(probs * np.log2(probs + 1e-30)))
    cap = np.log2(min(n, n_total))
    return len(cnt), ent, ent / cap if cap > 0 else 0.0


# ---------------------------------------------------------------------------
# Word sequence loading + context-window builder
# ---------------------------------------------------------------------------

def load_word_sequences(stories):
    """Load per-story word DataSequences from TextGrids."""
    grids = load_textgrids(stories, DATA_DIR)
    with open(os.path.join(DERIVATIVE_DIR, "respdict.json")) as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict)
    return make_word_ds(grids, trfiles)


def build_context_text(wordseq, tr_idx, context_trs=3):
    """Build a context-window string for (story, tr_idx). Mirrors the format in
    the existing atom_contexts/ files (target words from the active TR plus
    several TRs of context on each side).

    Parameters
    ----------
    wordseq : DataSequence
    tr_idx : int (feature TR — already TRIM-offset compensated externally)
    context_trs : int

    Returns
    -------
    text : str   single-line context
    n_words : int
    """
    raw_idx = tr_idx + TRIM_OFFSET
    chunks = wordseq.chunks()
    n = len(chunks)
    if not (0 <= raw_idx < n):
        return "", 0
    pieces = []
    for i in range(max(0, raw_idx - context_trs), min(n, raw_idx + context_trs + 1)):
        pieces.extend(chunks[i])
    text = " ".join(str(w) for w in pieces).strip()
    return text, len(pieces)


def story_length_features_trs(wordseq):
    """Length of a story in feature TRs (i.e. n_chunks - TRIM_OFFSET, clipped)."""
    n = len(wordseq.chunks())
    return max(0, n - TRIM_OFFSET)


# ---------------------------------------------------------------------------
# Random baseline — sample uniformly from ALL training-story TRs (Fix A)
# ---------------------------------------------------------------------------

def build_uniform_tr_pool(wordseqs):
    """List of all (story, feature_tr_idx) across training stories. Each TR is
    a candidate uniformly sampleable point; weighting is implicitly by story
    length (longer stories contribute more TRs)."""
    pool = []
    for story, ws in wordseqs.items():
        n = story_length_features_trs(ws)
        for tr in range(n):
            pool.append((story, tr))
    return pool


def random_coherence_baseline(pool, wordseqs, embedder, n_per_sample=20,
                               n_samples=10, context_trs=3, rng=None):
    """Sample N random (story, feature_tr) triples uniformly from the pool of
    ALL training TRs, build context windows, embed, and compute coherence.
    Repeated n_samples times.

    Returns mean, std of coherence across draws, and a list of unique-story
    counts per draw.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_pool = len(pool)
    print(f"[random] sampling {n_samples} draws of {n_per_sample} contexts from "
          f"the FULL pool of {n_pool} TRs across {len(wordseqs)} training stories")
    cohs, uniqs = [], []
    for _ in range(n_samples):
        idx = rng.choice(n_pool, size=n_per_sample, replace=False)
        triples = [pool[i] for i in idx]
        texts = [build_context_text(wordseqs[s], t, context_trs)[0]
                 for (s, t) in triples]
        # Filter out empties (out-of-range TRs)
        texts = [tx for tx in texts if tx]
        if len(texts) < 2:
            continue
        embs = embed_texts(embedder, texts)
        cohs.append(coherence(embs))
        uniqs.append(len(set(s for (s, _) in triples)))
    return float(np.mean(cohs)), float(np.std(cohs, ddof=1)), uniqs


# ---------------------------------------------------------------------------
# Lexical / POS / speech-rate / silence-distance metrics
# ---------------------------------------------------------------------------

_SPACY_NLP = None
_PYPHEN_DICT = None


def _get_spacy():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        _SPACY_NLP = spacy.load("en_core_web_sm",
                                 disable=["parser", "ner", "lemmatizer"])
        # parser disabled → doc.sents not set; add lightweight sentencizer
        if "sentencizer" not in _SPACY_NLP.pipe_names:
            _SPACY_NLP.add_pipe("sentencizer")
    return _SPACY_NLP


def _get_pyphen():
    global _PYPHEN_DICT
    if _PYPHEN_DICT is None:
        import pyphen
        _PYPHEN_DICT = pyphen.Pyphen(lang="en_US")
    return _PYPHEN_DICT


def count_syllables(word):
    """Count syllables in a word using pyphen hyphenation.

    Returns 0 for empty input or words with no alphabetic chars (e.g. punctuation),
    so a comma or ellipsis doesn't get counted as 1 syllable.
    """
    if not word:
        return 0
    s = word.lower()
    if not any(c.isalpha() for c in s):
        return 0
    d = _get_pyphen()
    inserted = d.inserted(s)
    return inserted.count("-") + 1


def precompute_silence_features(wordseq):
    """Per raw-chunk metrics; index by raw chunk idx (= feature_tr + TRIM_OFFSET).

    Returns dict with arrays of length n_chunks:
      n_words[i]:           number of words in raw chunk i
      n_syllables[i]:       sum of syllables of words in raw chunk i
      is_silent[i]:         bool, True if n_words[i] == 0
      trs_since_silence[i]: 0 if silent; otherwise # raw TRs back to most recent silent
      trs_since_sentence_end[i]: like silence-distance but for sentence ends from spaCy
    """
    chunks = wordseq.chunks()
    n = len(chunks)
    n_words = np.array([len(c) for c in chunks], dtype=np.int32)
    n_syll = np.array([sum(count_syllables(str(w)) for w in c) for c in chunks],
                       dtype=np.int32)
    is_silent = (n_words == 0)

    # Distance to most recent silence (in raw TRs). 0 at silent TRs themselves.
    trs_since_silence = np.zeros(n, dtype=np.int32)
    last_sil = -10**9
    for i in range(n):
        if is_silent[i]:
            last_sil = i
            trs_since_silence[i] = 0
        else:
            trs_since_silence[i] = i - last_sil

    # NOTE: sentence-end detection from these TextGrid transcripts is unreliable
    # because the transcripts contain no punctuation (so spaCy's sentencizer
    # finds no sentence boundaries). We use silence-distance as the closest
    # proxy: most natural-speech sentence boundaries align with brief pauses.
    # An "utterance-length" view would compute time-since-silence + time-to-next-silence.
    # Compute trs_to_next_silence as a complementary metric.
    trs_to_next_silence = np.zeros(n, dtype=np.int32)
    next_sil = 10**9
    for i in range(n - 1, -1, -1):
        if is_silent[i]:
            next_sil = i
            trs_to_next_silence[i] = 0
        else:
            trs_to_next_silence[i] = next_sil - i if next_sil < 10**9 else -1
    # Cap "no silence ever ahead" to a sensible value (story length)
    if (trs_to_next_silence < 0).any():
        trs_to_next_silence[trs_to_next_silence < 0] = n

    return dict(n_words=n_words, n_syllables=n_syll, is_silent=is_silent,
                trs_since_silence=trs_since_silence,
                trs_to_next_silence=trs_to_next_silence)


def pos_distribution(words):
    """Count POS tags for a list of words. Joins words with space and lets
    spaCy retokenize.

    Note: spaCy's English tokenizer splits contractions ("i'm" -> "i" + "'m";
    "don't" -> "do" + "n't"). This means contractions are counted as 2 tokens
    each (PRON+AUX, AUX+PART, etc.). We accept this because the tagger is
    trained on the split form and gives accurate POS labels — passing "i'm"
    as a single token to the tagger via Doc(vocab, words=...) gives wrong
    POS labels (tagger has no training signal for that surface form).

    Returns dict tag -> count over spaCy's tokenization (not the input).
    """
    if not words:
        return {}
    nlp = _get_spacy()
    text = " ".join(str(w) for w in words)
    doc = nlp(text)
    out = {}
    for t in doc:
        if t.is_space:
            continue
        out[t.pos_] = out.get(t.pos_, 0) + 1
    return out


def low_freq_word_metrics(words, low_zipf=3.0):
    """Word-frequency metrics using `wordfreq` (large multilingual freq tables).

    Zipf scale: ~7 for very common words, ~3 for somewhat uncommon, <3 for rare /
    foreign / non-English. words with zipf == 0 don't appear in the English
    frequency list at all (likely foreign or non-words).

    Returns dict with:
      mean_zipf:        mean Zipf frequency (lower = more obscure/foreign overall)
      frac_low_zipf:    fraction of words with zipf < `low_zipf` (≈ rare/foreign)
      frac_zero_zipf:   fraction of words with zipf == 0 (almost certainly foreign / non-words)
    """
    from wordfreq import zipf_frequency
    if not words:
        return dict(mean_zipf=np.nan, frac_low_zipf=np.nan, frac_zero_zipf=np.nan)
    zipfs = []
    for w in words:
        s = str(w).strip().lower()
        if not s or not any(c.isalpha() for c in s):
            continue
        zipfs.append(zipf_frequency(s, "en"))
    if not zipfs:
        return dict(mean_zipf=np.nan, frac_low_zipf=np.nan, frac_zero_zipf=np.nan)
    zipfs = np.array(zipfs)
    return dict(
        mean_zipf=float(zipfs.mean()),
        frac_low_zipf=float((zipfs < low_zipf).mean()),
        frac_zero_zipf=float((zipfs == 0).mean()),
    )


_STOP_WORDS_CACHE = None


def _stop_words():
    """spaCy English stop-word set (lower-cased)."""
    global _STOP_WORDS_CACHE
    if _STOP_WORDS_CACHE is None:
        from spacy.lang.en.stop_words import STOP_WORDS
        _STOP_WORDS_CACHE = set(w.lower() for w in STOP_WORDS)
    return _STOP_WORDS_CACHE


def lexical_jaccard_pairwise(contexts_word_lists, content_only=True, min_len=2):
    """Mean pairwise Jaccard over word sets (case-folded, deduped).

    With ``content_only=True`` (default), excludes spaCy English stop words
    and tokens shorter than ``min_len`` so the metric measures recurring
    *content* words, not the constant baseline from {the, and, of, i, ...}.

    Returns the mean Jaccard, or NaN if fewer than 2 non-empty sets remain.
    """
    stop = _stop_words() if content_only else set()
    sets = []
    for ws in contexts_word_lists:
        toks = [str(w).lower() for w in ws]
        if content_only:
            toks = [t for t in toks if t not in stop and len(t) >= min_len]
        sets.append(set(toks))
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return np.nan
    sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = sets[i] & sets[j]
            union = sets[i] | sets[j]
            sims.append(len(inter) / len(union) if union else 0.0)
    return float(np.mean(sims))


def context_window_words(wordseq, tr_idx, context_trs=3):
    """Return the list of word strings within ±context_trs of feature TR."""
    raw_idx = tr_idx + TRIM_OFFSET
    chunks = wordseq.chunks()
    n = len(chunks)
    if not (0 <= raw_idx < n):
        return []
    out = []
    for i in range(max(0, raw_idx - context_trs), min(n, raw_idx + context_trs + 1)):
        out.extend(str(w) for w in chunks[i])
    return out


# ---------------------------------------------------------------------------
# Per-atom positional metrics
# ---------------------------------------------------------------------------

def positional_metrics(contexts, story_lengths, atom_idx=None):
    """For one atom's contexts, compute positional metrics over the normalized
    TR position (tr / story_length).

    story_lengths : dict[str] -> int (feature-TR length per story)

    Clamps norm_pos to [0, 1] (logs a warning the first time it has to clamp,
    since this indicates a TRIM-convention mismatch worth investigating).
    """
    norms = []
    out_of_range = []
    for c in contexts:
        L = story_lengths.get(c["story"], 0)
        if L > 0:
            raw = c["tr"] / L
            if raw < 0 or raw > 1:
                out_of_range.append((c["story"], c["tr"], L, raw))
            norms.append(min(1.0, max(0.0, raw)))
    if out_of_range and atom_idx is not None:
        s = out_of_range[0]
        print(f"  [warn] atom {atom_idx}: {len(out_of_range)} norm_pos values "
              f"outside [0,1] (clamped). e.g. {s[0]} tr={s[1]} story_len={s[2]} -> "
              f"{s[3]:.3f}. Likely a small TRIM-convention mismatch.")
    if not norms:
        return dict(mean_norm_pos=np.nan, std_norm_pos=np.nan,
                    frac_first_decile=np.nan, frac_first_quartile=np.nan,
                    frac_last_quartile=np.nan, norms=[])
    arr = np.array(norms)
    return dict(
        mean_norm_pos=float(arr.mean()),
        std_norm_pos=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        frac_first_decile=float((arr < 0.10).mean()),
        frac_first_quartile=float((arr < 0.25).mean()),
        frac_last_quartile=float((arr > 0.75).mean()),
        norms=arr.tolist(),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True,
                   help="Path to model directory; expects a sister "
                        "atom_contexts/ dir under reports/figures/<base>/")
    p.add_argument("--ckpt_name", default="model_last.pt")
    p.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--n_random_samples", type=int, default=200)
    p.add_argument("--n_per_sample", type=int, default=20)
    p.add_argument("--context_trs", type=int, default=5,
                   help="±N TRs of context to include in random baseline windows. "
                        "Default 5 matches scripts/18_visualize_multisubject_atoms.py "
                        "which generated the existing atom_contexts files.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--cache_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base = os.path.basename(os.path.normpath(args.model_dir))
    out_dir = args.out_dir or join(PROJECT_DIR, "reports", "figures",
                                    f"coherence_embedding_{base}")
    cache_dir = args.cache_dir or join(PROJECT_DIR, "outputs", "cache",
                                        "coherence", "embedding", base)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Locate atom_contexts dir (under reports/figures/<base>/)
    # ------------------------------------------------------------------
    figs_dir = join(PROJECT_DIR, "reports", "figures", base)
    if not os.path.exists(join(figs_dir, "atom_contexts")):
        sys.exit(f"ERROR: no atom_contexts/ found at {figs_dir}/atom_contexts/")

    print(f"[parse] {figs_dir}/atom_contexts/")
    parsed = parse_atom_contexts(figs_dir)
    n_atoms = max(parsed.keys()) + 1 if parsed else 0
    print(f"  parsed {len(parsed)} atoms (max idx {max(parsed.keys())})")
    n_contexts_per_atom = [len(parsed[k]["contexts"]) for k in parsed]
    print(f"  contexts/atom: min={min(n_contexts_per_atom)}, "
          f"max={max(n_contexts_per_atom)}, median={int(np.median(n_contexts_per_atom))}")

    # Train story count for entropy normalization
    ckpt = torch.load(join(args.model_dir, args.ckpt_name),
                      map_location="cpu", weights_only=False)
    train_stories = ckpt.get("train_stories", [])
    n_train = len(train_stories)
    print(f"  n_train_stories: {n_train}")

    # ------------------------------------------------------------------
    # Embed top contexts per atom
    # ------------------------------------------------------------------
    embedder = load_embedder(args.embedder)

    rows = []
    print("\n[per-atom] computing coherence...")
    all_atom_texts = []
    atom_text_idx = {}  # atom -> list of indices into all_atom_texts
    for k in sorted(parsed.keys()):
        idxs = list(range(len(all_atom_texts), len(all_atom_texts) + len(parsed[k]["contexts"])))
        atom_text_idx[k] = idxs
        all_atom_texts.extend([c["text"] for c in parsed[k]["contexts"]])
    print(f"  embedding {len(all_atom_texts)} contexts in one batch...")
    all_embs = embed_texts(embedder, all_atom_texts, batch_size=64)
    print(f"  embeddings shape: {all_embs.shape}")

    # ------------------------------------------------------------------
    # Load wordseqs for ALL training stories (true random baseline + positional metrics)
    # ------------------------------------------------------------------
    print(f"\n[wordseq] loading TextGrids for {n_train} training stories...")
    wordseqs = load_word_sequences(train_stories)
    story_lengths = {s: story_length_features_trs(ws) for s, ws in wordseqs.items()}
    total_trs = sum(story_lengths.values())
    print(f"  loaded {len(wordseqs)} wordseqs, total TRs = {total_trs}")

    # Precompute silence / sentence-end / syllable arrays per story
    print(f"[silence/sent/syllables] precomputing per-story features (one-time, may take ~30s)...")
    sil_features = {}
    for s, ws in wordseqs.items():
        sil_features[s] = precompute_silence_features(ws)
    print("  done")

    # ------------------------------------------------------------------
    # True random baseline: uniform over all training TRs (Fix A)
    # ------------------------------------------------------------------
    pool = build_uniform_tr_pool(wordseqs)
    rng = np.random.default_rng(args.seed)
    rand_mean, rand_std, rand_uniqs = random_coherence_baseline(
        pool, wordseqs, embedder, n_per_sample=args.n_per_sample,
        n_samples=args.n_random_samples, context_trs=args.context_trs, rng=rng,
    )
    print(f"  random baseline coherence: {rand_mean:.4f} ± {rand_std:.4f}")
    print(f"  random baseline unique stories per draw: "
          f"mean={np.mean(rand_uniqs):.1f}, range={min(rand_uniqs)}-{max(rand_uniqs)}")

    # ------------------------------------------------------------------
    # Per-atom metrics (coherence + diversity + position + lexical/POS + silence)
    # ------------------------------------------------------------------
    contexts_dump = {}  # for per-context JSON dump
    POS_TAGS = ["NOUN", "VERB", "PROPN", "PRON", "ADJ", "ADV", "ADP", "DET",
                "CCONJ", "SCONJ", "AUX", "INTJ", "NUM", "PART", "PUNCT", "SYM", "X"]
    for k in sorted(parsed.keys()):
        info = parsed[k]
        idxs = atom_text_idx[k]
        embs = all_embs[idxs]
        coh = coherence(embs)
        z = (coh - rand_mean) / rand_std if rand_std > 0 else np.nan
        n_uniq, ent, ent_n = story_diversity(
            [c["story"] for c in info["contexts"]], n_train,
        )
        pos = positional_metrics(info["contexts"], story_lengths, atom_idx=k)

        # Per-context: silence/utterance metrics, syllables, words, POS, freq metrics
        per_ctx_silence = []
        per_ctx_to_next_sil = []
        per_ctx_syll = []
        per_ctx_nwords = []
        per_ctx_words = []
        per_ctx_pos = []
        per_ctx_zipf = []  # list of dicts
        for c in info["contexts"]:
            story = c["story"]
            tr = c["tr"]
            raw_idx = tr + TRIM_OFFSET
            sf = sil_features.get(story)
            if sf is None or not (0 <= raw_idx < len(sf["n_words"])):
                continue
            per_ctx_silence.append(int(sf["trs_since_silence"][raw_idx]))
            per_ctx_to_next_sil.append(int(sf["trs_to_next_silence"][raw_idx]))
            per_ctx_syll.append(int(sf["n_syllables"][raw_idx]))
            per_ctx_nwords.append(int(sf["n_words"][raw_idx]))
            words = context_window_words(wordseqs[story], tr,
                                          context_trs=args.context_trs)
            per_ctx_words.append(words)
            per_ctx_pos.append(pos_distribution(words))
            per_ctx_zipf.append(low_freq_word_metrics(words))

        # Aggregate POS counts → distribution per atom (mean fraction of each tag)
        pos_frac_per_ctx = []
        for posdict in per_ctx_pos:
            total = sum(posdict.values())
            if total > 0:
                pos_frac_per_ctx.append(
                    {tag: posdict.get(tag, 0) / total for tag in POS_TAGS}
                )
        pos_dist_mean = {
            f"pos_{tag}": float(np.mean([p[tag] for p in pos_frac_per_ctx]))
                          if pos_frac_per_ctx else np.nan
            for tag in POS_TAGS
        }
        dominant_pos = (
            max(pos_dist_mean, key=lambda k: pos_dist_mean[k])
            if pos_frac_per_ctx else "NA"
        )

        jaccard_content = lexical_jaccard_pairwise(per_ctx_words, content_only=True)
        jaccard_all = lexical_jaccard_pairwise(per_ctx_words, content_only=False)

        rows.append({
            "atom": k, "label": info["label"], "rank": info["rank"],
            "dr": info["dr"], "freq": info["freq"], "mean_act": info["mean_act"],
            "kw": info["kw"],
            "n_contexts": len(info["contexts"]),
            "n_unique_stories": n_uniq,
            "story_entropy": ent,
            "story_entropy_normalized": ent_n,
            "top_coherence": coh,
            "random_coh_mean": rand_mean,
            "random_coh_std": rand_std,
            "coherence_z": z,
            "mean_norm_pos": pos["mean_norm_pos"],
            "std_norm_pos": pos["std_norm_pos"],
            "frac_first_decile": pos["frac_first_decile"],
            "frac_first_quartile": pos["frac_first_quartile"],
            "frac_last_quartile": pos["frac_last_quartile"],
            # New: silence-distance / utterance-length / speech-rate / lexical / POS / wordfreq
            "mean_trs_since_silence": float(np.mean(per_ctx_silence))
                                       if per_ctx_silence else np.nan,
            "std_trs_since_silence": float(np.std(per_ctx_silence, ddof=1))
                                      if len(per_ctx_silence) > 1 else np.nan,
            "mean_trs_to_next_silence": float(np.mean(per_ctx_to_next_sil))
                                         if per_ctx_to_next_sil else np.nan,
            "std_trs_to_next_silence": float(np.std(per_ctx_to_next_sil, ddof=1))
                                        if len(per_ctx_to_next_sil) > 1 else np.nan,
            "mean_syllables_per_tr": float(np.mean(per_ctx_syll))
                                      if per_ctx_syll else np.nan,
            "std_syllables_per_tr": float(np.std(per_ctx_syll, ddof=1))
                                     if len(per_ctx_syll) > 1 else np.nan,
            "mean_nwords_per_tr": float(np.mean(per_ctx_nwords))
                                  if per_ctx_nwords else np.nan,
            "lexical_jaccard_content": jaccard_content,
            "lexical_jaccard_all": jaccard_all,
            "mean_zipf": float(np.mean([z["mean_zipf"] for z in per_ctx_zipf
                                          if not np.isnan(z["mean_zipf"])]))
                          if per_ctx_zipf else np.nan,
            "frac_low_zipf": float(np.mean([z["frac_low_zipf"] for z in per_ctx_zipf
                                              if not np.isnan(z["frac_low_zipf"])]))
                              if per_ctx_zipf else np.nan,
            "frac_zero_zipf": float(np.mean([z["frac_zero_zipf"] for z in per_ctx_zipf
                                               if not np.isnan(z["frac_zero_zipf"])]))
                               if per_ctx_zipf else np.nan,
            "dominant_pos": dominant_pos.replace("pos_", ""),
            **pos_dist_mean,
        })
        # also dump per-context details (extended)
        contexts_dump[str(k)] = []
        for i, c in enumerate(info["contexts"]):
            sf = sil_features.get(c["story"])
            raw_idx = c["tr"] + TRIM_OFFSET
            silence_d = (int(sf["trs_since_silence"][raw_idx])
                          if sf is not None and 0 <= raw_idx < len(sf["n_words"]) else None)
            to_next_sil = (int(sf["trs_to_next_silence"][raw_idx])
                            if sf is not None and 0 <= raw_idx < len(sf["n_words"]) else None)
            syll = (int(sf["n_syllables"][raw_idx])
                     if sf is not None and 0 <= raw_idx < len(sf["n_words"]) else None)
            nw = (int(sf["n_words"][raw_idx])
                   if sf is not None and 0 <= raw_idx < len(sf["n_words"]) else None)
            contexts_dump[str(k)].append({
                "rank": i + 1, "story": c["story"], "tr": c["tr"],
                "story_length_trs": int(story_lengths.get(c["story"], 0)),
                "norm_pos": (c["tr"] / story_lengths[c["story"]])
                             if story_lengths.get(c["story"], 0) > 0 else None,
                "trs_since_silence": silence_d,
                "trs_to_next_silence": to_next_sil,
                "n_syllables_at_tr": syll,
                "n_words_at_tr": nw,
                "act": c["act"], "text": c["text"],
            })

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = join(cache_dir, "coherence.csv")
    POS_TAGS_OUT = ["NOUN", "VERB", "PROPN", "PRON", "ADJ", "ADV", "ADP", "DET",
                    "CCONJ", "SCONJ", "AUX", "INTJ", "NUM", "PART", "PUNCT", "SYM", "X"]
    cols = ["atom", "label", "rank", "dr", "freq", "mean_act",
            "n_contexts", "n_unique_stories", "story_entropy", "story_entropy_normalized",
            "top_coherence", "random_coh_mean", "random_coh_std", "coherence_z",
            "mean_norm_pos", "std_norm_pos",
            "frac_first_decile", "frac_first_quartile", "frac_last_quartile",
            "mean_trs_since_silence", "std_trs_since_silence",
            "mean_trs_to_next_silence", "std_trs_to_next_silence",
            "mean_syllables_per_tr", "std_syllables_per_tr",
            "mean_nwords_per_tr",
            "lexical_jaccard_content", "lexical_jaccard_all",
            "mean_zipf", "frac_low_zipf", "frac_zero_zipf",
            "dominant_pos",
            *(f"pos_{t}" for t in POS_TAGS_OUT),
            "kw"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(f"\n[save] {csv_path}")

    # Per-context dump (story, tr, normalized position, text) for inspection
    contexts_path = join(cache_dir, "atom_contexts_with_position.json")
    with open(contexts_path, "w") as f:
        json.dump(contexts_dump, f, indent=2)
    print(f"[save] {contexts_path}")

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    cohs = np.array([r["top_coherence"] for r in rows])
    zs = np.array([r["coherence_z"] for r in rows])
    divs = np.array([r["story_entropy_normalized"] for r in rows])
    drs = np.array([r["dr"] for r in rows])
    summary = {
        "model_dir": args.model_dir,
        "embedder": args.embedder,
        "n_atoms": int(len(rows)),
        "n_train_stories": int(n_train),
        "random_baseline": {
            "coherence_mean": rand_mean, "coherence_std": rand_std,
            "n_samples": args.n_random_samples, "n_per_sample": args.n_per_sample,
        },
        "atoms": {
            "median_coherence": float(np.median(cohs)),
            "mean_coherence": float(cohs.mean()),
            "median_z": float(np.median(zs)),
            "mean_z": float(zs.mean()),
            "frac_z_above_2": float((zs > 2).mean()),
            "frac_z_above_3": float((zs > 3).mean()),
            "frac_z_above_5": float((zs > 5).mean()),
            "median_story_entropy_normalized": float(np.median(divs)),
        },
    }
    with open(join(cache_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {join(cache_dir, 'summary.json')}")

    print("\n=== Coherence summary ===")
    print(f"  random baseline : {rand_mean:.4f} ± {rand_std:.4f}")
    print(f"  median atom coh : {np.median(cohs):.4f}  (z={np.median(zs):.2f})")
    print(f"  mean   atom coh : {cohs.mean():.4f}    (z={zs.mean():.2f})")
    print(f"  z > 2           : {(zs > 2).sum()}/{len(zs)}")
    print(f"  z > 3           : {(zs > 3).sum()}/{len(zs)}")
    print(f"  z > 5           : {(zs > 5).sum()}/{len(zs)}")
    print(f"  median story entropy (normalized): {np.median(divs):.3f}")

    print("\n=== Top-10 atoms by coherence ===")
    top = sorted(rows, key=lambda r: -r["top_coherence"])[:10]
    for r in top:
        print(f"  a{r['atom']:2d} coh={r['top_coherence']:.3f} z={r['coherence_z']:5.1f} "
              f"div={r['story_entropy_normalized']:.2f} "
              f"({r['n_unique_stories']:2d} stories) Δr={r['dr']:+.4f} | {r['label']:<25s} | {r['kw'][:50]}")

    print("\n=== Bottom-10 atoms by coherence ===")
    bot = sorted(rows, key=lambda r: r["top_coherence"])[:10]
    for r in bot:
        print(f"  a{r['atom']:2d} coh={r['top_coherence']:.3f} z={r['coherence_z']:5.1f} "
              f"div={r['story_entropy_normalized']:.2f} "
              f"({r['n_unique_stories']:2d} stories) Δr={r['dr']:+.4f} | {r['label']:<25s} | {r['kw'][:50]}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # 1) Distribution: top vs random
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(cohs, bins=20, alpha=0.7, label=f"atoms (n={len(cohs)})", color="tab:blue", edgecolor="black", linewidth=0.3)
    ax.axvline(rand_mean, color="grey", lw=2, label=f"random mean = {rand_mean:.3f}")
    ax.axvspan(rand_mean - 2*rand_std, rand_mean + 2*rand_std, color="grey",
               alpha=0.2, label="random ±2σ")
    ax.set_xlabel("Top-20 context coherence (mean pairwise cosine)")
    ax.set_ylabel("# atoms")
    ax.set_title(f"Per-atom semantic coherence — {base[:60]}...\n"
                 f"embedder: {args.embedder}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(join(out_dir, "coherence_distribution.png"), dpi=120)
    plt.close(fig)
    print(f"\n[save] {join(out_dir, 'coherence_distribution.png')}")

    # 2) Coherence vs story diversity scatter, color by Δr, label noteworthy atoms
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(divs, cohs, c=drs, cmap="RdBu_r", s=70, alpha=0.85,
                     edgecolor="black", linewidth=0.4,
                     vmin=-max(abs(drs.min()), abs(drs.max())),
                     vmax=max(abs(drs.min()), abs(drs.max())))
    ax.axhline(rand_mean, color="grey", lw=1, ls="--", alpha=0.6, label="random mean")
    ax.axhspan(rand_mean - 2*rand_std, rand_mean + 2*rand_std, color="grey",
               alpha=0.15)
    plt.colorbar(sc, ax=ax, label="Δr (LOO importance)")

    # Label: top-K by Δr (importance), and most extreme on each axis
    label_set = set()
    for k_dr in np.argsort(-drs)[:5]:
        label_set.add(int(k_dr))
    for k_coh in np.argsort(-cohs)[:3]:
        label_set.add(int(k_coh))
    for k_div in np.argsort(-divs)[:3]:
        label_set.add(int(k_div))
    for k_low in np.argsort(divs)[:3]:
        label_set.add(int(k_low))
    for k in label_set:
        r = rows[k]
        ax.annotate(f"a{r['atom']}", (divs[k], cohs[k]),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Story diversity (entropy normalized to [0, 1])")
    ax.set_ylabel("Top-20 context coherence (mean pairwise cosine)")
    ax.set_title("Coherence vs. story diversity per atom\n"
                 "top-right = cross-story semantic feature; "
                 "bottom-right = noisy across stories; "
                 "top-left = story-specific")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(join(out_dir, "coherence_vs_diversity.png"), dpi=120)
    plt.close(fig)
    print(f"[save] {join(out_dir, 'coherence_vs_diversity.png')}")

    # 3) Z-score bar chart, ranked
    order = np.argsort(-zs)
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["tab:green" if zs[k] > 2 else "tab:grey" for k in order]
    ax.bar(range(len(zs)), zs[order], color=colors, edgecolor="black", linewidth=0.2)
    ax.axhline(2, color="red", lw=1, ls="--", label="z = 2")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(len(zs)))
    ax.set_xticklabels([f"a{rows[k]['atom']}" for k in order], rotation=90, fontsize=7)
    ax.set_ylabel("Coherence z-score (vs random baseline)")
    ax.set_title("Per-atom coherence z-score, ranked")
    ax.legend()
    fig.tight_layout()
    fig.savefig(join(out_dir, "coherence_z_ranked.png"), dpi=120)
    plt.close(fig)
    print(f"[save] {join(out_dir, 'coherence_z_ranked.png')}")

    print(f"\nDone. Outputs: {out_dir} and {cache_dir}")


if __name__ == "__main__":
    main()
