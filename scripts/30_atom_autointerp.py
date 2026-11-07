#!/usr/bin/env python
"""AutoInterp (nameability) for the 50 components, judged blind by a language model.

Three steps; only the middle one calls the judge, and the judge never sees the answer key.

  --mode prepare            For each component: its 10 strongest activating contexts (shown to
                            the judge to write a description), and 20 held-out contexts in
                            random order: its 11th-20th strongest and 10 random time points
                            where it is inactive. Writes prepared_atoms.jsonl (no labels) and
                            held_out_answer_key.jsonl (labels, never shown to the judge).
  --mode emit_batch_prompt  Writes batch_prompt.txt: one prompt covering all components.
  (judge)                   Give batch_prompt.txt to the judge model and save its answer as
                            results.tsv (component, description, 20 scores 1-5) next to it.
                            The paper used Claude Fable 5.1 for every model.
  --mode aggregate          Joins results.tsv with the answer key and computes AUC per
                            component and the median over components.

Usage:
    uv run python scripts/30_atom_autointerp.py --mode prepare --model_dir <run>
    uv run python scripts/30_atom_autointerp.py --mode emit_batch_prompt --model_dir <run>
    uv run python scripts/30_atom_autointerp.py --mode aggregate --model_dir <run>
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

from sparsebrain.data import load_features  # noqa: E402

TRIM_OFFSET = 10

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# atom_contexts/rankNN_*.txt parsing  (KEEP **target** markers)
# ---------------------------------------------------------------------------

CTX_HEADER_RE = re.compile(
    r"=== Atom (\d+) \(([^)]+)\) "
    r"\(rank (\d+), [^=]*Δr=([0-9.\-]+), freq=([0-9.]+)%, mean_act=([0-9.]+)\) ==="
)
CTX_LINE_RE = re.compile(r'^\s*\d+\.\s+\[act=([0-9.\-]+)\]\s+"([^"]+)"\s+TR\s+(\d+)\s*$')


def parse_atom_contexts(model_figs_dir):
    """Parse atom_contexts/rank{NN}_*.txt files. Keeps **target** markers."""
    ctx_dir = join(model_figs_dir, "atom_contexts")
    out = {}
    for path in sorted(glob.glob(join(ctx_dir, "*.txt"))):
        lines = open(path).readlines()
        if not lines:
            continue
        m = CTX_HEADER_RE.search(lines[0])
        if not m:
            continue
        atom = int(m.group(1))
        contexts = []
        i = 0
        while i < len(lines):
            mm = CTX_LINE_RE.match(lines[i])
            if mm:
                act = float(mm.group(1)); story = mm.group(2); tr = int(mm.group(3))
                if i + 1 < len(lines):
                    text = lines[i + 1].strip()
                    text = re.sub(r"^\.\.\.\s*", "", text)
                    text = re.sub(r"\s*\.\.\.$", "", text)
                    # Keep **target** markers — Bills et al. AutoInterp convention
                    contexts.append({"story": story, "tr": tr, "act": act, "text": text})
                i += 2
            else:
                i += 1
        out[atom] = {"label": m.group(2), "rank": int(m.group(3)),
                     "dr": float(m.group(4)), "freq": float(m.group(5)),
                     "mean_act": float(m.group(6)), "contexts": contexts}
    return out


def trim_marked_text(text, words_each_side=12):
    """Trim a **target**-marked context to ~words_each_side words on each side
    of the marked target span. Returns the trimmed text with markers preserved.

    If markers are missing, just truncates the start/end.
    """
    marker_match = re.search(r"\*\*([^*]+)\*\*", text)
    if not marker_match:
        # No marker — fall back to taking first ~25 words
        words = text.split()
        return " ".join(words[: 2 * words_each_side])

    pre = text[: marker_match.start()].strip()
    target = marker_match.group(0)  # includes the **
    post = text[marker_match.end():].strip()

    pre_words = pre.split()[-words_each_side:]
    post_words = post.split()[:words_each_side]

    parts = []
    if pre_words:
        parts.append(" ".join(pre_words))
    parts.append(target)
    if post_words:
        parts.append(" ".join(post_words))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Word sequences + context-window builder for negatives (mark target with **)
# ---------------------------------------------------------------------------

def load_word_sequences(stories):
    grids = load_textgrids(stories, DATA_DIR)
    with open(os.path.join(DERIVATIVE_DIR, "respdict.json")) as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict)
    return make_word_ds(grids, trfiles)


def build_marked_context_text(wordseq, tr_idx, context_trs=2):
    """Build a context window with target TR's words wrapped in **markers**.
    Empty (silent) target TR is rendered as **[silence]** so the LLM can see it.
    Used uniformly for both positives and negatives so silence handling matches.
    """
    raw_idx = tr_idx + TRIM_OFFSET
    chunks = wordseq.chunks()
    n = len(chunks)
    if not (0 <= raw_idx < n):
        return ""
    parts = []
    for i in range(max(0, raw_idx - context_trs), min(n, raw_idx + context_trs + 1)):
        words = [str(w) for w in chunks[i]]
        if i == raw_idx:
            parts.append("**" + (" ".join(words) if words else "[silence]") + "**")
        elif words:
            parts.append(" ".join(words))
    return " ".join(parts).strip()


def story_length_features_trs(wordseq):
    return max(0, len(wordseq.chunks()) - TRIM_OFFSET)


# Pyphen-based syllable count (matches script 29's per-TR speech-rate metric).
_PYPHEN = None
def _count_syllables(word):
    global _PYPHEN
    if _PYPHEN is None:
        import pyphen
        _PYPHEN = pyphen.Pyphen(lang="en_US")
    if not word or not any(c.isalpha() for c in word):
        return 0
    return _PYPHEN.inserted(word.lower()).count("-") + 1


def excerpt_meta(wordseq, story_len_trs, tr_idx):
    """Per-excerpt metadata: story position (0..1) and syllables in target TR."""
    raw_idx = tr_idx + TRIM_OFFSET
    chunks = wordseq.chunks()
    if 0 <= raw_idx < len(chunks):
        words = [str(w) for w in chunks[raw_idx]]
        n_syll = sum(_count_syllables(w) for w in words)
        n_words = len(words)
    else:
        n_syll = 0
        n_words = 0
    pos = float(tr_idx) / story_len_trs if story_len_trs > 0 else 0.0
    pos = max(0.0, min(1.0, pos))
    return {"pos": round(pos, 3), "spr": int(n_syll), "n_words": int(n_words)}


# ---------------------------------------------------------------------------
# Compute Z (atom activations) for sampling true negatives
# ---------------------------------------------------------------------------

def compute_atom_activations(model_dir, ckpt_name, train_stories):
    from sparsebrain.models.coupled_sae import MultiSubjectCoupledSAE
    ckpt = torch.load(join(model_dir, ckpt_name), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    args = ckpt["args"]
    model = MultiSubjectCoupledSAE(
        n_sem=sd["encoder.weight"].shape[1],
        n_voxels_per_subject=ckpt["n_voxels_per_subject"],
        n_atoms=sd["encoder.weight"].shape[0], k=args["k"],
        n_delays=sd["delay_scales"].shape[0],
        encoder_bias=("encoder_bias" in sd),
        brain_voxel_scale=any(key.startswith("voxel_scales.") for key in sd),
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    do_rms = args.get("input_rms_normalize", False)
    if do_rms:
        X_full = load_features(train_stories, embedding=args["embedding"], demean_per_story=True)
        rms = float(np.sqrt(np.mean(X_full ** 2)))
    else:
        rms = 1.0
    Z_per_story = {}
    for s in train_stories:
        X = load_features([s], embedding=args["embedding"], demean_per_story=True)
        X = (X / rms).astype(np.float32)
        with torch.no_grad():
            Z = model.encode(torch.tensor(X)).numpy()
        Z_per_story[s] = Z
    return Z_per_story


def sample_negatives(Z_per_story, atom_idx, n_negatives=10, rng=None, excluded_keys=None):
    if rng is None:
        rng = np.random.default_rng(0)
    excluded = set(excluded_keys or [])
    candidates = []
    for s, Z in Z_per_story.items():
        zeros = np.where(Z[:, atom_idx] == 0)[0]
        for t in zeros:
            if (s, int(t)) not in excluded:
                candidates.append((s, int(t)))
    if len(candidates) <= n_negatives:
        return candidates
    idx = rng.choice(len(candidates), size=n_negatives, replace=False)
    return [candidates[i] for i in idx]


# ---------------------------------------------------------------------------
# Per-atom aggregate metrics (trimmed to 6 high-signal fields, parsed to float)
# ---------------------------------------------------------------------------

# Keep only metrics with high LLM-interpretive signal. Drop redundant /
# correlated ones (frac_first_quartile, std_syllables_per_tr, etc.).
KEEP_METRICS = [
    ("mean_norm_pos",          float, "story position 0=start..1=end"),
    ("std_norm_pos",           float, "low=consistent position"),
    ("mean_syllables_per_tr",  float, "speech rate; <=1 silence, >=13 rapid"),
    ("std_syllables_per_tr",   float, "low=consistent speech rate"),
    ("mean_zipf",              float, "vocab commonness; <6 unusual/foreign"),
    ("frac_zero_zipf",         float, "fraction not in English freq tables"),
    ("dominant_pos",           str,   "most common POS tag"),
    ("n_unique_stories",       int,   "atom diversity (1..n_train_stories)"),
]


def load_coherence_metrics(model_base):
    path = join(PROJECT_DIR, "outputs", "cache", "coherence", "embedding",
                model_base, "coherence.csv")
    if not os.path.exists(path):
        print(f"  [warn] coherence.csv not found at {path}")
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        atom = int(r["atom"])
        clean = {}
        for field, caster, _desc in KEEP_METRICS:
            if field in r and r[field] != "":
                try:
                    clean[field] = caster(float(r[field]) if caster is int else r[field])
                    if caster is float:
                        clean[field] = round(float(r[field]), 4)
                    elif caster is int:
                        clean[field] = int(float(r[field]))
                except (ValueError, TypeError):
                    pass
        out[atom] = clean
    return out


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------

def compute_auc(scores, labels):
    from scipy.stats import rankdata
    s = np.asarray(scores, dtype=float); y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    ranks = rankdata(s)
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    return float((ranks[y == 1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg))


# ---------------------------------------------------------------------------
# Phase 1: --mode prepare
# ---------------------------------------------------------------------------

def cmd_prepare(args):
    base = os.path.basename(os.path.normpath(args.model_dir))
    cache_dir = args.cache_dir or join(PROJECT_DIR, "outputs", "cache",
                                        "coherence", "autointerp", base)
    os.makedirs(cache_dir, exist_ok=True)

    figs_dir = join(PROJECT_DIR, "reports", "figures", base)
    parsed = parse_atom_contexts(figs_dir)
    print(f"[parse] {len(parsed)} atoms (markers preserved)")

    ckpt = torch.load(join(args.model_dir, args.ckpt_name),
                      map_location="cpu", weights_only=False)
    train_stories = ckpt["train_stories"]
    print(f"[ckpt] n_train_stories = {len(train_stories)}")

    z_cache = join(cache_dir, "Z_per_story_train.npz")
    if os.path.exists(z_cache):
        loaded = np.load(z_cache)
        Z_per_story = {k: loaded[k] for k in loaded.files}
        print(f"[Z] loaded cache ({len(Z_per_story)} stories)")
    else:
        print("[Z] computing per-story atom activations...")
        Z_per_story = compute_atom_activations(args.model_dir, args.ckpt_name, train_stories)
        np.savez(z_cache, **Z_per_story)
        print(f"  cached -> {z_cache}")

    print(f"[wordseq] loading TextGrids for {len(train_stories)} stories...")
    wordseqs = load_word_sequences(train_stories)
    story_lengths = {s: story_length_features_trs(ws) for s, ws in wordseqs.items()}

    coherence_metrics = load_coherence_metrics(base)
    print(f"[coherence] aggregate metrics for {len(coherence_metrics)} atoms")

    rng = np.random.default_rng(args.seed)
    out_path = join(cache_dir, "prepared_atoms.jsonl")
    answer_key_path = join(cache_dir, "held_out_answer_key.jsonl")
    n_written = 0

    with open(out_path, "w") as fout, open(answer_key_path, "w") as fkey:
        for atom in sorted(parsed.keys()):
            info = parsed[atom]
            ctxs = info["contexts"]
            if len(ctxs) < 20:
                continue

            # train (10) and held positives (10) both come from the
            # atom_contexts file (top-20 firings) — already have **target** markers
            train_pos = ctxs[:10]
            held_pos = ctxs[10:20]
            train_keys = {(c["story"], c["tr"]) for c in train_pos}
            held_keys = {(c["story"], c["tr"]) for c in held_pos}

            # Sample 10 true negatives (atom does not fire there)
            negs = sample_negatives(Z_per_story, atom, n_negatives=10,
                                      rng=rng, excluded_keys=train_keys | held_keys)

            # Build per-context entries: text + per-excerpt (pos, spr, n_words).
            # Rebuild ALL excerpts (pos + neg) from wordseqs so silence handling is uniform.
            def _entry(story, tr):
                raw_text = build_marked_context_text(wordseqs[story], tr,
                                                      context_trs=args.context_trs)
                return {
                    "text": trim_marked_text(raw_text, words_each_side=args.context_words_each_side),
                    **excerpt_meta(wordseqs[story], story_lengths[story], tr),
                }
            train_entries = [_entry(c["story"], c["tr"]) for c in train_pos]
            held_pos_entries = [_entry(c["story"], c["tr"]) for c in held_pos]
            neg_entries = [_entry(s, t) for (s, t) in negs]

            # Combine held-out (positives + negatives), then SHUFFLE
            all_held = held_pos_entries + neg_entries
            all_held_labels = [1] * len(held_pos_entries) + [0] * len(neg_entries)
            perm = rng.permutation(len(all_held))
            shuffled_held = [all_held[i] for i in perm]
            shuffled_labels = [int(all_held_labels[i]) for i in perm]

            # Aggregate metrics: trimmed to 6, parsed to floats/ints
            agg = coherence_metrics.get(atom, {})

            record = {
                "atom": atom,
                "label": info["label"],
                "rank": info["rank"],
                "dr": info["dr"],
                "freq_pct": info["freq"],
                "mean_act": info["mean_act"],
                "aggregate_metrics": agg,
                "train_contexts": train_entries,
                "held_out_contexts": shuffled_held,
                # NO held_out_labels in this file — answer key is separate
            }
            fout.write(json.dumps(record) + "\n")

            answer = {"atom": atom, "held_out_labels": shuffled_labels,
                      "permutation": perm.tolist()}
            fkey.write(json.dumps(answer) + "\n")

            n_written += 1

    print(f"\n[save] {out_path} ({n_written} atoms, no labels)")
    print(f"[save] {answer_key_path}  ({n_written} atoms, labels + perm)")
    # Quick size report
    p_size = os.path.getsize(out_path)
    print(f"  prepared_atoms.jsonl: {p_size/1024:.1f} KB "
          f"(~{p_size//n_written} bytes/atom)")


# ---------------------------------------------------------------------------
# Phase 2: --mode emit_batch_prompt
# ---------------------------------------------------------------------------

BATCH_PROMPT_HEADER = """SAE atom AutoInterp — score {n_atoms} atoms in ONE pass.

You're identifying what each SAE atom (learned latent in a coupled sparse autoencoder) responds to in fMRI brain responses to spoken Moth-style autobiographical stories. The **bold** word(s) in each excerpt mark the words at the activation peak (firing TR); **[silence]** marks an empty firing TR.

Each atom block has: a `metrics:` line (atom-level mean±std), 10 train sentences (T1..T10) where the atom fires, 20 held-out sentences (H1..H20, randomized mix of fire/no-fire), and one-line `T-stats` and `H-stats` summaries showing per-excerpt `pos/spr` values aligned to the same indices.

For each atom:
1. READ THE TRAIN SENTENCES FIRST (T1..T10). Look at the **bold** target words — write 1 sentence (≤25 words) describing what semantic / syntactic / lexical pattern this atom responds to. Be specific, name the actual content. Most atoms have a clear lexical concept here.
2. THEN check T-stats. If train sentences had no clear lexical pattern but T-stats DO show a tight pattern (all pos near 0 → story-opening atom; all spr=0 → silence-firing atom; all spr≥13 → rapid-speech atom), that IS the concept — name it accordingly. Silence-firing is a valid mono concept, NOT "dead". Don't add metadata qualifiers ("early in story", "high speech rate") to lexical descriptions unless those are the primary signal.
3. Label monosemanticity: "mono" (one coherent concept — lexical or structural), "poly" (train has truly distinct concepts that don't unify), "dead" (no consistent pattern at all — almost no atoms). Most atoms should be mono.
4. Score each H1..H20 from 1–5 (1=def NOT fire, 5=def fire). For lexical atoms, score by whether the held-out's **bold** target matches the lexical pattern (use H1..H20 sentences). For structural atoms, score by H-stats (pos/spr). Held-out order is randomized — do NOT assume positives come first. Score blind, independently.

Metric legend:
  metrics: line = atom-level mean±std across all firings.
  pos       = story position of the firing TR (0=start, 1=end). Low std = atom is positionally consistent.
  spr       = syllables in the firing TR (0 silence, 1–3 slow, 4–9 normal, 10+ rapid). Low std = atom is rate-consistent.
  zipf      = vocab commonness (<6 unusual). pos_tag = dominant POS. n_stories = atom diversity.

== ATOMS ==

{atom_blocks}
== OUTPUT ==
Write all {n_atoms} results in ONE bash heredoc to a TSV file. Each row has 4 tab-separated fields: atom_index<TAB>mono|poly|dead<TAB>description<TAB>{n_held} comma-separated integer scores. Description must NOT contain tabs or newlines. mono/poly/dead must be exactly one of those three lowercase tokens. Exactly {n_atoms} rows, in atom-index order.

```bash
cat > {result_path} <<'EOF'
0\tmono\t<description for atom 0>\ts1,s2,...,s{n_held}
1\tpoly\t<description for atom 1>\ts1,s2,...,s{n_held}
...
EOF
```

After writing, reply only with "done". No commentary, no thinking, no other tools.
"""


def _format_metrics_compact(agg):
    """Compact one-line metrics, e.g. pos=0.57±0.31 spr=10.5±2.1 zipf=6.09 pos_tag=PRON n_stories=13."""
    parts = []
    if "mean_norm_pos" in agg:
        if "std_norm_pos" in agg:
            parts.append(f"pos={agg['mean_norm_pos']:.2f}±{agg['std_norm_pos']:.2f}")
        else:
            parts.append(f"pos={agg['mean_norm_pos']:.2f}")
    if "mean_syllables_per_tr" in agg:
        if "std_syllables_per_tr" in agg:
            parts.append(f"spr={agg['mean_syllables_per_tr']:.1f}±{agg['std_syllables_per_tr']:.1f}")
        else:
            parts.append(f"spr={agg['mean_syllables_per_tr']:.1f}")
    if "mean_zipf" in agg:
        parts.append(f"zipf={agg['mean_zipf']:.2f}")
    if "dominant_pos" in agg:
        parts.append(f"pos_tag={agg['dominant_pos']}")
    if "n_unique_stories" in agg:
        parts.append(f"n_stories={agg['n_unique_stories']}")
    return " ".join(parts)


def _excerpt_text(prefix, idx, e):
    if isinstance(e, str):
        return f"{prefix}{idx} {e}"
    return f"{prefix}{idx} {e['text']}"


def _excerpt_stats_inline(e):
    if isinstance(e, str):
        return "?/?"
    return f"{e['pos']:.2f}/{e['spr']}"


def _atom_block(r):
    metric_str = _format_metrics_compact(r["aggregate_metrics"])
    train_txt = "\n".join(_excerpt_text("T", i+1, t) for i, t in enumerate(r["train_contexts"]))
    held_txt = "\n".join(_excerpt_text("H", i+1, t) for i, t in enumerate(r["held_out_contexts"]))
    train_stats = " ".join(f"T{i+1}={_excerpt_stats_inline(t)}"
                            for i, t in enumerate(r["train_contexts"]))
    held_stats = " ".join(f"H{i+1}={_excerpt_stats_inline(t)}"
                           for i, t in enumerate(r["held_out_contexts"]))
    return (f"[atom {r['atom']}]\n"
            f"metrics: {metric_str}\n"
            f"{train_txt}\n"
            f"{held_txt}\n"
            f"T-stats (pos/spr): {train_stats}\n"
            f"H-stats (pos/spr): {held_stats}\n")


def cmd_emit_batch_prompt(args):
    base = os.path.basename(os.path.normpath(args.model_dir))
    cache_dir = args.cache_dir or join(PROJECT_DIR, "outputs", "cache",
                                        "coherence", "autointerp", base)

    src = join(cache_dir, "prepared_atoms.jsonl")
    if not os.path.exists(src):
        sys.exit(f"ERROR: {src} not found. Run --mode prepare first.")

    records = [json.loads(line) for line in open(src)]
    records.sort(key=lambda r: r["atom"])

    n_held = len(records[0]["held_out_contexts"]) if records else 0
    blocks = "\n".join(_atom_block(r) for r in records)
    result_path = join(cache_dir, "results.tsv")

    prompt = BATCH_PROMPT_HEADER.format(
        n_atoms=len(records),
        n_held=n_held,
        atom_blocks=blocks,
        result_path=result_path,
    )

    out = join(cache_dir, "batch_prompt.txt")
    with open(out, "w") as f:
        f.write(prompt)

    sz = os.path.getsize(out)
    print(f"[save] {out}")
    print(f"  size: {sz} bytes (~{sz//4} tokens)  n_atoms={len(records)}  n_held={n_held}")
    print(f"[note] give this prompt to the judge model; save its answer as {result_path}")


# ---------------------------------------------------------------------------
# Phase 3: --mode aggregate
# ---------------------------------------------------------------------------

def cmd_aggregate(args):
    base = os.path.basename(os.path.normpath(args.model_dir))
    cache_dir = args.cache_dir or join(PROJECT_DIR, "outputs", "cache",
                                        "coherence", "autointerp", base)
    out_dir = args.out_dir or join(PROJECT_DIR, "reports", "figures",
                                    f"coherence_autointerp_{base}")
    os.makedirs(out_dir, exist_ok=True)

    # Load prepared (for atom metadata)
    prepared = {}
    for line in open(join(cache_dir, "prepared_atoms.jsonl")):
        r = json.loads(line)
        prepared[r["atom"]] = r

    # Load answer key (labels in shuffled order)
    labels_by_atom = {}
    for line in open(join(cache_dir, "held_out_answer_key.jsonl")):
        r = json.loads(line)
        labels_by_atom[r["atom"]] = r["held_out_labels"]

    # Load results from results.tsv (atom_index<TAB>description<TAB>scores_csv)
    results = {}
    tsv_path = join(cache_dir, "results.tsv")
    if not os.path.exists(tsv_path):
        sys.exit(f"ERROR: {tsv_path} not found. Run the judge on batch_prompt.txt first.")
    for raw in open(tsv_path):
        line = raw.rstrip("\n").rstrip("\r")
        if not line:
            continue
        parts = line.split("\t")
        # 4-col format: atom\tmono|poly|dead\tdescription\tscores_csv
        # 3-col format (legacy): atom\tdescription\tscores_csv
        if len(parts) == 4:
            atom_s, mono, desc, scores_csv = parts
        elif len(parts) == 3:
            atom_s, desc, scores_csv = parts
            mono = ""
        else:
            print(f"  [warn] malformed line: {line[:80]!r}")
            continue
        try:
            atom = int(atom_s)
            scores = [int(s.strip()) for s in scores_csv.split(",") if s.strip() != ""]
        except ValueError as e:
            print(f"  [warn] parse error on line {line[:80]!r}: {e}")
            continue
        results[atom] = {"atom": atom, "mono": mono, "description": desc, "scores": scores}
    print(f"[aggregate] loaded {len(results)} atoms from {tsv_path}")

    rows = []
    for atom in sorted(prepared.keys()):
        if atom not in results:
            continue
        prep = prepared[atom]
        res = results[atom]
        labels = labels_by_atom.get(atom)
        if labels is None:
            print(f"  [warn] no answer key for atom {atom}")
            continue
        scores = res["scores"]
        if len(scores) != len(labels):
            print(f"  [warn] atom {atom}: scores ({len(scores)}) != labels ({len(labels)})")
            continue
        auc = compute_auc(scores, labels)
        rows.append({
            "atom": atom, "label": prep["label"], "rank": prep["rank"],
            "dr": prep["dr"], "freq_pct": prep["freq_pct"], "mean_act": prep["mean_act"],
            "auc": auc, "mono": res.get("mono", ""), "description": res["description"],
        })

    csv_path = join(cache_dir, "autointerp.csv")
    cols = ["atom", "label", "rank", "dr", "freq_pct", "mean_act", "auc", "mono", "description"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(f"[save] {csv_path}")

    aucs = np.array([r["auc"] for r in rows if not np.isnan(r["auc"])])
    monos = [r["mono"] for r in rows]
    summary = {
        "n_atoms": len(rows),
        "median_auc": float(np.median(aucs)) if len(aucs) else None,
        "mean_auc": float(aucs.mean()) if len(aucs) else None,
        "frac_above_0.7": float((aucs > 0.7).mean()) if len(aucs) else None,
        "frac_above_0.8": float((aucs > 0.8).mean()) if len(aucs) else None,
        "frac_above_0.9": float((aucs > 0.9).mean()) if len(aucs) else None,
        "n_mono": sum(m == "mono" for m in monos),
        "n_poly": sum(m == "poly" for m in monos),
        "n_dead": sum(m == "dead" for m in monos),
    }
    with open(join(cache_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {join(cache_dir, 'summary.json')}")
    print("\n=== AutoInterp summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v if isinstance(v, int) or v is None else f'{v:.3f}'}")

    if len(rows) > 0:
        print("\n=== Top-10 atoms by AUC ===")
        for r in sorted(rows, key=lambda x: -x["auc"])[:10]:
            print(f"  a{r['atom']:2d}  AUC={r['auc']:.3f}  Δr={r['dr']:+.4f}  | {r['description'][:90]}")
        print("\n=== Bottom-10 atoms by AUC ===")
        for r in sorted(rows, key=lambda x: x["auc"])[:10]:
            print(f"  a{r['atom']:2d}  AUC={r['auc']:.3f}  Δr={r['dr']:+.4f}  | {r['description'][:90]}")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(aucs, bins=20, alpha=0.7, color="tab:blue", edgecolor="black", linewidth=0.3)
        ax.axvline(0.5, color="grey", lw=1, ls="--", label="chance")
        ax.axvline(0.7, color="tab:green", lw=1, ls="--", label="0.7 threshold")
        ax.set_xlabel("Held-out AUC")
        ax.set_ylabel("# atoms")
        ax.set_title(f"AutoInterp AUC — {base[:60]}...")
        ax.legend()
        fig.tight_layout()
        fig.savefig(join(out_dir, "auc_distribution.png"), dpi=120)
        plt.close(fig)
        print(f"[save] {join(out_dir, 'auc_distribution.png')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["prepare", "emit_batch_prompt", "aggregate"],
                   required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--ckpt_name", default="model_last.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--context_trs", type=int, default=2,
                   help="±N TRs around target for negatives. Default 2 (~25 words).")
    p.add_argument("--context_words_each_side", type=int, default=8,
                   help="Trim each context to ~N words on each side of **target**.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "prepare":
        cmd_prepare(args)
    elif args.mode == "emit_batch_prompt":
        cmd_emit_batch_prompt(args)
    elif args.mode == "aggregate":
        cmd_aggregate(args)


if __name__ == "__main__":
    main()
