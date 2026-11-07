#!/usr/bin/env python
"""02b_save_features_llm.py — Compute contextual LLM embeddings for stories.

Runs story text through a causal LM (default: Gemma 2 2B), extracts hidden
states, average-pools subword tokens back to word level, then downsamples to
TR rate via Lanczos interpolation — identical output format to 02_save_features.py.

Usage:
    # Default: Gemma 2 2B, last hidden layer, LeBel dataset
    uv run python scripts/02b_save_features_llm.py --subject UTS03

    # Narratives dataset (merlin, pieman, sherlock)
    uv run python scripts/02b_save_features_llm.py --dataset narratives

    # Average last 4 layers
    uv run python scripts/02b_save_features_llm.py --subject UTS03 --layer avg_last4

    # Different model
    uv run python scripts/02b_save_features_llm.py --model_id meta-llama/Llama-2-7b-hf
"""
import argparse
import json
import os
import re
import sys
from bisect import bisect_right
from os.path import join

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = join(PROJECT_DIR, "external", "deep-fMRI-dataset")
DATA_DIR = join(DATASET_DIR, "data")
ENCODING_DIR = join(DATASET_DIR, "encoding")
DERIVATIVE_DIR = join(DATA_DIR, "ds003020", "derivative")
OUTPUT_DIR = join(PROJECT_DIR, "outputs", "features")

# Add encoding utilities to path
sys.path.insert(0, ENCODING_DIR)
sys.path.insert(0, PROJECT_DIR)
from ridge_utils.dsutils import make_word_ds
from ridge_utils.interpdata import lanczosinterp2D
from ridge_utils.stimulus_utils import load_simulated_trfiles, load_textgrids, TRFile
from ridge_utils.textgrid import TextGrid
from sparsebrain.paths import paths as _proj_paths

TRIM = slice(10, -5)

# Default story ordering (matches 02_save_features.py)
STORIES_TRAIN = [
    "alternateithicatom", "souls", "avatar", "legacy", "odetostepfather",
    "undertheinfluence", "howtodraw", "myfirstdaywiththeyankees", "naked",
    "life", "stagefright", "tildeath", "fromboyhoodtofatherhood", "sloth",
    "exorcism", "haveyoumethimyet", "adollshouse", "inamoment",
    "theclosetthatateeverything", "adventuresinsayingyes", "buck",
    "swimmingwithastronauts", "thatthingonmyarm", "eyespy", "itsabox",
    "hangtime",
]
STORIES_TEST = ["wheretheressmoke"]

# Narratives dataset constants
NARRATIVES_STORIES = ["pieman", "merlin", "sherlock"]
NARRATIVES_TR = 1.5
NARRATIVES_TRIM_START = 10
NARRATIVES_TRIM_END = 5
NARRATIVES_TRIM = slice(NARRATIVES_TRIM_START, -NARRATIVES_TRIM_END)
NARRATIVES_BAD_WORDS = {"sp", "sil", "{br}", "{lg}", "{ns}", "", "spn", "<unk>"}


# ---------------------------------------------------------------------------
# Token-to-word alignment
# ---------------------------------------------------------------------------
def align_tokens_to_words(
    tokenizer, words: list[str]
) -> tuple[list[int], list[int]]:
    """Tokenize a word list and map each token back to its source word.

    Builds the full text by joining words with spaces, tokenizes, then uses
    character offsets to assign each token to the word it came from.

    Handles tokenizers (like GPT-2) that include the preceding space in a
    token's character span by mapping each character to its source word and
    looking up the first non-space character in the token.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerFast
        HuggingFace tokenizer (must support return_offsets_mapping).
    words : list of str
        Words in presentation order.

    Returns
    -------
    input_ids : list of int
        Token IDs for the full text.
    word_indices : list of int
        For each token, the index of the source word (0-based).
        Special tokens (BOS/EOS) get index -1.
    """
    text = " ".join(words)

    # Build char -> word mapping (spaces between words map to -1)
    char_to_word = [-1] * len(text)
    pos = 0
    for i, w in enumerate(words):
        for j in range(len(w)):
            char_to_word[pos + j] = i
        pos += len(w) + 1  # +1 for space

    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    word_indices = []
    for token_start, token_end in offsets:
        if token_start == 0 and token_end == 0:
            # Special token (BOS, EOS, etc.)
            word_indices.append(-1)
        else:
            # Find first character in token span that belongs to a word
            # (skips leading spaces that some tokenizers include)
            wi = -1
            for c in range(token_start, token_end):
                if c < len(char_to_word) and char_to_word[c] >= 0:
                    wi = char_to_word[c]
                    break
            word_indices.append(wi)

    return input_ids, word_indices


# ---------------------------------------------------------------------------
# Model forward pass
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_and_extract(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: str,
    device: str,
) -> np.ndarray:
    """Run forward pass and extract hidden states from a specific layer.

    Parameters
    ----------
    model : PreTrainedModel
        HuggingFace causal LM.
    input_ids : Tensor, shape (1, seq_len)
        Token IDs.
    attention_mask : Tensor, shape (1, seq_len)
        Attention mask.
    layer : str
        Which layer to extract: "-1" for last, "avg_last4" for average of
        last 4 layers, or an integer string for a specific layer index.
    device : str
        Device string.

    Returns
    -------
    ndarray, shape (seq_len, hidden_dim)
        Hidden states for each token.
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    # outputs.hidden_states is a tuple of (n_layers + 1) tensors
    # index 0 = embedding layer, 1..n_layers = transformer layers
    hidden_states = outputs.hidden_states

    if layer == "avg_last4":
        # Average the last 4 transformer layers
        stacked = torch.stack(hidden_states[-4:], dim=0)  # (4, 1, seq, dim)
        hidden = stacked.mean(dim=0).squeeze(0)  # (seq, dim)
    else:
        layer_idx = int(layer)
        hidden = hidden_states[layer_idx].squeeze(0)  # (seq, dim)

    return hidden.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Pooling subword tokens -> words
# ---------------------------------------------------------------------------
def pool_tokens_to_words(
    hidden: np.ndarray,
    word_indices: list[int],
    n_words: int,
) -> np.ndarray:
    """Average-pool subword token embeddings back to word level.

    Parameters
    ----------
    hidden : ndarray, shape (n_tokens, hidden_dim)
        Per-token hidden states.
    word_indices : list of int
        Word index for each token (-1 for special tokens).
    n_words : int
        Total number of words.

    Returns
    -------
    ndarray, shape (n_words, hidden_dim)
        Per-word embeddings (average of constituent subword tokens).
        Words with no tokens get zero vectors.
    """
    hidden_dim = hidden.shape[1]
    word_embeds = np.zeros((n_words, hidden_dim), dtype=np.float64)
    word_counts = np.zeros(n_words, dtype=np.int32)

    for tok_idx, wi in enumerate(word_indices):
        if wi < 0:
            continue  # skip special tokens
        word_embeds[wi] += hidden[tok_idx]
        word_counts[wi] += 1

    # Average where we have tokens; leave zeros otherwise
    mask = word_counts > 0
    word_embeds[mask] /= word_counts[mask, None]

    return word_embeds.astype(np.float32)


# ---------------------------------------------------------------------------
# Full extraction pipeline for one word list
# ---------------------------------------------------------------------------
def extract_word_embeddings(
    model,
    tokenizer,
    words: list[str],
    layer: str,
    max_length: int,
    device: str,
) -> np.ndarray:
    """Extract contextual word embeddings with per-word optimal context.

    For causal LMs, each token's hidden state depends only on preceding
    tokens. This function guarantees every word gets the maximum possible
    left context:

    1. Forward pass on tokens[0:max_length] — gives optimal embeddings for
       all words whose last token falls before max_length (the full left
       history is visible).
    2. For each remaining word, runs a dedicated forward pass on a window
       ending at that word's last token, giving it exactly max_length tokens
       of left context.

    Parameters
    ----------
    model : PreTrainedModel
        HuggingFace causal LM.
    tokenizer : PreTrainedTokenizerFast
        Corresponding tokenizer.
    words : list of str
        Words in order.
    layer : str
        Layer specification (see forward_and_extract).
    max_length : int
        Maximum sequence length per forward pass.
    device : str
        Device string.

    Returns
    -------
    ndarray, shape (n_words, hidden_dim)
        Contextual embedding per word.
    """
    n_words = len(words)
    input_ids, word_indices = align_tokens_to_words(tokenizer, words)
    n_tokens = len(input_ids)

    if n_tokens <= max_length:
        # Single forward pass — already optimal for causal LMs
        ids_tensor = torch.tensor([input_ids], dtype=torch.long)
        mask_tensor = torch.ones(1, n_tokens, dtype=torch.long)
        hidden = forward_and_extract(model, ids_tensor, mask_tensor, layer, device)
        return pool_tokens_to_words(hidden, word_indices, n_words)

    # --- Per-word optimal context for long sequences ---
    hidden_dim = model.config.hidden_size
    word_embeds = np.zeros((n_words, hidden_dim), dtype=np.float64)
    word_counts = np.zeros(n_words, dtype=np.int32)

    # Find last token position for each word
    word_last_token = np.full(n_words, -1, dtype=np.int64)
    for tok_idx, wi in enumerate(word_indices):
        if wi >= 0:
            word_last_token[wi] = tok_idx

    # Pass 1: tokens[0:max_length] — optimal for all words ending here
    first_end = min(max_length, n_tokens)
    ids_tensor = torch.tensor([input_ids[:first_end]], dtype=torch.long)
    mask_tensor = torch.ones(1, first_end, dtype=torch.long)
    hidden_first = forward_and_extract(model, ids_tensor, mask_tensor, layer, device)

    claimed = np.zeros(n_words, dtype=bool)
    for tok_idx in range(first_end):
        wi = word_indices[tok_idx]
        if wi >= 0 and word_last_token[wi] < first_end:
            # This word's last token is in the first window — claim it
            word_embeds[wi] += hidden_first[tok_idx]
            word_counts[wi] += 1
            claimed[wi] = True

    # Pass 2+: per-word optimal windows for remaining words
    unclaimed = np.where(~claimed & (word_last_token >= 0))[0]
    if len(unclaimed) > 0:
        # Group unclaimed words by their required window to minimize passes.
        # Words sharing the same window end can share a single forward pass.
        # Window end = word_last_token[wi] + 1
        window_end_to_words = {}
        for wi in unclaimed:
            w_end = int(word_last_token[wi]) + 1
            if w_end not in window_end_to_words:
                window_end_to_words[w_end] = []
            window_end_to_words[w_end].append(wi)

        # Process unique windows in order
        for w_end in sorted(window_end_to_words.keys()):
            w_start = max(0, w_end - max_length)
            chunk_ids = input_ids[w_start:w_end]

            ids_tensor = torch.tensor([chunk_ids], dtype=torch.long)
            mask_tensor = torch.ones(1, len(chunk_ids), dtype=torch.long)
            hidden = forward_and_extract(model, ids_tensor, mask_tensor, layer, device)

            # Extract embeddings for words claimed by this window
            target_words = set(window_end_to_words[w_end])
            for tok_idx_in_chunk in range(len(chunk_ids)):
                tok_idx_global = w_start + tok_idx_in_chunk
                wi = word_indices[tok_idx_global]
                if wi in target_words:
                    word_embeds[wi] += hidden[tok_idx_in_chunk]
                    word_counts[wi] += 1

    # Average subword tokens per word
    mask = word_counts > 0
    word_embeds[mask] /= word_counts[mask, None]
    return word_embeds.astype(np.float32)


# ---------------------------------------------------------------------------
# Compute features for all stories
# ---------------------------------------------------------------------------
def compute_features_llm(
    stories: list[str],
    model,
    tokenizer,
    layer: str,
    max_length: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Compute LLM-based semantic features for a list of stories.

    Pipeline per story: TextGrid -> words -> LLM forward -> pool to words ->
    Lanczos interpolation -> trim.

    Parameters
    ----------
    stories : list of str
        Story names.
    model : PreTrainedModel
        HuggingFace causal LM.
    tokenizer : PreTrainedTokenizerFast
        Corresponding tokenizer.
    layer : str
        Layer to extract.
    max_length : int
        Max sequence length per forward pass.
    device : str
        Device string.

    Returns
    -------
    dict
        Mapping story name -> array of shape (n_trs, hidden_dim).
    """
    # Load TextGrids and create word sequences
    print("Loading TextGrids and building word sequences...")
    grids = load_textgrids(stories, DATA_DIR)
    with open(join(DERIVATIVE_DIR, "respdict.json")) as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict)
    wordseqs = make_word_ds(grids, trfiles)

    features = {}
    for story in stories:
        if story not in wordseqs:
            print(f"  {story}: SKIPPED (no TextGrid)")
            continue

        ws = wordseqs[story]
        words = list(ws.data)
        n_words = len(words)
        print(f"  {story}: {n_words} words, ", end="", flush=True)

        # Get token count for info
        input_ids, _ = align_tokens_to_words(tokenizer, words)
        n_tokens = len(input_ids)
        print(f"{n_tokens} tokens -> ", end="", flush=True)

        # Extract word-level embeddings
        word_embeds = extract_word_embeddings(
            model, tokenizer, words, layer, max_length, device
        )

        # Lanczos interpolation: word times -> TR times
        downsampled = lanczosinterp2D(
            word_embeds, ws.data_times, ws.tr_times, window=3
        )

        # Trim to match preprocessed response data
        trimmed = downsampled[TRIM]
        print(f"{downsampled.shape[0]} TRs -> {trimmed.shape[0]} trimmed")
        features[story] = trimmed.astype(np.float32)

    return features


# ---------------------------------------------------------------------------
# Narratives dataset feature extraction
# ---------------------------------------------------------------------------
def _parse_textgrid_words_narratives(story: str) -> tuple[list[str], np.ndarray]:
    """Parse words and midpoint times from a Narratives TextGrid (tiers[0])."""
    path = os.path.join(_proj_paths.narratives_textgrids_dir, f"{story}.TextGrid")
    tg = TextGrid(open(path).read())
    transcript = tg.tiers[0].make_simple_transcript()
    words, times = [], []
    for start_s, end_s, word in transcript:
        if word.lower().strip("{}").strip() in NARRATIVES_BAD_WORDS:
            continue
        words.append(word.lower())
        times.append((float(start_s) + float(end_s)) / 2.0)
    return words, np.array(times)


def _make_tr_center_times_narratives(n_trs_simulate: int) -> np.ndarray:
    """TR center times for Narratives (soundstarttime=0)."""
    trf = TRFile(None, NARRATIVES_TR)
    trf.soundstarttime = 0.0
    trf.simulate(n_trs_simulate)
    return trf.get_reltriggertimes() + NARRATIVES_TR / 2.0


def compute_features_narratives(
    stories: list[str],
    model,
    tokenizer,
    layer: str,
    max_length: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Compute LLM features for Narratives dataset stories (TR=1.5s)."""
    respdict_path = os.path.join(
        _proj_paths.narratives_dir, "derivative", "respdict.json"
    )
    with open(respdict_path) as f:
        respdict = json.load(f)

    features = {}
    for story in stories:
        if story not in respdict:
            print(f"  {story}: not in respdict, SKIP")
            continue
        tg_path = os.path.join(
            _proj_paths.narratives_textgrids_dir, f"{story}.TextGrid"
        )
        if not os.path.exists(tg_path):
            print(f"  {story}: TextGrid missing, SKIP")
            continue

        words, word_times = _parse_textgrid_words_narratives(story)
        n_trs = respdict[story]
        n_simulate = n_trs + NARRATIVES_TRIM_START + NARRATIVES_TRIM_END
        tr_times = _make_tr_center_times_narratives(n_simulate)

        print(f"  {story}: {len(words)} words, ", end="", flush=True)
        input_ids, _ = align_tokens_to_words(tokenizer, words)
        print(f"{len(input_ids)} tokens -> ", end="", flush=True)

        word_embeds = extract_word_embeddings(
            model, tokenizer, words, layer, max_length, device
        )
        downsampled = lanczosinterp2D(word_embeds, word_times, tr_times, window=3)
        trimmed = downsampled[NARRATIVES_TRIM]

        if trimmed.shape[0] != n_trs:
            print(f"\n  WARNING: expected {n_trs} TRs, got {trimmed.shape[0]}")

        print(f"{downsampled.shape[0]} -> {trimmed.shape[0]} TRs")
        features[story] = trimmed.astype(np.float32)

    return features


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def get_all_stories_for_subject(subject: str) -> list[str]:
    """Get all story names available for a subject."""
    subject_dir = join(DERIVATIVE_DIR, "preprocessed_data", subject)
    stories = []
    for f in sorted(os.listdir(subject_dir)):
        if f.endswith(".hf5"):
            stories.append(f.replace(".hf5", ""))
    return stories


def model_short_name(model_id: str) -> str:
    """Extract a short name from a HuggingFace model ID.

    Examples
    --------
    >>> model_short_name("google/gemma-2-2b")
    'gemma2'
    >>> model_short_name("meta-llama/Llama-2-7b-hf")
    'llama2'
    """
    name = model_id.split("/")[-1].lower()
    # Remove size suffixes and "hf" marker
    name = re.sub(r"-\d+b(-hf)?$", "", name)
    # Collapse hyphens and version numbers into compact form
    name = name.replace("-", "")
    return name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model_id", type=str, default="google/gemma-2-2b",
        help="HuggingFace model ID (default: google/gemma-2-2b)",
    )
    parser.add_argument(
        "--layer", type=str, default="-1",
        help="Layer to extract: '-1' (last), 'avg_last4', or integer index (default: -1)",
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        help="Compute features for all stories available for this subject",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: float16)",
    )
    parser.add_argument(
        "--max_length", type=int, default=8192,
        help="Max sequence length per forward pass (default: 8192)",
    )
    parser.add_argument(
        "--dataset", type=str, default="lebel",
        choices=["lebel", "narratives"],
        help="Dataset to compute features for (default: lebel)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    short_name = model_short_name(args.model_id)
    if args.layer not in ("-1",):
        short_name += f"_L{args.layer}"

    if args.dataset == "narratives":
        all_stories = NARRATIVES_STORIES
        out_fn = join(OUTPUT_DIR, f"stories-narratives-{short_name}.npz")
        compute_fn = compute_features_narratives
    else:
        if args.subject:
            all_stories = get_all_stories_for_subject(args.subject)
            suffix = f"-{args.subject}"
        else:
            all_stories = STORIES_TRAIN + STORIES_TEST
            suffix = ""
        out_fn = join(OUTPUT_DIR, f"stories-lebel-{short_name}{suffix}.npz")
        compute_fn = compute_features_llm

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_id}")
    print(f"Layer: {args.layer}")
    print(f"Device: {args.device}, dtype: {args.dtype}")
    print(f"Max length: {args.max_length}")
    print(f"Stories: {len(all_stories)}")
    print(f"Output: {out_fn}")
    print()

    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading tokenizer from {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"Loading model from {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device,
    )
    model.eval()
    print(f"Model loaded: {model.config.hidden_size}-dim hidden states, "
          f"{model.config.num_hidden_layers} layers\n")

    # Compute features
    features = compute_fn(
        all_stories, model, tokenizer, args.layer, args.max_length, args.device
    )

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nSaving features to {out_fn}")
    np.savez(out_fn, **features)

    # Summary
    total_trs = sum(v.shape[0] for v in features.values())
    dim = next(iter(features.values())).shape[1]
    print(f"Total TRs across all stories: {total_trs}")
    print(f"Feature dimensionality: {dim}")
    print("Done.")


if __name__ == "__main__":
    main()
