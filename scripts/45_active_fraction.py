"""Active fraction per component of the coupled multi-subject SAE (3 seeds, model_last.pt).

Reproduces the input pipeline of scripts/17_fit_multisubject_sae.py:
per-story demeaning (args.no_demean False), per-story trim (args trim_start/trim_end),
concatenation over checkpoint['train_stories'], global RMS normalisation
rms = sqrt(mean(X_train**2)), then SemanticSAE.encode:
z = TopK_k(ReLU(W_enc (x - b_pre) + b_enc)).
Active fraction of j = mean over TRs of 1[j in top-k at t].
Run: uv run python scripts/45_active_fraction.py
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from sparsebrain.runs import find_runs, seed_of, PAPER_COUPLED  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--model_dirs", nargs="+", default=None,
                help="runs to use; default: seeds 0-2 with the paper's coupled-SAE settings")
ap.add_argument("--features", default=os.path.join(PROJECT, "outputs", "features", "stories-lebel-gemma2.npz"))
ap.add_argument("--out", default=os.path.join(PROJECT, "reports", "active_fraction.json"))
cli = ap.parse_args()
DIRS = cli.model_dirs or find_runs(require="model_last.pt", seeds=(0, 1, 2), **PAPER_COUPLED)
FEAT, OUT = cli.features, cli.out

feats = np.load(FEAT)
result = {"feature_file": FEAT, "seeds": {}}
ref_stories = None
for run in DIRS:
    seed = seed_of(run)
    path = os.path.join(run, "model_last.pt")
    ck = torch.load(path, weights_only=False, map_location="cpu")
    a = ck["args"]
    a = a if isinstance(a, dict) else vars(a)
    assert a["sparsity_mode"] == "topk" and a["encoder_mode"] == "linear"
    assert not a.get("whiten", False) and a.get("feature_scale", 1.0) == 1.0
    assert a["input_rms_normalize"]
    k = int(a["k"])
    stories = list(ck["train_stories"])
    ts, te = int(a.get("trim_start", 5)), int(a.get("trim_end", 5))
    parts = []
    for s in stories:
        x = feats[s].astype(np.float64)
        if not a["no_demean"]:
            x = x - x.mean(axis=0, keepdims=True)
        x = np.nan_to_num(x).astype(np.float32)
        parts.append(x[slice(ts, -te if te > 0 else None)])
    X = np.concatenate(parts, axis=0)
    rms = float(np.sqrt(np.mean(X ** 2)))
    Xn = torch.from_numpy((X / rms).astype(np.float32))
    sd = ck["state_dict"]
    W, b_enc, b_pre = sd["encoder.weight"].float(), sd["encoder_bias"].float(), sd["b_pre"].float()
    with torch.no_grad():
        pre = torch.relu((Xn - b_pre) @ W.T + b_enc)
        _, idx = torch.topk(pre, k, dim=-1)
        mask = torch.zeros_like(pre).scatter_(-1, idx, 1.0)
        n_zero_in_topk = int((torch.gather(pre, 1, idx) <= 0).sum())
    frac = mask.mean(0).numpy().astype(float)
    entry = {
        "checkpoint": path, "k": k, "trim_start": ts, "trim_end": te,
        "demean_per_story": not a["no_demean"], "n_stories": len(stories),
        "rms": rms, "n_TRs": int(X.shape[0]),
        "fractions": [round(f, 6) for f in frac.tolist()],
        "sum_fractions": float(frac.sum()),
        "median_fraction": float(np.median(frac)),
        "n_topk_slots_with_zero_preactivation": n_zero_in_topk,
    }
    if seed == 0:
        order = np.argsort(-frac)
        entry["sorted_components"] = [{"component": f"c{int(j)}", "fraction": round(float(frac[j]), 6)} for j in order]
    result["seeds"][str(seed)] = entry
    if ref_stories is None:
        ref_stories = stories
    result.setdefault("same_train_stories_across_seeds", True)
    if stories != ref_stories:
        result["same_train_stories_across_seeds"] = False
    print(f"seed {seed}: rms={rms:.4f} n_TRs={X.shape[0]} sum={frac.sum():.6f} median={np.median(frac):.4f} zero-slots={n_zero_in_topk}")
    print("  top8:", [(f"c{j}", round(float(frac[j]), 4)) for j in np.argsort(-frac)[:8]])

result["rms"] = result["seeds"]["0"]["rms"]
result["n_TRs"] = result["seeds"]["0"]["n_TRs"]
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(result, f, indent=1)
print("wrote", OUT)
