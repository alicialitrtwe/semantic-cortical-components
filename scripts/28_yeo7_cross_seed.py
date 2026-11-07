#!/usr/bin/env python
"""28_yeo7_cross_seed.py — Cross-seed reproducibility of atom × Yeo7 mapping.

Takes 3 seed-variant model dirs, matches atoms across seeds via Hungarian
on encoder-row cosine similarity, then for each matched-atom triple compares
their Yeo7 network distributions.

Outputs:
    cross_seed_matching.csv         — per seed pair: rank, atomA, atomB, encoder cos
    cross_seed_reproducibility.csv  — per matched triple: encoder cosines, Yeo7
                                       Spearman per pair, dominant network per
                                       seed, agreement flag
    cross_seed_yeo7_scatter.png     — scatter of dominant-network % across pairs

Usage:
    uv run python scripts/28_yeo7_cross_seed.py \\
        --model_dirs <s0_dir> <s1_dir> <s2_dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import combinations
from os.path import join

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

YEO7_NAMES = ["Visual", "Somatomotor", "DorsalAttn", "VentralAttn",
              "Limbic", "Frontoparietal", "Default"]


def load_encoder(model_dir, ckpt_name="model_last.pt"):
    sd = torch.load(join(model_dir, ckpt_name),
                    map_location="cpu", weights_only=False)["state_dict"]
    return sd["encoder.weight"].numpy()  # (n_atoms, n_sem)


def load_decoder_sem(model_dir, ckpt_name="model_last.pt"):
    """Semantic decoder: 50 atoms in n_sem-d space (n_sem = 2304 for Gemma2)."""
    sd = torch.load(join(model_dir, ckpt_name),
                    map_location="cpu", weights_only=False)["state_dict"]
    if "decoder_sem.weight" in sd:
        return sd["decoder_sem.weight"].numpy().T  # (n_atoms, n_sem)
    # Tied case: encoder rows are the decoder
    return sd["encoder.weight"].numpy()


def load_decoder_brain_fsaverage(yeo7_dir, subjects=("UTS01", "UTS02",
                                                      "UTS03", "UTS04",
                                                      "UTS05", "UTS06")):
    """Brain decoder atoms in fsaverage space, averaged across subjects.

    Returns (n_atoms, n_fsav_vertices) array. Uses cached fsav_D from
    `outputs/cache/yeo7_<base>/fsav_D_*.npy`.
    """
    base = os.path.basename(os.path.normpath(yeo7_dir)).replace("yeo7_", "")
    cache_dir = join(PROJECT_DIR, "outputs", "cache", f"yeo7_{base}")
    arrays = []
    for s in subjects:
        path = join(cache_dir, f"fsav_D_{s}.npy")
        if os.path.exists(path):
            arrays.append(np.load(path))  # (n_fsav, n_atoms)
    if not arrays:
        raise FileNotFoundError(f"No fsav_D_*.npy in {cache_dir}")
    avg = np.mean(arrays, axis=0)         # (n_fsav, n_atoms)
    return avg.T                           # (n_atoms, n_fsav)


def load_cos_csv(yeo7_dir):
    """Load atom_yeo7_cos.csv into a (n_atoms, 7) array of cosine similarities."""
    rows = list(csv.DictReader(open(join(yeo7_dir, "atom_yeo7_cos.csv"))))
    n_atoms = len(rows)
    cos = np.zeros((n_atoms, 7))
    for i, n in enumerate(YEO7_NAMES):
        for r in rows:
            cos[int(r["atom"]), i] = float(r[f"{n}_cos"])
    return cos, rows


def load_qvals_csv(yeo7_dir):
    """Load atom_yeo7_pvalues.csv into a (n_atoms, 7) q-value array, or None."""
    path = join(yeo7_dir, "atom_yeo7_pvalues.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    n_atoms = len(rows)
    q = np.zeros((n_atoms, 7))
    for i, n in enumerate(YEO7_NAMES):
        for r in rows:
            q[int(r["atom"]), i] = float(r[f"{n}_q"])
    return q


def hungarian_match(W_a, W_b):
    """Match rows of W_a to rows of W_b by cosine similarity. Return (idx_a, idx_b, sims)."""
    Wa = W_a / (np.linalg.norm(W_a, axis=1, keepdims=True) + 1e-12)
    Wb = W_b / (np.linalg.norm(W_b, axis=1, keepdims=True) + 1e-12)
    sim = Wa @ Wb.T   # (K, K)
    row, col = linear_sum_assignment(-sim)
    return row, col, sim[row, col]


def yeo7_dir_for(model_dir):
    base = os.path.basename(os.path.normpath(model_dir))
    return join(PROJECT_DIR, "reports", "figures", f"yeo7_{base}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dirs", nargs=3, required=True,
                   help="Three seed-variant model directories")
    p.add_argument("--out_dir", default=join(PROJECT_DIR, "reports", "figures",
                                              "yeo7_cross_seed"))
    p.add_argument("--seed_labels", nargs=3, default=["s0", "s1", "s2"])
    p.add_argument("--match_by", choices=["encoder", "decoder_sem",
                                           "decoder_brain"],
                   default="encoder",
                   help="Which atom representation to use for Hungarian matching")
    p.add_argument("--out_suffix", default="",
                   help="Suffix appended to output filenames (use to keep "
                        "outputs from different --match_by runs separate)")
    return p.parse_args()


def get_atom_vectors(model_dir, match_by):
    """Return (n_atoms, D) array used for Hungarian matching."""
    if match_by == "encoder":
        return load_encoder(model_dir)
    if match_by == "decoder_sem":
        return load_decoder_sem(model_dir)
    if match_by == "decoder_brain":
        return load_decoder_brain_fsaverage(yeo7_dir_for(model_dir))
    raise ValueError(match_by)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = args.seed_labels
    model_dirs = args.model_dirs

    # Load atom vectors for matching + per-seed Yeo7 data
    atom_vecs = [get_atom_vectors(d, args.match_by) for d in model_dirs]
    coss = [load_cos_csv(yeo7_dir_for(d))[0] for d in model_dirs]   # list of (50, 7)
    qvals = [load_qvals_csv(yeo7_dir_for(d)) for d in model_dirs]   # may be None
    n_atoms = atom_vecs[0].shape[0]
    print(f"Loaded {len(seeds)} seeds, {n_atoms} atoms each.")
    print(f"  match_by = {args.match_by!r}")
    print(f"  atom vector shapes: {[v.shape for v in atom_vecs]}")
    print(f"  cos shapes: {[c.shape for c in coss]}")

    # Pairwise Hungarian matching on chosen atom vectors
    pairs = list(combinations(range(len(seeds)), 2))
    matchings = {}
    for i, j in pairs:
        row, col, sims = hungarian_match(atom_vecs[i], atom_vecs[j])
        matchings[(i, j)] = (row, col, sims)
        print(f"\n[{seeds[i]} ↔ {seeds[j]}]")
        print(f"  matched {args.match_by} cos: mean={sims.mean():.3f}, median={np.median(sims):.3f}, "
              f"min={sims.min():.3f}, max={sims.max():.3f}")
        print(f"  cos > 0.7 fraction: {(sims > 0.7).mean():.2%}")

    # Save the pairwise matching CSV
    match_csv = join(args.out_dir, f"cross_seed_matching{args.out_suffix}.csv")
    with open(match_csv, "w") as f:
        f.write("seed_pair,rank,atomA,atomB,encoder_cos\n")
        for (i, j) in pairs:
            row, col, sims = matchings[(i, j)]
            order = np.argsort(-sims)
            for r, idx in enumerate(order):
                f.write(f"{seeds[i]}-{seeds[j]},{r},{int(row[idx])},{int(col[idx])},"
                        f"{sims[idx]:.4f}\n")
    print(f"\n[save] {match_csv}")

    # Build "matched triples" anchored on s0 atoms.
    # s0 atom k → s1 atom (s0-s1 mapping) → s2 atom (s0-s2 mapping)
    row_01, col_01, sims_01 = matchings[(0, 1)]
    row_02, col_02, sims_02 = matchings[(0, 2)]
    row_12, col_12, sims_12 = matchings[(1, 2)]

    s0_to_s1 = dict(zip(row_01, col_01))
    s0_to_s2 = dict(zip(row_02, col_02))
    cos_01 = dict(zip(row_01, sims_01))
    cos_02 = dict(zip(row_02, sims_02))

    # For each s0 atom, also need cos between (s1 partner, s2 partner) — look up in matchings[(1,2)]
    s1_to_s2_full = dict(zip(row_12, col_12))
    s1s2_cos_full = dict(zip(row_12, sims_12))

    # Per-triple records
    repro_csv = join(args.out_dir, f"cross_seed_reproducibility{args.out_suffix}.csv")
    triples = []
    with open(repro_csv, "w") as f:
        cols = ["s0_atom", "s1_atom", "s2_atom",
                "enc_cos_s0_s1", "enc_cos_s0_s2", "enc_cos_s1_to_partner_s2",
                "yeo_spearman_s0_s1", "yeo_spearman_s0_s2", "yeo_spearman_s1_s2",
                "mean_yeo_spearman",
                "dom_net_s0", "dom_net_s1", "dom_net_s2",
                "dom_cos_s0", "dom_cos_s1", "dom_cos_s2",
                "dom_agree_3_seeds"]
        f.write(",".join(cols) + "\n")
        for k in range(n_atoms):
            s1_k = int(s0_to_s1[k])
            s2_k = int(s0_to_s2[k])
            # cos between s1_k and the s2 partner inferred via the s1↔s2 hungarian
            # (note: this s2 partner may differ from s2_k = s0's s2-partner)
            s1s2_partner = int(s1_to_s2_full[s1_k])
            s1s2_cos = float(s1s2_cos_full[s1_k])

            v_s0 = coss[0][k]
            v_s1 = coss[1][s1_k]
            v_s2 = coss[2][s2_k]
            sp_01 = spearmanr(v_s0, v_s1).statistic if v_s0.std() and v_s1.std() else np.nan
            sp_02 = spearmanr(v_s0, v_s2).statistic if v_s0.std() and v_s2.std() else np.nan
            sp_12 = spearmanr(v_s1, v_s2).statistic if v_s1.std() and v_s2.std() else np.nan
            mean_sp = np.nanmean([sp_01, sp_02, sp_12])
            dom_s0 = int(v_s0.argmax()); dom_s1 = int(v_s1.argmax()); dom_s2 = int(v_s2.argmax())
            agree = int(dom_s0 == dom_s1 == dom_s2)

            f.write(",".join([
                str(k), str(s1_k), str(s2_k),
                f"{cos_01[k]:.4f}", f"{cos_02[k]:.4f}", f"{s1s2_cos:.4f}",
                f"{sp_01:.4f}", f"{sp_02:.4f}", f"{sp_12:.4f}", f"{mean_sp:.4f}",
                YEO7_NAMES[dom_s0], YEO7_NAMES[dom_s1], YEO7_NAMES[dom_s2],
                f"{v_s0[dom_s0]:.4f}", f"{v_s1[dom_s1]:.4f}", f"{v_s2[dom_s2]:.4f}",
                str(agree),
            ]) + "\n")
            triples.append(dict(s0=k, s1=s1_k, s2=s2_k,
                                cos_01=cos_01[k], cos_02=cos_02[k], cos_12=s1s2_cos,
                                sp_01=sp_01, sp_02=sp_02, sp_12=sp_12,
                                mean_sp=mean_sp,
                                dom_s0=dom_s0, dom_s1=dom_s1, dom_s2=dom_s2,
                                cos_s0=v_s0, cos_s1=v_s1, cos_s2=v_s2,
                                agree=agree))
    print(f"[save] {repro_csv}")

    # ---------- Headline stats ----------
    print("\n" + "=" * 60)
    print("CROSS-SEED REPRODUCIBILITY — headline numbers")
    print("=" * 60)
    mean_sps = np.array([t["mean_sp"] for t in triples])
    cos_min = np.array([min(t["cos_01"], t["cos_02"], t["cos_12"]) for t in triples])
    agree = np.array([t["agree"] for t in triples])
    print(f"\nMean per-triple cross-seed Yeo7 Spearman:")
    print(f"  median={np.nanmedian(mean_sps):.3f}  mean={np.nanmean(mean_sps):.3f}  "
          f"std={np.nanstd(mean_sps):.3f}")
    print(f"\nFraction of triples with mean Spearman > X:")
    for thr in [0.3, 0.5, 0.7, 0.9]:
        print(f"  > {thr}: {(mean_sps > thr).mean():.2%} ({int((mean_sps > thr).sum())}/50)")
    print(f"\nDominant-network agreement (all 3 seeds same):")
    print(f"  {agree.sum()}/{len(triples)} ({agree.mean():.1%}) triples")
    print(f"\nMin pairwise encoder cos in triple:")
    print(f"  median={np.median(cos_min):.3f}, mean={cos_min.mean():.3f}")
    print(f"  fraction with min cos > 0.5: {(cos_min > 0.5).mean():.2%}")
    print(f"  fraction with min cos > 0.7: {(cos_min > 0.7).mean():.2%}")
    print(f"\nHEADLINE: triples with (mean Spearman > 0.7 AND all 3 seeds agree on dominant net):")
    headline = (mean_sps > 0.7) & (agree == 1)
    print(f"  {headline.sum()}/50 ({headline.mean():.1%})")

    # Per-network agreement breakdown
    print(f"\nPer-network: # triples whose all-3-seeds-dominant network is this network:")
    for i, n in enumerate(YEO7_NAMES):
        n_for_this = sum(1 for t in triples
                         if t["dom_s0"] == i and t["dom_s1"] == i and t["dom_s2"] == i)
        print(f"  {n:<16s}: {n_for_this}")

    # ---------- Figure: scatter of dominant-net % across seed pairs ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, (i, j), title in zip(axes, [(0, 1), (0, 2), (1, 2)],
                                  [f"{seeds[0]} vs {seeds[1]}",
                                   f"{seeds[0]} vs {seeds[2]}",
                                   f"{seeds[1]} vs {seeds[2]}"]):
        x, y, c = [], [], []
        for t in triples:
            v_i = t[f"cos_s{i}"]; v_j = t[f"cos_s{j}"]
            dom_i = int(v_i.argmax())
            x.append(v_i[dom_i])
            y.append(v_j[dom_i])  # same network index
            c.append("tab:blue" if dom_i == int(v_j.argmax()) else "tab:red")
        ax.scatter(x, y, c=c, alpha=0.7, s=40, edgecolor="black", linewidth=0.3)
        ax.plot([0, 0.7], [0, 0.7], "k--", lw=0.5)
        ax.set_title(title)
        ax.set_xlabel(f"{seeds[i]} cos  (in {seeds[i]}'s dominant net)")
        ax.set_ylabel(f"{seeds[j]} cos  (matched atom, same net)")
        ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.5)
        ax.grid(alpha=0.3)
    fig.suptitle("Cross-seed reproducibility of Yeo7 dominance "
                 "(blue = same dominant net, red = different)")
    fig.tight_layout()
    fig.savefig(join(args.out_dir, f"cross_seed_yeo7_scatter{args.out_suffix}.png"), dpi=120)
    plt.close(fig)
    print(f"\n[save] {join(args.out_dir, 'cross_seed_yeo7_scatter.png')}")

    # JSON summary
    summary = {
        "model_dirs": model_dirs,
        "seed_labels": seeds,
        "n_atoms": int(n_atoms),
        "n_triples": int(len(triples)),
        "encoder_cosine": {
            "mean_per_pair": {f"{seeds[i]}_{seeds[j]}": float(matchings[(i,j)][2].mean())
                              for (i, j) in pairs},
            "median_per_pair": {f"{seeds[i]}_{seeds[j]}": float(np.median(matchings[(i,j)][2]))
                                for (i, j) in pairs},
        },
        "yeo7_spearman": {
            "median_per_triple_mean": float(np.nanmedian(mean_sps)),
            "mean_per_triple_mean": float(np.nanmean(mean_sps)),
            "frac_above_0.5": float((mean_sps > 0.5).mean()),
            "frac_above_0.7": float((mean_sps > 0.7).mean()),
            "frac_above_0.9": float((mean_sps > 0.9).mean()),
        },
        "dominant_network_agreement_3_seeds": {
            "n_triples_agreeing": int(agree.sum()),
            "frac": float(agree.mean()),
        },
        "headline_count": {
            "definition": "triples with mean cross-seed Spearman > 0.7 AND same dominant network across all 3 seeds",
            "n": int(headline.sum()),
            "frac": float(headline.mean()),
        },
    }
    summary["match_by"] = args.match_by
    with open(join(args.out_dir, f"summary{args.out_suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {join(args.out_dir, f'summary{args.out_suffix}.json')}")


if __name__ == "__main__":
    main()
