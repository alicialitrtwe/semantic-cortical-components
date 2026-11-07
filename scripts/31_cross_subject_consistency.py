#!/usr/bin/env python
"""Cross-subject consistency of cortical maps for any set of model folders (Section 4.3 of the paper and its cross-subject appendix).

For each model: build each subject's cortical map per component (delay-weighted brain-decoder
column, absolute value), project to fsaverage with the subject's mapper, and average the Pearson
correlation of the same component's map over the 15 subject pairs. Reports the mean over
components per model and, for the first model, a shuffled-component null (1,000 permutations,
add-one p-values, BH-FDR over components).

Needs outputs/mappers/UTS0x_voxel_to_fsaverage.hdf for the six subjects.

  python scripts/31_cross_subject_consistency.py --model_dirs <dir_s0> <dir_s1> <dir_s2> --label NAME
"""
import argparse, json, os, sys
from itertools import combinations
from os.path import join
import h5py, numpy as np, scipy.sparse, torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NV = {"UTS01": 81126, "UTS02": 94251, "UTS03": 95556, "UTS04": 109469, "UTS05": 99322, "UTS06": 92198}
SUBJ = list(NV)


def mapper(root, s):
    with h5py.File(join(root, "outputs", "mappers", f"{s}_voxel_to_fsaverage.hdf"), "r") as f:
        return scipy.sparse.csr_matrix((f["data"][:], f["indices"][:], f["indptr"][:]), shape=tuple(f["shape"][:]))


def crosscorr(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    A = A / np.maximum(np.sqrt((A ** 2).sum(0)), 1e-12); B = B / np.maximum(np.sqrt((B ** 2).sum(0)), 1e-12)
    return A.T @ B


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for rank, i in zip(range(n, 0, -1), o[::-1]):
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dirs", nargs="+", required=True)
    ap.add_argument("--ckpt", default="model_last.pt")
    ap.add_argument("--label", default="models")
    ap.add_argument("--root", default=PROJECT, help="folder that contains outputs/mappers")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--out_json", default=None)
    a = ap.parse_args()
    MAP = {s: mapper(a.root, s) for s in SUBJ}
    per_seed = []; null_info = None
    for mi, d in enumerate(a.model_dirs):
        p = join(d, a.ckpt)
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
        ck = torch.load(p, map_location="cpu", weights_only=False); sd = ck["state_dict"]; vi = ck["voxel_indices"]
        lam = sd["delay_scales"].numpy(); nd, K = lam.shape
        ind, maps = {}, {}
        for s in SUBJ:
            i = np.zeros(NV[s], dtype=np.float32); i[vi[s]] = 1.0; ind[s] = MAP[s].dot(i) > 0
            W = sd[f"brain_decoders.{s}.weight"].numpy()
            D = np.abs((W.reshape(W.shape[0], nd, K) * lam[None]).sum(1))
            full = np.zeros((NV[s], K), dtype=np.float32); full[vi[s]] = D
            maps[s] = np.clip(np.asarray(MAP[s].dot(full)), 0, None)
        C = {}
        for s1, s2 in combinations(SUBJ, 2):
            m = ind[s1] & ind[s2]; C[(s1, s2)] = crosscorr(maps[s1][m], maps[s2][m])
        obs = np.mean([np.diag(c) for c in C.values()], 0)
        per_seed.append(float(obs.mean()))
        print(f"  {os.path.basename(d)[-75:]}  {a.ckpt}: mean cross-subject r = {obs.mean():.4f}", flush=True)
        if mi == 0:
            rng = np.random.default_rng(0); nullc = np.zeros((a.n_perm, K))
            for q in range(a.n_perm):
                perm = {s: rng.permutation(K) for s in SUBJ}
                nullc[q] = np.mean([C[(s1, s2)][perm[s1], perm[s2]] for s1, s2 in C], 0)
            ng = nullc.mean(1); pc = ((nullc >= obs[None]).sum(0) + 1) / (a.n_perm + 1); qv = bh(pc)
            null_info = {"null_mean": float(ng.mean()), "null_std": float(ng.std()),
                         "p_global": float(((ng >= obs.mean()).sum() + 1) / (a.n_perm + 1)),
                         "n_components_q05": int((qv < 0.05).sum()), "n_components": int(K),
                         "per_component": [{"component": int(k), "r": float(obs[k]), "p": float(pc[k]),
                                            "q": float(qv[k]), "sig_q05": bool(qv[k] < 0.05)} for k in range(K)]}
    res = {"label": a.label, "ckpt": a.ckpt, "per_seed": per_seed, "mean": float(np.mean(per_seed)),
           "pop_std": float(np.std(per_seed)), "first_model_null": null_info}
    print(f"== {a.label}: {res['mean']:.3f} ± {res['pop_std']:.3f} (population std over {len(per_seed)} seeds); first model: "
          f"null {null_info['null_mean']:.3f} ± {null_info['null_std']:.3f}, p = {null_info['p_global']:.4f}, "
          f"{null_info['n_components_q05']}/{null_info['n_components']} components at q < 0.05")
    if a.out_json:
        json.dump(res, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
