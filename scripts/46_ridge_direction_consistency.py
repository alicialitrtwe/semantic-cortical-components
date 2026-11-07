"""Cross-subject consistency of ridge weights projected onto component directions, against the
coupled SAE's cortical maps: ridge projected onto (a) encoder rows and (b) semantic-decoder columns.

Usage: uv run python scripts/46_ridge_direction_consistency.py <project dir> <out.json>
Needs the ridge refits of scripts/23b_ridge_refit_trainonly.py and seeds 0-2 of the coupled SAE.
"""
import glob, json, os, sys
from itertools import combinations
from os.path import join
import h5py, numpy as np, scipy.sparse, torch
from scipy.stats import ttest_rel

P = sys.argv[1]; OUT = sys.argv[2]
MAPPER_DIR = join(P, "outputs", "mappers")
NV = {"UTS01": 81126, "UTS02": 94251, "UTS03": 95556, "UTS04": 109469, "UTS05": 99322, "UTS06": 92198}
SUBJ = list(NV)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparsebrain.runs import find_runs, PAPER_COUPLED  # noqa: E402
MODEL_DIRS = find_runs(root=join(P, "outputs", "models"), require="model.pt", seeds=(0, 1, 2), **PAPER_COUPLED)

def mapper(s):
    with h5py.File(join(MAPPER_DIR, f"{s}_voxel_to_fsaverage.hdf"), "r") as f:
        return scipy.sparse.csr_matrix((f["data"][:], f["indices"][:], f["indptr"][:]), shape=tuple(f["shape"][:]))
MAP = {s: mapper(s) for s in SUBJ}

def pearson(M1, M2):
    a = M1 - M1.mean(0); b = M2 - M2.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return np.where(den > 1e-12, (a * b).sum(0) / den, 0.0)

def consistency(maps, masks):
    acc = 0
    for s1, s2 in combinations(SUBJ, 2):
        m = masks[(s1, s2)]; acc = acc + pearson(maps[s1][m], maps[s2][m])
    return acc / 15

def to_fs(s, vox_idx, D):
    full = np.zeros((NV[s], D.shape[1]), dtype=np.float32); full[vox_idx] = D
    return np.clip(np.asarray(MAP[s].dot(full)), 0, None)

RIDGE = {}
for s in SUBJ:
    d = sorted(glob.glob(join(P, f"outputs/models/ridge_{s}_*stories_gemma2_cv_matched2")))[-1]
    RIDGE[s] = (np.load(join(d, "primal_weights_sum.npy")), np.load(join(d, "voxel_indices_sel.npy")))
    print("ridge", s, RIDGE[s][0].shape, os.path.basename(d), flush=True)

res = {}
for ckname in ("model.pt", "model_last.pt"):
    per = {k: [] for k in ("sae", "ridge_enc", "ridge_dec")}
    for d in MODEL_DIRS:
        ck = torch.load(join(d, ckname), map_location="cpu", weights_only=False); sd = ck["state_dict"]
        E = sd["encoder.weight"].numpy().astype(np.float32)                 # (K, F)
        Dm = sd["decoder_sem.weight"].numpy().astype(np.float32).T          # (K, F) columns of W_sem
        Dm = Dm / np.linalg.norm(Dm, axis=1, keepdims=True)
        lam = sd["delay_scales"].numpy(); nd, K = lam.shape
        vi = ck["voxel_indices"]
        ind = {}
        for s in SUBJ:
            i = np.zeros(NV[s], dtype=np.float32); i[vi[s]] = 1.0; ind[s] = MAP[s].dot(i) > 0
        masks = {(a, b): ind[a] & ind[b] for a, b in combinations(SUBJ, 2)}
        sae, renc, rdec = {}, {}, {}
        for s in SUBJ:
            W = sd[f"brain_decoders.{s}.weight"].numpy()
            D = np.abs((W.reshape(W.shape[0], nd, K) * lam[None]).sum(1))
            sae[s] = to_fs(s, vi[s], D)
            Ws, vsel = RIDGE[s]
            assert np.array_equal(np.sort(vsel), np.sort(np.asarray(vi[s]))), s
            renc[s] = to_fs(s, vsel, np.abs(Ws @ E.T))
            rdec[s] = to_fs(s, vsel, np.abs(Ws @ Dm.T))
        per["sae"].append(consistency(sae, masks)); per["ridge_enc"].append(consistency(renc, masks)); per["ridge_dec"].append(consistency(rdec, masks))
        print(ckname, os.path.basename(d)[-60:-40], {k: round(float(v[-1].mean()), 4) for k, v in per.items()}, flush=True)
    out = {}
    for k, v in per.items():
        seeds = [float(x.mean()) for x in v]
        out[k] = {"per_seed": seeds, "mean": float(np.mean(seeds)), "pop_std": float(np.std(seeds))}
        np.save(join(OUT, f"{ckname.replace('.pt','')}_{k}_per_component_3seed.npy"), np.mean(v, 0))
    ms = np.mean(per["sae"], 0)
    for k in ("ridge_enc", "ridge_dec"):
        mr = np.mean(per[k], 0); t, p2 = ttest_rel(ms, mr)
        out[f"sae_vs_{k}"] = {"mean_diff": float((ms - mr).mean()), "sd_diff": float((ms - mr).std(ddof=1)), "t": float(t),
                              "p_one_sided_sae_greater": float(p2 / 2 if t > 0 else 1 - p2 / 2), "n_components_sae_greater": int((ms > mr).sum())}
    res[ckname] = out
json.dump(res, open(join(OUT, "ridge_direction_check.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
