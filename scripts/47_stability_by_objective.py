#!/usr/bin/env python
"""Cross-seed stability of a set of models, as in Table `tab:configstab` of the paper.

Same computation as scripts/43_config_stability.py (Hungarian-matched mean cosine between
seed pairs), but for explicit model folders, so it can be run on the retrained brain-only models.

  --stage1  folders whose encoder and brain decoders are compared (e.g. the three
            Brain->Sem stage-1 runs)
  --stage2  optional folders whose semantic decoder is compared (e.g. the three
            Brain->Sem stage-2 runs, i.e. semantic decoders fit on those encoders).
            If omitted, the semantic decoder is read from --stage1.
  --ckpt    model_last.pt (final epoch, default) or model.pt (best validation epoch)

Prints, for the three faces: mean Hungarian-matched cosine over seed pairs, the fraction of
components whose best match exceeds 0.7, and for the semantic decoder a rotation null.
"""
import argparse, glob, os, sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

torch.set_grad_enabled(False)
rng = np.random.default_rng(0)
SUBJS = ["UTS01", "UTS02", "UTS03", "UTS04", "UTS05", "UTS06"]


def unit(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def dec_rows(sd):
    return unit(sd["decoder_sem.weight"].float().numpy().T)


def enc_rows(sd):
    return unit(sd["encoder.weight"].float().numpy())


def brain_rows(sd):
    na = sd["encoder.weight"].shape[0]
    ds = sd["delay_scales"].float().numpy()
    maps = []
    for s in SUBJS:
        W = sd[f"brain_decoders.{s}.weight"].float().numpy().reshape(-1, ds.shape[0], na)
        M = (W * ds[None, :, :]).sum(1).T
        maps.append(unit(M - M.mean(0, keepdims=True)))
    return maps


def hung(A, B):
    S = A @ B.T
    r, c = linear_sum_assignment(-S)
    return S[r, c].mean()


def frac(A, B, th=0.7):
    return ((A @ B.T).max(1) > th).mean()


def rot(A):
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    r = int((S > 1e-8).sum())
    G = U[:, :r] * S[:r]
    Vr, _ = np.linalg.qr(rng.standard_normal((A.shape[1], r)))
    return unit(G @ Vr.T)


def load(d, ckpt):
    p = os.path.join(d, ckpt)
    if not os.path.exists(p):
        sys.exit(f"missing {p}")
    c = torch.load(p, map_location="cpu", weights_only=False)
    return c["state_dict"], c.get("best_epoch"), len(c["history"]["train_total"]) if "history" in c and "train_total" in c["history"] else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", nargs="+", required=True)
    ap.add_argument("--stage2", nargs="+", default=None)
    ap.add_argument("--ckpt", default="model_last.pt")
    ap.add_argument("--label", default="models")
    a = ap.parse_args()
    s1 = [load(d, a.ckpt) for d in a.stage1]
    s2 = [load(d, a.ckpt) for d in (a.stage2 or a.stage1)]
    n = len(s1)
    pairs = [(i, j) for i in range(n) for j in range(n) if i < j]
    E = [enc_rows(x[0]) for x in s1]
    B = [brain_rows(x[0]) for x in s1]
    D = [dec_rows(x[0]) for x in s2]
    if a.stage2:
        for i in range(n):  # stage 2 must sit on the stage-1 encoder
            d = np.abs(s1[i][0]["encoder.weight"].float().numpy() - s2[i][0]["encoder.weight"].float().numpy()).max()
            print(f"  seed {i}: max |encoder(stage1) - encoder(stage2)| = {d:.2e}")
    print(f"== {a.label}  ({n} seeds, {a.ckpt}; stage-1 best epochs {[x[1] for x in s1]}, epochs run {[x[2] for x in s1]})")
    eh = np.mean([hung(E[i], E[j]) for i, j in pairs]); ef = np.mean([frac(E[i], E[j]) for i, j in pairs])
    dh = np.mean([hung(D[i], D[j]) for i, j in pairs]); df = np.mean([frac(D[i], D[j]) for i, j in pairs])
    dn = np.mean([hung(D[i], rot(D[j])) for i, j in pairs])
    bh = np.mean([np.mean([hung(B[i][s], B[j][s]) for s in range(6)]) for i, j in pairs])
    bf = np.mean([np.mean([frac(B[i][s], B[j][s]) for s in range(6)]) for i, j in pairs])
    print(f"  semantic decoder : Hungarian cos {dh:.3f}   best-match > 0.7: {100*df:.0f}%   rotation null {dn:.3f}")
    print(f"  cortical map     : Hungarian cos {bh:.3f}   best-match > 0.7: {100*bf:.0f}%")
    print(f"  encoder row      : Hungarian cos {eh:.3f}   best-match > 0.7: {100*ef:.0f}%")
    per_pair = [(i, j, round(float(hung(D[i], D[j])), 3), round(float(np.mean([hung(B[i][s], B[j][s]) for s in range(6)])), 3)) for i, j in pairs]
    print(f"  per seed pair (i, j, semantic decoder, cortical map): {per_pair}")


if __name__ == "__main__":
    main()
