#!/usr/bin/env python
"""Cross-seed stability of components across the 20 seeds of the coupled SAE (SAE-stability literature metrics).

20 production-recipe models, SAME data (6 subj, all_stories, gemma2 L25, k5, 50 atoms, a1),
only `seed` varies -> pure init/optimization noise, the clean cross-seed setting the SAE papers
use (Paulo & Belrose 2501.16615; Unstable-Features-Reproducible-Subspaces 2606.12138).

Metrics, all pairwise over the C(20,2)=190 seed pairs:
  - MMCS  (mean max cosine sim, greedy) on encoder rows AND semantic-decoder columns.
  - Hungarian-matched mean cosine (bijective, no double-counting) for both spaces.
  - fraction of atoms PAIRED > 0.7 (field-standard reproducibility threshold).
  - SHARED-feature criterion: atom counts as reproduced only if encoder AND decoder BOTH
    match the SAME partner atom > theta (Paulo/Belrose "shared" = agreement in both spaces).
  - p-hat reappearance distribution across the 20 seeds + bimodality (stable vs unstable split).
  - rotation NULL floor (random orthogonal map) for calibration.
All cosines are signed (ReLU + TopK makes the sign of a component meaningful). Decoder columns normalized.
"""
import glob, re, os, sys
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
PROJECT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,PROJECT)
torch.set_grad_enabled(False); rng=np.random.default_rng(0)
NA=50; THETA=0.7


def load_dirs(runs=None, ckpt="model_last.pt"):
    from sparsebrain.runs import find_runs, seed_of, PAPER_COUPLED
    dirs = find_runs(runs, require=ckpt, **({} if runs else PAPER_COUPLED))
    return [os.path.join(d, ckpt) for d in dirs], [seed_of(d) for d in dirs]

def _cli(description):
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--runs", default=None,
                    help="glob of run folders; default: every run under outputs/models with the paper's coupled-SAE settings")
    ap.add_argument("--ckpt", default="model_last.pt",
                    help="checkpoint per run; the paper uses the final-epoch model_last.pt")
    return ap.parse_args()

def feats(path):
    sd=torch.load(path,map_location='cpu',weights_only=False)['state_dict']
    E=sd['encoder.weight'].float().numpy()                    # (50,2304) rows=atoms
    D=sd['decoder_sem.weight'].float().numpy().T              # (50,2304) cols->rows=atoms
    E=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
    D=D/(np.linalg.norm(D,axis=1,keepdims=True)+1e-9)
    return E,D

def simmat(A,B): return A@B.T                                  # (50,50) SIGNED cos:
    # ReLU+TopK => codes >=0 => atom sign is meaningful; genuine matches are POSITIVE.
    # abs() would count anti-aligned atoms as matches and inflates the mid-range means.

def mmcs(A,B):    return simmat(A,B).max(1).mean()             # greedy mean-max
def hungarian(A,B):
    S=simmat(A,B); r,c=linear_sum_assignment(-S); return S[r,c].mean()
def frac_paired(A,B,th): return (simmat(A,B).max(1)>th).mean()

def shared(EA,DA,EB,DB,th):
    """atom a in A is 'shared' if its BEST encoder partner in B also matches in decoder>th."""
    Se=simmat(EA,EB); Sd=simmat(DA,DB)
    part=Se.argmax(1)
    enc_ok=Se.max(1)>th
    dec_ok=Sd[np.arange(NA),part]>th
    return (enc_ok & dec_ok).mean()

def rot_null(A):
    Q,_=np.linalg.qr(rng.standard_normal((A.shape[1],A.shape[1])))
    Ar=A@Q; return Ar/(np.linalg.norm(Ar,axis=1,keepdims=True)+1e-9)

def main():
    cli=_cli(__doc__); ds,seeds=load_dirs(cli.runs, cli.ckpt)
    print(f'Loaded {len(ds)} cross-seed models: seeds {seeds}')
    E=[]; D=[]
    for p in ds:
        e,d=feats(p); E.append(e); D.append(d)
    n=len(E); pairs=[(i,j) for i in range(n) for j in range(n) if i<j]

    mE=[mmcs(E[i],E[j]) for i,j in pairs]; mD=[mmcs(D[i],D[j]) for i,j in pairs]
    hE=[hungarian(E[i],E[j]) for i,j in pairs]; hD=[hungarian(D[i],D[j]) for i,j in pairs]
    fE=[frac_paired(E[i],E[j],THETA) for i,j in pairs]; fD=[frac_paired(D[i],D[j],THETA) for i,j in pairs]
    sh=[shared(E[i],D[i],E[j],D[j],THETA) for i,j in pairs]

    # NULL: rotate one member of each pair (rigid rotation floor) — for BOTH spaces,
    # since the decoder MMCS needs its own chance floor (decoder cols may span a lower-dim subspace).
    nE=[mmcs(E[i],rot_null(E[j])) for i,j in pairs]
    nhE=[hungarian(E[i],rot_null(E[j])) for i,j in pairs]
    nfE=[frac_paired(E[i],rot_null(E[j]),THETA) for i,j in pairs]
    nD=[mmcs(D[i],rot_null(D[j])) for i,j in pairs]
    nhD=[hungarian(D[i],rot_null(D[j])) for i,j in pairs]
    nfD=[frac_paired(D[i],rot_null(D[j]),THETA) for i,j in pairs]

    def st(x): return f'{np.mean(x):.3f} ± {np.std(x):.3f}'
    print('\n============ CROSS-SEED (20 seeds, same data) — 190 pairs, SIGNED cos ============')
    print(f'{"metric":34}| {"ENCODER":>16} {"null":>13} | {"SEM-DECODER":>16} {"null":>13}')
    print('-'*100)
    print(f'{"MMCS (greedy mean-max cos)":34}| {st(mE):>16} {st(nE):>13} | {st(mD):>16} {st(nD):>13}')
    print(f'{"Hungarian mean cos (bijective)":34}| {st(hE):>16} {st(nhE):>13} | {st(hD):>16} {st(nhD):>13}')
    print(f'{"frac atoms paired > 0.7":34}| {st(fE):>16} {st(nfE):>13} | {st(fD):>16} {st(nfD):>13}')
    print(f'{"SHARED (enc & dec both > 0.7)":34}| {st(sh):>16} {"":>13} |')

    # ---- p-hat reappearance across the 20 seeds (encoder space) ----
    # per (ref model, atom): fraction of OTHER seeds with an encoder match > theta
    ph=np.full((n,NA),np.nan)
    for i in range(n):
        hits=np.zeros(NA)
        for j in range(n):
            if i==j: continue
            hits+=(simmat(E[i],E[j]).max(1)>THETA)
        ph[i]=hits/(n-1)
    phf=ph.ravel()
    print('\n----- reappearance p-hat @theta=0.7 (encoder), across 20 seeds -----')
    print(f'  per-atom p-hat: mean={phf.mean():.3f}  median={np.median(phf):.3f}')
    hist=np.histogram(phf,bins=[0,.1,.3,.5,.7,.9,1.01])[0]
    print(f'  p-hat hist [0,.1,.3,.5,.7,.9,1]: {hist.tolist()} (n={len(phf)})')
    print(f'  STABLE (p-hat>=0.9)={np.mean(phf>=.9):.2f}   UNSTABLE (p-hat<=0.1)={np.mean(phf<=.1):.2f}')
    print(f'  bimodality check: frac in middle band [.3,.7)={np.mean((phf>=.3)&(phf<.7)):.2f} '
          f'(low => bimodal/split; high => graded continuum)')
    # NOTE: p-hat is a property of (ref model, atom); atom INDEX is not aligned across seeds,
    # so we summarize the pooled distribution (above), not per-slot means.

    import os; os.makedirs('reports/stability', exist_ok=True)
    np.savez('reports/stability/crossseed_metrics.npz',
             mmcs_enc=mE,mmcs_dec=mD,hung_enc=hE,hung_dec=hD,frac_enc=fE,frac_dec=fD,
             shared=sh,phat=ph,null_mmcs_enc=nE,null_mmcs_dec=nD,seeds=seeds)
    print('\nSaved -> reports/stability/crossseed_metrics.npz')

if __name__=='__main__':
    main()
