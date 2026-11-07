#!/usr/bin/env python
"""Table B: cross-seed component stability across the training-configuration sweep (atom count, sparsity
k, coupling alpha), on BOTH the semantic-decoder path and the brain-decoder path -- the second is our
addition beyond the SAE-stability literature (which measures decoder/activation only). Production recipe
(6 subjects, all-stories); 3 seeds per config. No retraining -- all cells already on disk.

Per config, per face: mean over the 3 seed-pairs of the mean best-match |signed cosine| (Hungarian), and
% of components with a match > 0.7. Decoder = decoder_sem columns; brain = delay_scales-weighted map,
averaged over the 6 subjects. Rotation null per face for calibration.
"""
import glob, re, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
torch.set_grad_enabled(False); rng=np.random.default_rng(0)
SUBJS=['UTS01','UTS02','UTS03','UTS04','UTS05','UTS06']

def find(natoms, k, alpha, ckpt="model_last.pt"):
    """Seeds 0-2 of one configuration, selected by saved settings (sparsebrain.runs)."""
    from sparsebrain.runs import find_runs, PAPER_COUPLED
    want = {**PAPER_COUPLED, "n_atoms": natoms, "k": k, "alpha": alpha}
    return [os.path.join(d, ckpt) for d in find_runs(require=ckpt, seeds=(0, 1, 2), **want)]

def dec_rows(sd):
    W=sd['decoder_sem.weight'].float().numpy().T; return W/(np.linalg.norm(W,axis=1,keepdims=True)+1e-9)
def brain_rows(sd):
    NA=sd['encoder.weight'].shape[0]; ds=sd['delay_scales'].float().numpy()
    maps=[]
    for s in SUBJS:
        k=[x for x in sd if x.startswith(f'brain_decoders.{s}')]
        if not k: return None
        W=sd[k[0]].float().numpy().reshape(-1,4,NA); M=(W*ds[None,:,:]).sum(1).T
        M=M-M.mean(0,keepdims=True); maps.append(M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9))
    return maps  # list of (NA,nvox) per subject

def hung(A,B): S=A@B.T; r,c=linear_sum_assignment(-S); return S[r,c].mean()
def frac(A,B,th=0.7): return (A@B.T).max(1).mean()>-9 and ((A@B.T).max(1)>th).mean()
def rot(A):
    U,S,_=np.linalg.svd(A,full_matrices=False); r=int((S>1e-8).sum()); G=U[:,:r]*S[:r]
    Vr,_=np.linalg.qr(rng.standard_normal((A.shape[1],r))); R=G@Vr.T
    return R/(np.linalg.norm(R,axis=1,keepdims=True)+1e-9)

def config_stability(paths):
    D=[]; B=[]
    for p in paths:
        sd=torch.load(p,map_location='cpu',weights_only=False)['state_dict']
        D.append(dec_rows(sd)); B.append(brain_rows(sd))
    pairs=[(i,j) for i in range(len(D)) for j in range(len(D)) if i<j]
    dh=np.mean([hung(D[i],D[j]) for i,j in pairs]); df=np.mean([frac(D[i],D[j]) for i,j in pairs])
    dn=np.mean([hung(D[i],rot(D[j])) for i,j in pairs])
    bh=np.mean([np.mean([hung(B[i][s],B[j][s]) for s in range(6)]) for i,j in pairs])
    bf=np.mean([np.mean([frac(B[i][s],B[j][s]) for s in range(6)]) for i,j in pairs])
    return dh,df,dn,bh,bf,len(D)

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="model_last.pt", help="model_last.pt (final epoch, as in the paper) or model.pt (best validation epoch)")
    ckpt = ap.parse_args().ckpt
    CONFIGS=[('atom count','25',25,5,1.0),('atom count','50 (prod)',50,5,1.0),('atom count','100',100,5,1.0),
             ('coupling a','0',50,5,0.0),('coupling a','0.3',50,5,0.3),('coupling a','1 (prod)',50,5,1.0),
             ('coupling a','3',50,5,3.0),('coupling a','10',50,5,10.0),
             ('sparsity k','1',50,1,1.0),('sparsity k','3',50,3,1.0),('sparsity k','5 (prod)',50,5,1.0),
             ('sparsity k','8',50,8,1.0),('sparsity k','10',50,10,1.0)]
    print(f'{"axis":11} {"cfg":9} | {"DECODER hung / %>.7 / null":>28} | {"BRAIN hung / %>.7":>20} | n')
    print('-'*80)
    for axis,lab,na,k,a in CONFIGS:
        ps=find(na,k,a,ckpt)
        if len(ps)<2: print(f'{axis:11} {lab:9} | MISSING ({len(ps)})'); continue
        dh,df,dn,bh,bf,n=config_stability(ps)
        print(f'{axis:11} {lab:9} | {dh:.3f} / {100*df:4.0f}% / {dn:.3f}      | {bh:.3f} / {100*bf:4.0f}%    | {n}')

if __name__=='__main__':
    main()
