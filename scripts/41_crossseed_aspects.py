#!/usr/bin/env python
"""Cross-seed stability of two aspects of each component, activation and cortical map, over the
20 seeds of the coupled SAE (encoder rows and semantic decoder: scripts 42 and 48).
Same 20 same-data seeds, same metrics: MMCS (greedy mean-max), Hungarian bijective match,
frac paired>0.7, reappearance p-hat, and a chance NULL. SIGNED similarity throughout
(ReLU+TopK => atom sign meaningful; matches are positive).

  ACTIVATION face: z-scored TopK codes on 5 probe stories -> (50,T) per-atom time-course;
    similarity = Pearson r across TRs. Null = TR-shuffle of the partner.
  CORTICAL MAP: delay-scale-weighted sum of each component's 4 delay columns -> (50,nvox) voxel map,
    demeaned across atoms (match masking convention); similarity = spatial cosine.
    Computed per subject then averaged over all 6. Null = voxel-permutation of the partner.
"""
import os
import glob, re, os, sys
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
PROJECT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,PROJECT)
from sparsebrain.data import load_features
torch.set_grad_enabled(False); rng=np.random.default_rng(0)
NA=50; THETA=0.7; K=5
SUBJS=['UTS01','UTS02','UTS03','UTS04','UTS05','UTS06']
PROBE=['wheretheressmoke','howtodraw','undertheinfluence','vixenandtheussr','whenmothersbullyback']


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

def feats(s):
    f=load_features([s],embedding='gemma2',subject='UTS01',demean_per_story=True).astype(np.float32)
    return f/np.sqrt((f**2).mean())
Xprobe={s:feats(s) for s in PROBE}

def enc_codes(sd,Xf):
    z=torch.relu((torch.tensor(Xf)-sd['b_pre'].float())@sd['encoder.weight'].float().T+sd['encoder_bias'].float())
    _,idx=torch.topk(z,K,-1); return (z*torch.zeros_like(z).scatter_(-1,idx,1.0)).numpy()   # (T,50)

def act_rows(sd):
    Z=np.concatenate([enc_codes(sd,Xprobe[s]) for s in PROBE],0)      # (T,50)
    Z=(Z-Z.mean(0))/(Z.std(0)+1e-8); R=Z.T                            # (50,T)
    return R/(np.linalg.norm(R,axis=1,keepdims=True)+1e-9)
def brain_rows(sd,subj):
    # Column j of the brain decoder is delay*50 + component; the cortical map is the delay-scale-weighted
    # sum of a component's four delay columns (as in the model's forward pass).
    W=sd[f'brain_decoders.{subj}.weight'].float().numpy().reshape(-1,4,NA)          # (nvox,4,50)
    M=(W*sd['delay_scales'].float().numpy()[None,:,:]).sum(1).T                      # (50,nvox)
    M=M-M.mean(0,keepdims=True); return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)

def simmat(A,B): return A@B.T                                          # signed cos
def mmcs(A,B): return simmat(A,B).max(1).mean()
def hung(A,B): S=simmat(A,B); r,c=linear_sum_assignment(-S); return S[r,c].mean()
def frac(A,B,th): return (simmat(A,B).max(1)>th).mean()

def summarize(name, feats_by_model, null_fn):
    n=len(feats_by_model); pairs=[(i,j) for i in range(n) for j in range(n) if i<j]
    m=[mmcs(feats_by_model[i],feats_by_model[j]) for i,j in pairs]
    h=[hung(feats_by_model[i],feats_by_model[j]) for i,j in pairs]
    f=[frac(feats_by_model[i],feats_by_model[j],THETA) for i,j in pairs]
    nm=[mmcs(feats_by_model[i],null_fn(feats_by_model[j])) for i,j in pairs]
    nf=[frac(feats_by_model[i],null_fn(feats_by_model[j]),THETA) for i,j in pairs]
    # p-hat per (ref,atom): fraction of other seeds with a >theta match
    ph=np.zeros((n,NA))
    for i in range(n):
        hit=np.zeros(NA)
        for j in range(n):
            if i!=j: hit+=(simmat(feats_by_model[i],feats_by_model[j]).max(1)>THETA)
        ph[i]=hit/(n-1)
    def st(x): return f'{np.mean(x):.3f} ± {np.std(x):.3f}'
    print(f'  {name:11}| MMCS {st(m):>16} (null {np.mean(nm):.3f}) | Hung {st(h):>16} '
          f'| frac>0.7 {st(f):>14} (null {np.mean(nf):.3f})')
    phf=ph.ravel()
    print(f'  {"":11}| p-hat@0.7 mean={phf.mean():.3f} med={np.median(phf):.3f} '
          f'STABLE(>=.9)={np.mean(phf>=.9):.2f} UNSTABLE(<=.1)={np.mean(phf<=.1):.2f} '
          f'mid[.3,.7)={np.mean((phf>=.3)&(phf<.7)):.2f}')
    return dict(mmcs=m,hung=h,frac=f,phat=ph,null_mmcs=nm)

def tr_shuffle(R):  # activation null: permute TR axis of partner (R is (50,T))
    return R[:, rng.permutation(R.shape[1])]
def vox_perm(M):    # brain null: permute voxel axis of partner (M is (50,nvox))
    return M[:, rng.permutation(M.shape[1])]

def main():
    cli=_cli(__doc__); ds,seeds=load_dirs(cli.runs, cli.ckpt); print(f'Loaded {len(ds)} cross-seed models: seeds {seeds}')
    # load each checkpoint once; keep activation rows + per-subject brain rows
    ACT=[]; BR={s:[] for s in SUBJS}
    for p in ds:
        sd=torch.load(p,map_location='cpu',weights_only=False)['state_dict']
        sd={k:(v.float() if torch.is_tensor(v) else v) for k,v in sd.items()}
        ACT.append(act_rows(sd))
        for s in SUBJS: BR[s].append(brain_rows(sd,s))
    print('\n============ CROSS-SEED faces (20 seeds, same data) — SIGNED cos ============')
    out={}
    out['activation']=summarize('ACTIVATION', ACT, tr_shuffle)
    # brain: per-subject metrics, then averaged across subjects
    print('  --- brain-decoder (per subject, then averaged) ---')
    persub={}
    for s in SUBJS:
        persub[s]=summarize(f'brain {s}', BR[s], vox_perm)
    # averaged-over-subjects headline
    def avg(metric): return np.mean([np.mean(persub[s][metric]) for s in SUBJS])
    print(f'  BRAIN avg-over-6-subj | MMCS {avg("mmcs"):.3f} | Hung {avg("hung"):.3f} | frac>0.7 {avg("frac"):.3f} '
          f'(null {avg("null_mmcs"):.3f})')
    # As in the paper: each component's best-match cosine averaged over the six subjects, then the
    # mean (MMCS) and the fraction above 0.7 over components and seed pairs.
    n=len(BR[SUBJS[0]])
    best=np.array([np.mean([(BR[s][i]@BR[s][j].T).max(1) for s in SUBJS],0)
                   for i in range(n) for j in range(n) if i!=j])
    print(f'  CORTICAL MAP (best match averaged over subjects) | MMCS {best.mean():.3f} | frac>0.7 {(best>0.7).mean():.3f}')
    phall=np.concatenate([persub[s]['phat'].ravel() for s in SUBJS])
    print(f'  BRAIN pooled p-hat@0.7 mean={phall.mean():.3f} STABLE(>=.9)={np.mean(phall>=.9):.2f} '
          f'UNSTABLE(<=.1)={np.mean(phall<=.1):.2f} mid={np.mean((phall>=.3)&(phall<.7)):.2f}')
    os.makedirs('reports/stability', exist_ok=True)
    np.savez('reports/stability/crossseed_aspects.npz',
             act_mmcs=out['activation']['mmcs'], act_frac=out['activation']['frac'],
             act_phat=out['activation']['phat'],
             cortical_mmcs=best.mean(), cortical_frac=(best>0.7).mean(), brain_phat=phall, seeds=seeds)
    print('\nSaved -> reports/stability/crossseed_aspects.npz')

if __name__=='__main__':
    main()
