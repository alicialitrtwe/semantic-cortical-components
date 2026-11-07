#!/usr/bin/env python
"""Stable subset of components across the 20 seeds of the coupled SAE, with a seed-level bootstrap.

For each reference seed and component, p-hat = fraction of the other seeds that contain a match
with signed cosine > theta. The stable subset is the fraction of (reference seed, component) pairs
with p-hat >= tau, pooled over all 20 reference seeds. 95% CI: resample the 20 seeds with
replacement (the seed is the unit of replication), recompute p-hat over the other distinct original
seeds, 1,000 resamples, 2.5th/97.5th percentiles. Reported at theta = 0.7 and tau in {0.5, 0.7} for
four aspects of a component (semantic decoder, activation, cortical map, encoder) and a rotation
null. Cortical map = delay-scale-weighted sum of the four delay columns, demeaned over components,
best match averaged over the six subjects. Also writes a theta-sweep figure.

Usage: uv run python scripts/48_stable_subset_bootstrap.py [--runs "<glob>"] [--ckpt model.pt]
"""
import glob, re, os, sys
import numpy as np, torch
PROJECT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,PROJECT)
from sparsebrain.data import load_features
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import argparse
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--runs", default=None, help="glob of the 20 seed runs; default: runs with the paper's coupled-SAE settings")
_ap.add_argument("--ckpt", default="model_last.pt", help="model_last.pt (final epoch, as in the paper) or model.pt (best validation epoch)")
_CLI = _ap.parse_args()
CKPT = _CLI.ckpt
torch.set_grad_enabled(False); rng=np.random.default_rng(0)
NA=50; K=5; NB=1000
SUBJS=['UTS01','UTS02','UTS03','UTS04','UTS05','UTS06']
PROBE=['wheretheressmoke','howtodraw','undertheinfluence','vixenandtheussr','whenmothersbullyback']
FIGDIR='reports/figures/stability'; os.makedirs(FIGDIR,exist_ok=True)

def load_dirs():
    from sparsebrain.runs import find_runs, seed_of, PAPER_COUPLED
    dirs = find_runs(_CLI.runs, require=CKPT, **({} if _CLI.runs else PAPER_COUPLED))
    return [os.path.join(d, CKPT) for d in dirs], [seed_of(d) for d in dirs]
def feats(s):
    f=load_features([s],embedding='gemma2',subject='UTS01',demean_per_story=True).astype(np.float32)
    return f/np.sqrt((f**2).mean())
Xprobe={s:feats(s) for s in PROBE}
def enc_codes(sd,Xf):
    z=torch.relu((torch.tensor(Xf)-sd['b_pre'].float())@sd['encoder.weight'].float().T+sd['encoder_bias'].float())
    _,idx=torch.topk(z,K,-1); return (z*torch.zeros_like(z).scatter_(-1,idx,1.0)).numpy()
def r_enc(sd): W=sd['encoder.weight'].float().numpy(); return W/(np.linalg.norm(W,axis=1,keepdims=True)+1e-9)
def r_dec(sd): W=sd['decoder_sem.weight'].float().numpy().T; return W/(np.linalg.norm(W,axis=1,keepdims=True)+1e-9)
def r_act(sd):
    Z=np.concatenate([enc_codes(sd,Xprobe[s]) for s in PROBE],0); Z=(Z-Z.mean(0))/(Z.std(0)+1e-8)
    R=Z.T; return R/(np.linalg.norm(R,axis=1,keepdims=True)+1e-9)
def r_brain(sd,subj):
    W=sd[f'brain_decoders.{subj}.weight'].float().numpy().reshape(-1,4,NA)   # delay_scales-WEIGHTED
    M=(W*sd['delay_scales'].float().numpy()[None,:,:]).sum(1).T
    M=M-M.mean(0,keepdims=True); return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)

def maxsim_tensor(rows_list):
    """rows_list[i]=(50,d) -> maxsim[i,a,j] = best signed cos of atom a in model i to any atom in j."""
    n=len(rows_list); M=np.zeros((n,NA,n))
    for i in range(n):
        for j in range(n):
            M[i,:,j]= (rows_list[i]@rows_list[j].T).max(1)
    return M
def maxsim_brain(brain_by_subj):
    n=len(brain_by_subj[SUBJS[0]]); M=np.zeros((n,NA,n))
    for i in range(n):
        for j in range(n):
            M[i,:,j]=np.mean([ (brain_by_subj[s][i]@brain_by_subj[s][j].T).max(1) for s in SUBJS],0)
    return M
def rot_null_maxsim(rows_list):
    n=len(rows_list); d=rows_list[0].shape[1]; Q,_=np.linalg.qr(rng.standard_normal((d,d)))
    rot=[ (r@Q)/(np.linalg.norm(r@Q,axis=1,keepdims=True)+1e-9) for r in rows_list]
    M=np.zeros((n,NA,n))
    for i in range(n):
        for j in range(n):
            M[i,:,j]=(rows_list[i]@rot[j].T).max(1)
    return M

def stable_frac(M, ids, theta, tau):
    """M[i,a,j] maxsim; ids = list of ORIGINAL model indices (bootstrap sample). p-hat over
    other DISTINCT original ids; return fraction of (ref,atom) with p-hat>=tau."""
    n=len(ids); out=[]
    for p in range(n):
        i=ids[p]; others=[ids[q] for q in range(n) if ids[q]!=i]
        if not others: continue
        phat=(M[i][:,others]>theta).mean(1)     # (50,)
        out.append(phat>=tau)
    return float(np.mean(np.concatenate(out)))

def boot_ci(M, theta, tau):
    n=M.shape[0]; base=stable_frac(M,list(range(n)),theta,tau)
    bs=[stable_frac(M,rng.choice(n,n,replace=True).tolist(),theta,tau) for _ in range(NB)]
    lo,hi=np.percentile(bs,[2.5,97.5]); return base,lo,hi

def main():
    ds,seeds=load_dirs(); print(f'Loaded {len(ds)} models: seeds {seeds}')
    E=[];D=[];A=[];B={s:[] for s in SUBJS}
    for p in ds:
        sd=torch.load(p,map_location='cpu',weights_only=False)['state_dict']
        E.append(r_enc(sd)); D.append(r_dec(sd)); A.append(r_act(sd))
        for s in SUBJS: B[s].append(r_brain(sd,s))
    MS={'sem_decoder':maxsim_tensor(D),'activation':maxsim_tensor(A),
        'brain_map':maxsim_brain(B),'encoder':maxsim_tensor(E)}
    NULL={'encoder':rot_null_maxsim(E),'sem_decoder':rot_null_maxsim(D)}

    print('\n===== STABLE SUBSET averaged over all 20 reference seeds — seed-level bootstrap 95% CI =====')
    print(f'{"face":12}| theta=0.7,tau=0.7           | theta=0.7,tau=0.5           | null(tau=.7)')
    for f in ['sem_decoder','activation','brain_map','encoder']:
        b7,l7,h7=boot_ci(MS[f],0.7,0.7); b5,l5,h5=boot_ci(MS[f],0.7,0.5)
        nb=''
        if f in NULL:
            n7,_,_=boot_ci(NULL[f],0.7,0.7); nb=f'{n7:.3f}'
        print(f'{f:12}| {b7:.3f} [{l7:.3f}, {h7:.3f}]        | {b5:.3f} [{l5:.3f}, {h5:.3f}]        | {nb}')

    # ---------- (B) theta-sweep figure ----------
    thetas=np.linspace(0,1,101)
    colors={'sem_decoder':'#1b7837','activation':'#762a83','brain_map':'#2166ac','encoder':'#b2182b'}
    labels={'sem_decoder':'semantic decoder','activation':'activation','brain_map':'brain map','encoder':'encoder'}
    fig,ax=plt.subplots(1,2,figsize=(13,4.6))
    # left: mean p-hat (=mean frac paired) vs theta, per face + null
    for f in ['sem_decoder','activation','brain_map','encoder']:
        M=MS[f]; n=M.shape[0]
        off=M[~np.eye(n,dtype=bool)[:,None,:].repeat(NA,1)].reshape(n,NA,n-1)  # drop self
        curve=[(off>th).mean() for th in thetas]
        ax[0].plot(thetas,curve,color=colors[f],lw=2,label=labels[f])
    for f in ['encoder','sem_decoder']:
        Nn=NULL[f]; n=Nn.shape[0]
        offn=Nn[~np.eye(n,dtype=bool)[:,None,:].repeat(NA,1)].reshape(n,NA,n-1)
        ax[0].plot(thetas,[(offn>th).mean() for th in thetas],color=colors[f],ls=':',lw=1.2,alpha=.7)
    ax[0].axvline(0.7,color='gray',ls='--',lw=1); ax[0].text(0.705,0.9,'θ=0.7',fontsize=9,color='gray')
    ax[0].set_xlabel('match threshold θ (cosine)'); ax[0].set_ylabel('mean reappearance p̂  (= mean frac paired)')
    ax[0].set_title('Reappearance vs θ (dotted = rotation null)'); ax[0].legend(fontsize=8); ax[0].set_xlim(0,1); ax[0].set_ylim(0,1)
    # right: histogram of matched max-cosine (best partner per atom, off-diagonal pooled)
    for f in ['sem_decoder','activation','brain_map','encoder']:
        M=MS[f]; n=M.shape[0]
        off=M[~np.eye(n,dtype=bool)[:,None,:].repeat(NA,1)].reshape(-1)
        ax[1].hist(off,bins=60,range=(-0.2,1),histtype='step',color=colors[f],lw=2,density=True,label=labels[f])
    ax[1].axvline(0.7,color='gray',ls='--',lw=1); ax[1].text(0.705,ax[1].get_ylim()[1]*0.9,'θ=0.7',fontsize=9,color='gray')
    ax[1].set_xlabel('matched max-cosine (best partner in another seed)'); ax[1].set_ylabel('density')
    ax[1].set_title('Distribution of best-match cosines (bimodality & where θ sits)'); ax[1].legend(fontsize=8)
    plt.tight_layout(); out=f'{FIGDIR}/theta_sweep_4faces.png'; plt.savefig(out,dpi=130,bbox_inches='tight')
    print(f'\nFigure saved -> {out}')

if __name__=='__main__':
    main()
