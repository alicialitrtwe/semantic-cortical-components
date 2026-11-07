#!/usr/bin/env python
"""Cross-seed atom stability (Hungarian cosine + fair feature-covariance null) for every
hyperparameter-sweep configuration with >=2 seeds. Fair null = null atoms are random
linear combinations of the config's own pooled atoms (preserves subspace/covariance)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparsebrain.runs import find_runs, PAPER_COUPLED
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import combinations

rng = np.random.default_rng(0)

def load(natoms, k, alpha):
    """Encoder rows of seeds 0-2 of one configuration, selected by saved settings."""
    want = {**PAPER_COUPLED, "n_atoms": natoms, "k": k, "alpha": alpha}
    d = {}
    for run in find_runs(require="encoder_dictionary.npy", seeds=(0, 1, 2), **want):
        from sparsebrain.runs import seed_of
        W = np.load(os.path.join(run, "encoder_dictionary.npy"))
        d[seed_of(run)] = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    return d, len(d)

def mc(A, B):
    S = A @ B.T; r, c = linear_sum_assignment(-S); return S[r, c].mean()

def within(d):
    return np.mean([mc(d[i], d[j]) for i, j in combinations(sorted(d), 2)]) if len(d) > 1 else float('nan')

def fair_null(d, n=15):
    pooled = np.vstack([d[s] for s in d]); ref = d[sorted(d)[0]]; out = []
    for _ in range(n):
        G = rng.standard_normal((50, pooled.shape[0])); X = G @ pooled
        out.append(mc(ref, X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)))
    return np.mean(out)

CONFIGS = [
    ('atom-count', '25 atoms',  (25, 5, 1.0)),
    ('atom-count', '50 atoms*', (50, 5, 1.0)),
    ('atom-count', '100 atoms', (100, 5, 1.0)),
    ('coupling a', 'a=0',       (50, 5, 0.0)),
    ('coupling a', 'a=0.3',     (50, 5, 0.3)),
    ('coupling a', 'a=3',       (50, 5, 3.0)),
    ('coupling a', 'a=10',      (50, 5, 10.0)),
    ('sparsity k', 'k=1',       (50, 1, 1.0)),
    ('sparsity k', 'k=3',       (50, 3, 1.0)),
    ('sparsity k', 'k=8',       (50, 8, 1.0)),
    ('sparsity k', 'k=10',      (50, 10, 1.0)),
]

print(f"{'axis':12} {'config':10} {'seeds':7} {'cosine':>7} {'fairnull':>9} {'signal':>7}")
for axis, lab, cfg in CONFIGS:
    d, npaths = load(*cfg)
    if len(d) < 2:
        print(f'{axis:12} {lab:10} {str(sorted(d)):7} SKIP (<2 seeds; {npaths} paths)'); continue
    flag = '' if npaths == len(d) else f'  !! {npaths} paths / {len(d)} seeds (glob collision?)'
    w, fn = within(d), fair_null(d)
    print(f'{axis:12} {lab:10} {str(sorted(d)):7} {w:7.3f} {fn:9.3f} {w-fn:+7.3f}{flag}')
print('* 50 components, alpha = 1: the coupled SAE (shared point of the component-count and alpha axes).')
