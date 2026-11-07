#!/usr/bin/env python
"""Build catalog/components.csv: cross-seed stability over the 20 seeds plus every per-component value of the paper."""
import glob, re, os, sys, csv, json
import numpy as np, torch
PROJECT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,PROJECT)
torch.set_grad_enabled(False)
NA=50; SUBJS=['UTS01','UTS02','UTS03','UTS04','UTS05','UTS06']

def r_dec(sd): W=sd['decoder_sem.weight'].float().numpy().T; return W/(np.linalg.norm(W,axis=1,keepdims=True)+1e-9)
def r_brain(sd,subj):
    # cortical map = delay_scales-WEIGHTED sum of the 4 delay columns (matches the model forward:
    # y_hat = decoder_brain(cat_d s_d * delay_scales[d])). Column j = delay*50 + component.
    W=sd[f'brain_decoders.{subj}.weight'].float().numpy().reshape(-1,4,NA)   # (nvox,4,50)
    ds=sd['delay_scales'].float().numpy()                                    # (4,50)
    M=(W*ds[None,:,:]).sum(1).T                                             # (50,nvox) weighted sum
    M=M-M.mean(0,keepdims=True); return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)

def main():
    import argparse
    from sparsebrain.runs import find_runs, seed_of, PAPER_COUPLED
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=None, help="glob of the 20 seed runs; default: runs with the paper's coupled-SAE settings")
    ap.add_argument("--ckpt", default="model_last.pt", help="the paper's catalog uses final-epoch checkpoints")
    ap.add_argument("--autointerp_csv", required=True, help="autointerp.csv of seed 0 (scripts/30_atom_autointerp.py --mode aggregate)")
    ap.add_argument("--consistency_json", required=True, help="output of scripts/31_cross_subject_consistency.py")
    ap.add_argument("--active_fraction_json", required=True, help="output of scripts/45_active_fraction.py")
    ap.add_argument("--neurosynth_csv", required=True, help="atom summary of scripts/31_neurosynth_spatial_perm.py (seed 0)")
    ap.add_argument("--coherence_csv", required=True, help="coherence.csv of scripts/29_atom_coherence_embedding.py (seed 0)")
    ap.add_argument("--yeo7_dir", required=True, help="output folder of scripts/27_atom_yeo7_mapping.py (seed 0), with atom_yeo7_cos.csv and atom_yeo7_pvalues.csv")
    ap.add_argument("--out", default=os.path.join(PROJECT, "catalog", "components.csv"))
    a = ap.parse_args()

    dirs = find_runs(a.runs, require=a.ckpt, **({} if a.runs else PAPER_COUPLED))
    seeds = [seed_of(d) for d in dirs]
    if 0 not in seeds:
        raise SystemExit("no seed-0 run found")
    ref = os.path.join(dirs[seeds.index(0)], a.ckpt)
    others = [os.path.join(d, a.ckpt) for d, s in zip(dirs, seeds) if s != 0]
    print(f"reference seed 0; comparison seeds {sorted(s for s in seeds if s != 0)}")
    sd0 = torch.load(ref, map_location='cpu', weights_only=False)['state_dict']
    D0 = r_dec(sd0); B0 = {s: r_brain(sd0, s) for s in SUBJS}
    dec_best = np.zeros((NA, len(others))); brn_best = np.zeros((NA, len(others)))
    for j, p in enumerate(others):
        sdj = torch.load(p, map_location='cpu', weights_only=False)['state_dict']
        dec_best[:, j] = (D0 @ r_dec(sdj).T).max(1)
        brn_best[:, j] = np.mean([(B0[s] @ r_brain(sdj, s).T).max(1) for s in SUBJS], 0)
    dec_cos = dec_best.mean(1); brn_cos = brn_best.mean(1)

    ai = {int(r['atom']): r for r in csv.DictReader(open(a.autointerp_csv))}
    pc = {d['component']: d for d in json.load(open(a.consistency_json))['first_model_null']['per_component']}
    af = json.load(open(a.active_fraction_json))['seeds']['0']['fractions']
    ns = {int(r['atom']): r for r in csv.DictReader(open(a.neurosynth_csv))}
    coh = {int(r['atom']): float(r['coherence_z']) for r in csv.DictReader(open(a.coherence_csv))}
    ycos = {int(r['atom']): r for r in csv.DictReader(open(os.path.join(a.yeo7_dir, 'atom_yeo7_cos.csv')))}
    ypv = {int(r['atom']): r for r in csv.DictReader(open(os.path.join(a.yeo7_dir, 'atom_yeo7_pvalues.csv')))}
    NETS = ["Visual", "Somatomotor", "DorsalAttn", "VentralAttn", "Limbic", "Frontoparietal", "Default"]
    ABBR = {"Visual": "VIS", "Somatomotor": "SMN", "DorsalAttn": "DAN", "VentralAttn": "VAN",
            "Limbic": "LIM", "Frontoparietal": "FPN", "Default": "DMN"}

    def networks(j):
        """Significant Yeo 7 networks (BH q < 0.05), strongest first, as in the paper's catalog."""
        sig = [(float(ycos[j][f"{n}_cos"]), n) for n in NETS if int(float(ypv[j][f"{n}_sig"])) == 1]
        return ";".join(ABBR[n] for _, n in sorted(sig, reverse=True))
    rows = []
    for j in range(NA):
        n = ns.get(j, {})
        unstable = (brn_cos[j] < 0.5) or (dec_cos[j] < 0.5)
        rows.append(dict(
            component=f"c{j}", description=re.sub(r'\b([Aa])tom(s?)\b', lambda m: ('C' if m.group(1) == 'A' else 'c') + 'omponent' + m.group(2), ai[j]['description']), autointerp_auc=ai[j]['auc'],
            judge_label=ai[j].get('mono', ''), coherence_z=f"{coh[j]:.1f}",
            cross_subject_r=f"{pc[j]['r']:.4f}", cross_subject_q=f"{pc[j]['q']:.4g}", cross_subject_sig_q05=pc[j]['sig_q05'],
            cortical_map_stability_20seeds=f"{brn_cos[j]:.4f}", semantic_decoder_stability_20seeds=f"{dec_cos[j]:.4f}",
            stability_flag="unstable" if unstable else "",
            active_fraction=f"{af[j]:.4f}", yeo7_networks=networks(j),
            neurosynth_term=n.get('text_best_term', ''), neurosynth_r=f"{float(n.get('obs_r', 0) or 0):.4f}",
            neurosynth_q=f"{float(n.get('q_text_best', 1) or 1):.4g}"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}: {sum(r['cross_subject_sig_q05'] for r in rows)} of {NA} consistent across subjects (q < 0.05); "
          f"{int((brn_cos < 0.5).sum())} below 0.5 cortical-map stability")


if __name__=='__main__':
    main()
