# Component catalog

`components.csv`: the 50 components of the coupled SAE, with every per-component value in the
paper's catalog table (seed 0, final epoch unless stated). Built by `scripts/40_build_catalog.py`.

| Column | Meaning |
|---|---|
| component | c0 … c49 |
| description | the judge's one-line description (Claude Fable 5.1); the paper prints a shortened form |
| autointerp_auc | nameability: how well the description separates the component's held-out activating text from random text (AUC over 10 + 10 contexts) |
| judge_label | mono (one concept) or poly (several unrelated triggers) |
| coherence_z | semantic coherence of the 20 strongest contexts against 200 random draws (z), as printed in the paper |
| cross_subject_r | cross-subject consistency: mean Pearson r of the component's cortical maps over the 15 subject pairs |
| cross_subject_q, cross_subject_sig_q05 | BH-corrected permutation p value (shuffled-component null, 1,000 permutations) and q < 0.05 |
| cortical_map_stability_20seeds | cross-seed stability of the cortical map: best-match cosine averaged over the other 19 seeds (final epoch) |
| semantic_decoder_stability_20seeds | the same for the semantic-decoder column |
| stability_flag | "unstable" where the cortical map or the semantic-decoder column falls below 0.5 (the ‡ components of the paper) |
| active_fraction | fraction of training time points at which the component is among the 5 active ones |
| yeo7_networks | Yeo 7 networks significantly aligned with the cortical map (spin test, BH q < 0.05), strongest first; empty if none |
| neurosynth_term, neurosynth_r, neurosynth_q | Neurosynth term closest to the description, correlation of its map with the cortical map, BH-corrected spin-test q |

Coherence z is given to the precision printed in the paper; `scripts/29_atom_coherence_embedding.py`
reproduces it at full precision.
