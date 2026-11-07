# Semantic-Cortical Components

Code for *Semantic-Cortical Components: A Multisubject Coupled Sparse Autoencoder for
Interpretable Language Encoding in fMRI* (Alicia Zeng, NeurIPS 2026).

A coupled sparse autoencoder learns 50 components from Gemma-2-2b activations of the story
stimuli. Each component has a semantic direction (a column of the semantic decoder) and, for each
of six subjects, a cortical map (a column of that subject's brain decoder). The same sparse code
reconstructs the activations and predicts every subject's fMRI responses.

`catalog/components.csv` lists the 50 components with every per-component value reported in the
paper (description, nameability AUC, cross-subject consistency, cross-seed stability, active
fraction, Yeo network).

## Install

```bash
uv sync                          # core: training and most analyses
uv sync --extra text             # + Gemma-2-2b activations and sentence-embedding coherence
uv sync --extra surface          # + pycortex cortical maps (Figure 2, surface analyses)
```

Python 3.12, PyTorch 2.2 or later. Training was run on one GPU; a run takes 8–15 minutes.

## Data

1. The natural-listening fMRI dataset of LeBel et al. (2023), OpenNeuro ds003020, through the
   authors' repository, which also provides `ridge_utils` used by several scripts:

   ```bash
   git clone https://github.com/HuthLab/deep-fMRI-dataset external/deep-fMRI-dataset
   ```

   then download the preprocessed responses and TextGrids as that repository describes.
2. FreeSurfer (`FREESURFER_HOME` set) with the `fsaverage` subject, for the surface analyses.
3. Neurosynth v7 term maps, downloaded by `scripts/30_neurosynth_validation.py` on first use.

Everything the scripts write goes to `outputs/` (models, features, caches) and `reports/`
(analysis tables and plots). Neither is tracked.

## Reproducing the paper

Scores are means and population standard deviations over seeds 0, 1 and 2. Results use the
final-epoch checkpoint `model_last.pt`, except ablation rows A, C and D, which the paper reports at
the best-validation checkpoint `model.pt`. The new-subject decoders are scored at their own best
validation epoch, the transfer protocol's model selection.

### Inputs

```bash
uv run python scripts/02b_save_features_llm.py                       # Gemma-2-2b layer-25 activations
for s in UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08; do
  uv run python scripts/23_fit_ridge_voxsel.py --subject $s --embedding gemma2 --n_stories 10   # voxel selection
done
uv run python scripts/25_build_fsaverage_mappers.py                  # voxel -> fsaverage mappers
```

### Training

Configs are in `configs/`; `configs/multisubj.yaml` is the coupled SAE. One run per seed:

```bash
uv run python scripts/train_multisubj_hydra.py experiment=coupled_sae seed=0
```

Each run writes `outputs/models/<run>/` with `model_last.pt`, `model.pt` and `args.json`, and
prints its final-epoch test scores (`Last-epoch: test_r=…, sem_r=…`, the brain and semantic
scores) and per-subject scores (`TEST-last [UTSxx]`). The analysis scripts find runs by the settings
in `args.json` (`sparsebrain/runs.py`), so run folders can be named freely; scripts that take
`--runs` or `--model_dirs` also accept explicit folders. A run takes 8–15 minutes on one GPU.

| Model or table | Command (per seed unless stated) |
|---|---|
| Coupled SAE | `experiment=coupled_sae` |
| Sem→Brain | `experiment=sem2brain_stage1`, then `experiment=sem2brain_stage2 freeze_from=<stage-1 run>/model_last.pt` |
| Brain→Sem | `experiment=brain2sem_stage1`, then `experiment=brain2sem_stage2 freeze_from=<stage-1 run>/model_last.pt` |
| Random encoder | `experiment=random_encoder`; semantic score: `experiment=brain2sem_stage2 freeze_from=<that run>/model_last.pt` |
| Ablation rows A, C, D, E | `experiment=ablation_A_no_encoder_norm`, `ablation_C_matryoshka`, `ablation_D_matching_pursuit`, `ablation_E_voxel_scale` (A, C, D without AuxK, as in the paper) |
| Sweep (α, K, k = 2) | `alpha=0.3` (or 3, 10); `n_atoms=25` or `100`; `k=2` |
| k sweep | `k=1`, `k=3`, `k=8`, `k=10`, seeds 0–2 each |
| 20-seed stability | `experiment=coupled_sae seed=$s` for s = 0..19 |
| New-subject reference | `experiment=transfer_reference freeze_from=<seed-0 coupled SAE run>/model_last.pt` |
| Robustness check | `experiment=robustness_fixed` |

### Tables

| Paper item | Command | Expected |
|---|---|---|
| Table 1, sparse models | training logs above | coupled SAE 0.162 ± 0.005 / 0.523 ± 0.006; Sem→Brain 0.131 ± 0.006 / 0.537 ± 0.004; Brain→Sem 0.174 ± 0.003 / 0.394 ± 0.013; random encoder 0.064 ± 0.005 / 0.407 ± 0.005 (brain / semantic) |
| Table 1 and per-subject table, matched ridge | `scripts/23_fit_ridge_voxsel.py --subject <s> --embedding gemma2 --val_stories forgettingfear lifereimagined onlyonewaytofindout thefreedomridersandme --refit_on_all --suffix matched` for UTS01–UTS06; score = mean `corr_test` over the coupled SAE's 30,000 voxels | 0.181, 0.234, 0.295, 0.117, 0.120, 0.161; mean 0.185 |
| Ablation table | training logs; rows A, C, D at the best epoch (`CKPT=model.pt` for script 24) | A 0.145 ± 0.012, B 0.162 ± 0.005, C 0.158 ± 0.004, D 0.137 ± 0.004, E 0.156 ± 0.010 (brain) |
| k sweep | training logs | brain 0.072, 0.116, 0.143, 0.162, 0.167, 0.174 for k = 1, 2, 3, 5, 8, 10 |
| Nameability (AutoInterp AUC) | `scripts/18_visualize_multisubject_atoms.py --model_dir <run>`, then `scripts/30_atom_autointerp.py --mode prepare`, `--mode emit_batch_prompt`, the judge (Claude Fable 5.1) on `batch_prompt.txt`, `--mode aggregate` | median AUC 0.995 ± 0.004 (coupled), 0.992 ± 0.003 (Sem→Brain), 0.922 ± 0.012 (Brain→Sem) |
| Coherence | `scripts/29_atom_coherence_embedding.py --model_dir <run>` (200 random draws) | median z 4.4 ± 0.3, 5.0 ± 0.2, 4.7 ± 0.3 |
| Cross-subject consistency | `scripts/31_cross_subject_consistency.py --model_dirs <seed-0..2 runs> --out_json <file>` | coupled 0.205 ± 0.008, 42 of 50 components q < 0.05 |
| Consistency of ridge projections | `scripts/23b_ridge_refit_trainonly.py --subject <s> --embedding gemma2 --sae_model_dir <seed-0 run>` for each subject (writes the ridge weights), then `scripts/46_ridge_direction_consistency.py <repository folder> <output folder>` | ridge 0.174 ± 0.002 (encoder directions), 0.177 ± 0.004 (decoder directions) |
| Yeo 7 networks | `scripts/27_atom_yeo7_mapping.py --model_dir <run>`; across seeds `scripts/28_yeo7_cross_seed.py --model_dirs <seed-0..2 runs>` | |
| Neurosynth | `scripts/30_neurosynth_validation.py --model_dir <run>`, then `scripts/31_neurosynth_spatial_perm.py --model_dir <run>` | 21 of 50 components q < 0.05 |
| 20-seed stability, semantic decoder and encoder rows | `scripts/42_crossseed_metrics.py` | MMCS 0.73 and 0.48; frac > 0.7 0.61 and 0.16 |
| 20-seed stability, cortical map and activation | `scripts/41_crossseed_aspects.py` (cortical map: the line "best match averaged over subjects") | MMCS 0.71 and 0.58; frac > 0.7 0.54 and 0.36 |
| 20-seed stable subset (p̂ ≥ 0.7 at θ = 0.7, bootstrap CI) | `scripts/48_stable_subset_bootstrap.py` | semantic decoder 0.49 [0.47, 0.54], cortical map 0.46 [0.44, 0.50], activation 0.27 [0.24, 0.31], encoder 0.08 [0.06, 0.13] |
| Stability by configuration | `scripts/43_config_stability.py` and `scripts/44_stability_table.py` | semantic decoder / cortical map: α 0.3, 1, 3, 10: 0.57 / 0.61, 0.64 / 0.65, 0.74 / 0.67, 0.78 / 0.69 |
| Stability by training objective | `scripts/47_stability_by_objective.py --stage1 <runs> --stage2 <runs>` | coupled 0.64 / 0.65 / 0.41 (semantic decoder / cortical map / encoder row) |
| Encoder rows and CKA | `scripts/24_analyze_stability.py --model_dirs <runs>` | |
| Active fraction | `scripts/45_active_fraction.py` | c30 53%, c21, c47, c48 41%, c49 34%; median 6% |
| Catalog (`catalog/components.csv`) | `scripts/40_build_catalog.py --autointerp_csv <file> --coherence_csv <file> --consistency_json <file> --active_fraction_json <file> --neurosynth_csv <file> --yeo7_dir <folder>` | |
| New subjects (UTS07, UTS08) | `scripts/34_learning_curve.py --base_model_dirs <seed-0..2 runs> --random_encoder_dirs <runs> --transfer_ref_dir <new-subject reference run> --n_story_seeds 7` (defaults: each subject's own four validation stories, inputs scaled by the RMS of all 83 training stories, 11.86, as in the paper) | at 22 stories 0.110 (UTS07) and 0.099 (UTS08), against ridge 0.126 and 0.097 |

### Figures

| Figure | Command |
|---|---|
| Figure 1 | `uv run python figures/fig1/figure1.py` |
| Figure 2 | render the brain maps of c8, c23, c2 and c45: `uv run python figures/fig2/component_panel.py --model_dir <seed-0 run> --component <j> --cache_dir outputs/cache/figure2` (needs `--extra surface`); then `uv run python figures/fig2/figure2.py --model_dir <seed-0 run> --consistency_json <file>` |
| Stand-alone panels | one component: `figures/fig2/component_panel.py`; all 50 components' stability: `figures/fig2/strip.py` |
| Figure 3 | `uv run python figures/fig3/figure3.py` (paired one-sided t over distinct training sets: p = 0.065 at half a story, 0.041 at one story) |

## Tests

```bash
uv run --group dev pytest
```

The tests run on synthetic data. They check that the configs compose to the paper's recipe, that a
small model trains, and that the per-subject z-scoring does what it says. If pytest stops while
scanning installed packages for plugins, set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

## Citation

See `CITATION.cff`.

## Licence

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0); see `LICENSE`.
The code and the catalog may be used, shared and adapted for non-commercial purposes, with
attribution. Commercial use requires written permission from the author.
