"""Smoke test on synthetic data: configs, a few training steps, and the per-subject z-scoring."""
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sparsebrain.data import zscore_own_rows
from sparsebrain.models import MultiSubjectCoupledSAE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAPER_RECIPE = {
    "subjects": "UTS01,UTS02,UTS03,UTS04,UTS05,UTS06", "all_stories": True, "n_atoms": 50, "k": 5,
    "alpha": 1.0, "embedding": "gemma2", "voxel_selection": "ridge_gemma2_cv", "n_voxels": 30000,
    "delays": "1,2,3,4", "input_rms_normalize": True, "loss_normalization": "none",
    "normalize_every": "batch", "patience": 10, "epochs": 600, "val_mode": "story_level",
    "val_n_stories": 4, "val_seed": 0, "trim_start": 0, "trim_end": 0, "auxk": True,
    "grad_projection": True, "brain_decoder_unit_norm": True, "brain_voxel_scale": False,
    "no_encoder_norm": False, "brain_subj_mode": "round_robin", "tie_semantic_decoder": False,
    "encoder_bias": True, "val_masked": False, "zscore_own_rows": False, "train_mode": "coupled",
    "weight_decay": 0.0, "lr": 0.001, "batch_size": 1024, "grad_clip": 1.0,
}


def _compose(overrides=()):
    hydra = pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=os.path.join(ROOT, "configs"), version_base=None):
        return compose(config_name="multisubj", overrides=list(overrides))


def test_defaults_are_the_paper_recipe():
    cfg = _compose()
    for key, value in PAPER_RECIPE.items():
        assert cfg[key] == value, (key, cfg[key], value)


@pytest.mark.parametrize("name", sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "configs", "experiment"))))
def test_experiment_configs_compose(name):
    cfg = _compose([f"experiment={name}"])
    assert cfg.n_atoms == 50 and cfg.k == 5
    if name == "robustness_fixed":
        assert cfg.val_masked and cfg.zscore_own_rows


def test_zscore_own_rows_restores_unit_variance():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(300, 7)).astype(np.float32)
    padded = np.concatenate([real, np.zeros((600, 7), np.float32)])
    mask = np.r_[np.ones(300, bool), np.zeros(600, bool)]
    from scipy.stats import zscore
    z_padded = zscore(padded, axis=0)
    assert z_padded[mask].var(axis=0).mean() > 2.5          # the padding inflates the real rows
    fixed = zscore_own_rows(z_padded, mask)
    np.testing.assert_allclose(fixed[mask].var(axis=0), 1.0, atol=1e-4)
    np.testing.assert_allclose(fixed[mask], zscore(real, axis=0), atol=1e-4)
    assert np.all(fixed[~mask] == 0)


def test_few_training_steps_on_synthetic_data():
    torch.manual_seed(0)
    n_sem, n_atoms, k, n_delays = 32, 8, 2, 4
    vox = {"S1": 20, "S2": 15}
    model = MultiSubjectCoupledSAE(n_sem=n_sem, n_voxels_per_subject=vox, n_atoms=n_atoms, k=k,
                                   n_delays=n_delays, encoder_bias=True)
    T = 256
    x = torch.randn(T, n_sem)
    x_del = torch.randn(T, n_sem * n_delays)
    W_true = {s: torch.randn(n_sem * n_delays, v) * 0.1 for s, v in vox.items()}
    y = {s: x_del @ W for s, W in W_true.items()}
    mask = {"S1": torch.ones(T, dtype=torch.bool), "S2": torch.arange(T) < T // 3}  # S2 heard fewer stories
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def loss_fn():
        total = 0.0
        for s in vox:
            out = model(x, x_del, subject=s)
            m = mask[s]
            total = total + F.mse_loss(out["y_hat"][m], y[s][m]) + F.mse_loss(out["x_hat"], x)
        return total

    first = float(loss_fn())
    for step in range(60):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        model.normalize_weights(non_neg_brain=True)
    last = float(loss_fn())
    assert last < first
    z = model.encode(x)
    assert torch.all((z > 0).sum(dim=1) <= k)
    for s in vox:
        W = model.brain_decoders[s].weight
        assert torch.all(W >= 0)


def test_find_runs_selects_by_saved_settings(tmp_path):
    import json
    from sparsebrain.runs import find_runs, PAPER_COUPLED

    def make(name, seed, **over):
        d = tmp_path / name
        d.mkdir()
        args = {**PAPER_COUPLED, "seed": seed}
        args.pop("zscore_own_rows")            # older runs lack the new settings
        args.pop("val_masked")
        args.update(over)
        (d / "args.json").write_text(json.dumps(args))
        (d / "model_last.pt").write_text("")
        return str(d)

    a = make("anything_s1", 1)
    b = make("whatever_s0", 0)
    make("other_seed0_alpha3", 0, alpha=3.0)
    make("robust", 2, zscore_own_rows=True)    # explicitly different: must be excluded
    got = find_runs(str(tmp_path / "*"), **PAPER_COUPLED)
    assert got == [b, a]
    assert find_runs(str(tmp_path / "*"), seeds=(1,), **PAPER_COUPLED) == [a]


def test_encode_topk_matches_the_model():
    from sparsebrain.training import encode_topk
    torch.manual_seed(1)
    model = MultiSubjectCoupledSAE(n_sem=16, n_voxels_per_subject={"S1": 5}, n_atoms=10, k=3,
                                   n_delays=4, encoder_bias=True)
    with torch.no_grad():
        model.b_pre.normal_()
        model.encoder_bias.normal_()
        x = torch.randn(40, 16)
        z_model = model.encode(x).numpy()
    z = encode_topk(x.numpy(), model.encoder.weight.detach().numpy(), k=3,
                    b_pre=model.b_pre.detach().numpy(), encoder_bias=model.encoder_bias.detach().numpy())
    np.testing.assert_allclose(z, z_model, atol=1e-5)
