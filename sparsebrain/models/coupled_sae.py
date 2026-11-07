"""Coupled Sparse Autoencoder models for joint semantic-brain learning.

This module provides:
- CoupledSAEBase: Abstract base class with shared encode/decode logic
- CoupledSAE: Single-subject model with TV regularization
- MultiSubjectCoupledSAE: Multi-subject model with subject-specific decoders
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp_sparse
import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_scipy_to_torch(E: sp_sparse.csr_matrix, device: str = "cpu") -> torch.Tensor:
    """Convert scipy sparse matrix to PyTorch sparse tensor.

    Parameters
    ----------
    E : scipy sparse matrix
        Sparse matrix in CSR format.
    device : str
        Target device for the tensor.

    Returns
    -------
    E_torch : torch sparse tensor
        Coalesced sparse COO tensor.
    """
    E_coo = E.tocoo()
    indices = torch.LongTensor(np.vstack([E_coo.row, E_coo.col]))
    values = torch.FloatTensor(E_coo.data)
    shape = torch.Size(E_coo.shape)
    E_torch = torch.sparse_coo_tensor(indices, values, shape).to(device)
    return E_torch.coalesce()


class CoupledSAEBase(nn.Module, ABC):
    """Base class for coupled sparse autoencoders.

    Provides shared functionality for encoding semantic features to sparse
    codes and decoding to brain responses.

    Subclasses must implement:
    - forward()
    - normalize_weights()
    - get_brain_dictionary()
    """

    def __init__(
        self,
        n_sem: int,
        n_atoms: int,
        k: int,
        n_delays: int,
        tie_semantic_decoder: bool = False,
        sparsity_mode: str = "topk",
        encoder_bias: bool = False,
    ):
        """Initialize base coupled SAE.

        Parameters
        ----------
        n_sem : int
            Dimension of semantic features (e.g., 300 for GloVe).
        n_atoms : int
            Number of dictionary atoms.
        k : int
            TopK sparsity parameter (max active atoms per sample).
        n_delays : int
            Number of HRF temporal delays.

        Raises
        ------
        ValueError
            If parameters are invalid (e.g., k > n_atoms, negative values).
        """
        super().__init__()

        # Validate parameters
        if n_sem <= 0:
            raise ValueError(f"n_sem must be positive, got {n_sem}")
        if n_atoms <= 0:
            raise ValueError(f"n_atoms must be positive, got {n_atoms}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if k > n_atoms:
            raise ValueError(f"k ({k}) cannot exceed n_atoms ({n_atoms})")
        if n_delays <= 0:
            raise ValueError(f"n_delays must be positive, got {n_delays}")

        self.n_sem = n_sem
        self.n_atoms = n_atoms
        self.k = k
        self.n_delays = n_delays
        self.tie_semantic_decoder = tie_semantic_decoder
        self.sparsity_mode = sparsity_mode
        if sparsity_mode not in ("topk", "l1"):
            raise ValueError(f"sparsity_mode must be 'topk' or 'l1', got '{sparsity_mode}'")

        # Shared encoder: semantic features -> atoms
        self.encoder = nn.Linear(n_sem, n_atoms, bias=False)

        # Per-atom activation threshold (Bricken 2023 / Gao 2024).
        # Allows each atom to learn its own firing threshold; helps prevent
        # dead latents by letting the optimizer "rescue" marginal atoms.
        if encoder_bias:
            self.encoder_bias = nn.Parameter(torch.zeros(n_atoms))
        else:
            self.encoder_bias = None

        # Semantic decoder: untied by default (separate parameter).
        # When tie_semantic_decoder=True, the forward pass uses encoder.weight.T
        # directly (old-arch-style tied weights); decoder_sem is skipped.
        if not tie_semantic_decoder:
            self.decoder_sem = nn.Linear(n_atoms, n_sem, bias=False)
        else:
            self.decoder_sem = None

        # Pre-encoder bias (Gao 2024): subtracted before encoding, added back after decoding
        self.b_pre = nn.Parameter(torch.zeros(n_sem))

    def _init_encoder_weights(self):
        """Initialize encoder and semantic decoder weights.

        Follows Gao 2024: initialize decoder_sem with Xavier, then set
        encoder = decoder_sem^T (tied-at-init). Encoder rows are then
        normalized to unit norm. When tie_semantic_decoder=True there is
        no separate decoder_sem parameter so we just Xavier the encoder.
        """
        if self.decoder_sem is not None:
            nn.init.xavier_uniform_(self.decoder_sem.weight)
            self.encoder.weight.data = self.decoder_sem.weight.data.T.clone()
        else:
            nn.init.xavier_uniform_(self.encoder.weight)
        with torch.no_grad():
            norms = self.encoder.weight.data.norm(dim=1, keepdim=True)
            self.encoder.weight.data.div_(norms.clamp(min=1e-8))

    @torch.no_grad()
    def init_b_pre_from_data(
        self,
        x: torch.Tensor,
        method: str = "geometric_median",
        n_iter: int = 100,
        eps: float = 1e-8,
    ) -> None:
        """Initialize b_pre from a sample of training inputs.

        Gao 2024 (following Bricken 2023) initializes the pre-encoder bias to
        the geometric median of training data rather than zero. The geometric
        median is more robust to outliers than the arithmetic mean and gives
        a better starting point for the sparse code to build on.

        Parameters
        ----------
        x : torch.Tensor, shape (n_samples, n_sem)
            Sample of training semantic features.
        method : {"geometric_median", "mean"}
            "geometric_median" (default) uses Weiszfeld's iteration.
            "mean" uses the arithmetic mean (faster, less robust).
        n_iter : int
            Max iterations for Weiszfeld. Ignored if method="mean".
        eps : float
            Numerical stabilizer for 1/||x - y||.
        """
        if method == "mean":
            self.b_pre.data = x.mean(dim=0).to(self.b_pre.device)
            return

        # Weiszfeld's algorithm for geometric median
        y = x.mean(dim=0)
        for _ in range(n_iter):
            d = (x - y).norm(dim=1).clamp(min=eps)
            w = 1.0 / d
            y_new = (w.unsqueeze(1) * x).sum(dim=0) / w.sum()
            if (y_new - y).norm() < eps:
                y = y_new
                break
            y = y_new
        self.b_pre.data = y.to(self.b_pre.device)

    def encode(self, x: torch.Tensor, k_override: int = None) -> torch.Tensor:
        """Encode semantic features to sparse non-negative codes.

        In 'topk' mode: exactly k non-zeros per sample (hard selection).
        In 'l1' mode: dense ReLU output; sparsity comes from the external
        gauge-invariant L1 penalty in the training loss (Anthropic/Conerly style).

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_sem)
            Semantic features.
        k_override : int, optional
            Override sparsity level (for Matryoshka training).

        Returns
        -------
        z : torch.Tensor, shape (batch, n_atoms)
        """
        k = k_override if k_override is not None else self.k
        z = self.encoder(x - self.b_pre)
        if self.encoder_bias is not None:
            z = z + self.encoder_bias
        z = F.relu(z)

        if self.sparsity_mode == "topk":
            _, top_idx = torch.topk(z, k, dim=-1)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, top_idx, 1.0)
            z = z * mask

        return z

    def _decode_semantic(self, s: torch.Tensor) -> torch.Tensor:
        """Semantic reconstruction from sparse code. Handles tied vs untied."""
        if self.decoder_sem is not None:
            return self.decoder_sem(s) + self.b_pre
        return F.linear(s, self.encoder.weight.T) + self.b_pre

    def get_semantic_dictionary(self) -> np.ndarray:
        """Get semantic dictionary atoms.

        Returns
        -------
        D : ndarray, shape (n_atoms, n_sem)
            Semantic dictionary atoms. When decoder_sem is untied this is
            decoder_sem.weight.T; when tied it is encoder.weight.
        """
        if self.decoder_sem is not None:
            return self.decoder_sem.weight.data.T.detach().cpu().numpy()
        return self.encoder.weight.data.detach().cpu().numpy()

    def load_state_dict_compat(self, state_dict, strict: bool = False):
        """Load state_dict with backward compatibility for pre-refactor checkpoints.

        Older checkpoints lack ``decoder_sem.weight`` and ``b_pre`` (semantic
        decoder was tied to encoder, no pre-encoder bias). This method fills
        those in so old checkpoints can be loaded into the new model:

        - ``decoder_sem.weight`` defaults to ``encoder.weight.T`` (tied-init),
          matching the behavior of the old tied forward pass at load time.
        - ``b_pre`` defaults to zeros.

        Parameters
        ----------
        state_dict : dict
            State dict to load. May be from an old or new checkpoint.
        strict : bool
            Passed through to ``load_state_dict``. Default False because the
            brain decoder weight shape may differ when voxel counts change.

        Returns
        -------
        IncompatibleKeys
            The usual namedtuple returned by ``load_state_dict``.
        """
        state_dict = dict(state_dict)
        if ("decoder_sem.weight" not in state_dict and "encoder.weight" in state_dict
                and getattr(self, "decoder_sem", None) is not None):
            # Older checkpoints of untied models have no decoder_sem: initialise it from the encoder.
            state_dict["decoder_sem.weight"] = state_dict["encoder.weight"].T.clone()
        if "b_pre" not in state_dict:
            state_dict["b_pre"] = torch.zeros(self.n_sem)
        return self.load_state_dict(state_dict, strict=strict)

    @abstractmethod
    def forward(self, x: torch.Tensor, x_delayed: torch.Tensor, **kwargs) -> dict:
        """Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_sem)
            Current semantic features.
        x_delayed : torch.Tensor, shape (batch, n_sem * n_delays)
            Pre-computed delayed semantic features.

        Returns
        -------
        output : dict
            Dictionary containing at least 's', 'x_hat', 'y_hat'.
        """
        ...

    @abstractmethod
    @torch.no_grad()
    def normalize_weights(self, non_neg_brain: bool = True) -> None:
        """Normalize encoder rows to unit norm and optionally clamp brain decoder.

        Parameters
        ----------
        non_neg_brain : bool
            If True, clamp brain decoder weights to be non-negative.
        """
        ...

    @abstractmethod
    def get_brain_dictionary(self, **kwargs) -> np.ndarray:
        """Get brain dictionary atoms.

        Returns
        -------
        D : ndarray, shape (n_atoms, n_voxels)
            Brain dictionary atoms, averaged across delays.
        """
        ...


class CoupledSAE(CoupledSAEBase):
    """Coupled Sparse Autoencoder for single-subject brain encoding.

    Features:
    - Delay-specific scaling: each atom can have different contributions
      at different HRF delays.
    - Total Variation (TV) regularization: encourages spatial smoothness
      in brain decoder weights.
    - Adaptive sparsity: TopK with minimum threshold allows <=k active atoms.
    """

    def __init__(
        self,
        n_sem: int,
        n_voxels: int,
        n_atoms: int,
        k: int,
        n_delays: int,
        use_delay_scaling: bool = False,
        edge_incidence: sp_sparse.csr_matrix = None,
        tv_weight: float = 0.0,
        tie_semantic_decoder: bool = False,
        brain_voxel_scale: bool = False,
        temporal_first: bool = False,
        sparsity_mode: str = "topk",
        encoder_bias: bool = False,
    ):
        """Initialize single-subject coupled SAE.

        Parameters
        ----------
        n_sem : int
            Dimension of semantic features.
        n_voxels : int
            Number of brain voxels.
        n_atoms : int
            Number of dictionary atoms.
        k : int
            TopK sparsity parameter.
        n_delays : int
            Number of HRF temporal delays.
        use_delay_scaling : bool
            If True, learn delay-specific scaling for atoms.
        edge_incidence : scipy sparse matrix, optional
            Edge incidence matrix for TV regularization.
        tv_weight : float
            Weight for TV regularization loss.
        """
        super().__init__(
            n_sem=n_sem,
            n_atoms=n_atoms,
            k=k,
            n_delays=n_delays,
            tie_semantic_decoder=tie_semantic_decoder,
            sparsity_mode=sparsity_mode,
            encoder_bias=encoder_bias,
        )
        self.n_voxels = n_voxels
        self.use_delay_scaling = use_delay_scaling
        self.tv_weight = tv_weight
        self.temporal_first = temporal_first

        # Delay-specific scaling
        if use_delay_scaling:
            self.delay_scales = nn.Parameter(torch.ones(n_delays, n_atoms))
        else:
            self.delay_scales = None

        # Brain decoder. temporal_first: sum delays first → 50-dim → V.
        # Otherwise: concat delays → 250-dim → V.
        brain_in = n_atoms if temporal_first else n_atoms * n_delays
        self.decoder_brain = nn.Linear(brain_in, n_voxels, bias=False)

        # Optional per-voxel scale (paired with decoder unit-norm this
        # decouples direction from magnitude per voxel; ones-init).
        if brain_voxel_scale:
            self.voxel_scale = nn.Parameter(torch.ones(n_voxels))
        else:
            self.voxel_scale = None

        # TV regularization storage
        self._edge_incidence_scipy = edge_incidence
        self.edge_incidence = None

        self._init_weights()

    def _init_weights(self):
        """Initialize all weights."""
        self._init_encoder_weights()
        nn.init.xavier_uniform_(self.decoder_brain.weight)

    def init_tv_on_device(self, device: str):
        """Initialize TV edge incidence tensor on the target device.

        Must be called after model.to(device).

        Parameters
        ----------
        device : str
            Target device ("cpu" or "cuda").
        """
        if self._edge_incidence_scipy is not None and self.tv_weight > 0:
            self.edge_incidence = sparse_scipy_to_torch(
                self._edge_incidence_scipy, device
            )
        else:
            self.edge_incidence = None

    def tv_loss(self) -> torch.Tensor:
        """Compute Total Variation loss on brain decoder weights.

        TV(W) = mean(|E @ W|) where E is the edge incidence matrix.
        This penalizes differences between neighboring voxels.

        Returns
        -------
        loss : torch.Tensor
            Scalar TV loss (0 if TV is disabled).
        """
        if self.edge_incidence is None or self.tv_weight == 0:
            return torch.tensor(0.0, device=self.decoder_brain.weight.device)

        W = self.decoder_brain.weight
        diff = torch.sparse.mm(self.edge_incidence, W)
        tv = torch.abs(diff).mean()
        return self.tv_weight * tv

    def forward(
        self,
        x: torch.Tensor,
        x_delayed: torch.Tensor,
        **kwargs,
    ) -> dict:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_sem)
            Current semantic features.
        x_delayed : torch.Tensor, shape (batch, n_sem * n_delays)
            Pre-computed delayed semantic features.

        Returns
        -------
        output : dict
            Keys: 's' (sparse codes), 's_delayed' (delayed codes),
            'x_hat' (semantic reconstruction), 'y_hat' (brain prediction).
        """
        batch_size = x.shape[0]

        # Encode current X to sparse codes
        s = self.encode(x)

        # Semantic reconstruction (untied decoder + b_pre)
        x_hat = self._decode_semantic(s)

        # Encode each delayed X with optional scaling
        x_del_reshaped = x_delayed.view(batch_size, self.n_delays, self.n_sem)

        s_delayed_list = []
        for d in range(self.n_delays):
            s_d = self.encode(x_del_reshaped[:, d, :])
            if self.use_delay_scaling and self.delay_scales is not None:
                s_d = s_d * self.delay_scales[d]
            s_delayed_list.append(s_d)

        if self.temporal_first:
            s_delayed = torch.stack(s_delayed_list, dim=0).sum(dim=0)  # (batch, n_atoms)
        else:
            s_delayed = torch.cat(s_delayed_list, dim=-1)  # (batch, n_atoms * n_delays)

        # Brain prediction
        y_hat = self.decoder_brain(s_delayed)
        if self.voxel_scale is not None:
            y_hat = y_hat * self.voxel_scale

        return {
            's': s,
            's_delayed': s_delayed,
            'x_hat': x_hat,
            'y_hat': y_hat,
        }

    @torch.no_grad()
    def normalize_weights(self, non_neg_brain: bool = True,
                          encoder_unit_norm: bool = True,
                          brain_decoder_unit_norm: bool = False,
                          semantic_decoder_unit_norm: bool = False):
        """Optional gauge-fixing projections after each step.

        Parameters
        ----------
        non_neg_brain : bool
            If True, clamp brain decoder weights to be non-negative.
        encoder_unit_norm : bool
            If True, project encoder rows to unit norm.
        brain_decoder_unit_norm : bool
            If True, project brain decoder columns to unit norm
            (Gao/Bricken standard; applied AFTER non_neg clamp when both set).
        semantic_decoder_unit_norm : bool
            If True, project semantic decoder columns to unit norm.
            No-op when the semantic decoder is tied (decoder_sem is None).
        """
        if encoder_unit_norm:
            norms = self.encoder.weight.norm(dim=1, keepdim=True)
            self.encoder.weight.div_(norms.clamp(min=1e-8))
        if non_neg_brain:
            self.decoder_brain.weight.clamp_(min=0)
        if brain_decoder_unit_norm:
            col_norms = self.decoder_brain.weight.norm(dim=0, keepdim=True)
            self.decoder_brain.weight.div_(col_norms.clamp(min=1e-8))
        if semantic_decoder_unit_norm and self.decoder_sem is not None:
            col_norms = self.decoder_sem.weight.norm(dim=0, keepdim=True)
            self.decoder_sem.weight.div_(col_norms.clamp(min=1e-8))

    def gauge_invariant_l1(self, s_delayed: torch.Tensor) -> torch.Tensor:
        """Gauge-invariant L1 penalty on brain decoder (Conerly/Anthropic 2024).

        Penalty = sum_i ||W_b[:, i]||_2 * mean_batch(|s_delayed[:, i]|)
        Invariant under W_b[:,i] -> c*W_b[:,i], s[:,i] -> s[:,i]/c.
        """
        col_norms = self.decoder_brain.weight.norm(dim=0)  # (n_atoms * n_delays,)
        act_abs_mean = s_delayed.abs().mean(dim=0)          # (n_atoms * n_delays,)
        return (col_norms * act_abs_mean).sum()

    def get_brain_dictionary(self, **kwargs) -> np.ndarray:
        """Get brain atoms.

        For full FIR (temporal_first=False): average across delays.
        For temporal-first: return the 50-column decoder directly.

        Returns
        -------
        D : ndarray, shape (n_atoms, n_voxels)
            Brain dictionary atoms.
        """
        W = self.decoder_brain.weight.detach().cpu().numpy()
        if self.temporal_first:
            return W.T  # (n_atoms, n_voxels)
        W_reshaped = W.reshape(self.n_voxels, self.n_delays, self.n_atoms)
        return W_reshaped.mean(axis=1).T  # (n_atoms, n_voxels)


class MultiSubjectCoupledSAE(CoupledSAEBase):
    """Coupled SAE with shared encoder and subject-specific brain decoders.

    The encoder and semantic decoder are shared across subjects.
    Each subject has their own brain decoder (different voxel spaces).
    """

    def __init__(
        self,
        n_sem: int,
        n_voxels_per_subject: dict[str, int],
        n_atoms: int,
        k: int,
        n_delays: int,
        use_delay_scaling: bool = True,
        subject_specific_delays: bool = False,
        tie_semantic_decoder: bool = False,
        brain_voxel_scale: bool = False,
        temporal_first: bool = False,
        sparsity_mode: str = "topk",
        encoder_bias: bool = False,
        encoder_mode: str = "linear",
        mp_steps: int = None,
    ):
        """Initialize multi-subject coupled SAE.

        Parameters
        ----------
        n_sem : int
            Dimension of semantic features.
        n_voxels_per_subject : dict
            Mapping from subject ID to number of voxels.
        n_atoms : int
            Number of dictionary atoms (shared).
        k : int
            TopK sparsity parameter.
        n_delays : int
            Number of HRF temporal delays.
        use_delay_scaling : bool
            If True, learn delay-specific scaling.
        subject_specific_delays : bool
            If True, each subject has own delay scales.
        """
        # MP-SAE forces tied decoder, no encoder bias, unit-norm dictionary
        if encoder_mode == "mp":
            tie_semantic_decoder = True
            encoder_bias = False

        super().__init__(
            n_sem=n_sem,
            n_atoms=n_atoms,
            k=k,
            n_delays=n_delays,
            tie_semantic_decoder=tie_semantic_decoder,
            sparsity_mode=sparsity_mode,
            encoder_bias=encoder_bias,
        )
        self.encoder_mode = encoder_mode
        self.mp_steps = mp_steps if mp_steps is not None else k
        self.n_voxels_per_subject = n_voxels_per_subject
        self.subjects = list(n_voxels_per_subject.keys())
        self.use_delay_scaling = use_delay_scaling
        self.subject_specific_delays = subject_specific_delays
        self.temporal_first = temporal_first

        # Delay-specific scaling
        if use_delay_scaling:
            if subject_specific_delays:
                self.delay_scales = nn.ParameterDict({
                    subj: nn.Parameter(torch.ones(n_delays, n_atoms))
                    for subj in self.subjects
                })
            else:
                self.delay_scales = nn.Parameter(torch.ones(n_delays, n_atoms))
        else:
            self.delay_scales = None

        # Subject-specific brain decoders
        brain_in = n_atoms if temporal_first else n_atoms * n_delays
        self.brain_decoders = nn.ModuleDict()
        for subj, n_vox in n_voxels_per_subject.items():
            self.brain_decoders[subj] = nn.Linear(
                brain_in, n_vox, bias=False
            )

        # Optional per-subject per-voxel scale (ones-init).
        if brain_voxel_scale:
            self.voxel_scales = nn.ParameterDict({
                subj: nn.Parameter(torch.ones(n_vox))
                for subj, n_vox in n_voxels_per_subject.items()
            })
        else:
            self.voxel_scales = None

        # Per-subject TV regularization (set via init_tv_on_device)
        self._edge_incidence_per_subject = {}
        self.tv_weight = 0.0

        self._init_weights()

    def init_tv_on_device(self, edge_incidence_per_subject: dict, tv_weight: float, device: str):
        """Initialize TV edge incidence tensors on the target device.

        Parameters
        ----------
        edge_incidence_per_subject : dict
            {subject_id: scipy sparse edge incidence matrix}
        tv_weight : float
            Weight for TV regularization loss.
        device : str
            Target device.
        """
        self.tv_weight = tv_weight
        for subj, E in edge_incidence_per_subject.items():
            if E is not None:
                self._edge_incidence_per_subject[subj] = sparse_scipy_to_torch(E, device)

    def tv_loss(self) -> torch.Tensor:
        """Compute Total Variation loss across all subjects' brain decoders."""
        if self.tv_weight == 0 or not self._edge_incidence_per_subject:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        total = 0.0
        n = 0
        for subj, E in self._edge_incidence_per_subject.items():
            if subj in self.brain_decoders:
                W = self.brain_decoders[subj].weight
                diff = torch.sparse.mm(E, W)
                total = total + torch.abs(diff).mean()
                n += 1
        if n == 0:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return self.tv_weight * total / n

    def _init_weights(self):
        """Initialize all weights."""
        self._init_encoder_weights()
        for decoder in self.brain_decoders.values():
            nn.init.xavier_uniform_(decoder.weight)

    def encode(self, x: torch.Tensor, k_override: int = None) -> torch.Tensor:
        """Encode with linear (TopK) or matching pursuit.

        MP mode: iterative greedy selection from tied dictionary.
        Adaptive sparsity — stops when no atom has positive correlation
        with the residual. Coefficients are non-negative (ReLU).
        """
        if self.encoder_mode != "mp":
            return super().encode(x, k_override=k_override)

        # Matching Pursuit encoding using tied dictionary
        # Dictionary D = encoder.weight.T, shape (n_sem, n_atoms)
        D = self.encoder.weight.T
        T = k_override if k_override is not None else self.mp_steps
        batch = x.shape[0]

        r = x - self.b_pre                              # (batch, n_sem)
        z = torch.zeros(batch, self.n_atoms,
                        device=x.device, dtype=x.dtype)
        selected = torch.zeros(batch, self.n_atoms,
                               device=x.device, dtype=torch.bool)

        for t in range(T):
            scores = r @ D                               # (batch, n_atoms)
            scores = scores.masked_fill(selected.detach(), -float('inf'))

            # Adaptive sparsity: stop if no sample has positive correlation
            max_scores = scores.max(dim=-1).values       # (batch,)
            if (max_scores <= 0).all():
                break

            j = scores.argmax(dim=-1)                    # (batch,)
            coeff = scores.gather(1, j.unsqueeze(1))     # (batch, 1)
            coeff = F.relu(coeff)                         # non-negative

            # Mark as selected
            selected = selected.clone()
            selected.scatter_(1, j.unsqueeze(1), True)

            # Update sparse code and residual (only where coeff > 0)
            z = z.scatter(1, j.unsqueeze(1), coeff)
            D_j = D.T[j]                                 # (batch, n_sem)
            r = r - coeff * D_j

        return z

    def _get_delay_scales(self, subject: str = None) -> torch.Tensor:
        """Get delay scales, optionally subject-specific.

        Parameters
        ----------
        subject : str, optional
            Subject ID for subject-specific scales.

        Returns
        -------
        scales : torch.Tensor or None
            Delay scales of shape (n_delays, n_atoms).
        """
        if not self.use_delay_scaling or self.delay_scales is None:
            return None
        if self.subject_specific_delays and subject is not None:
            return self.delay_scales[subject]
        elif isinstance(self.delay_scales, nn.Parameter):
            return self.delay_scales
        else:
            return list(self.delay_scales.values())[0]

    def forward(
        self,
        x: torch.Tensor,
        x_delayed: torch.Tensor,
        subject: str = None,
        k_override: int = None,
        **kwargs,
    ) -> dict:
        """Forward pass for a specific subject.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_sem)
            Current semantic features.
        x_delayed : torch.Tensor, shape (batch, n_sem * n_delays)
            Pre-computed delayed semantic features.
        subject : str, optional
            Subject ID for brain prediction.

        Returns
        -------
        output : dict
            Keys: 's', 's_delayed', 'x_hat', and 'y_hat' if subject specified.
        """
        batch_size = x.shape[0]

        s = self.encode(x, k_override=k_override)
        x_hat = self._decode_semantic(s)

        x_del_reshaped = x_delayed.view(batch_size, self.n_delays, self.n_sem)
        delay_scales = self._get_delay_scales(subject)

        s_delayed_list = []
        for d in range(self.n_delays):
            s_d = self.encode(x_del_reshaped[:, d, :], k_override=k_override)
            if delay_scales is not None:
                s_d = s_d * delay_scales[d]
            s_delayed_list.append(s_d)

        if self.temporal_first:
            s_delayed = torch.stack(s_delayed_list, dim=0).sum(dim=0)
        else:
            s_delayed = torch.cat(s_delayed_list, dim=-1)

        result = {
            's': s,
            's_delayed': s_delayed,
            'x_hat': x_hat,
        }

        if subject is not None:
            y_hat = self.brain_decoders[subject](s_delayed)
            if self.voxel_scales is not None:
                y_hat = y_hat * self.voxel_scales[subject]
            result['y_hat'] = y_hat

        return result

    def forward_all_subjects(
        self,
        x: torch.Tensor,
        x_delayed: torch.Tensor,
        k_override: int = None,
        subjects: list = None,
    ) -> dict:
        """Forward pass predicting all subjects' brain responses.

        ``subjects`` (optional) restricts the brain decoders that are run; the
        encoder/semantic path is unchanged. Used by round-robin training, where
        only the active subject's prediction enters the loss.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_sem)
            Current semantic features.
        x_delayed : torch.Tensor, shape (batch, n_sem * n_delays)
            Pre-computed delayed semantic features.
        k_override : int, optional
            Override sparsity level for Matryoshka training.

        Returns
        -------
        output : dict
            Keys: 's', 'x_hat', 'y_hat' (dict per subject),
            's_delayed' (dict per subject).
        """
        batch_size = x.shape[0]

        s = self.encode(x, k_override=k_override)
        x_hat = self._decode_semantic(s)

        x_del_reshaped = x_delayed.view(batch_size, self.n_delays, self.n_sem)

        result = {
            's': s,
            'x_hat': x_hat,
            'y_hat': {},
            's_delayed': {},
        }

        for subj in (self.subjects if subjects is None else subjects):
            delay_scales = self._get_delay_scales(subj)
            s_delayed_list = []
            for d in range(self.n_delays):
                s_d = self.encode(x_del_reshaped[:, d, :], k_override=k_override)
                if delay_scales is not None:
                    s_d = s_d * delay_scales[d]
                s_delayed_list.append(s_d)
            if self.temporal_first:
                s_delayed = torch.stack(s_delayed_list, dim=0).sum(dim=0)
            else:
                s_delayed = torch.cat(s_delayed_list, dim=-1)
            result['s_delayed'][subj] = s_delayed
            y_hat_subj = self.brain_decoders[subj](s_delayed)
            if self.voxel_scales is not None:
                y_hat_subj = y_hat_subj * self.voxel_scales[subj]
            result['y_hat'][subj] = y_hat_subj

        return result

    @torch.no_grad()
    def normalize_weights(self, non_neg_brain: bool = True,
                          encoder_unit_norm: bool = True,
                          brain_decoder_unit_norm: bool = False,
                          semantic_decoder_unit_norm: bool = False):
        """Optional gauge-fixing projections (multi-subject version).

        Parameters
        ----------
        non_neg_brain : bool
            If True, clamp every subject's brain decoder to be non-negative.
        encoder_unit_norm : bool
            If True, project encoder rows to unit norm.
        brain_decoder_unit_norm : bool
            If True, project each subject's brain decoder columns to unit norm.
        semantic_decoder_unit_norm : bool
            If True, project semantic decoder columns to unit norm.
            No-op when the semantic decoder is tied.
        """
        if encoder_unit_norm:
            norms = self.encoder.weight.norm(dim=1, keepdim=True)
            self.encoder.weight.div_(norms.clamp(min=1e-8))
        for decoder in self.brain_decoders.values():
            if non_neg_brain:
                decoder.weight.clamp_(min=0)
            if brain_decoder_unit_norm:
                col_norms = decoder.weight.norm(dim=0, keepdim=True)
                decoder.weight.div_(col_norms.clamp(min=1e-8))
        if semantic_decoder_unit_norm and self.decoder_sem is not None:
            col_norms = self.decoder_sem.weight.norm(dim=0, keepdim=True)
            self.decoder_sem.weight.div_(col_norms.clamp(min=1e-8))

    def gauge_invariant_l1(self, s_delayed_per_subject: dict) -> torch.Tensor:
        """Gauge-invariant L1 across all subjects' brain decoders.

        ``s_delayed_per_subject`` is the ``s_delayed`` dict returned by
        ``forward_all_subjects`` (same sparse code for every subject).
        """
        total = None
        for subj, decoder in self.brain_decoders.items():
            col_norms = decoder.weight.norm(dim=0)
            s_del = s_delayed_per_subject[subj]
            act_abs_mean = s_del.abs().mean(dim=0)
            term = (col_norms * act_abs_mean).sum()
            total = term if total is None else total + term
        return total / len(self.brain_decoders)

    def get_brain_dictionary(self, subject: str, **kwargs) -> np.ndarray:
        """Get brain atoms for a specific subject, averaged across delays.

        Parameters
        ----------
        subject : str
            Subject ID.

        Returns
        -------
        D : ndarray, shape (n_atoms, n_voxels)
            Brain dictionary atoms.
        """
        W = self.brain_decoders[subject].weight.detach().cpu().numpy()
        if self.temporal_first:
            return W.T  # (n_atoms, n_voxels)
        n_vox = self.n_voxels_per_subject[subject]
        W_reshaped = W.reshape(n_vox, self.n_delays, self.n_atoms)
        return W_reshaped.mean(axis=1).T  # (n_atoms, n_voxels)
