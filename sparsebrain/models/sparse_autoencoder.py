"""Sparse Autoencoder for learning dictionary representations of brain responses.

A linear sparse autoencoder is equivalent to dictionary learning:
    Y ≈ Z @ D
where:
    - Y: brain responses (n_samples, n_voxels)
    - Z: sparse codes (n_samples, n_atoms)
    - D: dictionary atoms (n_atoms, n_voxels)

The encoder learns to produce sparse codes Z, and the decoder learns the dictionary D.
"""

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsityFunction(ABC, nn.Module):
    """Abstract base class for sparsity functions."""

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply sparsity to latent codes.

        Parameters
        ----------
        z : torch.Tensor
            Pre-activation latent codes, shape (batch, n_atoms).

        Returns
        -------
        torch.Tensor
            Sparse latent codes, shape (batch, n_atoms).
        """
        ...

    @abstractmethod
    def penalty(self, z: torch.Tensor) -> torch.Tensor:
        """Compute sparsity penalty for loss function.

        Parameters
        ----------
        z : torch.Tensor
            Latent codes (before or after sparsity, depending on method).

        Returns
        -------
        torch.Tensor
            Scalar penalty term.
        """
        ...


class L1Sparsity(SparsityFunction):
    """L1-regularized sparsity with ReLU activation.

    Applies ReLU to get non-negative activations, then adds L1 penalty to loss.

    Parameters
    ----------
    lambda_l1 : float
        L1 penalty coefficient.
    """

    def __init__(self, lambda_l1: float = 1e-3):
        super().__init__()
        self.lambda_l1 = lambda_l1

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply ReLU activation."""
        return F.relu(z)

    def penalty(self, z: torch.Tensor) -> torch.Tensor:
        """Compute L1 penalty on activations."""
        return self.lambda_l1 * z.abs().mean()

    def __repr__(self) -> str:
        return f"L1Sparsity(lambda_l1={self.lambda_l1})"


class TopKSparsity(SparsityFunction):
    """TopK sparsity with hard threshold.

    Keeps only the top-k activations per sample, sets rest to zero.
    This gives exact sparsity control.

    Parameters
    ----------
    k : int
        Number of atoms to keep active per sample.
    """

    def __init__(self, k: int = 50):
        super().__init__()
        self.k = k

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Keep top-k activations, zero out the rest."""
        # Apply ReLU first to get non-negative activations
        z = F.relu(z)

        # Get top-k indices
        _, top_indices = torch.topk(z, self.k, dim=-1)

        # Create mask
        mask = torch.zeros_like(z)
        mask.scatter_(-1, top_indices, 1.0)

        return z * mask

    def penalty(self, z: torch.Tensor) -> torch.Tensor:
        """No penalty needed — sparsity is enforced by TopK."""
        return torch.tensor(0.0, device=z.device)

    def __repr__(self) -> str:
        return f"TopKSparsity(k={self.k})"


class SparseAutoencoder(nn.Module):
    """Linear Sparse Autoencoder for brain response dictionary learning.

    Architecture:
        encoder: Y -> Z (linear, no bias by default)
        sparsity: Z -> Z_sparse (L1 or TopK)
        decoder: Z_sparse -> Y_hat (linear, no bias by default)

    The decoder weights can be interpreted as dictionary atoms.

    Parameters
    ----------
    n_voxels : int
        Number of input voxels.
    n_atoms : int
        Number of dictionary atoms (latent dimension).
    sparsity : SparsityFunction
        Sparsity function (L1Sparsity or TopKSparsity).
    tied_weights : bool
        If True, decoder weights are transpose of encoder weights.
    bias : bool
        If True, include bias terms in encoder/decoder.
    normalize_decoder : bool
        If True, normalize decoder columns to unit norm after each forward pass.

    Attributes
    ----------
    encoder : nn.Linear
        Encoder layer.
    decoder : nn.Linear
        Decoder layer (dictionary).
    sparsity : SparsityFunction
        Sparsity function.
    """

    def __init__(
        self,
        n_voxels: int,
        n_atoms: int,
        sparsity: SparsityFunction,
        tied_weights: bool = False,
        bias: bool = False,
        normalize_decoder: bool = True,
        non_negative: bool = False,
    ):
        super().__init__()

        # Validate parameters
        if n_voxels <= 0:
            raise ValueError(f"n_voxels must be positive, got {n_voxels}")
        if n_atoms <= 0:
            raise ValueError(f"n_atoms must be positive, got {n_atoms}")
        if not isinstance(sparsity, SparsityFunction):
            raise TypeError(f"sparsity must be a SparsityFunction, got {type(sparsity)}")

        self.n_voxels = n_voxels
        self.n_atoms = n_atoms
        self.tied_weights = tied_weights
        self.normalize_decoder = normalize_decoder
        self.non_negative = non_negative

        # Encoder: voxels -> atoms
        self.encoder = nn.Linear(n_voxels, n_atoms, bias=bias)

        # Decoder: atoms -> voxels
        if tied_weights:
            self.decoder = None  # Use encoder weights transposed
        else:
            self.decoder = nn.Linear(n_atoms, n_voxels, bias=bias)

        # Sparsity function
        self.sparsity = sparsity

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot uniform."""
        nn.init.xavier_uniform_(self.encoder.weight)
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)

        if self.decoder is not None:
            nn.init.xavier_uniform_(self.decoder.weight)
            if self.decoder.bias is not None:
                nn.init.zeros_(self.decoder.bias)

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """Encode brain responses to sparse codes.

        Parameters
        ----------
        y : torch.Tensor
            Brain responses, shape (batch, n_voxels).

        Returns
        -------
        torch.Tensor
            Sparse codes, shape (batch, n_atoms).
        """
        z = self.encoder(y)
        z_sparse = self.sparsity(z)
        return z_sparse

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse codes to reconstructed brain responses.

        Parameters
        ----------
        z : torch.Tensor
            Sparse codes, shape (batch, n_atoms).

        Returns
        -------
        torch.Tensor
            Reconstructed responses, shape (batch, n_voxels).
        """
        if self.tied_weights:
            return F.linear(z, self.encoder.weight.t())
        else:
            return self.decoder(z)

    def forward(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through autoencoder.

        Parameters
        ----------
        y : torch.Tensor
            Brain responses, shape (batch, n_voxels).

        Returns
        -------
        dict with keys:
            - 'y_hat': reconstructed responses
            - 'z': sparse codes
            - 'z_pre': pre-sparsity activations (for penalty computation)
        """
        # Encode
        z_pre = self.encoder(y)
        z = self.sparsity(z_pre)

        # Decode
        y_hat = self.decode(z)

        return {"y_hat": y_hat, "z": z, "z_pre": z_pre}

    def loss(
        self,
        y: torch.Tensor,
        output: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute loss components.

        Parameters
        ----------
        y : torch.Tensor
            Original brain responses.
        output : dict
            Output from forward pass.

        Returns
        -------
        dict with keys:
            - 'total': total loss
            - 'reconstruction': MSE reconstruction loss
            - 'sparsity': sparsity penalty
        """
        y_hat = output["y_hat"]
        z = output["z"]

        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(y_hat, y)

        # Sparsity penalty
        sparsity_penalty = self.sparsity.penalty(z)

        # Total loss
        total_loss = reconstruction_loss + sparsity_penalty

        return {
            "total": total_loss,
            "reconstruction": reconstruction_loss,
            "sparsity": sparsity_penalty,
        }

    @torch.no_grad()
    def normalize_decoder_weights(self):
        """Normalize decoder columns to unit norm and optionally enforce non-negativity."""
        if self.decoder is not None:
            # Enforce non-negativity first (if enabled)
            if self.non_negative:
                self.decoder.weight.clamp_(min=0)
            # decoder.weight shape: (n_voxels, n_atoms)
            norms = self.decoder.weight.norm(dim=0, keepdim=True)
            self.decoder.weight.div_(norms.clamp(min=1e-8))

    def get_dictionary(self) -> torch.Tensor:
        """Get dictionary atoms (decoder weights).

        Returns
        -------
        torch.Tensor
            Dictionary atoms, shape (n_atoms, n_voxels).
        """
        if self.tied_weights:
            return self.encoder.weight.detach()
        else:
            # decoder.weight shape is (n_voxels, n_atoms), transpose to (n_atoms, n_voxels)
            return self.decoder.weight.t().detach()

    def get_sparsity_stats(self, z: torch.Tensor) -> dict[str, float]:
        """Compute sparsity statistics for latent codes.

        Parameters
        ----------
        z : torch.Tensor
            Sparse codes, shape (batch, n_atoms).

        Returns
        -------
        dict with sparsity statistics.
        """
        with torch.no_grad():
            # Fraction of zeros
            sparsity = (z == 0).float().mean().item()

            # Average L0 (number of active atoms per sample)
            l0 = (z != 0).float().sum(dim=-1).mean().item()

            # Average L1
            l1 = z.abs().mean().item()

            return {
                "sparsity": sparsity,
                "l0_per_sample": l0,
                "l1_mean": l1,
            }

    def __repr__(self) -> str:
        return (
            f"SparseAutoencoder(\n"
            f"  n_voxels={self.n_voxels},\n"
            f"  n_atoms={self.n_atoms},\n"
            f"  sparsity={self.sparsity},\n"
            f"  tied_weights={self.tied_weights},\n"
            f"  normalize_decoder={self.normalize_decoder},\n"
            f"  non_negative={self.non_negative}\n"
            f")"
        )
