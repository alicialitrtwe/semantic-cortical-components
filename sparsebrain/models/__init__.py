"""Neural network models for sparse representations."""

from sparsebrain.models.sparse_autoencoder import (
    SparseAutoencoder,
    L1Sparsity,
    TopKSparsity,
)
from sparsebrain.models.coupled_sae import (
    CoupledSAEBase,
    CoupledSAE,
    MultiSubjectCoupledSAE,
    sparse_scipy_to_torch,
)

__all__ = [
    "SparseAutoencoder",
    "L1Sparsity",
    "TopKSparsity",
    "CoupledSAEBase",
    "CoupledSAE",
    "MultiSubjectCoupledSAE",
    "sparse_scipy_to_torch",
]
