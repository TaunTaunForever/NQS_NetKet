from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from netket.nn.symmetric_linear import DenseSymmMatrix
from netket.utils import HashableArray

from kitaev_honeycomb_vit_model import OutputHead, TransformerBlock


class HoneycombSymmInputViT(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    learn_phase: bool
    symmetries: Tuple[Tuple[int, ...], ...]
    permutation: Optional[Tuple[int, ...]] = None
    data_type: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, sigma):
        x = jnp.atleast_2d(sigma)

        if self.permutation is not None:
            perm_arr = jnp.asarray(self.permutation, dtype=jnp.int32)
            x = x[:, perm_arr]

        x = x.astype(self.data_type)
        x = jnp.expand_dims(x, axis=-2)  # (batch, in_features=1, n_sites)

        symm_tokens = DenseSymmMatrix(
            symmetries=HashableArray(np.asarray(self.symmetries, dtype=np.int32)),
            features=self.embed_dim,
            param_dtype=self.data_type,
            use_bias=True,
            name="symm_input_stem",
        )(x)

        # DenseSymmMatrix returns (batch, features, n_symm). We interpret the
        # symmetry dimension as the token axis for the transformer.
        x = jnp.swapaxes(symm_tokens, -2, -1)
        x = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="symm_token_norm",
        )(x)

        # No positional embeddings: the ordering of symmetry elements is
        # arbitrary, so we avoid learning order-specific biases here.
        for layer in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                data_type=self.data_type,
                name=f"TransformerBlock_{layer}",
            )(x)

        log_psi = OutputHead(
            d_model=self.embed_dim,
            data_type=self.data_type,
            name="OutputHead",
        )(x)

        if self.learn_phase:
            return log_psi
        return jnp.real(log_psi)
