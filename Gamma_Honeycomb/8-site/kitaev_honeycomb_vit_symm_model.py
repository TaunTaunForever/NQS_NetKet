from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp

from kitaev_honeycomb_vit_model import HoneycombPatchViT


def complex_logmeanexp(z, axis):
    """Stable log(mean(exp(z))) for complex-valued log-amplitudes."""
    max_real = jnp.max(jnp.real(z), axis=axis, keepdims=True)
    centered = jnp.exp(z - max_real)
    mean_centered = jnp.mean(centered, axis=axis)
    max_real = jnp.squeeze(max_real, axis=axis)
    return jnp.log(mean_centered) + max_real


class SymmetryProjectedHoneycombViT(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool
    symmetries: Tuple[Tuple[int, ...], ...]
    permutation: Tuple[int, ...] | None = None
    data_type: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, sigma):
        x = jnp.atleast_2d(sigma)
        symm = jnp.asarray(self.symmetries, dtype=jnp.int32)

        x_symm = x[:, symm].reshape(-1, x.shape[-1])

        base_model = HoneycombPatchViT(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_hidden_dim=self.mlp_hidden_dim,
            patch_size=self.patch_size,
            learn_phase=self.learn_phase,
            data_type=self.data_type,
            permutation=self.permutation,
            name="base_model",
        )
        log_psi = base_model(x_symm).reshape(x.shape[0], symm.shape[0])
        return complex_logmeanexp(log_psi, axis=1)


class CanonicalRepresentativeHoneycombViT(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool
    symmetries: Tuple[Tuple[int, ...], ...]
    permutation: Tuple[int, ...] | None = None
    data_type: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, sigma):
        x = jnp.atleast_2d(sigma)
        symm = jnp.asarray(self.symmetries, dtype=jnp.int32)

        x_orbit = x[:, symm]
        bits = (x_orbit > 0).astype(jnp.int64)
        weights = (1 << jnp.arange(x.shape[-1] - 1, -1, -1, dtype=jnp.int64))
        orbit_codes = jnp.tensordot(bits, weights, axes=([-1], [0]))
        canon_idx = jnp.argmin(orbit_codes, axis=1)
        canonical_x = jnp.take_along_axis(
            x_orbit, canon_idx[:, None, None], axis=1
        ).squeeze(axis=1)

        base_model = HoneycombPatchViT(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_hidden_dim=self.mlp_hidden_dim,
            patch_size=self.patch_size,
            learn_phase=self.learn_phase,
            data_type=self.data_type,
            permutation=self.permutation,
            name="base_model",
        )
        return base_model(canonical_x)
