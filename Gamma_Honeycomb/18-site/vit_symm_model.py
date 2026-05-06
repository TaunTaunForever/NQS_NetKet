from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import normal

from vit_model import HoneycombPatchViT, TransformerBlock, OutputHead


def _array_like_to_tuple(obj) -> Tuple[int, ...]:
    """
    Convert array-like objects, including NetKet HashableArray wrappers, into
    a flat tuple[int]. This avoids iterating over HashableArray directly.
    """
    candidates = [obj]

    # Common array-wrapper attributes/methods. Do not include deprecated
    # NetKet Permutation.permutation here.
    for name in ("to_array", "to_numpy", "array", "_array", "wrapped", "_wrapped", "data"):
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        try:
            value = value() if callable(value) else value
        except Exception:
            continue
        candidates.append(value)

    for value in candidates:
        try:
            arr = np.asarray(value)
            # np.asarray(HashableArray) may produce a scalar object array; skip that.
            if arr.dtype == object and arr.shape == ():
                continue
            return tuple(int(i) for i in arr.astype(np.int64).reshape(-1).tolist())
        except Exception:
            pass

        try:
            return tuple(int(i) for i in value)
        except Exception:
            pass

    raise TypeError(
        f"Could not convert symmetry permutation object of type {type(obj)} "
        "to tuple[int]."
    )


def permutation_to_tuple(p) -> Tuple[int, ...]:
    """
    Convert a NetKet Permutation / graph automorphism into tuple[int].

    NetKet deprecated `Permutation.permutation` in favor of
    `inverse_permutation_array` / `permutation_array`. Since these models apply
    a symmetry action through x[:, symm], we prefer inverse_permutation_array,
    which preserves the behavior of the old deprecated `permutation` field.
    """
    for name in ("inverse_permutation_array", "permutation_array"):
        try:
            value = getattr(p, name)
        except Exception:
            continue
        try:
            value = value() if callable(value) else value
        except Exception:
            continue
        return _array_like_to_tuple(value)

    return _array_like_to_tuple(p)


def normalise_symmetries(symm_group) -> Tuple[Tuple[int, ...], ...]:
    """
    Convert a NetKet symmetry group into the tuple-of-tuples format expected by
    the Flax modules below.
    """
    return tuple(permutation_to_tuple(p) for p in symm_group)


def complex_logmeanexp(z, axis):

    """Stable log(mean(exp(z))) for complex-valued log-amplitudes."""
    max_real = jnp.max(jnp.real(z), axis=axis, keepdims=True)
    centered = jnp.exp(z - max_real)
    mean_centered = jnp.mean(centered, axis=axis)
    max_real = jnp.squeeze(max_real, axis=axis)
    return jnp.log(mean_centered) + max_real


def _symmetry_array(symmetries):
    return jnp.asarray(symmetries, dtype=jnp.int32)


def _canonicalize_by_lexicographic_orbit(x, symm):
    x_orbit = x[:, symm]
    bits = (x_orbit > 0).astype(jnp.int64)
    weights = 1 << jnp.arange(x.shape[-1] - 1, -1, -1, dtype=jnp.int64)
    orbit_codes = jnp.tensordot(bits, weights, axes=([-1], [0]))
    canon_idx = jnp.argmin(orbit_codes, axis=1)
    canonical_x = jnp.take_along_axis(x_orbit, canon_idx[:, None, None], axis=1).squeeze(axis=1)
    return canonical_x, canon_idx


class SymmetryProjectedHoneycombViT(nn.Module):
    """Exact output symmetrization. Cost scales as O(|G|) base-model evaluations."""
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool
    symmetries: Tuple[Tuple[int, ...], ...]
    permutation: Tuple[int, ...] | None = None
    data_type: jnp.dtype = jnp.float64
    zero_imag_branch: bool = False

    @nn.compact
    def __call__(self, sigma):
        x = jnp.atleast_2d(sigma)
        symm = _symmetry_array(self.symmetries)
        x_symm = x[:, symm].reshape(-1, x.shape[-1])
        log_psi = HoneycombPatchViT(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_hidden_dim=self.mlp_hidden_dim,
            patch_size=self.patch_size,
            learn_phase=self.learn_phase,
            zero_imag_branch=self.zero_imag_branch,
            data_type=self.data_type,
            permutation=self.permutation,
            name="base_model",
        )(x_symm).reshape(x.shape[0], symm.shape[0])
        return complex_logmeanexp(log_psi, axis=1)


class CanonicalRepresentativeHoneycombViT(nn.Module):
    """One-forward-pass invariant model using a hard canonical orbit representative."""
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool
    symmetries: Tuple[Tuple[int, ...], ...]
    permutation: Tuple[int, ...] | None = None
    data_type: jnp.dtype = jnp.float64
    zero_imag_branch: bool = False

    @nn.compact
    def __call__(self, sigma):
        x = jnp.atleast_2d(sigma)
        symm = _symmetry_array(self.symmetries)
        canonical_x, _canon_idx = _canonicalize_by_lexicographic_orbit(x, symm)
        return HoneycombPatchViT(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_hidden_dim=self.mlp_hidden_dim,
            patch_size=self.patch_size,
            learn_phase=self.learn_phase,
            zero_imag_branch=self.zero_imag_branch,
            data_type=self.data_type,
            permutation=self.permutation,
            name="base_model",
        )(canonical_x)


class CanonicalTransformEmbeddingHoneycombViT(nn.Module):
    """Canonical representative plus learned embedding of selected symmetry operation."""
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool
    symmetries: Tuple[Tuple[int, ...], ...]
    permutation: Tuple[int, ...] | None = None
    data_type: jnp.dtype = jnp.float64
    zero_imag_branch: bool = False

    @nn.compact
    def __call__(self, sigma):
        x = jnp.atleast_2d(sigma)
        B, N = x.shape
        symm = _symmetry_array(self.symmetries)
        canonical_x, canon_idx = _canonicalize_by_lexicographic_orbit(x, symm)
        if self.permutation is not None:
            perm_arr = jnp.asarray(self.permutation, dtype=jnp.int32)
            canonical_x = canonical_x[:, perm_arr]
        if N % self.patch_size != 0:
            raise ValueError(f"Number of sites N={N} must be divisible by patch_size={self.patch_size}")
        n_patches = N // self.patch_size
        tokens = canonical_x.reshape(B, n_patches, self.patch_size).astype(self.data_type)
        tokens = nn.Dense(self.embed_dim, dtype=self.data_type, param_dtype=self.data_type,
                          kernel_init=nn.initializers.xavier_uniform(), name="patch_embed")(tokens)
        pos_emb = self.param("pos_emb", normal(stddev=0.02), (n_patches, self.embed_dim), self.data_type)
        tokens = tokens + pos_emb[None, :, :]
        frame_emb = nn.Embed(num_embeddings=symm.shape[0], features=self.embed_dim,
                             dtype=self.data_type, param_dtype=self.data_type,
                             embedding_init=normal(stddev=0.02),
                             name="canonical_transform_embedding")(canon_idx.astype(jnp.int32))
        tokens = tokens + frame_emb[:, None, :]
        for layer in range(self.num_layers):
            tokens = TransformerBlock(embed_dim=self.embed_dim, num_heads=self.num_heads,
                                      mlp_hidden_dim=self.mlp_hidden_dim, data_type=self.data_type,
                                      name=f"TransformerBlock_{layer}")(tokens)
        log_psi = OutputHead(d_model=self.embed_dim, data_type=self.data_type,
                             zero_imag_branch=self.zero_imag_branch, name="OutputHead")(tokens)
        return log_psi if self.learn_phase else jnp.real(log_psi)


class OrbitPooledInputHoneycombViT(nn.Module):
    """One-forward-pass ViT with raw spin plus simple orbit-pooled per-site features."""
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool
    symmetries: Tuple[Tuple[int, ...], ...]
    permutation: Tuple[int, ...] | None = None
    data_type: jnp.dtype = jnp.float64
    zero_imag_branch: bool = False

    @nn.compact
    def __call__(self, sigma):
        x0 = jnp.atleast_2d(sigma)
        B, N = x0.shape
        symm = _symmetry_array(self.symmetries)
        x = x0
        if self.permutation is not None:
            perm_arr = jnp.asarray(self.permutation, dtype=jnp.int32)
            x = x[:, perm_arr]
        if N % self.patch_size != 0:
            raise ValueError(f"Number of sites N={N} must be divisible by patch_size={self.patch_size}")
        x_orbit = x0[:, symm].astype(self.data_type)
        orbit_mean = jnp.mean(x_orbit, axis=1)
        orbit_var = jnp.var(x_orbit, axis=1)
        orbit_abs_mean = jnp.mean(jnp.abs(x_orbit), axis=1)
        features = jnp.stack([x.astype(self.data_type), orbit_mean, orbit_var, orbit_abs_mean], axis=-1)
        n_patches = N // self.patch_size
        features = features.reshape(B, n_patches, self.patch_size * 4)
        tokens = nn.Dense(self.embed_dim, dtype=self.data_type, param_dtype=self.data_type,
                          kernel_init=nn.initializers.xavier_uniform(), name="patch_embed")(features)
        pos_emb = self.param("pos_emb", normal(stddev=0.02), (n_patches, self.embed_dim), self.data_type)
        tokens = tokens + pos_emb[None, :, :]
        for layer in range(self.num_layers):
            tokens = TransformerBlock(embed_dim=self.embed_dim, num_heads=self.num_heads,
                                      mlp_hidden_dim=self.mlp_hidden_dim, data_type=self.data_type,
                                      name=f"TransformerBlock_{layer}")(tokens)
        log_psi = OutputHead(d_model=self.embed_dim, data_type=self.data_type,
                             zero_imag_branch=self.zero_imag_branch, name="OutputHead")(tokens)
        return log_psi if self.learn_phase else jnp.real(log_psi)


class StochasticSymmetryProjectedHoneycombViT(nn.Module):
    """Approximate output projection using a fixed subset of symmetries.

    For JAX/NetKet stability this uses a deterministic subset controlled by
    `num_symmetry_samples` and `symmetry_offset`. Vary offset between runs to
    emulate stochastic subsets without recompilation-time dynamic randomness.
    """
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool
    symmetries: Tuple[Tuple[int, ...], ...]
    num_symmetry_samples: int = 4
    symmetry_offset: int = 0
    permutation: Tuple[int, ...] | None = None
    data_type: jnp.dtype = jnp.float64
    zero_imag_branch: bool = False

    @nn.compact
    def __call__(self, sigma):
        x = jnp.atleast_2d(sigma)
        symm_all = _symmetry_array(self.symmetries)
        group_size = symm_all.shape[0]
        k = min(self.num_symmetry_samples, group_size)
        idx = (jnp.arange(k, dtype=jnp.int32) + int(self.symmetry_offset)) % group_size
        symm = symm_all[idx]
        x_symm = x[:, symm].reshape(-1, x.shape[-1])
        log_psi = HoneycombPatchViT(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_hidden_dim=self.mlp_hidden_dim,
            patch_size=self.patch_size,
            learn_phase=self.learn_phase,
            zero_imag_branch=self.zero_imag_branch,
            data_type=self.data_type,
            permutation=self.permutation,
            name="base_model",
        )(x_symm).reshape(x.shape[0], k)
        return complex_logmeanexp(log_psi, axis=1)
