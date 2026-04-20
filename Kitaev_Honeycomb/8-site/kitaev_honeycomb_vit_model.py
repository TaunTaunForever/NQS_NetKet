from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import normal


class MLP(nn.Module):
    hidden_dim: int
    out_dim: int
    data_type: jnp.dtype

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.hidden_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(x)
        x = nn.gelu(x)
        x = nn.Dense(
            self.out_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(x)
        return x


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_hidden_dim: int
    data_type: jnp.dtype
    residual_scale: float = 0.8

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            deterministic=True,
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(y)
        x = x + self.residual_scale * y

        y = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(x)
        y = MLP(
            hidden_dim=self.mlp_hidden_dim,
            out_dim=self.embed_dim,
            data_type=self.data_type,
        )(y)
        x = x + self.residual_scale * y
        return x


def log_cosh(z):
    return jnp.log(jnp.cosh(z))


class OutputHead(nn.Module):
    d_model: int
    data_type: jnp.dtype

    @nn.compact
    def __call__(self, x):
        z = x.sum(axis=1)

        z = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="out_layer_norm",
        )(z)

        out_real = nn.Dense(
            self.d_model,
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="output_layer_real",
        )(z)
        out_real = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="norm_real",
        )(out_real)

        out_imag = nn.Dense(
            self.d_model,
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="output_layer_imag",
        )(z)
        out_imag = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="norm_imag",
        )(out_imag)

        out = out_real + 1j * out_imag
        return jnp.sum(log_cosh(out), axis=-1)


class HoneycombPatchViT(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool

    data_type: jnp.dtype = jnp.float64
    permutation: Optional[Tuple[int, ...]] = None

    @nn.compact
    def __call__(self, sigma):
        x = jnp.atleast_2d(sigma)
        B, N = x.shape

        if self.permutation is not None:
            perm_arr = jnp.asarray(self.permutation, dtype=jnp.int32)
            x = x[:, perm_arr]

        if N % self.patch_size != 0:
            raise ValueError(
                f"Number of sites N={N} must be divisible by patch_size={self.patch_size}"
            )

        n_patches = N // self.patch_size

        x = x.reshape(B, n_patches, self.patch_size).astype(self.data_type)

        x = nn.Dense(
            self.embed_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
            kernel_init=nn.initializers.xavier_uniform(),
            name="patch_embed",
        )(x)

        pos_emb = self.param(
            "pos_emb",
            normal(stddev=0.02),
            (n_patches, self.embed_dim),
            self.data_type,
        )
        x = x + pos_emb[None, :, :]

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
