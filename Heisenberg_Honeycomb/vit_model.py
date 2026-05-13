from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import normal


class MLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, dtype=jnp.float64, param_dtype=jnp.float64)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, dtype=jnp.float64, param_dtype=jnp.float64)(x)
        return x


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_hidden: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            deterministic=True,
            dtype=jnp.float64,
            param_dtype=jnp.float64,
        )(y)
        x = x + y

        y = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)(x)
        y = MLP(self.mlp_hidden, self.embed_dim)(y)
        return x + y


class PatchViT(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden: int
    patch_size: int
    learn_phase: bool

    @nn.compact
    def __call__(self, sigma):
        batch_size, num_sites = sigma.shape
        if num_sites % self.patch_size != 0:
            raise ValueError("NUM_SITES must be divisible by PATCH_SIZE.")

        n_patches = num_sites // self.patch_size
        x = sigma.reshape(batch_size, n_patches, self.patch_size).astype(jnp.float64)
        x = nn.Dense(self.embed_dim, dtype=jnp.float64, param_dtype=jnp.float64)(x)

        pos_emb = self.param("pos_emb", normal(stddev=0.02), (n_patches, self.embed_dim), jnp.float64)
        x = x + pos_emb

        for _ in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden=self.mlp_hidden,
            )(x)

        x = jnp.mean(x, axis=1)
        log_amp = nn.Dense(1, name="amp_head", dtype=jnp.float64, param_dtype=jnp.float64)(x).squeeze(-1)
        log_phase = nn.Dense(1, name="phase_head", dtype=jnp.float64, param_dtype=jnp.float64)(x).squeeze(-1)

        if self.learn_phase:
            return log_amp + 1j * log_phase
        return log_amp

