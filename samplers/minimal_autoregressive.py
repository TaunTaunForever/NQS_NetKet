"""Small causal binary proposal used by NIS debug launchers."""
from __future__ import annotations
import flax.linen as nn
import jax.numpy as jnp

class MinimalAutoregressiveTransformer(nn.Module):
    n_sites: int
    embed_dim: int = 16
    num_heads: int = 2
    num_layers: int = 2
    probability_floor: float = 1e-8

    @nn.compact
    def __call__(self, sigma):
        tokens = ((jnp.asarray(sigma) + 1) // 2).astype(jnp.int32)
        shifted = tokens[:, :-1]
        x = nn.Embed(2, self.embed_dim, dtype=jnp.float64)(shifted)
        bos = self.param("bos", nn.initializers.normal(.02), (self.embed_dim,), jnp.float64)
        x = jnp.concatenate([jnp.broadcast_to(bos, (tokens.shape[0], 1, self.embed_dim)), x], axis=1)
        position = self.param("position", nn.initializers.normal(.02), (self.n_sites, self.embed_dim), jnp.float64)
        x = x + position[None]
        mask = nn.make_causal_mask(jnp.ones((tokens.shape[0], self.n_sites), dtype=jnp.bool_))
        for i in range(self.num_layers):
            y = nn.LayerNorm(dtype=jnp.float64, name=f"ln_a_{i}")(x)
            y = nn.SelfAttention(self.num_heads, dtype=jnp.float64, name=f"attn_{i}")(y, mask=mask)
            x = x + y
            y = nn.LayerNorm(dtype=jnp.float64, name=f"ln_b_{i}")(x)
            y = nn.Dense(2 * self.embed_dim, dtype=jnp.float64, name=f"ff1_{i}")(y)
            y = nn.gelu(y)
            x = x + nn.Dense(self.embed_dim, dtype=jnp.float64, name=f"ff2_{i}")(y)
        return nn.Dense(2, dtype=jnp.float64, name="logits")(x)
