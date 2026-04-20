from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


def spins_to_tokens(spins):
    spins = jnp.asarray(spins)
    return ((spins + 1) // 2).astype(jnp.int32)


def tokens_to_spins(tokens):
    tokens = jnp.asarray(tokens)
    return (2 * tokens - 1).astype(jnp.float64)


class ProposalTransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_hidden_dim: int

    @nn.compact
    def __call__(self, x, *, attention_mask):
        y = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            deterministic=True,
            dtype=jnp.float64,
            param_dtype=jnp.float64,
        )(y, mask=attention_mask)
        x = x + y

        y = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)(x)
        y = nn.Dense(self.mlp_hidden_dim, dtype=jnp.float64, param_dtype=jnp.float64)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.embed_dim, dtype=jnp.float64, param_dtype=jnp.float64)(y)
        x = x + y
        return x


class AutoregressiveProposalNet(nn.Module):
    n_sites: int
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int

    @nn.compact
    def __call__(self, sigma):
        tokens = spins_to_tokens(sigma)
        shifted = tokens[:, :-1]
        attention_mask = nn.make_causal_mask(
            jnp.ones((sigma.shape[0], self.n_sites), dtype=jnp.bool_)
        )

        tok_emb = nn.Embed(
            num_embeddings=2,
            features=self.embed_dim,
            dtype=jnp.float64,
            param_dtype=jnp.float64,
        )(shifted)
        bos_emb = self.param(
            "bos_emb",
            nn.initializers.normal(stddev=0.02),
            (self.embed_dim,),
            jnp.float64,
        )
        bos = jnp.broadcast_to(
            bos_emb[None, None, :], (sigma.shape[0], 1, self.embed_dim)
        )
        x = jnp.concatenate([bos, tok_emb], axis=1)
        pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (self.n_sites, self.embed_dim),
            jnp.float64,
        )
        x = x + pos_emb[None, :, :]

        for layer in range(self.num_layers):
            x = ProposalTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                name=f"ProposalTransformerBlock_{layer}",
            )(x, attention_mask=attention_mask)

        logits = nn.Dense(2, dtype=jnp.float64, param_dtype=jnp.float64)(x)
        return logits


def proposal_probs_from_logits(logits, prob_floor):
    probs = nn.softmax(logits, axis=-1)
    probs = jnp.clip(probs, prob_floor, 1.0)
    probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
    return probs


def proposal_log_prob(model, params, sigma, *, prob_floor):
    logits = model.apply({"params": params}, sigma)
    probs = proposal_probs_from_logits(logits, prob_floor)
    tokens = spins_to_tokens(sigma)
    gathered = jnp.take_along_axis(probs, tokens[..., None], axis=-1)[..., 0]
    return jnp.sum(jnp.log(gathered), axis=-1)


def sample_from_proposal(model, params, rng, n_samples, n_sites, *, prob_floor):
    sigma = jnp.ones((n_samples, n_sites), dtype=jnp.float64)
    for site in range(n_sites):
        logits = model.apply({"params": params}, sigma)
        probs = proposal_probs_from_logits(logits[:, site, :], prob_floor)
        rng, subkey = jax.random.split(rng)
        token = jax.random.categorical(subkey, jnp.log(probs), axis=-1)
        spin = tokens_to_spins(token)
        sigma = sigma.at[:, site].set(spin)
    return sigma, rng


def forward_kl_loss(model, params, samples, *, prob_floor):
    return -jnp.mean(proposal_log_prob(model, params, samples, prob_floor=prob_floor))


def train_proposal_step(model, params, opt_state, optimizer, samples, *, prob_floor):
    loss_fn = lambda p: forward_kl_loss(model, p, samples, prob_floor=prob_floor)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
