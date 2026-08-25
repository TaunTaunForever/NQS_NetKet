from __future__ import annotations

import functools
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

PROPOSAL_DTYPE = jnp.float32


def spins_to_tokens(spins):
    spins = jnp.asarray(spins)
    return ((spins + 1) // 2).astype(jnp.int32)


def tokens_to_spins(tokens):
    tokens = jnp.asarray(tokens)
    return (2 * tokens - 1).astype(jnp.float64)


def _ordered_tuple(values, order):
    if values is None:
        return None
    if order is None:
        return tuple(values)
    return tuple(values[index] for index in order)


def _ordered_relation_matrix(relation_matrix, order):
    if relation_matrix is None:
        return None
    if order is None:
        return tuple(tuple(row) for row in relation_matrix)
    return tuple(
        tuple(relation_matrix[row_index][column_index] for column_index in order)
        for row_index in order
    )


def _ordered_tokens(model, sigma):
    tokens = spins_to_tokens(sigma)
    if model.site_order is not None:
        tokens = tokens[:, jnp.asarray(model.site_order, dtype=jnp.int32)]
    return tokens


class CausalGraphContext(nn.Module):
    relation_matrix: Tuple[Tuple[int, ...], ...]
    site_type_ids: Tuple[int, ...]
    embed_dim: int

    @nn.compact
    def __call__(self, tokens, *, position=None):
        relation = jnp.asarray(self.relation_matrix, dtype=jnp.int32)
        site_types = jnp.asarray(self.site_type_ids, dtype=jnp.int32)
        num_relation_types = max(max(row) for row in self.relation_matrix) + 1
        num_site_types = max(self.site_type_ids) + 1

        relation_spin_embedding = nn.Embed(
            num_embeddings=2 * num_relation_types,
            features=self.embed_dim,
            dtype=PROPOSAL_DTYPE,
            param_dtype=PROPOSAL_DTYPE,
            name="relation_spin_embedding",
        )
        site_type_embedding = nn.Embed(
            num_embeddings=num_site_types,
            features=self.embed_dim,
            dtype=PROPOSAL_DTYPE,
            param_dtype=PROPOSAL_DTYPE,
            name="site_type_embedding",
        )

        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        n_sites = relation.shape[0]
        if position is None:
            relation_token_ids = 2 * relation[None, :, :] + tokens[:, None, :]
            messages = relation_spin_embedding(relation_token_ids)
            causal_mask = jnp.tril(
                jnp.ones((n_sites, n_sites), dtype=PROPOSAL_DTYPE),
                k=-1,
            )
            normalizer = jnp.maximum(
                jnp.arange(n_sites, dtype=PROPOSAL_DTYPE),
                1.0,
            )
            context = jnp.sum(
                messages * causal_mask[None, :, :, None],
                axis=2,
            )
            context = context / normalizer[None, :, None]
            return context + site_type_embedding(site_types)[None, :, :]

        relation_token_ids = 2 * relation[position][None, :] + tokens
        messages = relation_spin_embedding(relation_token_ids)
        causal_mask = (
            jnp.arange(n_sites, dtype=jnp.int32) < jnp.asarray(position)
        ).astype(PROPOSAL_DTYPE)
        normalizer = jnp.maximum(
            jnp.asarray(position, dtype=PROPOSAL_DTYPE),
            1.0,
        )
        context = jnp.sum(messages * causal_mask[None, :, None], axis=1)
        context = context / normalizer
        return context + site_type_embedding(site_types[position])[None, :]


class ProposalTransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_hidden_dim: int
    decode: bool = False

    @nn.compact
    def __call__(self, x, *, attention_mask=None):
        y = nn.LayerNorm(dtype=PROPOSAL_DTYPE, param_dtype=PROPOSAL_DTYPE)(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            deterministic=True,
            dtype=PROPOSAL_DTYPE,
            param_dtype=PROPOSAL_DTYPE,
            decode=self.decode,
        )(y, mask=attention_mask)
        x = x + y

        y = nn.LayerNorm(dtype=PROPOSAL_DTYPE, param_dtype=PROPOSAL_DTYPE)(x)
        y = nn.Dense(self.mlp_hidden_dim, dtype=PROPOSAL_DTYPE, param_dtype=PROPOSAL_DTYPE)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.embed_dim, dtype=PROPOSAL_DTYPE, param_dtype=PROPOSAL_DTYPE)(y)
        x = x + y
        return x


class AutoregressiveProposalNet(nn.Module):
    n_sites: int
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    relation_matrix: Optional[Tuple[Tuple[int, ...], ...]] = None
    site_type_ids: Optional[Tuple[int, ...]] = None
    site_order: Optional[Tuple[int, ...]] = None

    @nn.compact
    def __call__(self, sigma):
        tokens = _ordered_tokens(self, sigma)
        shifted = tokens[:, :-1]
        attention_mask = nn.make_causal_mask(jnp.ones((sigma.shape[0], self.n_sites), dtype=jnp.bool_))

        tok_emb = nn.Embed(
            num_embeddings=2,
            features=self.embed_dim,
            dtype=PROPOSAL_DTYPE,
            param_dtype=PROPOSAL_DTYPE,
        )(shifted)
        bos_emb = self.param(
            "bos_emb",
            nn.initializers.normal(stddev=0.02),
            (self.embed_dim,),
            PROPOSAL_DTYPE,
        )
        bos = jnp.broadcast_to(bos_emb[None, None, :], (sigma.shape[0], 1, self.embed_dim))
        x = jnp.concatenate([bos, tok_emb], axis=1)
        pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (self.n_sites, self.embed_dim),
            PROPOSAL_DTYPE,
        )
        x = x + pos_emb[None, :, :]
        if self.relation_matrix is not None and self.site_type_ids is not None:
            relation_matrix = _ordered_relation_matrix(
                self.relation_matrix,
                self.site_order,
            )
            site_type_ids = _ordered_tuple(self.site_type_ids, self.site_order)
            x = x + CausalGraphContext(
                relation_matrix=relation_matrix,
                site_type_ids=site_type_ids,
                embed_dim=self.embed_dim,
                name="CausalGraphContext",
            )(tokens)

        for layer in range(self.num_layers):
            x = ProposalTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                decode=False,
                name=f"ProposalTransformerBlock_{layer}",
            )(x, attention_mask=attention_mask)

        logits = nn.Dense(2, dtype=PROPOSAL_DTYPE, param_dtype=PROPOSAL_DTYPE)(x)
        return logits


class AutoregressiveProposalDecodeNet(nn.Module):
    n_sites: int
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    relation_matrix: Optional[Tuple[Tuple[int, ...], ...]] = None
    site_type_ids: Optional[Tuple[int, ...]] = None
    site_order: Optional[Tuple[int, ...]] = None

    @nn.compact
    def init_decode_cache(self, dummy_tokens):
        batch_size = dummy_tokens.shape[0]
        tok_emb = nn.Embed(
            num_embeddings=2,
            features=self.embed_dim,
            dtype=PROPOSAL_DTYPE,
            param_dtype=PROPOSAL_DTYPE,
        )(dummy_tokens)
        bos_emb = self.param(
            "bos_emb",
            nn.initializers.normal(stddev=0.02),
            (self.embed_dim,),
            PROPOSAL_DTYPE,
        )
        x = tok_emb.at[:, 0, :].set(bos_emb)
        pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (self.n_sites, self.embed_dim),
            PROPOSAL_DTYPE,
        )
        x = x + pos_emb[None, :, :]
        if self.relation_matrix is not None and self.site_type_ids is not None:
            relation_matrix = _ordered_relation_matrix(
                self.relation_matrix,
                self.site_order,
            )
            site_type_ids = _ordered_tuple(self.site_type_ids, self.site_order)
            x = x + CausalGraphContext(
                relation_matrix=relation_matrix,
                site_type_ids=site_type_ids,
                embed_dim=self.embed_dim,
                name="CausalGraphContext",
            )(dummy_tokens)

        for layer in range(self.num_layers):
            x = ProposalTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                decode=True,
                name=f"ProposalTransformerBlock_{layer}",
            )(x)

        logits = nn.Dense(2, dtype=PROPOSAL_DTYPE, param_dtype=PROPOSAL_DTYPE)(x)
        return logits

    @nn.compact
    def decode_step(self, prev_tokens, history_tokens, position):
        prev_tokens = jnp.asarray(prev_tokens, dtype=jnp.int32)
        tok_emb = nn.Embed(
            num_embeddings=2,
            features=self.embed_dim,
            dtype=PROPOSAL_DTYPE,
            param_dtype=PROPOSAL_DTYPE,
        )(prev_tokens[:, None])
        bos_emb = self.param(
            "bos_emb",
            nn.initializers.normal(stddev=0.02),
            (self.embed_dim,),
            PROPOSAL_DTYPE,
        )
        x = jnp.where(
            jnp.asarray(position == 0)[None, None, None],
            bos_emb[None, None, :],
            tok_emb,
        )
        pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (self.n_sites, self.embed_dim),
            PROPOSAL_DTYPE,
        )
        x = x + pos_emb[position][None, None, :]
        if self.relation_matrix is not None and self.site_type_ids is not None:
            relation_matrix = _ordered_relation_matrix(
                self.relation_matrix,
                self.site_order,
            )
            site_type_ids = _ordered_tuple(self.site_type_ids, self.site_order)
            graph_context = CausalGraphContext(
                relation_matrix=relation_matrix,
                site_type_ids=site_type_ids,
                embed_dim=self.embed_dim,
                name="CausalGraphContext",
            )(history_tokens, position=position)
            x = x + graph_context[:, None, :]

        for layer in range(self.num_layers):
            x = ProposalTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                decode=True,
                name=f"ProposalTransformerBlock_{layer}",
            )(x)

        logits = nn.Dense(2, dtype=PROPOSAL_DTYPE, param_dtype=PROPOSAL_DTYPE)(x)
        return logits[:, 0, :]


def proposal_probs_from_logits(logits, prob_floor):
    probs = nn.softmax(logits, axis=-1)
    probs = jnp.clip(probs, prob_floor, 1.0)
    probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
    return probs


def fixed_magnetization_token_mask(tokens, n_up):
    if n_up is None:
        return None
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    n_sites = tokens.shape[-1]
    positions = jnp.arange(n_sites, dtype=jnp.int32)
    previous_up = jnp.cumsum(tokens, axis=-1) - tokens
    up_remaining = jnp.asarray(n_up, dtype=jnp.int32) - previous_up
    sites_after = n_sites - positions - 1
    allow_down = up_remaining <= sites_after
    allow_up = up_remaining > 0
    return jnp.stack([allow_down, allow_up], axis=-1)


def constrained_proposal_probs_from_logits(logits, prob_floor, token_mask=None):
    probs = nn.softmax(logits, axis=-1)
    if token_mask is None:
        probs = jnp.clip(probs, prob_floor, 1.0)
    else:
        probs = jnp.where(token_mask, jnp.clip(probs, prob_floor, 1.0), 0.0)
    norm = jnp.sum(probs, axis=-1, keepdims=True)
    probs = probs / jnp.maximum(norm, jnp.finfo(probs.dtype).tiny)
    return probs


@functools.partial(jax.jit, static_argnames=("model", "prob_floor"))
def proposal_log_prob(model, params, sigma, *, prob_floor):
    logits = model.apply({"params": params}, sigma)
    probs = proposal_probs_from_logits(logits, prob_floor)
    tokens = _ordered_tokens(model, sigma)
    gathered = jnp.take_along_axis(probs, tokens[..., None], axis=-1)[..., 0]
    return jnp.asarray(jnp.sum(jnp.log(gathered), axis=-1), dtype=jnp.float64)


@functools.partial(jax.jit, static_argnames=("model", "prob_floor", "n_up"))
def proposal_log_prob_fixed_magnetization(model, params, sigma, *, prob_floor, n_up):
    logits = model.apply({"params": params}, sigma)
    tokens = _ordered_tokens(model, sigma)
    token_mask = fixed_magnetization_token_mask(tokens, n_up)
    probs = constrained_proposal_probs_from_logits(logits, prob_floor, token_mask)
    gathered = jnp.take_along_axis(probs, tokens[..., None], axis=-1)[..., 0]
    return jnp.asarray(jnp.sum(jnp.log(gathered), axis=-1), dtype=jnp.float64)


@functools.partial(
    jax.jit,
    static_argnames=("model", "n_samples", "n_sites", "prob_floor", "n_up"),
)
def sample_from_proposal(model, params, rng, n_samples, n_sites, *, prob_floor, n_up=None):
    sigma = jnp.ones((n_samples, n_sites), dtype=jnp.float64)
    history_tokens = jnp.zeros((n_samples, n_sites), dtype=jnp.int32)
    prev_tokens = jnp.zeros((n_samples,), dtype=jnp.int32)
    up_count = jnp.zeros((n_samples,), dtype=jnp.int32)
    decode_model = AutoregressiveProposalDecodeNet(
        n_sites=model.n_sites,
        embed_dim=model.embed_dim,
        num_heads=model.num_heads,
        num_layers=model.num_layers,
        mlp_hidden_dim=model.mlp_hidden_dim,
        relation_matrix=model.relation_matrix,
        site_type_ids=model.site_type_ids,
        site_order=model.site_order,
    )
    dummy_tokens = jnp.zeros((n_samples, n_sites), dtype=jnp.int32)
    _, cache_vars = decode_model.apply(
        {"params": params},
        dummy_tokens,
        method=decode_model.init_decode_cache,
        mutable=["cache"],
    )
    cache = cache_vars["cache"]

    def sample_site(carry, site):
        sigma, history_tokens, prev_tokens, up_count, cache, rng = carry
        logits, updated = decode_model.apply(
            {"params": params, "cache": cache},
            prev_tokens,
            history_tokens,
            site,
            method=decode_model.decode_step,
            mutable=["cache"],
        )
        cache = updated["cache"]
        if n_up is None:
            token_mask = None
        else:
            up_remaining = jnp.asarray(n_up, dtype=jnp.int32) - up_count
            sites_after = n_sites - site - 1
            allow_down = up_remaining <= sites_after
            allow_up = up_remaining > 0
            token_mask = jnp.stack([allow_down, allow_up], axis=-1)
        probs = constrained_proposal_probs_from_logits(logits, prob_floor, token_mask)
        rng, subkey = jax.random.split(rng)
        token = jax.random.categorical(subkey, jnp.log(probs), axis=-1).astype(jnp.int32)
        spin = tokens_to_spins(token)
        sigma = sigma.at[:, site].set(spin)
        history_tokens = history_tokens.at[:, site].set(token)
        up_count = up_count + token
        return (sigma, history_tokens, token, up_count, cache, rng), None

    (sigma, _history, _prev_tokens, _up_count, _cache, rng), _ = jax.lax.scan(
        sample_site,
        (sigma, history_tokens, prev_tokens, up_count, cache, rng),
        jnp.arange(n_sites),
    )
    if model.site_order is not None:
        physical_sigma = jnp.empty_like(sigma)
        physical_sigma = physical_sigma.at[
            :, jnp.asarray(model.site_order, dtype=jnp.int32)
        ].set(sigma)
        sigma = physical_sigma
    return sigma, rng


def forward_kl_loss(model, params, samples, *, prob_floor, n_up=None, sample_weights=None):
    if n_up is None:
        log_prob = proposal_log_prob(model, params, samples, prob_floor=prob_floor)
    else:
        log_prob = proposal_log_prob_fixed_magnetization(
            model,
            params,
            samples,
            prob_floor=prob_floor,
            n_up=n_up,
        )
    if sample_weights is None:
        return -jnp.mean(log_prob)
    weights = jnp.asarray(sample_weights, dtype=jnp.float64)
    weights = weights / jnp.maximum(jnp.sum(weights), jnp.finfo(weights.dtype).tiny)
    return -jnp.sum(weights * log_prob)


def train_proposal_step(
    model,
    params,
    opt_state,
    optimizer,
    samples,
    *,
    prob_floor,
    n_up=None,
    sample_weights=None,
    update_scale=1.0,
):
    loss_fn = lambda p: forward_kl_loss(
        model,
        p,
        samples,
        prob_floor=prob_floor,
        n_up=n_up,
        sample_weights=sample_weights,
    )
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    updates = jax.tree.map(
        lambda update: jnp.asarray(update_scale, dtype=update.dtype) * update,
        updates,
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
