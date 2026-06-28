from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _as_float64(values):
    return jnp.asarray(values, dtype=jnp.float64)


def normalised_importance_weights_from_log_probs(
    log_target_probs,
    log_proposal_probs,
):
    log_target_probs = _as_float64(log_target_probs)
    log_proposal_probs = _as_float64(log_proposal_probs)
    if log_target_probs.shape != log_proposal_probs.shape:
        raise ValueError("Target and proposal log-probabilities must have matching shapes.")

    log_weights = log_target_probs - log_proposal_probs
    log_norm = logsumexp(log_weights)
    return jnp.exp(log_weights - log_norm)


def effective_sample_size(weights):
    weights = _as_float64(weights)
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array.")
    return 1.0 / jnp.sum(jnp.square(weights))


def sampling_efficiency(weights):
    weights = _as_float64(weights)
    return effective_sample_size(weights) / weights.shape[0]


@functools.partial(jax.jit, static_argnames=("n_samples",))
def importance_resample(
    configs,
    log_target_probs,
    log_proposal_probs,
    *,
    n_samples: int,
    rng,
):
    configs = jnp.asarray(configs)
    weights = normalised_importance_weights_from_log_probs(log_target_probs, log_proposal_probs)
    safe_log_weights = jnp.log(jnp.clip(weights, jnp.finfo(weights.dtype).tiny, 1.0))
    rng, subkey = jax.random.split(rng)
    indices = jax.random.categorical(subkey, safe_log_weights, shape=(n_samples,))
    return configs[indices], indices, weights, rng
