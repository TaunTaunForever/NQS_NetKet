from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _as_float64(values):
    return jnp.asarray(values, dtype=jnp.float64)


def _normalise_weights(weights):
    weights = _as_float64(weights)
    weights = weights / jnp.sum(weights)
    return weights


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


def _resample_multinomial(weights, n_samples, rng):
    safe_log_weights = jnp.log(jnp.clip(weights, jnp.finfo(weights.dtype).tiny, 1.0))
    return jax.random.categorical(rng, safe_log_weights, shape=(n_samples,))


def _resample_systematic(weights, n_samples, rng):
    u0 = jax.random.uniform(
        rng,
        shape=(),
        minval=0.0,
        maxval=1.0 / n_samples,
        dtype=weights.dtype,
    )
    positions = u0 + jnp.arange(n_samples, dtype=weights.dtype) / n_samples
    cdf = jnp.cumsum(weights)
    cdf = cdf.at[-1].set(1.0)
    return jnp.searchsorted(cdf, positions, side="right")


def _resample_stratified(weights, n_samples, rng):
    offsets = jax.random.uniform(
        rng,
        shape=(n_samples,),
        minval=0.0,
        maxval=1.0,
        dtype=weights.dtype,
    )
    positions = (jnp.arange(n_samples, dtype=weights.dtype) + offsets) / n_samples
    cdf = jnp.cumsum(weights)
    cdf = cdf.at[-1].set(1.0)
    return jnp.searchsorted(cdf, positions, side="right")


@functools.partial(jax.jit, static_argnames=("n_samples", "method"))
def importance_resample(
    configs,
    log_target_probs,
    log_proposal_probs,
    *,
    n_samples: int,
    method: str = "multinomial",
    rng,
):
    configs = jnp.asarray(configs)
    weights = _normalise_weights(
        normalised_importance_weights_from_log_probs(log_target_probs, log_proposal_probs)
    )
    rng, subkey = jax.random.split(rng)
    if method == "multinomial":
        indices = _resample_multinomial(weights, n_samples, subkey)
    elif method == "systematic":
        indices = _resample_systematic(weights, n_samples, subkey)
    elif method == "stratified":
        indices = _resample_stratified(weights, n_samples, subkey)
    else:
        raise ValueError(
            f"Unsupported importance resampling method {method!r}; "
            "expected 'multinomial', 'systematic', or 'stratified'."
        )
    return configs[indices], indices, weights, rng
