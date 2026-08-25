from __future__ import annotations
import jax
import jax.numpy as jnp

def _cdf(weights):
    w = jnp.asarray(weights, dtype=jnp.float64)
    w = w / jnp.sum(w)
    cdf = jnp.cumsum(w).at[-1].set(1.0)
    return cdf

def systematic_resample(key, weights_normalized, n_samples: int):
    cdf = _cdf(weights_normalized)
    u0 = jax.random.uniform(key, (), minval=0., maxval=1. / n_samples)
    return jnp.searchsorted(cdf, u0 + jnp.arange(n_samples) / n_samples, side="right")

def stratified_resample(key, weights_normalized, n_samples: int):
    cdf = _cdf(weights_normalized)
    u = jax.random.uniform(key, (n_samples,))
    return jnp.searchsorted(cdf, (jnp.arange(n_samples) + u) / n_samples, side="right")

def resample_batch(key, sigma, weights_normalized, *, n_samples: int, method="systematic"):
    methods = {"systematic": systematic_resample, "stratified": stratified_resample}
    if method not in methods: raise ValueError("method must be 'systematic' or 'stratified'")
    idx = methods[method](key, weights_normalized, n_samples)
    return jnp.asarray(sigma)[idx], idx
