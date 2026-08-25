"""Log-domain importance weights.  Target probabilities are always |psi|**2."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def compute_target_logabs2(apply_fun, variables, sigma: jax.Array) -> jax.Array:
    logpsi = apply_fun(variables, sigma)
    return 2.0 * jnp.real(jnp.asarray(logpsi))


def effective_sample_size(weights_normalized: jax.Array) -> jax.Array:
    w = jnp.asarray(weights_normalized, dtype=jnp.float64)
    return 1.0 / jnp.sum(jnp.square(w))


def stable_log_weights(logp_tilde, logq, *, exact_logZ=None, clip_weights=False,
                       max_weight_ratio=None) -> dict:
    """Return raw OIS weights when Z is known and normalized SNIS weights always.

    Clipping is performed in log space and both clipped and unclipped extrema are
    reported.  It is deliberately opt-in because it biases an estimator.
    """
    lp, lq = (jnp.asarray(x, dtype=jnp.float64) for x in (logp_tilde, logq))
    if lp.shape != lq.shape or lp.ndim != 1:
        raise ValueError("logp_tilde and logq must be matching one-dimensional arrays")
    if max_weight_ratio is not None and max_weight_ratio < 1:
        raise ValueError("max_weight_ratio must be >= 1")
    raw = lp - lq - (0.0 if exact_logZ is None else jnp.asarray(exact_logZ))
    raw_max, raw_min = jnp.max(raw), jnp.min(raw)
    shifted = raw - raw_max
    unclipped_ratio = jnp.exp(jnp.minimum(raw_max - raw_min, 700.0))
    clipped = bool(clip_weights and max_weight_ratio is not None)
    used = jnp.maximum(raw, raw_max - jnp.log(max_weight_ratio)) if clipped else raw
    log_norm = logsumexp(used)
    wn = jnp.exp(used - log_norm)
    # For OIS these are the actual unbiased weights; SNIS has no known scale.
    weights = jnp.exp(jnp.clip(used, -745.0, 709.0)) if exact_logZ is not None else wn
    ess = effective_sample_size(wn)
    return {"logw_raw": raw, "logw_shifted": used - jnp.max(used), "weights": weights,
            "weights_normalized": wn, "ess": ess, "ess_frac": ess / raw.size,
            "logw_max": raw_max, "logw_min": raw_min, "logw_mean": jnp.mean(raw),
            "logw_var": jnp.var(raw), "max_weight_ratio_observed": unclipped_ratio,
            "clipped": jnp.asarray(clipped)}


def stable_log_weights_sharded(
    logp_tilde,
    logq,
    *,
    axis_name: str = "S",
    clip_weights: bool = False,
    max_weight_ratio=None,
) -> dict:
    """Globally normalize log-domain importance weights inside a JAX shard map.

    ``logp_tilde`` and ``logq`` contain one local sample shard.  All returned
    sample-shaped arrays remain local to that shard, while scalar diagnostics
    are replicated across the named mesh.  This is the SNIS-only counterpart
    of :func:`stable_log_weights`; an exact OIS normalizer is deliberately not
    supported because the weighted-NIS production path never supplies one.
    """
    lp, lq = (jnp.asarray(value, dtype=jnp.float64) for value in (logp_tilde, logq))
    if lp.shape != lq.shape or lp.ndim != 1:
        raise ValueError("logp_tilde and logq must be matching one-dimensional arrays")
    if max_weight_ratio is not None and max_weight_ratio < 1:
        raise ValueError("max_weight_ratio must be >= 1")

    raw = lp - lq
    raw_max = jax.lax.pmax(jnp.max(raw), axis_name)
    raw_min = jax.lax.pmin(jnp.min(raw), axis_name)
    clipped = bool(clip_weights and max_weight_ratio is not None)
    used = jnp.maximum(raw, raw_max - jnp.log(max_weight_ratio)) if clipped else raw
    used_max = jax.lax.pmax(jnp.max(used), axis_name)
    shifted = jnp.exp(used - used_max)
    normalizer = jax.lax.psum(jnp.sum(shifted), axis_name)
    weights_normalized = shifted / normalizer

    n_total = jax.lax.psum(jnp.asarray(raw.size, dtype=jnp.float64), axis_name)
    raw_mean = jax.lax.psum(jnp.sum(raw), axis_name) / n_total
    raw_var = jax.lax.psum(jnp.sum(jnp.square(raw - raw_mean)), axis_name) / n_total
    ess = 1.0 / jax.lax.psum(jnp.sum(jnp.square(weights_normalized)), axis_name)
    finite = jax.lax.pmin(
        jnp.asarray(jnp.all(jnp.isfinite(lp)) & jnp.all(jnp.isfinite(lq)), dtype=jnp.int32),
        axis_name,
    )
    return {
        "logw_raw": raw,
        "logw_shifted": used - used_max,
        "weights": weights_normalized,
        "weights_normalized": weights_normalized,
        "ess": ess,
        "ess_frac": ess / n_total,
        "logw_max": raw_max,
        "logw_min": raw_min,
        "logw_mean": raw_mean,
        "logw_var": raw_var,
        "max_weight_ratio_observed": jnp.exp(jnp.minimum(raw_max - raw_min, 700.0)),
        "clipped": jnp.asarray(clipped),
        "finite": finite,
    }


def estimate_reverse_kl_proxy(logp_tilde, logq, weights_normalized):
    """SNIS estimate of E_p[log p - log q], with p normalized empirically."""
    logp = jnp.asarray(logp_tilde, dtype=jnp.float64)
    return jnp.sum(jnp.asarray(weights_normalized) *
                   ((logp - logsumexp(logp)) - jnp.asarray(logq)))
