from __future__ import annotations
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from samplers.importance_weights import effective_sample_size


def weighted_variational_statistics(local_estimators, weights_normalized):
    """NetKet-style expectation statistics for weighted (SNIS/OIS) samples.

    ``Mean``, ``Variance`` and ``ErrorOfMean`` mirror the fields exposed by
    ``nk.stats.Stats``.  The error estimate uses ESS in place of the raw sample
    count because NIS weights reduce the number of effective target samples.
    """
    estimators = jnp.asarray(local_estimators)
    weights = jnp.asarray(weights_normalized, dtype=jnp.float64)
    mean = jnp.sum(weights * estimators)
    variance = jnp.real(jnp.sum(weights * jnp.abs(estimators - mean) ** 2))
    ess = effective_sample_size(weights)
    return {
        "Mean": mean,
        "Variance": variance,
        "ErrorOfMean": jnp.sqrt(variance / ess),
        "ESS": ess,
    }

def compare_distributions_smallN(logp_exact, logq_exact):
    lp, lq = (jnp.asarray(x, dtype=jnp.float64) for x in (logp_exact, logq_exact))
    p, q = jnp.exp(lp-logsumexp(lp)), jnp.exp(lq-logsumexp(lq))
    m = .5*(p+q)
    klpq, klqp = jnp.sum(p*(jnp.log(p)-jnp.log(q))), jnp.sum(q*(jnp.log(q)-jnp.log(p)))
    return {"KL_pq": klpq, "KL_qp": klqp, "JS": .5*jnp.sum(p*(jnp.log(p)-jnp.log(m)))+.5*jnp.sum(q*(jnp.log(q)-jnp.log(m))),
            "TV": .5*jnp.sum(jnp.abs(p-q)), "exact_ESS_from_q_to_p": 1/jnp.sum(q*(p/q)**2)}

def summarize_weight_batch(logw_raw, weights_normalized, *, target_logp_exact=None, proposal_logq_exact=None):
    lw, w = jnp.asarray(logw_raw), jnp.asarray(weights_normalized)
    out = {"ESS": effective_sample_size(w), "ESS_frac": effective_sample_size(w)/w.size,
           "logw_min": jnp.min(lw), "logw_max": jnp.max(lw), "logw_mean": jnp.mean(lw), "logw_var": jnp.var(lw),
           "entropy_of_weights": -jnp.sum(w*jnp.log(jnp.maximum(w, jnp.finfo(w.dtype).tiny))),
           "participation_ratio": effective_sample_size(w), "top1_weight": jnp.max(w),
           "top5_weight_sum": jnp.sum(jnp.sort(w)[-min(5,w.size):]),
           "max_weight_ratio": jnp.exp(jnp.minimum(jnp.max(lw)-jnp.min(lw), 700.))}
    if target_logp_exact is not None and proposal_logq_exact is not None: out.update(compare_distributions_smallN(target_logp_exact, proposal_logq_exact))
    return out
