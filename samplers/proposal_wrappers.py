from __future__ import annotations
from typing import Protocol
import jax
import jax.numpy as jnp

class ProposalModelProtocol(Protocol):
    def sample_with_log_prob(self, variables, key, batch_size: int): ...
    def log_prob(self, variables, sigma: jax.Array): ...

class AutoregressiveProposalWrapper:
    """Adapter for repository AR models exposing ``sample``/``log_prob`` or logits."""
    def __init__(self, hilbert, model, dtype=jnp.int8, probability_floor=1e-8,
                 temperature=1.0):
        if probability_floor < 0 or probability_floor >= .5:
            raise ValueError("probability_floor must be in [0, .5)")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.hilbert, self.model, self.dtype = hilbert, model, dtype
        self.probability_floor, self.temperature = probability_floor, temperature
        self._causal_sampler_cache = {}

    def _log_probs_from_logits(self, logits):
        logits = jnp.asarray(logits, dtype=jnp.float64) / self.temperature
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        if self.probability_floor:
            probs = jnp.maximum(jnp.exp(log_probs), self.probability_floor)
            log_probs = jnp.log(probs / jnp.sum(probs, axis=-1, keepdims=True))
        return log_probs

    def log_prob(self, variables, sigma):
        if hasattr(self.model, "log_prob"):
            return jnp.asarray(self.model.log_prob(variables, sigma), dtype=jnp.float64)
        logits = self.model.apply(variables, sigma)
        tokens = ((jnp.asarray(sigma) + 1) // 2).astype(jnp.int32)
        return jnp.sum(self._log_probs_from_logits(logits)[jnp.arange(tokens.shape[0])[:, None],
                                                       jnp.arange(tokens.shape[1]), tokens], axis=-1)

    def _sample_from_logits(self, variables, key, batch_size):
        """Causal AR sampling without an inner ``jax.jit`` wrapper.

        The normal single-device path wraps this fixed-shape computation in a
        cached JIT below.  NetKet's native sharding invokes this version inside
        its outer ``shard_map`` so each GPU decodes its own proposal shard
        without a nested JIT/mesh transition.
        """
        sigma = -jnp.ones((batch_size, self.hilbert.size), dtype=self.dtype)
        logq = jnp.zeros((batch_size,), dtype=jnp.float64)
        site_keys = jax.random.split(key, self.hilbert.size)
        for site, site_key in enumerate(site_keys):
            logits = jnp.asarray(self.model.apply(variables, sigma), dtype=jnp.float64)
            log_probs = self._log_probs_from_logits(logits)
            token = jax.random.categorical(site_key, log_probs[:, site, :])
            logq = logq + jnp.take_along_axis(
                log_probs[:, site, :], token[:, None], axis=-1
            )[:, 0]
            sigma = sigma.at[:, site].set(jnp.asarray(2 * token - 1, dtype=self.dtype))
        return sigma, logq

    def sample_with_log_prob_uncompiled(self, variables, key, batch_size):
        """Sample proposals without creating a nested JIT.

        This is used by the NetKet native-sharding path.  It retains support
        for proposal models that provide their own sampling method.
        """
        if hasattr(self.model, "sample"):
            sigma = self.model.sample(variables, key, batch_size)
            sigma = jnp.asarray(sigma, self.dtype)
            return sigma, self.log_prob(variables, sigma)
        if hasattr(self.model, "sample_with_log_prob"):
            return self.model.sample_with_log_prob(variables, key, batch_size)
        return self._sample_from_logits(variables, key, batch_size)

    def sample_with_log_prob(self, variables, key, batch_size):
        if hasattr(self.model, "sample"):
            sigma = self.model.sample(variables, key, batch_size)
        elif hasattr(self.model, "sample_with_log_prob"):
            return self.model.sample_with_log_prob(variables, key, batch_size)
        else:
            # Compile one fixed-size causal decode for each requested batch shape.
            # Accumulating logq here avoids a ninth full proposal forward pass.
            if batch_size not in self._causal_sampler_cache:
                @jax.jit
                def causal_sampler(model_variables, sample_key):
                    return self._sample_from_logits(model_variables, sample_key, batch_size)

                self._causal_sampler_cache[batch_size] = causal_sampler
            return self._causal_sampler_cache[batch_size](variables, key)
        sigma = jnp.asarray(sigma, self.dtype)
        return sigma, self.log_prob(variables, sigma)
