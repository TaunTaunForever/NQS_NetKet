from __future__ import annotations
import typing
import flax.struct
import jax
import jax.numpy as jnp
try:
    import netket as nk
    _Sampler = nk.sampler.Sampler
except ImportError:  # allows standalone utility tests without NetKet
    _Sampler = object
from .importance_weights import compute_target_logabs2, stable_log_weights, estimate_reverse_kl_proxy
from .resampling import resample_batch

@flax.struct.dataclass
class NISSamplerState:
    key: jax.Array
    proposal_state: typing.Any = None
    last_sigma_prop: jax.Array | None = None
    last_logq_prop: jax.Array | None = None
    last_logp_prop: jax.Array | None = None
    last_weights: jax.Array | None = None
    last_metrics: dict | None = None
    iteration: int = 0
    fallback_count: int = 0

def sample_proposals_and_metrics(apply_fun, target_variables, proposal_wrapper, proposal_variables, key, *, n_proposals, exact_logZ=None, clip_weights=False, max_weight_ratio=None):
    key, draw_key = jax.random.split(key)
    sigma, logq = proposal_wrapper.sample_with_log_prob(proposal_variables, draw_key, n_proposals)
    logpsi = apply_fun(target_variables, sigma)
    logp = 2.0 * jnp.real(jnp.asarray(logpsi))
    info = stable_log_weights(logp, logq, exact_logZ=exact_logZ, clip_weights=clip_weights, max_weight_ratio=max_weight_ratio)
    if not bool(jnp.all(jnp.isfinite(logp))) or not bool(jnp.all(jnp.isfinite(logq))):
        raise FloatingPointError("NIS received non-finite target or proposal log probabilities")
    metrics = dict(info)
    metrics["KL_estimate"] = estimate_reverse_kl_proxy(logp, logq, info["weights_normalized"])
    return {"key": key, "sigma_prop": sigma, "logq_prop": logq,
            "logpsi_prop": logpsi, "logp_tilde_prop": logp,
            "weight_info": info, "metrics": metrics}

class NeuralImportanceSampler(_Sampler):
    """Compatibility sampler; use :class:`WeightedNISState` for unbiased gradients."""
    proposal_model: typing.Any
    proposal_variables: typing.Any
    proposal_wrapper: typing.Any
    # Backing storage for NetKet's required ``n_chains`` property.
    _n_chains: int
    n_proposals: int = 4096
    resample_size: int = 1024
    resample_method: str = "systematic"
    ESS_threshold: float = .2
    resample_interval: int = 1
    clip_weights: bool = False
    max_weight_ratio: float | None = None
    allow_mcmc_fallback: bool = False
    debug_mode: bool = True

    def __init__(self, hilbert, proposal_model, proposal_variables, proposal_wrapper, *,
                 n_proposals=4096, resample_size=1024, resample_method="systematic",
                 ESS_threshold=.2, resample_interval=1, clip_weights=False,
                 max_weight_ratio=None, allow_mcmc_fallback=False, debug_mode=True,
                 n_chains=1, machine_pow=2, dtype=None):
        # NetKet's base class owns Hilbert/dtype validation and public sampling API.
        if _Sampler is not object:
            super().__init__(hilbert, machine_pow=machine_pow, dtype=dtype)
        else:
            self.hilbert, self.machine_pow, self.dtype = hilbert, machine_pow, dtype
        self.proposal_model, self.proposal_variables, self.proposal_wrapper = proposal_model, proposal_variables, proposal_wrapper
        self.n_proposals, self.resample_size, self.resample_method = int(n_proposals), int(resample_size), resample_method
        self.ESS_threshold, self.resample_interval = float(ESS_threshold), int(resample_interval)
        self.clip_weights, self.max_weight_ratio = bool(clip_weights), max_weight_ratio
        self.allow_mcmc_fallback, self.debug_mode, self._n_chains = bool(allow_mcmc_fallback), bool(debug_mode), int(n_chains)
        if self.n_proposals < 1 or self.resample_size < 1 or self._n_chains < 1:
            raise ValueError("n_proposals, resample_size, and n_chains must be positive")

    @property
    def n_chains(self):
        return self._n_chains

    def _init_state(self, machine, parameters, seed):
        return NISSamplerState(key=jax.random.PRNGKey(seed) if isinstance(seed, int) else seed)
    def _reset(self, machine, parameters, state): return state
    def _sample_chain(self, machine, parameters, state, chain_length, *, return_log_probabilities=False):
        data = sample_proposals_and_metrics(machine.apply, parameters, self.proposal_wrapper, self.proposal_variables, state.key,
            n_proposals=self.n_proposals, clip_weights=self.clip_weights, max_weight_ratio=self.max_weight_ratio)
        key, rkey = jax.random.split(data["key"])
        n = chain_length * getattr(self, "n_chains_per_rank", 1)
        samples, _ = resample_batch(rkey, data["sigma_prop"], data["weight_info"]["weights_normalized"], n_samples=n, method=self.resample_method)
        samples = samples.reshape(chain_length, -1, samples.shape[-1])
        new = state.replace(key=key, last_sigma_prop=data["sigma_prop"], last_logq_prop=data["logq_prop"], last_logp_prop=data["logp_tilde_prop"], last_weights=data["weight_info"]["weights_normalized"], last_metrics=data["metrics"], iteration=state.iteration+1)
        if return_log_probabilities:
            return (samples, compute_target_logabs2(machine.apply, parameters, samples.reshape(-1,samples.shape[-1])).reshape(samples.shape[:-1])), new
        return samples, new
