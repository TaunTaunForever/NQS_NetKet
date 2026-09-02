"""NetKet-facing variational state for neural importance sampling.

The samples held by this state are drawn from an autoregressive proposal, not
from ``|psi|^2``.  Consequently this is intentionally *not* an ``MCState``:
all expectation values and gradients below use their normalized importance
weights explicitly.  This keeps the usual NetKet ``Stats``/driver interface
without incorrectly presenting resampled NIS configurations as iid samples.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from flax import core as fcore
import netket as nk
import optax
from netket.jax.sharding import sharding_decorator

from diagnostics.nis_metrics import weighted_variational_statistics
from samplers.importance_weights import stable_log_weights, stable_log_weights_sharded
from samplers.neural_importance_sampling import sample_proposals_and_metrics


class WeightedNISState(nk.vqs.VariationalState):
    """A scalable, NetKet-compatible state evaluated with weighted NIS.

    Parameters
    ----------
    hilbert, model, sampler:
        The NetKet Hilbert space, the target log-amplitude model, and a
        :class:`samplers.NeuralImportanceSampler`.  The sampler supplies the
        autoregressive proposal and its variables.
    n_samples:
        Number of *proposal* configurations per weighted estimate.  It is
        retained under this familiar name for compatibility with NetKet, but it
        is not the number of iid target samples.  Inspect ``last_diagnostics``
        and especially ESS before trusting an estimate.
    chunk_size:
        Maximum number of configurations evaluated in one reverse-mode target
        gradient chunk.  It avoids constructing a batch-by-parameter Jacobian.
    local_energy_chunk_size:
        Maximum number of connected configurations in one forward evaluation
        for local energies.  This can be larger than ``chunk_size`` because it
        does not retain reverse-mode intermediates.
    use_sharding:
        Distribute the proposal pool over NetKet's native ``S`` mesh. The
        proposal count must be divisible by the number of local devices.
        Parameters remain replicated; weighted expectation values and
        gradients are reduced over the complete global pool.

    The target parameters are exposed through the usual ``parameters``
    property, so this state can be passed directly to ``nk.driver.VMC`` with a
    first-order optimizer.  Proposal updates are supplied separately by
    :class:`vqs.WeightedNISVMC`.
    """

    def __init__(
        self,
        hilbert,
        model,
        sampler,
        *,
        variables: Mapping[str, Any] | None = None,
        proposal_variables: Mapping[str, Any] | None = None,
        n_samples: int | None = None,
        seed: int | jax.Array = 0,
        chunk_size: int | None = None,
        local_energy_chunk_size: int | None = None,
        use_sharding: bool = False,
        mutable: Any = False,
        exact_enumeration_max_states: int = 2**18,
    ):
        super().__init__(hilbert)

        if sampler.hilbert != hilbert:
            raise ValueError("sampler and variational state must use the same Hilbert space")
        if not hasattr(sampler, "proposal_wrapper"):
            raise TypeError("sampler must expose an autoregressive proposal_wrapper")
        if n_samples is None:
            n_samples = sampler.n_proposals
        if int(n_samples) < 1:
            raise ValueError("n_samples must be positive")
        if chunk_size is not None and int(chunk_size) < 1:
            raise ValueError("chunk_size must be positive or None")
        if local_energy_chunk_size is not None and int(local_energy_chunk_size) < 1:
            raise ValueError("local_energy_chunk_size must be positive or None")

        self._model = model
        self.sampler = sampler
        self.mutable = mutable
        self.n_samples = int(n_samples)
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.local_energy_chunk_size = (
            self.chunk_size
            if local_energy_chunk_size is None
            else int(local_energy_chunk_size)
        )
        self.exact_enumeration_max_states = int(exact_enumeration_max_states)
        self.requested_sharding = bool(use_sharding)
        if self.requested_sharding and jax.device_count() > 1:
            if self.n_samples % jax.device_count() != 0:
                raise ValueError(
                    "n_samples must be divisible by the number of JAX devices when "
                    f"native NetKet sharding is enabled ({self.n_samples} samples, "
                    f"{jax.device_count()} devices)"
                )
        self._uses_native_sharding = (
            self.requested_sharding
            and jax.device_count() > 1
        )
        self.n_samples_per_device = (
            self.n_samples // jax.device_count()
            if self._uses_native_sharding
            else self.n_samples
        )

        self._key = jax.random.PRNGKey(seed) if isinstance(seed, int) else seed
        self._last_batch: dict[str, Any] | None = None
        self._last_local_energies: jax.Array | None = None
        self._last_statistics: nk.stats.Stats | None = None
        self._last_diagnostics: dict[str, Any] = {}

        self._key, target_init_key, proposal_init_key = jax.random.split(self._key, 3)
        # Keep held-out diagnostics on an independent PRNG stream so enabling
        # them never changes the training proposal pools or optimisation path.
        self._heldout_key = jax.random.fold_in(self._key, 0x484F4C44)
        dummy_input = self.hilbert.random_state(target_init_key, 1)

        if variables is None:
            variables = model.init({"params": target_init_key}, dummy_input)
        self.variables = variables

        if proposal_variables is None:
            proposal_variables = sampler.proposal_variables
        if proposal_variables is None:
            proposal_model = sampler.proposal_model
            proposal_variables = proposal_model.init({"params": proposal_init_key}, dummy_input)
        self._set_proposal_variables(proposal_variables)

        self._distributed_local_energy_kernels: dict[int, Any] = {}

        # Keep the differentiation function stable so it is compiled once per
        # target architecture and input shape instead of once per iteration.
        target_model = self._model

        @jax.jit
        def chunk_force(parameters, model_state, sigma, coefficients):
            def objective(params):
                log_psi = target_model.apply({"params": params, **model_state}, sigma)
                return jnp.sum(
                    jnp.real(coefficients) * jnp.real(log_psi)
                    + jnp.imag(coefficients) * jnp.imag(log_psi)
                )

            return jax.grad(objective)(parameters)

        self._chunk_force = chunk_force

        proposal_wrapper = self.sampler.proposal_wrapper

        # Retaining a stable loss/gradient closure avoids recompiling proposal
        # training inside every optimisation iteration.
        self._proposal_loss_and_grad = jax.jit(
            jax.value_and_grad(
                lambda parameters, proposal_state, training_sigma: -jnp.mean(
                    proposal_wrapper.log_prob(
                        {"params": parameters, **proposal_state}, training_sigma
                    )
                )
            )
        )

        if self._uses_native_sharding:
            self._init_native_sharding_kernels()

    @property
    def model(self):
        """The target Flax/NetKet model evaluated by this variational state."""
        return self._model

    @property
    def uses_native_sharding(self) -> bool:
        """Whether proposal pools and weighted reductions span all local GPUs."""
        return self._uses_native_sharding

    @property
    def proposal_parameters(self):
        """Trainable parameters of the autoregressive proposal."""
        return fcore.copy(self._proposal_parameters, {})

    @proposal_parameters.setter
    def proposal_parameters(self, parameters):
        self._proposal_parameters = parameters

    @property
    def proposal_model_state(self):
        return fcore.copy(self._proposal_model_state, {})

    @property
    def proposal_variables(self):
        return {"params": self.proposal_parameters, **self.proposal_model_state}

    def _set_proposal_variables(self, variables):
        if "params" not in variables:
            raise ValueError("proposal_variables must contain a 'params' entry")
        self._proposal_model_state, self._proposal_parameters = fcore.pop(variables, "params")
        # NetKet sampler instances are immutable pytrees.  The state owns the
        # evolving proposal variables and passes them explicitly to the
        # wrapper, while ``sampler.proposal_variables`` remains the immutable
        # initial compatibility snapshot for legacy MCState-style use.

    def _init_native_sharding_kernels(self):
        """Build NetKet-native SPMD kernels for the weighted proposal pool.

        Parameters remain replicated while configurations, local energies, and
        importance weights are sharded over NetKet's ``S`` mesh axis.  Every
        scalar expectation and parameter gradient is reduced explicitly, so a
        weighted estimate is identical to evaluating one global proposal pool.
        """
        target_model = self._model
        target_model_state = self.model_state
        proposal_wrapper = self.sampler.proposal_wrapper
        proposal_model_state = self.proposal_model_state
        n_per_device = self.n_samples_per_device

        def draw_proposals(proposal_variables, key):
            return proposal_wrapper.sample_with_log_prob_uncompiled(
                proposal_variables, key, n_per_device
            )

        self._distributed_draw_proposals = jax.jit(
            sharding_decorator(
                draw_proposals,
                sharded_args_tree=(False, "key"),
                reduction_op_tree=(False, False),
            )
        )

        def target_logpsi(parameters, sigma):
            return target_model.apply({"params": parameters, **target_model_state}, sigma)

        self._distributed_target_logpsi = jax.jit(
            sharding_decorator(
                target_logpsi,
                sharded_args_tree=(False, True),
                reduction_op_tree=False,
            )
        )

        weight_reductions = {
            "logw_raw": False,
            "logw_shifted": False,
            "weights": False,
            "weights_normalized": False,
            "ess": jax.lax.pmax,
            "ess_frac": jax.lax.pmax,
            "logw_max": jax.lax.pmax,
            "logw_min": jax.lax.pmax,
            "logw_mean": jax.lax.pmax,
            "logw_var": jax.lax.pmax,
            "max_weight_ratio_observed": jax.lax.pmax,
            "clipped": jax.lax.pmax,
            "finite": jax.lax.pmax,
        }

        def normalize_weights(logp, logq):
            return stable_log_weights_sharded(
                logp,
                logq,
                clip_weights=self.sampler.clip_weights,
                max_weight_ratio=self.sampler.max_weight_ratio,
            )

        self._distributed_normalize_weights = jax.jit(
            sharding_decorator(
                normalize_weights,
                sharded_args_tree=(True, True),
                reduction_op_tree=weight_reductions,
            )
        )

        def weighted_statistics(weights, local_energies):
            mean = jax.lax.psum(jnp.sum(weights * local_energies), "S")
            variance = jax.lax.psum(
                jnp.sum(weights * jnp.abs(local_energies - mean) ** 2), "S"
            )
            ess = 1.0 / jax.lax.psum(jnp.sum(jnp.square(weights)), "S")
            return jnp.real(mean), jnp.imag(mean), variance, ess

        self._distributed_statistics = jax.jit(
            sharding_decorator(
                weighted_statistics,
                sharded_args_tree=(True, True),
                reduction_op_tree=(jax.lax.pmax,) * 4,
            )
        )

        def global_sum(values):
            return jnp.sum(values)

        self._distributed_sum = jax.jit(
            sharding_decorator(
                global_sum,
                sharded_args_tree=(True,),
                reduction_op_tree=jax.lax.psum,
            )
        )

        def weighted_target_force(
            parameters, sigma, weights, local_energies, energy_real, energy_imag
        ):
            energy = energy_real + 1j * energy_imag
            total = jax.tree_util.tree_map(jnp.zeros_like, parameters)
            chunk_size = self.chunk_size or sigma.shape[0]
            for start in range(0, sigma.shape[0], chunk_size):
                stop = min(start + chunk_size, sigma.shape[0])
                sigma_chunk = sigma[start:stop]
                coefficients = weights[start:stop] * jax.lax.stop_gradient(
                    local_energies[start:stop] - energy
                )

                def objective(params):
                    log_psi = target_model.apply(
                        {"params": params, **target_model_state}, sigma_chunk
                    )
                    return jnp.sum(
                        jnp.real(coefficients) * jnp.real(log_psi)
                        + jnp.imag(coefficients) * jnp.imag(log_psi)
                    )

                partial = jax.grad(objective)(parameters)
                total = jax.tree_util.tree_map(
                    lambda old, new: old + 2.0 * new, total, partial
                )
            # The target parameters are replicated and each shard only sees
            # its local proposal configurations.  Explicitly sum the local
            # force contributions so the weighted force is the global NIS
            # estimate on every device.
            return jax.tree_util.tree_map(
                lambda leaf: jax.lax.psum(leaf, "S"), total
            )

        self._distributed_target_force = jax.jit(
            sharding_decorator(
                weighted_target_force,
                sharded_args_tree=(False, True, True, True, False, False),
                # Reverse-mode through replicated parameters already sums the
                # cotangent across NetKet's mesh. The leaves are therefore
                # identical on every device; pmax only declares replication.
                reduction_op_tree=jax.lax.pmax,
            )
        )

        def proposal_loss_and_grad(parameters, sigma, weights):
            def objective(params):
                logq = proposal_wrapper.log_prob(
                    {"params": params, **proposal_model_state}, sigma
                )
                return -jnp.sum(weights * logq)

            return jax.value_and_grad(objective)(parameters)

        self._distributed_proposal_loss_and_grad = jax.jit(
            sharding_decorator(
                proposal_loss_and_grad,
                sharded_args_tree=(False, True, True),
                # Both the scalar loss and the replicated parameter gradient
                # are sums over the sharded weighted proposal pool.
                reduction_op_tree=(jax.lax.psum, jax.lax.psum),
            )
        )

    def _distributed_local_energy(self, operator):
        """Return a cached local-energy kernel sharded over proposal samples."""
        key = id(operator)
        if key in self._distributed_local_energy_kernels:
            return self._distributed_local_energy_kernels[key]

        target_model = self._model
        target_model_state = self.model_state
        chunk_size = self.local_energy_chunk_size

        def local_energy(parameters, sigma, log_psi):
            connected, matrix_elements = operator.get_conn_padded(sigma)
            flat_connected = connected.reshape(-1, connected.shape[-1])
            if chunk_size is None or flat_connected.shape[0] <= chunk_size:
                connected_log_psi = target_model.apply(
                    {"params": parameters, **target_model_state}, flat_connected
                )
            else:
                connected_log_psi = jnp.concatenate(
                    [
                        target_model.apply(
                            {"params": parameters, **target_model_state},
                            flat_connected[start : start + chunk_size],
                        )
                        for start in range(0, flat_connected.shape[0], chunk_size)
                    ],
                    axis=0,
                )
            connected_log_psi = connected_log_psi.reshape(connected.shape[:-1])
            return jnp.sum(
                matrix_elements * jnp.exp(connected_log_psi - log_psi[:, None]), axis=1
            )

        kernel = jax.jit(
            sharding_decorator(
                local_energy,
                sharded_args_tree=(False, True, True),
                reduction_op_tree=False,
            )
        )
        self._distributed_local_energy_kernels[key] = kernel
        return kernel

    def global_sum(self, values):
        """Sum a sample-shaped array over the global proposal pool."""
        if self._uses_native_sharding:
            return self._distributed_sum(values)
        return jnp.sum(values)

    @property
    def last_batch(self):
        """Most recent proposal pool and normalized importance weights."""
        return self._last_batch

    @property
    def last_local_energies(self):
        """Local energies evaluated on :attr:`last_batch`.

        The NetKet-style weighted minimum-stochastic-reconfiguration backend
        uses these raw values together with the retained proposal weights to
        construct its sample-space right-hand side.  They are valid only until
        the next call to :meth:`expect` or :meth:`expect_and_grad`.
        """
        return self._last_local_energies

    @property
    def last_diagnostics(self) -> dict[str, Any]:
        """NIS diagnostics for the estimate most recently evaluated."""
        return dict(self._last_diagnostics)

    @property
    def last_statistics(self):
        return self._last_statistics

    def reset(self):
        """Invalidate parameter-dependent summaries without dropping the last pool.

        NetKet calls ``reset`` after every parameter update.  The most recent
        proposal pool belongs to the just-completed estimate and is required by
        the driver/logger to report its ESS and to write compatibility
        resamples.  A subsequent ``expect``/``expect_and_grad`` always replaces
        it, so retaining that completed pool is safe and avoids losing it during
        NetKet's bookkeeping reset.
        """
        self._last_statistics = None

    def _draw_proposal_batch(self, draw_key):
        """Draw a weighted proposal pool without changing training state."""
        if self._uses_native_sharding:
            sigma, logq = self._distributed_draw_proposals(self.proposal_variables, draw_key)
            logpsi = self._distributed_target_logpsi(self.parameters, sigma)
            logp = 2.0 * jnp.real(jnp.asarray(logpsi))
            weight_info = self._distributed_normalize_weights(logp, logq)
            if not bool(jax.device_get(weight_info["finite"])):
                raise FloatingPointError("NIS received non-finite target or proposal log probabilities")
            batch = {
                "key": draw_key,
                "sigma_prop": sigma,
                "logq_prop": logq,
                "logpsi_prop": logpsi,
                "logp_tilde_prop": logp,
                "weight_info": weight_info,
                "metrics": dict(weight_info),
            }
            return batch

        return sample_proposals_and_metrics(
            self._model.apply,
            self.variables,
            self.sampler.proposal_wrapper,
            self.proposal_variables,
            draw_key,
            n_proposals=self.n_samples,
            clip_weights=self.sampler.clip_weights,
            max_weight_ratio=self.sampler.max_weight_ratio,
        )

    def sample_proposals(self):
        """Draw one proposal pool and retain it as the training estimate."""
        self._key, draw_key = jax.random.split(self._key)
        batch = self._draw_proposal_batch(draw_key)
        batch["key"] = self._key
        self._last_batch = batch
        return batch

    def _apply_in_chunks(self, variables, configurations):
        chunk_size = self.local_energy_chunk_size
        if chunk_size is None or configurations.shape[0] <= chunk_size:
            return self._model.apply(variables, configurations)
        return jnp.concatenate(
            [
                self._model.apply(variables, configurations[start : start + chunk_size])
                for start in range(0, configurations.shape[0], chunk_size)
            ],
            axis=0,
        )

    def local_energy(self, operator, sigma, *, log_psi=None):
        """Evaluate NetKet's padded connected-state local-energy estimator."""
        connected, matrix_elements = operator.get_conn_padded(sigma)
        if log_psi is None:
            log_psi = self._model.apply(self.variables, sigma)
        flat_connected = connected.reshape(-1, connected.shape[-1])
        connected_log_psi = self._apply_in_chunks(self.variables, flat_connected)
        connected_log_psi = connected_log_psi.reshape(connected.shape[:-1])
        return jnp.sum(matrix_elements * jnp.exp(connected_log_psi - log_psi[:, None]), axis=1)

    def _evaluate_batch(self, operator, batch):
        """Evaluate one supplied proposal pool without mutating cached state."""
        weights = batch["weight_info"]["weights_normalized"]
        if self._uses_native_sharding:
            local_energies = self._distributed_local_energy(operator)(
                self.parameters, batch["sigma_prop"], batch["logpsi_prop"]
            )
            mean_real, mean_imag, variance, ess = self._distributed_statistics(
                weights, local_energies
            )
            weighted_stats = {
                "Mean": mean_real + 1j * mean_imag,
                "Variance": variance,
                "ErrorOfMean": jnp.sqrt(variance / ess),
                "ESS": ess,
            }
        else:
            local_energies = self.local_energy(
                operator, batch["sigma_prop"], log_psi=batch["logpsi_prop"]
            )
            weighted_stats = weighted_variational_statistics(local_energies, weights)
        stats = nk.stats.Stats(
            mean=weighted_stats["Mean"],
            variance=weighted_stats["Variance"],
            error_of_mean=weighted_stats["ErrorOfMean"],
        )
        diagnostics = {
            "ESS": weighted_stats["ESS"],
            "ESSFrac": batch["weight_info"]["ess_frac"],
            "logw_min": jnp.min(batch["weight_info"]["logw_raw"]),
            "logw_max": jnp.max(batch["weight_info"]["logw_raw"]),
            "top_weight": jnp.max(weights),
        }
        return stats, local_energies, diagnostics

    def _evaluate(self, operator):
        batch = self.sample_proposals()
        stats, local_energies, diagnostics = self._evaluate_batch(operator, batch)
        self._last_local_energies = local_energies
        self._last_statistics = stats
        self._last_diagnostics = diagnostics
        return stats, batch, local_energies

    def expect(self, operator):
        """Return a NetKet ``Stats`` object using the SNIS energy estimator."""
        stats, _, _ = self._evaluate(operator)
        return stats

    def expect_energy_weighted(self, operator):
        """Compatibility alias for :meth:`expect`."""
        return self.expect(operator)

    def _target_gradient(self, batch, local_energies):
        sigma = batch["sigma_prop"]
        weights = batch["weight_info"]["weights_normalized"]
        if self._uses_native_sharding:
            energy_real, energy_imag, _, _ = self._distributed_statistics(weights, local_energies)
            return self._distributed_target_force(
                self.parameters,
                sigma,
                weights,
                local_energies,
                energy_real,
                energy_imag,
            )

        energy = jnp.sum(weights * local_energies)
        total = jax.tree_util.tree_map(jnp.zeros_like, self.parameters)
        chunk_size = self.chunk_size or sigma.shape[0]
        model_state = self.model_state
        for start in range(0, sigma.shape[0], chunk_size):
            stop = min(start + chunk_size, sigma.shape[0])
            coefficients = weights[start:stop] * jax.lax.stop_gradient(
                local_energies[start:stop] - energy
            )
            partial = self._chunk_force(
                self.parameters, model_state, sigma[start:stop], coefficients
            )
            total = jax.tree_util.tree_map(lambda old, new: old + 2.0 * new, total, partial)
        return total

    def expect_and_grad(self, operator, *, mutable=None, **_):
        """Return weighted energy statistics and a target-parameter gradient tree."""
        if mutable not in (None, False):
            raise NotImplementedError("mutable model collections are not yet supported by WeightedNISState")
        stats, batch, local_energies = self._evaluate(operator)
        return stats, self._target_gradient(batch, local_energies)

    def evaluate_current_target_on_last_batch(self, operator):
        """Evaluate the current target on the retained proposal pool.

        This is intentionally read-only: it does not replace ``last_batch``
        or alter the proposal state.  It supports a paired, post-update energy
        comparison in which the proposal configurations and their original
        log-probabilities are shared by the pre- and post-update estimates.
        Reusing that pool makes an update-rejection check substantially less
        expensive than drawing a second pool, while retaining the correct
        self-normalized weights for the current target parameters.
        """
        if self._last_batch is None:
            raise RuntimeError("evaluate an operator before evaluating the current target")

        batch = self._last_batch
        sigma = batch["sigma_prop"]
        if self._uses_native_sharding:
            logpsi = self._distributed_target_logpsi(self.parameters, sigma)
            logp = 2.0 * jnp.real(jnp.asarray(logpsi))
            weight_info = self._distributed_normalize_weights(logp, batch["logq_prop"])
            finite = bool(jax.device_get(weight_info["finite"]))
        else:
            logpsi = self._apply_in_chunks(self.variables, sigma)
            logp = 2.0 * jnp.real(jnp.asarray(logpsi))
            weight_info = stable_log_weights(
                logp,
                batch["logq_prop"],
                clip_weights=self.sampler.clip_weights,
                max_weight_ratio=self.sampler.max_weight_ratio,
            )
            finite = bool(
                jax.device_get(
                    jnp.all(jnp.isfinite(logpsi))
                    & jnp.all(jnp.isfinite(batch["logq_prop"]))
                )
            )

        candidate_batch = dict(batch)
        candidate_batch.update({
            "logpsi_prop": logpsi,
            "logp_tilde_prop": logp,
            "weight_info": weight_info,
            "metrics": dict(weight_info),
        })
        stats, _local_energies, diagnostics = self._evaluate_batch(operator, candidate_batch)
        return stats, candidate_batch, diagnostics, finite

    def evaluate_heldout_and_grad(self, operator):
        """Evaluate an independent NIS pool for non-invasive diagnostics.

        The returned energy, force, and pool are deliberately not assigned to
        ``last_batch`` or ``last_diagnostics``.  Training, proposal MLE, and
        compatibility resampling therefore continue to use only the primary
        per-iteration pool.
        """
        self._heldout_key, draw_key = jax.random.split(self._heldout_key)
        batch = self._draw_proposal_batch(draw_key)
        stats, local_energies, diagnostics = self._evaluate_batch(operator, batch)
        return stats, self._target_gradient(batch, local_energies), batch, diagnostics

    def expect_and_grad_weighted(self, operator):
        """Compatibility helper returning Stats, gradient, and NIS diagnostics."""
        stats, gradient = self.expect_and_grad(operator)
        return stats, gradient, self.last_diagnostics

    def train_proposal(self, optimizer, optimizer_state, *, n_steps=1, batch_size=None):
        """Fit the proposal to the weighted pool from the latest target estimate.

        The training configurations are sampled from the normalized NIS weights,
        so this is a weighted maximum-likelihood update of ``q_phi``.  It makes
        no claim that the resampled batch is an iid target batch for energy or
        gradient estimation.
        """
        if self._last_batch is None:
            raise RuntimeError("evaluate an operator before training the proposal")
        if int(n_steps) < 1:
            return self.proposal_parameters, optimizer_state, None

        sigma = self._last_batch["sigma_prop"]
        weights = self._last_batch["weight_info"]["weights_normalized"]
        if self._uses_native_sharding:
            # This is the exact weighted MLE objective over the global pool.
            # Resampling a sharded categorical minibatch would add communication
            # and estimator noise without changing its expectation.
            loss_value = None
            for _ in range(int(n_steps)):
                loss_value, gradient = self._distributed_proposal_loss_and_grad(
                    self.proposal_parameters, sigma, weights
                )
                updates, optimizer_state = optimizer.update(
                    gradient, optimizer_state, self.proposal_parameters
                )
                self.proposal_parameters = optax.apply_updates(self.proposal_parameters, updates)
            return self.proposal_parameters, optimizer_state, loss_value

        if batch_size is None:
            batch_size = sigma.shape[0]
        batch_size = min(int(batch_size), sigma.shape[0])
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        loss_value = None
        for _ in range(int(n_steps)):
            self._key, batch_key = jax.random.split(self._key)
            indices = jax.random.choice(
                batch_key, sigma.shape[0], shape=(batch_size,), replace=True, p=weights
            )
            training_sigma = sigma[indices]
            loss_value, gradient = self._proposal_loss_and_grad(
                self.proposal_parameters, self.proposal_model_state, training_sigma
            )
            updates, optimizer_state = optimizer.update(
                gradient, optimizer_state, self.proposal_parameters
            )
            self.proposal_parameters = optax.apply_updates(self.proposal_parameters, updates)
        return self.proposal_parameters, optimizer_state, loss_value

    def quantum_geometric_tensor(self, qgt_T=None):
        del qgt_T
        raise NotImplementedError(
            "WeightedNISState does not expose NetKet's ordinary QGT because its "
            "samples are importance-weighted. Use optim.WeightedSR through "
            "WeightedNISVMC for the scalable matrix-free weighted-SR update."
        )

    def overlap_with_exact(self, exact_state_vector=None):
        if exact_state_vector is None:
            raise ValueError("provide an exact state vector")
        if self.hilbert.n_states > self.exact_enumeration_max_states:
            raise ValueError("exact overlap is disabled above exact_enumeration_max_states")
        psi = self.to_array()
        exact = jnp.asarray(exact_state_vector)
        exact = exact / jnp.linalg.norm(exact)
        return jnp.abs(jnp.vdot(exact, psi)) ** 2

    def to_array(self, normalize: bool = True):
        if self.hilbert.n_states > self.exact_enumeration_max_states:
            raise ValueError("dense state enumeration is disabled above exact_enumeration_max_states")
        psi = jnp.exp(self._model.apply(self.variables, self.hilbert.all_states()))
        return psi / jnp.linalg.norm(psi) if normalize else psi
