"""Importance-weighted stochastic reconfiguration without a dense QGT.

The target samples in neural importance sampling are drawn from ``q_phi``.
They must therefore retain their normalized importance weights in both the VMC
force and the quantum geometric tensor.  In particular, this module must not
be applied to a resampled ``MCState`` batch as though it were iid target data.
"""
from __future__ import annotations

from collections.abc import Callable
import math

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from netket.jax.sharding import gather, sharding_decorator


def weighted_force_vector(log_derivs, local_energies, weights_normalized):
    """Explicit weighted force, retained for tiny-system reference tests."""
    o, e, w = map(jnp.asarray, (log_derivs, local_energies, weights_normalized))
    om, em = jnp.einsum("b,bi->i", w, o), jnp.sum(w * e)
    return jnp.einsum("b,bi,b->i", w, jnp.conj(o - om), e - em)


def weighted_qgt(log_derivs, weights_normalized, diag_shift=1e-4):
    """Dense weighted QGT for small-system validation only."""
    o, w = jnp.asarray(log_derivs), jnp.asarray(weights_normalized)
    centered = o - jnp.einsum("b,bi->i", w, o)
    return (
        jnp.einsum("b,bi,bj->ij", w, jnp.conj(centered), centered)
        + diag_shift * jnp.eye(o.shape[-1], dtype=o.dtype)
    )


def solve_weighted_sr_step(qgt, force, *, solver="pinv_smooth", diag_shift=1e-4):
    """Dense weighted SR solve for debugging; not suitable for production."""
    del solver
    qgt = jnp.asarray(qgt) + diag_shift * jnp.eye(qgt.shape[-1], dtype=qgt.dtype)
    return jnp.linalg.pinv(qgt, hermitian=True) @ jnp.asarray(force)


def _tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_add(left, right):
    return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def _tree_scale(tree, scalar):
    return jax.tree_util.tree_map(lambda value: scalar * value, tree)


def _tree_multiply(left, right):
    return jax.tree_util.tree_map(lambda a, b: a * b, left, right)


def _tree_divide(left, right):
    return jax.tree_util.tree_map(lambda a, b: a / b, left, right)


def _tree_ones_like(tree):
    return jax.tree_util.tree_map(jnp.ones_like, tree)


def _tree_dot_real(left, right):
    return sum(
        jnp.real(jnp.vdot(a, b))
        for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    )


def _tree_rademacher_like(tree, key):
    """Create a real Rademacher pytree with the same leaves as ``tree``."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    values = [
        (2.0 * jax.random.bernoulli(leaf_key, shape=leaf.shape) - 1.0).astype(leaf.dtype)
        for leaf, leaf_key in zip(leaves, keys)
    ]
    return jax.tree_util.tree_unflatten(treedef, values)


def _assert_real_parameters(parameters):
    if any(jnp.iscomplexobj(leaf) for leaf in jax.tree_util.tree_leaves(parameters)):
        raise NotImplementedError(
            "WeightedSR currently implements the real tangent space used by the "
            "repository's real-parameter, complex-output ViTs. Split genuinely "
            "complex parameters into real coordinates before using it."
        )


class WeightedSR:
    """Matrix-free weighted SR preconditioner for ``WeightedNISState``.

    It solves a weighted quantum-geometric-tensor linear system in the real
    tangent space. With ``relative_damping=True`` this system is
    ``(S_w + lambda D) direction = gradient``, where ``D`` is a positive
    diagonal estimate of ``diag(S_w)``. The same diagonal is used as the
    preconditioner for preconditioned conjugate gradient. ``S_w`` is evaluated
    by JVP/VJP products over the current proposal pool, never by materialising
    a sample-by-parameter Jacobian or a dense quantum geometric tensor.

    ``trust_region`` bounds the actual outer SGD update in QGT norm, so it
    assumes plain SGD with the same ``learning_rate`` supplied here.
    """

    def __init__(
        self,
        *,
        diag_shift: float = 1.0e-3,
        maxiter: int = 12,
        tol: float = 1.0e-5,
        chunk_size: int | None = None,
        trust_region: float | None = 0.05,
        learning_rate: float | Callable[[int], float] = 1.0,
        adaptive: bool = True,
        adaptive_maxiter: int = 32,
        adaptive_diag_shift_factor: float = 4.0,
        adaptive_max_diag_shift: float = 1.0e-1,
        adaptive_diag_shift_decay: float = 1.5,
        adaptive_ess_threshold: float = 0.10,
        adaptive_healthy_residual: float = 1.0e-3,
        adaptive_trust_region_min_scale: float = 0.10,
        adaptive_trust_region_growth: float = 1.25,
        adaptive_trust_region_shrink: float = 0.5,
        warm_start: bool = False,
        diagonal_preconditioner: bool = False,
        relative_damping: bool = False,
        diagonal_probes: int = 4,
        diagonal_update_interval: int = 25,
        diagonal_ema: float = 0.9,
        diagonal_floor: float = 1.0e-6,
        diagonal_mode: str = "per_parameter",
        residual_target: float | None = None,
        residual_replacement_interval: int = 0,
    ):
        if diag_shift <= 0.0:
            raise ValueError("diag_shift must be positive")
        if maxiter < 1:
            raise ValueError("maxiter must be positive")
        if tol <= 0.0:
            raise ValueError("tol must be positive")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be positive or None")
        if trust_region is not None and trust_region <= 0.0:
            raise ValueError("trust_region must be positive or None")
        if not callable(learning_rate) and learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if adaptive_maxiter < maxiter:
            raise ValueError("adaptive_maxiter must be at least maxiter")
        if adaptive_diag_shift_factor <= 1.0:
            raise ValueError("adaptive_diag_shift_factor must exceed one")
        if adaptive_max_diag_shift < diag_shift:
            raise ValueError("adaptive_max_diag_shift must be at least diag_shift")
        if adaptive_diag_shift_decay <= 1.0:
            raise ValueError("adaptive_diag_shift_decay must exceed one")
        if not 0.0 <= adaptive_ess_threshold <= 1.0:
            raise ValueError("adaptive_ess_threshold must be between zero and one")
        if adaptive_healthy_residual <= 0.0:
            raise ValueError("adaptive_healthy_residual must be positive")
        if not 0.0 < adaptive_trust_region_min_scale <= 1.0:
            raise ValueError("adaptive_trust_region_min_scale must be in (0, 1]")
        if adaptive_trust_region_growth <= 1.0:
            raise ValueError("adaptive_trust_region_growth must exceed one")
        if not 0.0 < adaptive_trust_region_shrink <= 1.0:
            raise ValueError("adaptive_trust_region_shrink must be in (0, 1]")
        if relative_damping and not diagonal_preconditioner:
            raise ValueError("relative_damping requires diagonal_preconditioner")
        if diagonal_probes < 1:
            raise ValueError("diagonal_probes must be positive")
        if diagonal_update_interval < 1:
            raise ValueError("diagonal_update_interval must be positive")
        if not 0.0 <= diagonal_ema < 1.0:
            raise ValueError("diagonal_ema must be in [0, 1)")
        if diagonal_floor <= 0.0:
            raise ValueError("diagonal_floor must be positive")
        if diagonal_mode not in {"per_parameter", "per_leaf"}:
            raise ValueError("diagonal_mode must be 'per_parameter' or 'per_leaf'")
        if residual_target is not None and residual_target <= 0.0:
            raise ValueError("residual_target must be positive or None")
        if residual_replacement_interval < 0:
            raise ValueError("residual_replacement_interval must be non-negative")

        self.diag_shift = float(diag_shift)
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.trust_region = trust_region
        self.learning_rate = learning_rate
        self.adaptive = bool(adaptive)
        self.adaptive_maxiter = int(adaptive_maxiter)
        self.adaptive_diag_shift_factor = float(adaptive_diag_shift_factor)
        self.adaptive_max_diag_shift = float(adaptive_max_diag_shift)
        self.adaptive_diag_shift_decay = float(adaptive_diag_shift_decay)
        self.adaptive_ess_threshold = float(adaptive_ess_threshold)
        self.adaptive_healthy_residual = float(adaptive_healthy_residual)
        self.adaptive_trust_region_min_scale = float(adaptive_trust_region_min_scale)
        self.adaptive_trust_region_growth = float(adaptive_trust_region_growth)
        self.adaptive_trust_region_shrink = float(adaptive_trust_region_shrink)
        # Reusing the previous *unclipped* SR direction is effective once the
        # damping has settled.  It is opt-in so existing runs retain their
        # historical cold-start CG behaviour.
        self.warm_start = bool(warm_start)
        self.diagonal_preconditioner = bool(diagonal_preconditioner)
        self.relative_damping = bool(relative_damping)
        self.diagonal_probes = int(diagonal_probes)
        self.diagonal_update_interval = int(diagonal_update_interval)
        self.diagonal_ema = float(diagonal_ema)
        self.diagonal_floor = float(diagonal_floor)
        self.diagonal_mode = diagonal_mode
        # This is the numerical stopping target. ``tol`` is retained as the
        # backward-compatible spelling when no explicit target is supplied.
        self.residual_target = float(tol if residual_target is None else residual_target)
        self.residual_replacement_interval = int(residual_replacement_interval)
        # These carry solver confidence between iterations. ``diag_shift`` and
        # ``trust_region`` remain the user-selected nominal values: the
        # adaptive controller only raises damping and reduces the cap when the
        # weighted sample pool or linear solve is unreliable.
        self._active_diag_shift = self.diag_shift
        self._trust_region_multiplier = 1.0
        self.last_info: dict[str, float | int | bool | None] | None = None
        self._kernel_cache: dict[tuple[int, int, bool], tuple[Callable, ...]] = {}
        self._device_cg_kernel_cache: dict[tuple[int, int, bool], Callable] = {}
        self._last_solver_direction = None
        self._last_solver_diag_shift: float | None = None
        self._last_solver_metric_diagonal = None
        self._metric_diagonal = None
        self._metric_diagonal_step: int | None = None
        self._diagonal_key = jax.random.PRNGKey(0)

    def _kernels(self, vstate, chunk_length: int):
        """Cache JVP/VJP kernels by target model and static chunk shape."""
        key = (id(vstate.model), chunk_length, False)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        model = vstate.model
        model_state = vstate.model_state

        def apply_logpsi(parameters, sigma):
            return jnp.asarray(model.apply({"params": parameters, **model_state}, sigma)).reshape(-1)

        @jax.jit
        def directional(parameters, vector, sigma):
            _, output = jax.jvp(
                lambda params: apply_logpsi(params, sigma),
                (parameters,),
                (vector,),
            )
            return output

        @jax.jit
        def adjoint(parameters, cotangent, sigma):
            _, pullback = jax.vjp(lambda params: apply_logpsi(params, sigma), parameters)
            # JAX's complex VJP convention conjugates the supplied cotangent.
            # The real tangent-space adjoint required by SR is
            # Re[J^dagger cotangent], hence we supply its conjugate here.  This
            # is equivalent to differentiating
            # Re(cotangent) * Re(logpsi) + Im(cotangent) * Im(logpsi), while
            # retaining the efficient reverse-mode VJP implementation.
            return pullback(jnp.conj(cotangent))[0]

        self._kernel_cache[key] = directional, adjoint
        return directional, adjoint

    def _sharded_kernels(self, vstate, local_chunk_size: int):
        """Cache native-NetKet JVP/VJP kernels for a sharded proposal pool."""
        key = (id(vstate.model), local_chunk_size, True)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        model = vstate.model
        model_state = vstate.model_state

        def apply_logpsi(parameters, sigma):
            return jnp.asarray(
                model.apply({"params": parameters, **model_state}, sigma)
            ).reshape(-1)

        def directional_local(parameters, vector, sigma):
            outputs = []
            for start in range(0, sigma.shape[0], local_chunk_size):
                stop = min(start + local_chunk_size, sigma.shape[0])
                _, output = jax.jvp(
                    lambda params: apply_logpsi(params, sigma[start:stop]),
                    (parameters,),
                    (vector,),
                )
                outputs.append(output)
            return jnp.concatenate(outputs, axis=0)

        directional = jax.jit(
            sharding_decorator(
                directional_local,
                sharded_args_tree=(False, False, True),
                reduction_op_tree=False,
            )
        )

        def adjoint_local(parameters, cotangent, sigma):
            result = _tree_zeros_like(parameters)
            for start in range(0, sigma.shape[0], local_chunk_size):
                stop = min(start + local_chunk_size, sigma.shape[0])
                _, pullback = jax.vjp(
                    lambda params: apply_logpsi(params, sigma[start:stop]), parameters
                )
                partial = pullback(jnp.conj(cotangent[start:stop]))[0]
                result = _tree_add(result, partial)
            return result

        adjoint = jax.jit(
            sharding_decorator(
                adjoint_local,
                sharded_args_tree=(False, True, True),
                # The VJP of a replicated parameter tree already performs the
                # sample-shard cotangent sum. Its identical per-device leaves
                # only need to be marked replicated, not summed again.
                reduction_op_tree=jax.lax.pmax,
            )
        )

        def weighted_sum_local(weights, values):
            return jnp.sum(weights * values)

        weighted_sum = jax.jit(
            sharding_decorator(
                weighted_sum_local,
                sharded_args_tree=(True, True),
                reduction_op_tree=jax.lax.psum,
            )
        )

        self._kernel_cache[key] = directional, adjoint, weighted_sum
        return directional, adjoint, weighted_sum

    def _weighted_qgt_matvec(
        self,
        vstate,
        sigma,
        weights,
        vector,
        *,
        diag_shift=None,
        metric_diagonal=None,
        relative_damping: bool = False,
    ):
        """Apply ``S_w + lambda I`` or ``S_w + lambda D`` matrix-free."""
        if diag_shift is None:
            diag_shift = self.diag_shift
        if getattr(vstate, "uses_native_sharding", False):
            local_chunk_size = self.chunk_size or vstate.n_samples_per_device
            directional, adjoint, weighted_sum = self._sharded_kernels(
                vstate, local_chunk_size
            )
            jvp_value = directional(vstate.parameters, vector, sigma)
            weighted_directional_mean = weighted_sum(weights, jvp_value)
            result = adjoint(
                vstate.parameters,
                weights * (jvp_value - weighted_directional_mean),
                sigma,
            )
            if relative_damping:
                if metric_diagonal is None:
                    raise ValueError("relative damping requires a metric diagonal")
                return _tree_add(
                    result,
                    _tree_scale(_tree_multiply(metric_diagonal, vector), diag_shift),
                )
            return _tree_add(result, _tree_scale(vector, diag_shift))

        n_samples = sigma.shape[0]
        chunk_size = self.chunk_size or n_samples
        directional_values = []
        weighted_directional_mean = None

        for start in range(0, n_samples, chunk_size):
            stop = min(start + chunk_size, n_samples)
            sigma_chunk = sigma[start:stop]
            weight_chunk = weights[start:stop]
            directional, _ = self._kernels(vstate, sigma_chunk.shape[0])
            jvp_value = directional(vstate.parameters, vector, sigma_chunk)
            directional_values.append((sigma_chunk, weight_chunk, jvp_value))
            if weighted_directional_mean is None:
                # The usual Gamma target is complex, but NetKet-compatible
                # real-output ansaetze are valid too.  Keep the cotangent in
                # exactly the model output dtype for VJP compatibility.
                weighted_directional_mean = jnp.zeros((), dtype=jvp_value.dtype)
            weighted_directional_mean = weighted_directional_mean + jnp.sum(weight_chunk * jvp_value)

        result = _tree_zeros_like(vector)
        for sigma_chunk, weight_chunk, jvp_value in directional_values:
            _, adjoint = self._kernels(vstate, sigma_chunk.shape[0])
            partial = adjoint(
                vstate.parameters,
                weight_chunk * (jvp_value - weighted_directional_mean),
                sigma_chunk,
            )
            result = _tree_add(result, partial)

        if relative_damping:
            if metric_diagonal is None:
                raise ValueError("relative damping requires a metric diagonal")
            return _tree_add(
                result,
                _tree_scale(_tree_multiply(metric_diagonal, vector), diag_shift),
            )
        return _tree_add(result, _tree_scale(vector, diag_shift))

    def _metric_diagonal_for_solver(self, vstate, sigma, weights, step):
        """Return a positive, smoothed Hutchinson estimate of ``diag(S_w)``.

        A Rademacher probe ``z`` satisfies
        ``E[z * (S_w z)] = diag(S_w)``.  A few matrix-free products therefore
        provide a scalable diagonal preconditioner without exposing a
        sample-by-parameter Jacobian.  The estimate is refreshed only at the
        configured interval and exponential-smoothed between refreshes.
        """
        if not self.diagonal_preconditioner:
            return _tree_ones_like(vstate.parameters), False

        step_value = 0 if step is None else int(step)
        refresh = (
            self._metric_diagonal is None
            or self._metric_diagonal_step is None
            or step_value - self._metric_diagonal_step >= self.diagonal_update_interval
        )
        if not refresh:
            return self._metric_diagonal, False

        estimate = _tree_zeros_like(vstate.parameters)
        self._diagonal_key, *probe_keys = jax.random.split(
            self._diagonal_key, self.diagonal_probes + 1
        )
        for probe_key in probe_keys:
            probe = _tree_rademacher_like(vstate.parameters, probe_key)
            product = self._weighted_qgt_matvec(
                vstate, sigma, weights, probe, diag_shift=0.0
            )
            estimate = _tree_add(estimate, _tree_multiply(probe, product))
        estimate = _tree_scale(estimate, 1.0 / self.diagonal_probes)

        # Finite-probe noise can make individual estimates slightly negative.
        # A relative floor preserves positive definiteness while retaining the
        # parameter-scale information needed by relative damping.
        total_abs = sum(
            jnp.sum(jnp.abs(leaf)) for leaf in jax.tree_util.tree_leaves(estimate)
        )
        total_size = sum(leaf.size for leaf in jax.tree_util.tree_leaves(estimate))
        mean_scale = total_abs / max(total_size, 1)
        floor = jnp.maximum(
            jnp.asarray(self.diagonal_floor, dtype=mean_scale.dtype),
            jnp.asarray(self.diagonal_floor, dtype=mean_scale.dtype) * mean_scale,
        )
        estimate = jax.tree_util.tree_map(
            lambda leaf: jnp.maximum(leaf, floor), estimate
        )
        if self.diagonal_mode == "per_leaf":
            # Averaging within each parameter leaf estimates its trace scale,
            # which has much lower Hutchinson variance than individual
            # entries for strongly correlated transformer parameters.
            estimate = jax.tree_util.tree_map(
                lambda leaf: jnp.full_like(leaf, jnp.maximum(jnp.mean(leaf), floor)),
                estimate,
            )
        if self._metric_diagonal is not None:
            estimate = jax.tree_util.tree_map(
                lambda previous, current: self.diagonal_ema * previous
                + (1.0 - self.diagonal_ema) * current,
                self._metric_diagonal,
                estimate,
            )
        self._metric_diagonal = estimate
        self._metric_diagonal_step = step_value
        return estimate, True

    def _device_cg_kernel(self, vstate, local_chunk_size: int):
        """Build a cached, fully device-side weighted-QGT CG kernel.

        The old host loop synchronised after every curvature and residual
        check.  This kernel instead traces the weighted JVP/VJP matvec and the
        complete CG recurrence into one ``lax.while_loop``.  Parameters,
        samples, weights, damping, tolerance, and the iteration budget remain
        dynamic arguments, so the same compiled executable is reused at every
        optimisation step.
        """
        sharded = bool(getattr(vstate, "uses_native_sharding", False))
        key = (id(vstate.model), local_chunk_size, sharded)
        if key in self._device_cg_kernel_cache:
            return self._device_cg_kernel_cache[key]

        model = vstate.model
        model_state = vstate.model_state

        def apply_logpsi(parameters, sigma):
            return jnp.asarray(
                model.apply({"params": parameters, **model_state}, sigma)
            ).reshape(-1)

        def matvec(
            parameters,
            sigma,
            weights,
            vector,
            diag_shift,
            metric_diagonal,
            use_relative_damping,
        ):
            directional_values = []
            weighted_directional_mean = None
            for start in range(0, sigma.shape[0], local_chunk_size):
                stop = min(start + local_chunk_size, sigma.shape[0])
                sigma_chunk = sigma[start:stop]
                weight_chunk = weights[start:stop]
                _, directional = jax.jvp(
                    lambda params: apply_logpsi(params, sigma_chunk),
                    (parameters,),
                    (vector,),
                )
                directional_values.append((sigma_chunk, weight_chunk, directional))
                if weighted_directional_mean is None:
                    weighted_directional_mean = jnp.zeros((), dtype=directional.dtype)
                weighted_directional_mean = weighted_directional_mean + jnp.sum(
                    weight_chunk * directional
                )

            if sharded:
                weighted_directional_mean = jax.lax.psum(
                    weighted_directional_mean, "S"
                )

            result = _tree_zeros_like(vector)
            for sigma_chunk, weight_chunk, directional in directional_values:
                _, pullback = jax.vjp(
                    lambda params: apply_logpsi(params, sigma_chunk), parameters
                )
                partial = pullback(
                    jnp.conj(weight_chunk * (directional - weighted_directional_mean))
                )[0]
                result = _tree_add(result, partial)
            # ``parameters`` are replicated while the proposal pool is sharded.
            # A reverse-mode pullback therefore produces a *local* parameter
            # contribution on every device.  Sum those contributions before
            # entering the conjugate-gradient recurrence so every replica has
            # the same global weighted quantum-geometric-tensor product.
            #
            # Without this collective, different replicas can reach the local
            # residual tolerance at different iterations.  One replica then
            # exits ``lax.while_loop`` while another enters its next matvec and
            # waits forever at ``psum``.  This appears as a run stuck at a
            # particular outer iteration with all GPUs fully occupied.
            if sharded:
                result = jax.tree_util.tree_map(
                    lambda leaf: jax.lax.psum(leaf, "S"), result
                )
            damping_diagonal = jax.tree_util.tree_map(
                lambda diagonal, parameter: diag_shift
                * jnp.where(use_relative_damping, diagonal, jnp.ones_like(parameter)),
                metric_diagonal,
                vector,
            )
            return _tree_add(result, _tree_multiply(damping_diagonal, vector))

        def cg(
            parameters,
            sigma,
            weights,
            rhs,
            initial_solution,
            use_warm_start,
            diag_shift,
            metric_diagonal,
            preconditioner_diagonal,
            use_relative_damping,
            maxiter,
            residual_target,
            residual_replacement_interval,
        ):
            # The weighted force can have a narrower dtype than the target
            # parameters (notably in JAX's complex-output reverse mode). CG's
            # loop carry must have one fixed dtype, so solve in the real
            # parameter dtype just as the former host recurrence promoted it.
            rhs = jax.tree_util.tree_map(
                lambda value, parameter: jnp.asarray(value, dtype=parameter.dtype),
                rhs,
                parameters,
            )
            tangent_dtype = jax.tree_util.tree_leaves(rhs)[0].dtype
            # Keep scalar controls in the tangent dtype. In particular, a
            # float64 damping scalar must not promote only the accepted CG
            # branch while the untouched branch remains float32.
            diag_shift = jnp.asarray(diag_shift, dtype=tangent_dtype)
            residual_target = jnp.asarray(residual_target, dtype=tangent_dtype)
            residual_replacement_interval = jnp.asarray(
                residual_replacement_interval, dtype=jnp.int32
            )
            initial_solution = jax.tree_util.tree_map(
                lambda value, parameter: jnp.asarray(value, dtype=parameter.dtype),
                initial_solution,
                parameters,
            )
            metric_diagonal = jax.tree_util.tree_map(
                lambda value, parameter: jnp.asarray(value, dtype=parameter.dtype),
                metric_diagonal,
                parameters,
            )
            preconditioner_diagonal = jax.tree_util.tree_map(
                lambda value, parameter: jnp.asarray(value, dtype=parameter.dtype),
                preconditioner_diagonal,
                parameters,
            )
            rhs_norm_sq = _tree_dot_real(rhs, rhs)
            rhs_norm = jnp.sqrt(jnp.maximum(rhs_norm_sq, 0.0))

            def replicas_all(value):
                """Return true only when every shard agrees that ``value`` is true.

                The inner solver contains collectives.  Its branches and loop
                condition must therefore be identical across every shard even
                in the presence of small numerical differences.
                """
                value = jnp.asarray(value, dtype=jnp.bool_)
                if sharded:
                    return jax.lax.pmin(value.astype(jnp.int32), "S").astype(jnp.bool_)
                return value

            def cold_start(_):
                preconditioned_residual = _tree_divide(rhs, preconditioner_diagonal)
                return (
                    _tree_zeros_like(rhs),
                    rhs,
                    preconditioned_residual,
                    _tree_dot_real(rhs, preconditioned_residual),
                    rhs_norm_sq,
                )

            def warm_start(_):
                warm_residual = _tree_add(
                    rhs,
                    _tree_scale(
                        matvec(
                            parameters,
                            sigma,
                            weights,
                            initial_solution,
                            diag_shift,
                            metric_diagonal,
                            use_relative_damping,
                        ),
                        -1.0,
                    ),
                )
                preconditioned_residual = _tree_divide(
                    warm_residual, preconditioner_diagonal
                )
                return (
                    initial_solution,
                    warm_residual,
                    preconditioned_residual,
                    _tree_dot_real(warm_residual, preconditioned_residual),
                    _tree_dot_real(warm_residual, warm_residual),
                )

            # The stopping tolerance remains relative to ||rhs||, not to the
            # warm-start residual.  This keeps diagnostics comparable between
            # cold and warm solves and prevents a poor initial guess from
            # silently loosening the requested tolerance.
            use_warm_start = jnp.asarray(use_warm_start, dtype=jnp.bool_) & (rhs_norm > 0.0)
            (
                solution,
                residual,
                direction,
                preconditioned_residual_dot,
                residual_norm_sq,
            ) = jax.lax.cond(
                use_warm_start, warm_start, cold_start, operand=None
            )
            initial_residual_norm = rhs_norm
            residual_norm = jnp.sqrt(jnp.maximum(residual_norm_sq, 0.0))
            initially_converged = replicas_all(
                residual_norm <= residual_target * initial_residual_norm
            )

            # State: solution, residual, preconditioned search direction,
            # r^T M^-1 r, ||residual||^2, iteration, convergence/breakdown,
            # and residual-replacement count. All leaves remain on-device.
            initial_state = (
                solution,
                residual,
                direction,
                preconditioned_residual_dot,
                residual_norm_sq,
                jnp.asarray(0, dtype=jnp.int32),
                initially_converged,
                jnp.asarray(False),
                jnp.asarray(0, dtype=jnp.int32),
            )

            def cond(state):
                _, _, _, _, _, iteration, converged, breakdown, _ = state
                return (iteration < maxiter) & (~converged) & (~breakdown)

            def body(state):
                (
                    solution,
                    residual,
                    direction,
                    preconditioned_residual_dot,
                    residual_norm_sq,
                    iteration,
                    _,
                    _,
                    replacement_count,
                ) = state
                matvec_direction = matvec(
                    parameters,
                    sigma,
                    weights,
                    direction,
                    diag_shift,
                    metric_diagonal,
                    use_relative_damping,
                )
                denominator = _tree_dot_real(direction, matvec_direction)
                valid_curvature = replicas_all(
                    jnp.isfinite(denominator) & (denominator > 0.0)
                )

                def accepted_step(_):
                    step_size = preconditioned_residual_dot / denominator
                    next_solution = _tree_add(
                        solution, _tree_scale(direction, step_size)
                    )
                    recurrence_residual = _tree_add(
                        residual, _tree_scale(matvec_direction, -step_size)
                    )
                    replace_residual = (residual_replacement_interval > 0) & (
                        (iteration + 1) % residual_replacement_interval == 0
                    )

                    def exact_residual(_):
                        return _tree_add(
                            rhs,
                            _tree_scale(
                                matvec(
                                    parameters,
                                    sigma,
                                    weights,
                                    next_solution,
                                    diag_shift,
                                    metric_diagonal,
                                    use_relative_damping,
                                ),
                                -1.0,
                            ),
                        )

                    next_residual = jax.lax.cond(
                        replace_residual,
                        exact_residual,
                        lambda _: recurrence_residual,
                        operand=None,
                    )
                    next_residual_norm_sq = _tree_dot_real(
                        next_residual, next_residual
                    )
                    valid_residual = replicas_all(
                        jnp.isfinite(next_residual_norm_sq)
                        & (next_residual_norm_sq >= 0.0)
                    )

                    def finite_residual_step(_):
                        next_residual_norm = jnp.sqrt(next_residual_norm_sq)
                        next_converged = replicas_all(
                            next_residual_norm
                            <= residual_target * initial_residual_norm
                        )
                        next_preconditioned_residual = _tree_divide(
                            next_residual, preconditioner_diagonal
                        )
                        next_preconditioned_residual_dot = _tree_dot_real(
                            next_residual, next_preconditioned_residual
                        )
                        safe_preconditioned_residual_dot = jnp.maximum(
                            preconditioned_residual_dot,
                            jnp.finfo(next_preconditioned_residual_dot.dtype).tiny,
                        )
                        beta = (
                            next_preconditioned_residual_dot
                            / safe_preconditioned_residual_dot
                        )
                        next_direction = _tree_add(
                            next_preconditioned_residual,
                            _tree_scale(direction, beta),
                        )
                        return (
                            next_solution,
                            next_residual,
                            next_direction,
                            next_preconditioned_residual_dot,
                            next_residual_norm_sq,
                            iteration + 1,
                            next_converged,
                            jnp.asarray(False),
                            replacement_count + replace_residual.astype(jnp.int32),
                        )

                    def invalid_residual_step(_):
                        return (
                            solution,
                            residual,
                            direction,
                            preconditioned_residual_dot,
                            residual_norm_sq,
                            iteration + 1,
                            jnp.asarray(False),
                            jnp.asarray(True),
                            replacement_count,
                        )

                    return jax.lax.cond(
                        valid_residual,
                        finite_residual_step,
                        invalid_residual_step,
                        operand=None,
                    )

                def rejected_step(_):
                    return (
                        solution,
                        residual,
                        direction,
                        preconditioned_residual_dot,
                        residual_norm_sq,
                        iteration + 1,
                        jnp.asarray(False),
                        jnp.asarray(True),
                        replacement_count,
                    )

                return jax.lax.cond(
                    valid_curvature, accepted_step, rejected_step, operand=None
                )

            (
                solution,
                residual,
                _,
                _,
                residual_norm_sq,
                iteration,
                converged,
                breakdown,
                residual_replacements,
            ) = jax.lax.while_loop(cond, body, initial_state)
            final_residual_norm = jnp.sqrt(jnp.maximum(residual_norm_sq, 0.0))
            relative_residual = jnp.where(
                initial_residual_norm > 0.0,
                final_residual_norm / initial_residual_norm,
                0.0,
            )
            return (
                solution,
                residual,
                iteration,
                final_residual_norm,
                relative_residual,
                converged,
                breakdown,
                residual_replacements,
            )

        if sharded:
            kernel = jax.jit(
                sharding_decorator(
                    cg,
                    sharded_args_tree=(
                        False,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ),
                    # The parameter-space CG state is replicated. The VJP rule
                    # for a replicated parameter already combines sample-shard
                    # cotangents; pmax only declares this replication.
                    reduction_op_tree=(jax.lax.pmax,) * 8,
                )
            )
        else:
            kernel = jax.jit(cg)
        self._device_cg_kernel_cache[key] = kernel
        return kernel

    def _ess_fraction(self, vstate, weights):
        """Return the global ESS fraction for either local or sharded weights."""
        ess = 1.0 / vstate.global_sum(weights * weights)
        return ess / vstate.n_samples

    def _adapt_controls(self, *, reliable):
        """Update next iteration's damping and trust cap from solve quality."""
        if not self.adaptive:
            return self.diag_shift, 1.0
        if reliable:
            next_shift = max(
                self.diag_shift,
                self._active_diag_shift / self.adaptive_diag_shift_decay,
            )
            next_trust_multiplier = min(
                1.0,
                self._trust_region_multiplier * self.adaptive_trust_region_growth,
            )
        else:
            next_shift = min(
                self.adaptive_max_diag_shift,
                self._active_diag_shift * self.adaptive_diag_shift_factor,
            )
            next_trust_multiplier = max(
                self.adaptive_trust_region_min_scale,
                self._trust_region_multiplier * self.adaptive_trust_region_shrink,
            )
        self._active_diag_shift = next_shift
        self._trust_region_multiplier = next_trust_multiplier
        return next_shift, next_trust_multiplier

    def notify_rejected_update(self):
        """Make the next solve conservative after an energy-guard rejection.

        The rejected direction was computed at the restored parameters, but it
        demonstrably failed the post-update energy check. Discarding it avoids
        reusing a harmful warm start; increasing damping and shrinking the
        trust region makes the next matrix-free stochastic-reconfiguration
        proposal more conservative without changing the current parameters.
        """
        self._last_solver_direction = None
        self._last_solver_diag_shift = None
        self._last_solver_metric_diagonal = None
        if self.adaptive:
            self._active_diag_shift = min(
                self.adaptive_max_diag_shift,
                self._active_diag_shift * self.adaptive_diag_shift_factor,
            )
            self._trust_region_multiplier = max(
                self.adaptive_trust_region_min_scale,
                self._trust_region_multiplier * self.adaptive_trust_region_shrink,
            )
        if self.last_info is not None:
            self.last_info["PostUpdateRejected"] = True
            self.last_info["NextDiagShift"] = self._active_diag_shift
            self.last_info["NextTrustRegionMultiplier"] = self._trust_region_multiplier

    def _learning_rate_at(self, step):
        """Resolve the outer-SGD learning rate used by this SR application."""
        if callable(self.learning_rate):
            value = self.learning_rate(0 if step is None else step)
        else:
            value = self.learning_rate
        value = float(jax.device_get(value))
        if not math.isfinite(value) or value <= 0.0:
            raise FloatingPointError("weighted SR received a non-positive learning rate")
        return value

    def heldout_relative_residual(self, vstate, gradient, batch) -> float | None:
        """Evaluate the latest SR direction on an independent weighted pool.

        This is a diagnostic only: it performs one matrix-free QGT product on
        the held-out pool and never changes adaptive solver state.  It is
        useful for distinguishing an under-solved/noisy training pool from a
        direction that generalises to a fresh proposal draw.
        """
        if self._last_solver_direction is None or self._last_solver_diag_shift is None:
            return None
        _assert_real_parameters(vstate.parameters)
        sigma = batch["sigma_prop"]
        weights = jnp.asarray(batch["weight_info"]["weights_normalized"], dtype=jnp.float64)
        weights = weights / vstate.global_sum(weights)
        product = self._weighted_qgt_matvec(
            vstate,
            sigma,
            weights,
            self._last_solver_direction,
            diag_shift=self._last_solver_diag_shift,
            metric_diagonal=self._last_solver_metric_diagonal,
            relative_damping=self.relative_damping,
        )
        residual = _tree_add(gradient, _tree_scale(product, -1.0))
        rhs_norm = jnp.sqrt(jnp.maximum(_tree_dot_real(gradient, gradient), 0.0))
        residual_norm = jnp.sqrt(jnp.maximum(_tree_dot_real(residual, residual), 0.0))
        relative = jnp.where(rhs_norm > 0.0, residual_norm / rhs_norm, 0.0)
        return float(jax.device_get(relative))

    def __call__(self, vstate, gradient, step=None):
        """Return the weighted-SR-preconditioned gradient for an SGD update."""
        _assert_real_parameters(vstate.parameters)
        batch = vstate.last_batch
        if batch is None:
            raise RuntimeError("WeightedSR requires the current WeightedNISState proposal pool")
        sigma = batch["sigma_prop"]
        weights = jnp.asarray(batch["weight_info"]["weights_normalized"], dtype=jnp.float64)
        weights = weights / vstate.global_sum(weights)

        active_diag_shift = self._active_diag_shift if self.adaptive else self.diag_shift
        ess_fraction_device = self._ess_fraction(vstate, weights)
        requested_maxiter = jnp.asarray(self.maxiter, dtype=jnp.int32)
        if self.adaptive:
            requested_maxiter = jnp.where(
                ess_fraction_device >= self.adaptive_ess_threshold,
                jnp.asarray(self.adaptive_maxiter, dtype=jnp.int32),
                requested_maxiter,
            )
        local_chunk_size = (
            self.chunk_size
            or (
                vstate.n_samples_per_device
                if getattr(vstate, "uses_native_sharding", False)
                else sigma.shape[0]
            )
        )
        metric_diagonal, diagonal_refreshed = self._metric_diagonal_for_solver(
            vstate, sigma, weights, step
        )

        def preconditioner_diagonal(damping):
            if not self.diagonal_preconditioner:
                # This retains the historical unpreconditioned CG path unless
                # the new diagonal preconditioner is explicitly enabled.
                return _tree_ones_like(gradient)
            if self.relative_damping:
                return _tree_scale(metric_diagonal, 1.0 + damping)
            return jax.tree_util.tree_map(
                lambda diagonal: diagonal + damping, metric_diagonal
            )

        device_cg = self._device_cg_kernel(vstate, local_chunk_size)
        curvature_retries = 0
        update_skipped = False
        warm_start_used = (
            self.warm_start
            and self._last_solver_direction is not None
            and self._last_solver_diag_shift is not None
            and math.isclose(
                self._last_solver_diag_shift,
                active_diag_shift,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
        )
        initial_solution = (
            self._last_solver_direction if warm_start_used else _tree_zeros_like(gradient)
        )
        while True:
            active_preconditioner_diagonal = preconditioner_diagonal(active_diag_shift)
            (
                direction,
                residual,
                iterations_device,
                residual_norm_device,
                relative_residual_device,
                converged_device,
                breakdown_device,
                residual_replacements_device,
            ) = device_cg(
                vstate.parameters,
                sigma,
                weights,
                gradient,
                initial_solution,
                jnp.asarray(warm_start_used),
                jnp.asarray(active_diag_shift, dtype=jnp.float64),
                metric_diagonal,
                active_preconditioner_diagonal,
                jnp.asarray(self.relative_damping),
                requested_maxiter,
                jnp.asarray(self.residual_target, dtype=jnp.float64),
                jnp.asarray(self.residual_replacement_interval, dtype=jnp.int32),
            )
            (
                iterations,
                residual_norm,
                relative_residual,
                converged,
                breakdown,
                residual_replacements,
                ess_fraction,
                requested_maxiter_host,
            ) = jax.device_get(
                (
                    iterations_device,
                    residual_norm_device,
                    relative_residual_device,
                    converged_device,
                    breakdown_device,
                    residual_replacements_device,
                    ess_fraction_device,
                    requested_maxiter,
                )
            )
            iterations = int(iterations)
            residual_norm = float(residual_norm)
            relative_residual = float(relative_residual)
            converged = bool(converged)
            breakdown = bool(breakdown)
            residual_replacements = int(residual_replacements)
            ess_fraction = float(ess_fraction)
            requested_maxiter_host = int(requested_maxiter_host)
            if not breakdown:
                break
            # In exact arithmetic the damped QGT is positive definite. A
            # breakdown is therefore a numerical/sampling warning: retry once
            # per damping level, then leave target parameters untouched.
            if (
                self.adaptive
                and active_diag_shift < self.adaptive_max_diag_shift
            ):
                active_diag_shift = min(
                    self.adaptive_max_diag_shift,
                    active_diag_shift * self.adaptive_diag_shift_factor,
                )
                curvature_retries += 1
                # A direction solved with a different damping value is not a
                # valid CG initial guess for this retry.
                warm_start_used = False
                initial_solution = _tree_zeros_like(gradient)
                continue
            direction = _tree_zeros_like(gradient)
            residual = gradient
            iterations = 0
            residual_norm = math.inf
            relative_residual = math.inf
            converged = False
            residual_replacements = 0
            update_skipped = True
            break

        cg_extensions = int(iterations > self.maxiter)
        solver_reliable = (
            math.isfinite(relative_residual)
            and relative_residual <= self.adaptive_healthy_residual
        )
        reliable = solver_reliable and ess_fraction >= self.adaptive_ess_threshold
        # ``active_diag_shift`` can differ from the persistent value after a
        # same-iteration curvature retry. Make the next control decision from
        # the damping that actually produced this direction.
        self._active_diag_shift = active_diag_shift
        next_diag_shift, trust_multiplier = self._adapt_controls(reliable=reliable)

        # CG already computed the true residual of the damped linear system.
        # Reconstruct ``S direction`` without launching an additional full
        # JVP/VJP product solely for trust-region diagnostics.
        damping_product = (
            _tree_scale(_tree_multiply(metric_diagonal, direction), active_diag_shift)
            if self.relative_damping
            else _tree_scale(direction, active_diag_shift)
        )
        qgt_product = _tree_add(
            _tree_add(gradient, _tree_scale(residual, -1.0)),
            _tree_scale(damping_product, -1.0),
        )
        metric_norm = float(jax.device_get(jnp.sqrt(jnp.maximum(_tree_dot_real(direction, qgt_product), 0.0))))
        solver_direction = direction
        scale = 1.0
        current_learning_rate = self._learning_rate_at(step)
        effective_trust_region = (
            None
            if self.trust_region is None
            else self.trust_region * trust_multiplier
        )
        if effective_trust_region is not None and metric_norm > 0.0:
            actual_update_norm = current_learning_rate * metric_norm
            if actual_update_norm > effective_trust_region:
                scale = effective_trust_region / actual_update_norm
                direction = _tree_scale(direction, scale)

        # Cache the linear-system solution before trust-region clipping. It is
        # the appropriate initial guess for the next solve with the same
        # damping; the clipped outer update generally is not.
        if update_skipped:
            self._last_solver_direction = None
            self._last_solver_diag_shift = None
            self._last_solver_metric_diagonal = None
        else:
            self._last_solver_direction = solver_direction
            self._last_solver_diag_shift = active_diag_shift
            self._last_solver_metric_diagonal = metric_diagonal

        self.last_info = {
            "CGIterations": iterations,
            "CGResidualNorm": residual_norm,
            "CGRelativeResidual": relative_residual,
            "CGConverged": converged,
            "CGResidualTarget": self.residual_target,
            "CGResidualTargetReached": converged,
            "CGResidualReplacements": residual_replacements,
            "CGOnDevice": True,
            "CGExtensions": cg_extensions,
            "CGRequestedMaxiter": requested_maxiter_host,
            "CGCurvatureRetries": curvature_retries,
            "CGWarmStarted": warm_start_used,
            "DiagonalPreconditioner": self.diagonal_preconditioner,
            "DiagonalRefreshed": diagonal_refreshed,
            "DiagonalProbes": self.diagonal_probes if self.diagonal_preconditioner else 0,
            "DiagonalMode": self.diagonal_mode if self.diagonal_preconditioner else "none",
            "RelativeDamping": self.relative_damping,
            "ESSFraction": ess_fraction,
            "SolverReliable": solver_reliable,
            "UpdateSkipped": update_skipped,
            "AdaptiveEnabled": self.adaptive,
            "DiagShift": active_diag_shift,
            "NextDiagShift": next_diag_shift,
            "QGTDirectionNorm": metric_norm,
            "LearningRate": current_learning_rate,
            "EffectiveTrustRegion": effective_trust_region,
            "AdaptiveTrustMultiplier": trust_multiplier,
            "TrustRegionScale": scale,
        }
        return direction


class WeightedMinSR(WeightedSR):
    """NetKet-style, matrix-free minimum SR for weighted NIS samples.

    NetKet's ``VMC_SR(use_ntk=True, on_the_fly=True)`` solves stochastic
    reconfiguration in sample space.  Its public implementation deliberately
    rejects importance weights, because weighted centering changes the
    Jacobian, right-hand side, and final VJP.  This class implements that
    missing weighted variant without constructing either a dense QGT or a
    dense sample-by-sample neural-tangent kernel.

    For real parameter trees and complex log-wavefunctions, define the real
    sample-space Jacobian

    ``X_i v = sqrt(w_i) [Re(c_i(v)), Im(c_i(v))]``,

    where ``c_i(v) = J_i v - sum_j w_j J_j v``.  The update is the exact
    damped weighted-SR direction expressed in sample space. A positive
    diagonal ``D`` is formed from a Hutchinson estimate of ``diag(S_w)`` and
    the sole public damping value ``lambda``:

    ``D = lambda * (diag(S_w) + mean(diag(S_w)) I)``.

    This is the scale-invariant counterpart of NetKet's ``diag_scale`` plus
    ``diag_shift`` regularisation, without making users tune both quantities.
    The kernel identity is then

    ``D^-1 X.T @ (I + X @ D^-1 @ X.T)^-1 @ b``.

    ``proj_reg`` implements NetKet's optional low-rank projection
    regularisation in that same sample-space system. ``momentum`` implements
    the SPRING update used by :class:`netket.driver.VMC_SR`: the previous
    *unclipped* SR direction is projected through the current weighted
    Jacobian before the solve and added back afterwards. Both operations keep
    the original NIS weights in ``X``; neither turns the pool into resampled
    iid target configurations.

    The implementation applies this operator with JVP/VJP products and a
    fully device-side conjugate-gradient loop.  Only the conventional NetKet
    controls are public: damping, solve budget, tolerance, chunk size, and
    the existing QGT-norm trust region.  Proposal weights remain explicit at
    every step; no resampled ``MCState`` approximation is used.
    """

    def __init__(
        self,
        *,
        diag_shift: float = 1.0e-3,
        maxiter: int = 128,
        tol: float = 1.0e-2,
        chunk_size: int | None = None,
        trust_region: float | None = 0.05,
        learning_rate: float | Callable[[int], float] = 1.0,
        proj_reg: float | None = None,
        momentum: float | None = None,
        direct_solver: str = "auto",
        distributed_jacobian_chunk_size: int = 64,
    ):
        if proj_reg is not None and proj_reg < 0.0:
            raise ValueError("proj_reg must be non-negative or None")
        if momentum is not None and not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1) or None")
        if direct_solver not in {
            "auto",
            "matrix_free",
            "cholesky",
            "distributed_cholesky",
        }:
            raise ValueError(
                "direct_solver must be 'auto', 'matrix_free', 'cholesky', or "
                "'distributed_cholesky'"
            )
        if distributed_jacobian_chunk_size < 1:
            raise ValueError("distributed_jacobian_chunk_size must be positive")
        # Reuse the established weighted-QGT diagonal estimator and held-out
        # residual diagnostic, but keep the MinSR interface intentionally
        # compact. Four leafwise Hutchinson probes, refreshed every 25 target
        # updates, are an internal amortised cost rather than launcher knobs.
        super().__init__(
            diag_shift=diag_shift,
            maxiter=maxiter,
            tol=tol,
            chunk_size=chunk_size,
            trust_region=trust_region,
            learning_rate=learning_rate,
            adaptive=False,
            adaptive_maxiter=maxiter,
            residual_target=tol,
            diagonal_preconditioner=True,
            relative_damping=True,
            diagonal_probes=4,
            diagonal_update_interval=25,
            diagonal_ema=0.9,
            diagonal_mode="per_leaf",
        )
        self.proj_reg = proj_reg
        self.momentum = momentum
        self.direct_solver = direct_solver
        self.distributed_jacobian_chunk_size = distributed_jacobian_chunk_size
        self._device_minsr_kernel_cache: dict[
            tuple[int, int, bool, bool, bool], Callable
        ] = {}
        self._direct_cholesky_kernel_cache: dict[tuple[int, int], Callable] = {}
        self._distributed_cholesky_solve_cache: dict[tuple[int, str], Callable] = {}
        self._last_damping_diagonal = None
        # This is deliberately distinct from Krylov diagnostics. It is the
        # accepted, untrust-clipped update required by SPRING. A paired-energy
        # rejection clears it in ``notify_rejected_update``.
        self._last_momentum_direction = None
        self._max_auto_diag_shift = 1.0e-1
        # A failed Krylov solve is retried once immediately on exactly the
        # same weighted proposal pool. This has no cost on healthy updates.
        self._max_recovery_attempts = 1
        # NetKet's direct kernel/Cholesky solver is excellent for small
        # sample spaces, but its O((2N)^2) memory and O((2N)^3) factorisation
        # cannot serve as the scalable 18-/128-site path. Keep selection
        # automatic and deliberately conservative.
        self._direct_cholesky_max_sample_dimension = 2048
        self._direct_cholesky_max_work_bytes = 512 * 1024**2

    def _scale_invariant_damping_diagonal(self, metric_diagonal, diag_shift):
        """Build ``lambda * (diag(S_w) + mean(diag(S_w)) I)`` matrix-free."""
        total = sum(
            jnp.sum(leaf) for leaf in jax.tree_util.tree_leaves(metric_diagonal)
        )
        size = sum(leaf.size for leaf in jax.tree_util.tree_leaves(metric_diagonal))
        mean_diagonal = total / max(size, 1)
        floor = jnp.asarray(self.diagonal_floor, dtype=mean_diagonal.dtype)
        mean_diagonal = jnp.maximum(mean_diagonal, floor)
        return jax.tree_util.tree_map(
            lambda diagonal: jnp.asarray(
                diag_shift * (diagonal + mean_diagonal), dtype=diagonal.dtype
            ),
            metric_diagonal,
        )

    def _next_automatic_diag_shift(
        self, active_diag_shift: float, reliable: bool
    ) -> tuple[float, str]:
        """Restore the established recovery-aware relative-damping policy.

        A reliable solve always gently relaxes damping, including when it
        used much of the iteration budget.  Only a failed solve, a failed
        same-pool recovery, or the existing paired-energy rollback is allowed
        to make the next target update more conservative.
        """
        if reliable:
            return (
                max(self.diag_shift, active_diag_shift / 1.25),
                "successful_relax",
            )
        return (
            min(self._max_auto_diag_shift, 2.0 * active_diag_shift),
            "unreliable_increase",
        )

    def _direct_cholesky_is_safe(self, vstate) -> bool:
        """Whether NetKet-style dense kernel factorisation fits this pool.

        The estimate accounts for the realified Jacobian plus several dense
        sample-space matrices. It is intentionally a safety gate, not a user
        tuning knob: larger systems always retain matrix-free weighted MinSR.
        """
        if self.direct_solver == "matrix_free":
            return False
        if self.direct_solver == "cholesky":
            # Explicit experimental override. The caller is responsible for
            # selecting a single-device run with enough memory; this is useful
            # for controlled CG-versus-Cholesky ablations, not production.
            return True
        sample_dimension = 2 * int(vstate.n_samples)
        if sample_dimension > self._direct_cholesky_max_sample_dimension:
            return False
        leaves = jax.tree_util.tree_leaves(vstate.parameters)
        parameter_count = sum(leaf.size for leaf in leaves)
        element_bytes = max(jnp.dtype(leaf.dtype).itemsize for leaf in leaves)
        jacobian_bytes = sample_dimension * parameter_count * element_bytes
        kernel_bytes = sample_dimension * sample_dimension * element_bytes
        # Jacobian plus kernel, factor, and solve workspace.
        return jacobian_bytes + 3 * kernel_bytes <= self._direct_cholesky_max_work_bytes

    def _direct_cholesky_kernel(self, vstate, n_samples: int):
        """Build NetKet-style dense weighted-kernel Cholesky MinSR for small N."""
        key = (id(vstate.model), n_samples)
        if key in self._direct_cholesky_kernel_cache:
            return self._direct_cholesky_kernel_cache[key]

        model = vstate.model
        model_state = vstate.model_state
        _, unravel_parameters = jax.flatten_util.ravel_pytree(vstate.parameters)

        def logpsi_parts(parameters, sigma):
            values = jnp.asarray(
                model.apply({"params": parameters, **model_state}, sigma)
            ).reshape(-1)
            return jnp.stack((jnp.real(values), jnp.imag(values)), axis=-1)

        def solve(
            parameters,
            sigma,
            weights,
            local_energies,
            diag_shift,
            previous_direction,
            momentum,
            proj_reg,
        ):
            weights = jnp.asarray(weights, dtype=jnp.real(local_energies).dtype)
            weights = weights / jnp.sum(weights)
            local_energies = jnp.asarray(local_energies)
            # This is the explicit realified Jacobian used by NetKet's dense
            # kernel route, but centered and scaled with NIS weights rather
            # than assuming iid target samples.
            jacobian_tree = jax.jacrev(
                lambda params: logpsi_parts(params, sigma)
            )(parameters)
            jacobian = jnp.concatenate(
                [
                    leaf.reshape(2 * n_samples, -1)
                    for leaf in jax.tree_util.tree_leaves(jacobian_tree)
                ],
                axis=1,
            )
            jacobian = jacobian.reshape(n_samples, 2, -1)
            centered = jacobian - jnp.einsum("n,nkp->kp", weights, jacobian)
            sample_jacobian = (
                jnp.sqrt(jnp.maximum(weights, 0.0))[:, None, None] * centered
            ).reshape(2 * n_samples, -1)

            energy = jnp.sum(weights * local_energies)
            rhs = (
                2.0
                * jnp.sqrt(jnp.maximum(weights, 0.0))
                * (local_energies - energy)
            )
            rhs = jnp.stack((jnp.real(rhs), jnp.imag(rhs)), axis=-1).reshape(-1)

            previous_flat, _ = jax.flatten_util.ravel_pytree(previous_direction)
            rhs = rhs - momentum * (sample_jacobian @ previous_flat)

            kernel = sample_jacobian @ sample_jacobian.T
            kernel = 0.5 * (kernel + kernel.T)
            shifted_kernel = kernel + diag_shift * jnp.eye(
                kernel.shape[0], dtype=kernel.dtype
            )
            # NetKet adds ``proj_reg / N_mc * 11.T`` for uniform samples.
            # With NIS weights, the weighted-centering null vector is
            # ``sqrt(w)`` rather than ``1/sqrt(N_mc) * 1``. Regularising its
            # realified outer product preserves that property and reduces to
            # the NetKet expression exactly when all weights are uniform.
            projection_vector = jnp.repeat(
                jnp.sqrt(jnp.maximum(weights, 0.0)), 2
            )
            shifted_kernel = shifted_kernel + proj_reg * jnp.outer(
                projection_vector, projection_vector
            )
            factor = jnp.linalg.cholesky(shifted_kernel)
            auxiliary = jsp.linalg.solve_triangular(factor, rhs, lower=True)
            auxiliary = jsp.linalg.solve_triangular(
                factor.T, auxiliary, lower=False
            )
            direction_flat = (
                sample_jacobian.T @ auxiliary + momentum * previous_flat
            )
            direction = unravel_parameters(direction_flat)
            metric_vector = sample_jacobian @ direction_flat
            metric_norm = jnp.sqrt(
                jnp.maximum(jnp.vdot(metric_vector, metric_vector).real, 0.0)
            )
            finite = jnp.all(jnp.isfinite(factor)) & jnp.all(
                jnp.isfinite(auxiliary)
            )
            return direction, metric_norm, finite

        kernel = jax.jit(solve)
        self._direct_cholesky_kernel_cache[key] = kernel
        return kernel

    def _direct_cholesky_update(
        self, vstate, sigma, weights, local_energies, step
    ):
        """Use a weighted analogue of NetKet's direct kernel Cholesky solve."""
        if getattr(vstate, "uses_native_sharding", False):
            sigma = gather(sigma)
            weights = gather(weights)
            local_energies = gather(local_energies)
        n_samples = int(sigma.shape[0])
        solver = self._direct_cholesky_kernel(vstate, n_samples)
        initial_diag_shift = self._active_diag_shift
        active_diag_shift = initial_diag_shift
        recovery_attempts = 0
        previous_direction = self._last_momentum_direction
        momentum_enabled = self.momentum is not None and self.momentum > 0.0
        momentum = (
            self.momentum
            if momentum_enabled and previous_direction is not None
            else 0.0
        )
        if previous_direction is None:
            previous_direction = _tree_zeros_like(vstate.parameters)
        proj_reg = self.proj_reg if self.proj_reg is not None else 0.0

        direction, metric_norm, finite = solver(
            vstate.parameters,
            sigma,
            weights,
            local_energies,
            jnp.asarray(active_diag_shift, dtype=jnp.float64),
            previous_direction,
            jnp.asarray(momentum, dtype=jnp.float64),
            jnp.asarray(proj_reg, dtype=jnp.float64),
        )
        finite = bool(jax.device_get(finite))
        metric_norm = float(jax.device_get(metric_norm))
        while not finite and recovery_attempts < self._max_recovery_attempts:
            recovery_attempts += 1
            active_diag_shift = min(1.0e-1, 4.0 * active_diag_shift)
            direction, metric_norm, finite = solver(
                vstate.parameters,
                sigma,
                weights,
                local_energies,
                jnp.asarray(active_diag_shift, dtype=jnp.float64),
                previous_direction,
                jnp.asarray(momentum, dtype=jnp.float64),
                jnp.asarray(proj_reg, dtype=jnp.float64),
            )
            finite = bool(jax.device_get(finite))
            metric_norm = float(jax.device_get(metric_norm))

        if finite:
            self._active_diag_shift = max(
                self.diag_shift, active_diag_shift / 1.25
            )
        else:
            self._active_diag_shift = min(1.0e-1, 2.0 * active_diag_shift)
            direction = _tree_zeros_like(vstate.parameters)
            metric_norm = 0.0

        raw_direction = direction
        scale = 1.0
        learning_rate = self._learning_rate_at(step)
        if finite and self.trust_region is not None and metric_norm > 0.0:
            outer_norm = learning_rate * metric_norm
            if outer_norm > self.trust_region:
                scale = self.trust_region / outer_norm
                direction = _tree_scale(direction, scale)

        if finite:
            self._last_solver_direction = raw_direction
            self._last_solver_diag_shift = active_diag_shift
            self._last_solver_metric_diagonal = _tree_ones_like(raw_direction)
            self._last_momentum_direction = raw_direction
        else:
            self._last_solver_direction = None
            self._last_solver_diag_shift = None
            self._last_solver_metric_diagonal = None
            self._last_momentum_direction = None

        element_bytes = max(
            jnp.dtype(leaf.dtype).itemsize
            for leaf in jax.tree_util.tree_leaves(vstate.parameters)
        )
        self.last_info = {
            "SRMethod": "weighted_minsr_cholesky",
            "LinearSolver": "cholesky",
            "SolveOnDevice": True,
            "CGIterations": 0,
            "CGResidualNorm": None,
            "CGRelativeResidual": None,
            "CGConverged": None,
            "CGResidualTarget": None,
            "CGResidualTargetReached": None,
            "CGOnDevice": False,
            "CGExtensions": recovery_attempts,
            "CGRequestedMaxiter": 0,
            "CGCurvatureRetries": 0,
            "CGRecoveryAttempts": recovery_attempts,
            "CGWarmStarted": False,
            "SRMomentum": momentum,
            "SRMomentumEnabled": momentum_enabled,
            "SRMomentumActive": momentum > 0.0,
            "ProjectionRegularization": self.proj_reg,
            "ProjectionRegularizationEnabled": (
                self.proj_reg is not None and self.proj_reg > 0.0
            ),
            "DiagonalPreconditioner": False,
            "DiagonalRefreshed": False,
            "DiagonalProbes": 0,
            "DiagonalMode": "none",
            "RelativeDamping": False,
            "ScaleInvariantDamping": False,
            "ESSFraction": float(jax.device_get(self._ess_fraction(vstate, weights))),
            "SolverReliable": finite,
            "UpdateSkipped": not finite,
            "AutomaticDamping": True,
            "InitialDiagShift": initial_diag_shift,
            "DiagShift": active_diag_shift,
            "NextDiagShift": self._active_diag_shift,
            "QGTDirectionNorm": metric_norm,
            "LearningRate": learning_rate,
            "EffectiveTrustRegion": self.trust_region,
            "TrustRegionScale": scale,
            "MinSRSampleSpaceDimension": 2 * n_samples,
            "DenseKernelBytes": (2 * n_samples) ** 2 * element_bytes,
        }
        return direction

    def _distributed_cholesky_builder(
        self, vstate, number_of_devices: int, samples_per_device: int
    ):
        """Build cached multi-GPU primitives for a dense weighted kernel.

        The returned Jacobian is only materialised for one sample shard per
        device.  A ring exchange then forms the complete sample-space kernel
        without ever gathering the ``2*N_samples`` by ``N_parameters`` matrix
        on an accelerator. This is deliberately an 18-site diagnostic path:
        standard JAX Cholesky is replicated after construction rather than
        being a true distributed factorisation.
        """
        jacobian_chunk_size = math.gcd(
            samples_per_device, self.distributed_jacobian_chunk_size
        )
        number_of_jacobian_chunks = samples_per_device // jacobian_chunk_size
        parameter_leaves = jax.tree_util.tree_leaves(vstate.parameters)
        parameter_count = sum(leaf.size for leaf in parameter_leaves)
        parameter_dtype = parameter_leaves[0].dtype
        key = (
            id(vstate.model),
            number_of_devices,
            samples_per_device,
            jacobian_chunk_size,
        )
        cache = getattr(self, "_distributed_cholesky_builder_cache", {})
        if key in cache:
            return cache[key]

        model = vstate.model
        model_state = jax.device_get(vstate.model_state)
        mesh = jax.sharding.get_abstract_mesh()
        axis_name = "S"
        realified_samples_per_device = 2 * samples_per_device
        ring_permutation = tuple(
            (source, (source + 1) % number_of_devices)
            for source in range(number_of_devices)
        )

        def logpsi_parts(parameters, sigma):
            values = jnp.asarray(
                model.apply({"params": parameters, **model_state}, sigma)
            ).reshape(-1)
            return jnp.stack((jnp.real(values), jnp.imag(values)), axis=-1)

        def raw_jacobian_chunk(parameters, sigma_chunk):
            jacobian_tree = jax.jacrev(
                lambda current_parameters: logpsi_parts(
                    current_parameters, sigma_chunk
                )
            )(parameters)
            return jnp.concatenate(
                [
                    leaf.reshape(jacobian_chunk_size, 2, -1)
                    for leaf in jax.tree_util.tree_leaves(jacobian_tree)
                ],
                axis=2,
            )

        def local_weighted_jacobian(parameters, sigma, weights):
            # A batched jacrev over all local walkers makes reverse-mode
            # workspace scale with the whole proposal shard. Compute fixed
            # chunks twice instead: first for the global weighted mean, then
            # for the centered weighted rows retained by this device.
            jacobian_dtype = jnp.result_type(parameter_dtype, weights.dtype)
            def mean_body(chunk_index, local_mean):
                start = chunk_index * jacobian_chunk_size
                sigma_chunk = jax.lax.dynamic_slice_in_dim(
                    sigma, start, jacobian_chunk_size, axis=0
                )
                weights_chunk = jax.lax.dynamic_slice_in_dim(
                    weights, start, jacobian_chunk_size, axis=0
                )
                jacobian_chunk = raw_jacobian_chunk(parameters, sigma_chunk)
                return local_mean + jnp.einsum(
                    "n,ncp->cp", weights_chunk, jacobian_chunk
                )

            local_mean = jax.lax.fori_loop(
                0,
                number_of_jacobian_chunks,
                mean_body,
                jnp.zeros((2, parameter_count), dtype=jacobian_dtype),
            )
            weighted_mean = jax.lax.psum(
                local_mean, axis_name
            )
            local_jacobian = jnp.zeros(
                (realified_samples_per_device, parameter_count),
                dtype=jacobian_dtype,
            )

            def centered_body(chunk_index, result):
                start = chunk_index * jacobian_chunk_size
                sigma_chunk = jax.lax.dynamic_slice_in_dim(
                    sigma, start, jacobian_chunk_size, axis=0
                )
                weights_chunk = jax.lax.dynamic_slice_in_dim(
                    weights, start, jacobian_chunk_size, axis=0
                )
                jacobian_chunk = raw_jacobian_chunk(parameters, sigma_chunk)
                root_weights = jnp.sqrt(jnp.maximum(weights_chunk, 0.0))[
                    :, None, None
                ]
                centered_chunk = (
                    root_weights * (jacobian_chunk - weighted_mean)
                ).reshape(2 * jacobian_chunk_size, parameter_count)
                return jax.lax.dynamic_update_slice(
                    result,
                    centered_chunk,
                    (
                        2 * start,
                        jnp.asarray(0, dtype=start.dtype),
                    ),
                )

            return jax.lax.fori_loop(
                0, number_of_jacobian_chunks, centered_body, local_jacobian
            )

        def local_kernel_rows(local_jacobian):
            owner = jax.lax.axis_index(axis_name)
            transported_jacobian = local_jacobian
            kernel_rows = jnp.zeros(
                (
                    realified_samples_per_device,
                    number_of_devices * realified_samples_per_device,
                ),
                dtype=local_jacobian.dtype,
            )
            for block_index in range(number_of_devices):
                block = local_jacobian @ transported_jacobian.T
                kernel_rows = jax.lax.dynamic_update_slice(
                    kernel_rows,
                    block,
                    (
                        jnp.asarray(0, dtype=owner.dtype),
                        owner
                        * jnp.asarray(
                            realified_samples_per_device, dtype=owner.dtype
                        ),
                    ),
                )
                if block_index + 1 < number_of_devices:
                    transported_jacobian = jax.lax.ppermute(
                        transported_jacobian, axis_name, ring_permutation
                    )
                    owner = (owner - 1) % number_of_devices
            return kernel_rows

        def local_sample_product(local_jacobian, parameter_direction):
            return local_jacobian @ parameter_direction

        def local_adjoint(local_jacobian, sample_vector):
            local_partial = local_jacobian.T @ sample_vector
            return jax.lax.psum(local_partial, axis_name)

        def local_metric_norm_squared(local_jacobian, parameter_direction):
            local_product = local_jacobian @ parameter_direction
            return jax.lax.psum(jnp.vdot(local_product, local_product).real, axis_name)

        def shard(function, in_specs, out_specs):
            return jax.jit(
                jax.shard_map(
                    function,
                    mesh=mesh,
                    in_specs=in_specs,
                    out_specs=out_specs,
                    check_vma=False,
                )
            )

        builder = {
            "jacobian": shard(
                local_weighted_jacobian,
                (P(), P("S", None), P("S")),
                P("S", None),
            ),
            "kernel": shard(
                local_kernel_rows,
                P("S", None),
                P("S", None),
            ),
            "sample_product": shard(
                local_sample_product,
                (P("S", None), P()),
                P("S"),
            ),
            "adjoint": shard(
                local_adjoint,
                (P("S", None), P("S")),
                P(),
            ),
            "metric_norm_squared": shard(
                local_metric_norm_squared,
                (P("S", None), P()),
                P(),
            ),
            "replicate_parameter": jax.jit(
                lambda parameter_vector: parameter_vector,
                out_shardings=NamedSharding(mesh, P()),
            ),
            "shard_sample": jax.jit(
                lambda sample_vector: sample_vector,
                out_shardings=NamedSharding(mesh, P("S")),
            ),
        }
        cache[key] = builder
        self._distributed_cholesky_builder_cache = cache
        return builder

    def _distributed_cholesky_solver(self, dimension: int, dtype):
        """Return a single-device dense Cholesky solve cached by matrix shape."""
        key = (dimension, np.dtype(dtype).str)
        if key in self._distributed_cholesky_solve_cache:
            return self._distributed_cholesky_solve_cache[key]

        def solve(kernel, rhs, diag_shift, proj_reg, projection_vector):
            shifted_kernel = kernel + diag_shift * jnp.eye(
                dimension, dtype=kernel.dtype
            )
            shifted_kernel = shifted_kernel + proj_reg * jnp.outer(
                projection_vector, projection_vector
            )
            factor = jnp.linalg.cholesky(shifted_kernel)
            auxiliary = jsp.linalg.solve_triangular(factor, rhs, lower=True)
            auxiliary = jsp.linalg.solve_triangular(
                factor.T, auxiliary, lower=False
            )
            finite = jnp.all(jnp.isfinite(factor)) & jnp.all(
                jnp.isfinite(auxiliary)
            )
            return auxiliary, finite

        solver = jax.jit(solve)
        self._distributed_cholesky_solve_cache[key] = solver
        return solver

    def _distributed_cholesky_update(
        self, vstate, sigma, weights, local_energies, step
    ):
        """Dense weighted MinSR kernel from distributed local Jacobian blocks.

        This never materialises the full Jacobian on one GPU. Each local shard
        retains only ``2*N_samples/number_of_devices`` rows, and a ring
        exchange constructs its rows of the dense sample kernel. The final
        factorisation is replicated because JAX has no distributed Cholesky.
        """
        number_of_devices = jax.local_device_count()
        if number_of_devices < 2:
            raise RuntimeError(
                "distributed_cholesky requires at least two local JAX devices"
            )

        if not getattr(vstate, "uses_native_sharding", False):
            raise RuntimeError(
                "distributed_cholesky requires a NetKet-native sharded proposal pool"
            )
        sigma_host = np.asarray(jax.device_get(gather(sigma)))
        weights_host = np.asarray(jax.device_get(gather(weights)))
        local_energies_host = np.asarray(jax.device_get(gather(local_energies)))

        sample_count = int(sigma_host.shape[0])
        if sample_count % number_of_devices != 0:
            raise ValueError(
                "distributed_cholesky requires the proposal count to be divisible "
                f"by the local device count ({sample_count} vs {number_of_devices})"
            )
        samples_per_device = sample_count // number_of_devices
        realified_samples_per_device = 2 * samples_per_device
        dimension = 2 * sample_count
        weights_host = np.asarray(weights_host, dtype=local_energies_host.real.dtype)
        weights_host = weights_host / np.sum(weights_host)
        builder = self._distributed_cholesky_builder(
            vstate, number_of_devices, samples_per_device
        )
        local_jacobians = builder["jacobian"](vstate.parameters, sigma, weights)
        local_kernel_rows = builder["kernel"](local_jacobians)
        # The full kernel is only O((2*N_samples)^2), so it is inexpensive to
        # replicate after the distributed Jacobian contractions. In contrast,
        # the full Jacobian is never gathered.
        kernel = gather(local_kernel_rows)
        kernel = 0.5 * (kernel + kernel.T)

        energy = np.sum(weights_host * local_energies_host)
        rhs_complex = (
            2.0
            * np.sqrt(np.maximum(weights_host, 0.0))
            * (local_energies_host - energy)
        )
        rhs = np.stack((rhs_complex.real, rhs_complex.imag), axis=-1).reshape(-1)
        projection_vector = np.repeat(np.sqrt(np.maximum(weights_host, 0.0)), 2)

        previous_direction = self._last_momentum_direction
        momentum_enabled = self.momentum is not None and self.momentum > 0.0
        use_momentum = momentum_enabled and previous_direction is not None
        if use_momentum:
            previous_flat, _ = jax.flatten_util.ravel_pytree(
                jax.device_get(previous_direction)
            )
            previous_flat = np.asarray(previous_flat, dtype=kernel.dtype)
            previous_products = builder["sample_product"](
                local_jacobians,
                builder["replicate_parameter"](
                    jnp.asarray(previous_flat, dtype=kernel.dtype)
                ),
            )
            rhs = rhs - self.momentum * np.concatenate(
                list(
                    np.asarray(
                        jax.device_get(gather(previous_products))
                    ).reshape(number_of_devices, realified_samples_per_device)
                ),
                axis=0,
            )
        else:
            previous_flat, _ = jax.flatten_util.ravel_pytree(vstate.parameters)
            previous_flat = np.zeros_like(np.asarray(previous_flat), dtype=kernel.dtype)

        initial_diag_shift = self._active_diag_shift
        active_diag_shift = initial_diag_shift
        recovery_attempts = 0
        solver = self._distributed_cholesky_solver(dimension, kernel.dtype)
        rhs_device = builder["replicate_parameter"](
            jnp.asarray(rhs, dtype=kernel.dtype)
        )
        projection_device = builder["replicate_parameter"](
            jnp.asarray(projection_vector, dtype=kernel.dtype)
        )
        proj_reg = self.proj_reg if self.proj_reg is not None else 0.0

        auxiliary, finite = solver(
            kernel,
            rhs_device,
            jnp.asarray(active_diag_shift, dtype=kernel.dtype),
            jnp.asarray(proj_reg, dtype=kernel.dtype),
            projection_device,
        )
        finite = bool(jax.device_get(finite))
        while not finite and recovery_attempts < self._max_recovery_attempts:
            recovery_attempts += 1
            active_diag_shift = min(1.0e-1, 4.0 * active_diag_shift)
            auxiliary, finite = solver(
                kernel,
                rhs_device,
                jnp.asarray(active_diag_shift, dtype=kernel.dtype),
                jnp.asarray(proj_reg, dtype=kernel.dtype),
                projection_device,
            )
            finite = bool(jax.device_get(finite))

        if finite:
            auxiliary_shards = builder["shard_sample"](auxiliary)
            direction_replicas = builder["adjoint"](
                local_jacobians, auxiliary_shards
            )
            raw_flat_direction = np.asarray(jax.device_get(direction_replicas))
            if use_momentum:
                raw_flat_direction = raw_flat_direction + self.momentum * previous_flat
            metric_norm_squared = builder["metric_norm_squared"](
                local_jacobians,
                builder["replicate_parameter"](
                    jnp.asarray(raw_flat_direction, dtype=kernel.dtype)
                ),
            )
            metric_norm = float(
                np.sqrt(max(float(jax.device_get(metric_norm_squared)), 0.0))
            )
            self._active_diag_shift = max(
                self.diag_shift, active_diag_shift / 1.25
            )
        else:
            raw_flat_direction = np.zeros_like(previous_flat)
            metric_norm = 0.0
            self._active_diag_shift = min(1.0e-1, 2.0 * active_diag_shift)

        _, unravel_parameters = jax.flatten_util.ravel_pytree(vstate.parameters)
        raw_direction = unravel_parameters(
            builder["replicate_parameter"](
                jnp.asarray(raw_flat_direction, dtype=kernel.dtype)
            )
        )
        raw_direction = jax.tree_util.tree_map(
            lambda value, parameter: jnp.asarray(value, dtype=parameter.dtype),
            raw_direction,
            vstate.parameters,
        )
        direction = raw_direction
        scale = 1.0
        learning_rate = self._learning_rate_at(step)
        if finite and self.trust_region is not None and metric_norm > 0.0:
            outer_norm = learning_rate * metric_norm
            if outer_norm > self.trust_region:
                scale = self.trust_region / outer_norm
                direction = _tree_scale(direction, scale)

        if finite:
            self._last_solver_direction = raw_direction
            self._last_solver_diag_shift = active_diag_shift
            self._last_solver_metric_diagonal = _tree_ones_like(raw_direction)
            self._last_momentum_direction = raw_direction
        else:
            self._last_solver_direction = None
            self._last_solver_diag_shift = None
            self._last_solver_metric_diagonal = None
            self._last_momentum_direction = None

        parameter_count = sum(
            leaf.size for leaf in jax.tree_util.tree_leaves(vstate.parameters)
        )
        element_bytes = np.dtype(kernel.dtype).itemsize
        self.last_info = {
            "SRMethod": "weighted_minsr_distributed_cholesky",
            "LinearSolver": "cholesky",
            "SolveOnDevice": True,
            "KernelConstruction": "distributed_local_jacobian_ring",
            "KernelConstructionDevices": number_of_devices,
            "CholeskyFactorization": "replicated_per_device",
            "CGIterations": 0,
            "CGResidualNorm": None,
            "CGRelativeResidual": None,
            "CGConverged": None,
            "CGResidualTarget": None,
            "CGResidualTargetReached": None,
            "CGOnDevice": False,
            "CGExtensions": recovery_attempts,
            "CGRequestedMaxiter": 0,
            "CGCurvatureRetries": 0,
            "CGRecoveryAttempts": recovery_attempts,
            "CGWarmStarted": False,
            "SRMomentum": self.momentum if use_momentum else 0.0,
            "SRMomentumEnabled": momentum_enabled,
            "SRMomentumActive": use_momentum,
            "ProjectionRegularization": self.proj_reg,
            "ProjectionRegularizationEnabled": (
                self.proj_reg is not None and self.proj_reg > 0.0
            ),
            "DiagonalPreconditioner": False,
            "DiagonalRefreshed": False,
            "DiagonalProbes": 0,
            "DiagonalMode": "none",
            "RelativeDamping": False,
            "ScaleInvariantDamping": False,
            "ESSFraction": float(jax.device_get(self._ess_fraction(vstate, weights))),
            "SolverReliable": finite,
            "UpdateSkipped": not finite,
            "AutomaticDamping": True,
            "InitialDiagShift": initial_diag_shift,
            "DiagShift": active_diag_shift,
            "NextDiagShift": self._active_diag_shift,
            "QGTDirectionNorm": metric_norm,
            "LearningRate": learning_rate,
            "EffectiveTrustRegion": self.trust_region,
            "TrustRegionScale": scale,
            "MinSRSampleSpaceDimension": dimension,
            "DenseKernelBytes": dimension**2 * element_bytes,
            "DistributedJacobianBytes": dimension * parameter_count * element_bytes,
            "DistributedJacobianBytesPerDevice": (
                dimension * parameter_count * element_bytes // number_of_devices
            ),
        }
        return direction

    def _device_minsr_kernel(
        self, vstate, local_chunk_size: int, *, use_momentum: bool
    ):
        """Return a cached weighted sample-space conjugate-gradient kernel.

        The sample axis stays sharded under NetKet's native SPMD mesh.  The
        only collectives are the scalar sample reductions and the parameter
        VJP sum, so this remains a matrix-free multi-GPU implementation.
        """
        sharded = bool(getattr(vstate, "uses_native_sharding", False))
        use_projection_regularization = (
            self.proj_reg is not None and self.proj_reg > 0.0
        )
        key = (
            id(vstate.model),
            local_chunk_size,
            sharded,
            use_momentum,
            use_projection_regularization,
        )
        if key in self._device_minsr_kernel_cache:
            return self._device_minsr_kernel_cache[key]

        model = vstate.model
        model_state = vstate.model_state

        def apply_logpsi(parameters, sigma):
            return jnp.asarray(
                model.apply({"params": parameters, **model_state}, sigma)
            ).reshape(-1)

        def global_sum(values):
            result = jnp.sum(values)
            if sharded:
                result = jax.lax.psum(result, "S")
            return result

        def sample_dot(left, right):
            return global_sum(jnp.sum(left * right))

        def replicas_all(value):
            value = jnp.asarray(value, dtype=jnp.bool_)
            if sharded:
                return jax.lax.pmin(value.astype(jnp.int32), "S").astype(jnp.bool_)
            return value

        def tangent_to_sample(parameters, tangent, sigma, weights):
            directional_values = []
            directional_mean = None
            for start in range(0, sigma.shape[0], local_chunk_size):
                stop = min(start + local_chunk_size, sigma.shape[0])
                sigma_chunk = sigma[start:stop]
                weight_chunk = weights[start:stop]
                _, directional = jax.jvp(
                    lambda params: apply_logpsi(params, sigma_chunk),
                    (parameters,),
                    (tangent,),
                )
                directional_values.append((weight_chunk, directional))
                if directional_mean is None:
                    directional_mean = jnp.zeros((), dtype=directional.dtype)
                directional_mean = directional_mean + jnp.sum(weight_chunk * directional)
            if sharded:
                directional_mean = jax.lax.psum(directional_mean, "S")

            values = []
            for weight_chunk, directional in directional_values:
                centered = directional - directional_mean
                root_weight = jnp.sqrt(jnp.maximum(weight_chunk, 0.0))
                values.append(
                    root_weight[:, None]
                    * jnp.stack((jnp.real(centered), jnp.imag(centered)), axis=-1)
                )
            return jnp.concatenate(values, axis=0)

        def sample_adjoint(parameters, sample_vector, sigma, weights):
            root_weight = jnp.sqrt(jnp.maximum(weights, 0.0))
            cotangent = root_weight * (
                sample_vector[:, 0] + 1j * sample_vector[:, 1]
            )
            # ``sqrt(W) C`` is the sample-space Jacobian.  Its real tangent
            # adjoint is ``C^dagger sqrt(W)``, including this weighted
            # centering of the VJP cotangent.
            cotangent = cotangent - weights * global_sum(cotangent)

            result = _tree_zeros_like(parameters)
            for start in range(0, sigma.shape[0], local_chunk_size):
                stop = min(start + local_chunk_size, sigma.shape[0])
                sigma_chunk = sigma[start:stop]
                output, pullback = jax.vjp(
                    lambda params: apply_logpsi(params, sigma_chunk), parameters
                )
                cotangent_chunk = cotangent[start:stop]
                if jnp.issubdtype(output.dtype, jnp.complexfloating):
                    cotangent_chunk = jnp.conj(cotangent_chunk)
                else:
                    # A real-output ansatz has no imaginary tangent channel;
                    # JAX correctly requires a real VJP cotangent here.
                    cotangent_chunk = jnp.real(cotangent_chunk).astype(output.dtype)
                partial = pullback(cotangent_chunk)[0]
                result = _tree_add(result, partial)
            if sharded:
                result = jax.tree_util.tree_map(
                    lambda leaf: jax.lax.psum(leaf, "S"), result
                )
            return result

        def cg(
            parameters,
            sigma,
            weights,
            local_energies,
            damping_diagonal,
            previous_direction,
            momentum,
            proj_reg,
            maxiter,
            residual_target,
        ):
            local_energies = jnp.asarray(local_energies)
            # The sample-space CG vector is realified log-wavefunction output,
            # not a parameter vector.  Flax permits float32 parameters with a
            # float64 complex output, so its dtype must follow local energies
            # to keep VJP cotangents type-compatible.
            sample_dtype = jnp.real(local_energies).dtype
            weights = jnp.asarray(weights, dtype=sample_dtype)
            residual_target = jnp.asarray(residual_target, dtype=sample_dtype)
            damping_diagonal = jax.tree_util.tree_map(
                lambda damping, parameter: jnp.asarray(
                    damping, dtype=parameter.dtype
                ),
                damping_diagonal,
                parameters,
            )
            previous_direction = jax.tree_util.tree_map(
                lambda previous, parameter: jnp.asarray(
                    previous, dtype=parameter.dtype
                ),
                previous_direction,
                parameters,
            )
            momentum = jnp.asarray(momentum, dtype=sample_dtype)
            proj_reg = jnp.asarray(proj_reg, dtype=sample_dtype)
            energy = global_sum(weights * local_energies)
            root_weight = jnp.sqrt(jnp.maximum(weights, 0.0))
            rhs_complex = 2.0 * root_weight * (local_energies - energy)
            rhs = jnp.stack(
                (jnp.real(rhs_complex), jnp.imag(rhs_complex)), axis=-1
            ).astype(sample_dtype)
            if use_momentum:
                rhs = rhs - momentum * tangent_to_sample(
                    parameters, previous_direction, sigma, weights
                )

            def matvec(sample_vector):
                parameter_vector = sample_adjoint(
                    parameters, sample_vector, sigma, weights
                )
                parameter_vector = _tree_divide(
                    parameter_vector, damping_diagonal
                )
                product = tangent_to_sample(
                    parameters, parameter_vector, sigma, weights
                )
                result = product + sample_vector
                if use_projection_regularization:
                    # In the weighted realified kernel the centering null
                    # vector is [sqrt(w), sqrt(w)], not a vector of ones.
                    # ``proj_reg * uu.T`` is therefore the NIS-preserving
                    # extension of NetKet's uniform-sample projection term.
                    projection = proj_reg * global_sum(
                        jnp.sum(root_weight[:, None] * sample_vector)
                    )
                    result = result + projection * root_weight[:, None]
                return result

            rhs_norm_sq = sample_dot(rhs, rhs)
            rhs_norm = jnp.sqrt(jnp.maximum(rhs_norm_sq, 0.0))
            initial_residual = rhs
            # The production path deliberately uses the identity
            # preconditioner.  A four-probe sample-kernel diagonal estimate
            # proved too noisy for late-stage 18-site solves, causing valid
            # weighted SR updates to be skipped.  The metric diagonal above
            # remains in D for scale-invariant damping; it is not used as a
            # Krylov preconditioner here.
            kernel_diagonal = jnp.ones_like(rhs)
            initial_preconditioned = initial_residual / kernel_diagonal
            initial_preconditioned_dot = sample_dot(
                initial_residual, initial_preconditioned
            )
            initially_converged = replicas_all(
                rhs_norm <= residual_target * rhs_norm
            )
            initial_state = (
                jnp.zeros_like(rhs),
                initial_residual,
                initial_preconditioned,
                initial_preconditioned_dot,
                rhs_norm_sq,
                jnp.asarray(0, dtype=jnp.int32),
                initially_converged,
                jnp.asarray(False),
            )

            def condition(state):
                _, _, _, _, _, iteration, converged, breakdown = state
                return (iteration < maxiter) & (~converged) & (~breakdown)

            def body(state):
                (
                    solution,
                    residual,
                    direction,
                    preconditioned_dot,
                    residual_norm_sq,
                    iteration,
                    _,
                    _,
                ) = state
                matvec_direction = matvec(direction)
                curvature = sample_dot(direction, matvec_direction)
                valid_curvature = replicas_all(
                    jnp.isfinite(curvature) & (curvature > 0.0)
                )

                def accepted_step(_):
                    alpha = preconditioned_dot / curvature
                    next_solution = solution + alpha * direction
                    next_residual = residual - alpha * matvec_direction
                    next_residual_norm_sq = sample_dot(next_residual, next_residual)
                    valid_residual = replicas_all(
                        jnp.isfinite(next_residual_norm_sq)
                        & (next_residual_norm_sq >= 0.0)
                    )

                    def finite_residual_step(_):
                        next_preconditioned = next_residual / kernel_diagonal
                        next_preconditioned_dot = sample_dot(
                            next_residual, next_preconditioned
                        )
                        beta = next_preconditioned_dot / jnp.maximum(
                            preconditioned_dot,
                            jnp.finfo(sample_dtype).tiny,
                        )
                        next_direction = next_preconditioned + beta * direction
                        next_norm = jnp.sqrt(jnp.maximum(next_residual_norm_sq, 0.0))
                        next_converged = replicas_all(
                            next_norm <= residual_target * rhs_norm
                        )
                        return (
                            next_solution,
                            next_residual,
                            next_direction,
                            next_preconditioned_dot,
                            next_residual_norm_sq,
                            iteration + 1,
                            next_converged,
                            jnp.asarray(False),
                        )

                    def invalid_residual_step(_):
                        return (
                            solution,
                            residual,
                            direction,
                            preconditioned_dot,
                            residual_norm_sq,
                            iteration + 1,
                            jnp.asarray(False),
                            jnp.asarray(True),
                        )

                    return jax.lax.cond(
                        valid_residual,
                        finite_residual_step,
                        invalid_residual_step,
                        operand=None,
                    )

                def rejected_step(_):
                    return (
                        solution,
                        residual,
                        direction,
                        preconditioned_dot,
                        residual_norm_sq,
                        iteration + 1,
                        jnp.asarray(False),
                        jnp.asarray(True),
                    )

                return jax.lax.cond(
                    valid_curvature, accepted_step, rejected_step, operand=None
                )

            (
                solution,
                _,
                _,
                _,
                final_residual_norm_sq,
                iterations,
                converged,
                breakdown,
            ) = jax.lax.while_loop(condition, body, initial_state)

            raw_direction = _tree_divide(
                sample_adjoint(parameters, solution, sigma, weights),
                damping_diagonal,
            )
            if use_momentum:
                raw_direction = _tree_add(
                    raw_direction, _tree_scale(previous_direction, momentum)
                )
            raw_direction = jax.tree_util.tree_map(
                lambda value, parameter: jnp.asarray(value, dtype=parameter.dtype),
                raw_direction,
                parameters,
            )
            raw_direction = jax.lax.cond(
                breakdown,
                lambda _: _tree_zeros_like(raw_direction),
                lambda _: raw_direction,
                operand=None,
            )
            metric_vector = tangent_to_sample(
                parameters, raw_direction, sigma, weights
            )
            metric_norm = jnp.sqrt(
                jnp.maximum(sample_dot(metric_vector, metric_vector), 0.0)
            )
            residual_norm = jnp.sqrt(jnp.maximum(final_residual_norm_sq, 0.0))
            relative_residual = jnp.where(
                rhs_norm > 0.0, residual_norm / rhs_norm, 0.0
            )
            return (
                raw_direction,
                iterations,
                residual_norm,
                relative_residual,
                converged,
                breakdown,
                metric_norm,
            )

        if sharded:
            kernel = jax.jit(
                sharding_decorator(
                    cg,
                    sharded_args_tree=(
                        False,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ),
                    reduction_op_tree=(jax.lax.pmax,) * 7,
                )
            )
        else:
            kernel = jax.jit(cg)
        self._device_minsr_kernel_cache[key] = kernel
        return kernel

    def notify_rejected_update(self):
        """Increase the automatic scale-invariant damping after a rejection."""
        self._last_solver_direction = None
        self._last_solver_diag_shift = None
        self._last_solver_metric_diagonal = None
        self._last_momentum_direction = None
        self._active_diag_shift = min(1.0e-1, 2.0 * self._active_diag_shift)
        if self.last_info is not None:
            self.last_info["PostUpdateRejected"] = True
            self.last_info["NextDiagShift"] = self._active_diag_shift

    def __call__(self, vstate, gradient, step=None):
        """Return a NetKet-style weighted minimum-SR target direction."""
        del gradient
        _assert_real_parameters(vstate.parameters)
        batch = vstate.last_batch
        local_energies = getattr(vstate, "last_local_energies", None)
        if batch is None or local_energies is None:
            raise RuntimeError(
                "WeightedMinSR requires the current weighted proposal pool and local energies"
            )
        sigma = batch["sigma_prop"]
        weights = jnp.asarray(
            batch["weight_info"]["weights_normalized"], dtype=jnp.float64
        )
        weights = weights / vstate.global_sum(weights)
        if self.direct_solver == "distributed_cholesky":
            return self._distributed_cholesky_update(
                vstate, sigma, weights, local_energies, step
            )
        if self._direct_cholesky_is_safe(vstate):
            return self._direct_cholesky_update(
                vstate, sigma, weights, local_energies, step
            )

        initial_diag_shift = self._active_diag_shift
        metric_diagonal, diagonal_refreshed = self._metric_diagonal_for_solver(
            vstate, sigma, weights, step
        )
        local_chunk_size = (
            self.chunk_size
            or (
                vstate.n_samples_per_device
                if getattr(vstate, "uses_native_sharding", False)
                else sigma.shape[0]
            )
        )
        previous_direction = self._last_momentum_direction
        momentum_enabled = self.momentum is not None and self.momentum > 0.0
        use_momentum = momentum_enabled and previous_direction is not None
        momentum = self.momentum if use_momentum else 0.0
        if previous_direction is None:
            previous_direction = _tree_zeros_like(vstate.parameters)
        proj_reg = self.proj_reg if self.proj_reg is not None else 0.0
        device_cg = self._device_minsr_kernel(
            vstate, local_chunk_size, use_momentum=use_momentum
        )

        def solve_on_current_pool(diag_shift):
            damping_diagonal = self._scale_invariant_damping_diagonal(
                metric_diagonal, diag_shift
            )
            result = device_cg(
                vstate.parameters,
                sigma,
                weights,
                local_energies,
                damping_diagonal,
                previous_direction,
                jnp.asarray(momentum, dtype=jnp.float64),
                jnp.asarray(proj_reg, dtype=jnp.float64),
                jnp.asarray(self.maxiter, dtype=jnp.int32),
                jnp.asarray(self.residual_target, dtype=jnp.float64),
            )
            return damping_diagonal, jax.device_get(result)

        active_diag_shift = initial_diag_shift
        damping_diagonal, result = solve_on_current_pool(active_diag_shift)
        (
            direction,
            iterations,
            residual_norm,
            relative_residual,
            converged,
            breakdown,
            metric_norm,
        ) = result
        iterations = int(iterations)
        residual_norm = float(residual_norm)
        relative_residual = float(relative_residual)
        converged = bool(converged)
        breakdown = bool(breakdown)
        metric_norm = float(metric_norm)
        recovery_attempts = 0
        total_iterations = iterations

        # A retry uses the identical proposal pool and local energies. Only
        # the regularisation changes, so this neither draws extra samples nor
        # changes the stochastic objective seen by the candidate update.
        reliable = converged and not breakdown and math.isfinite(relative_residual)
        while not reliable and recovery_attempts < self._max_recovery_attempts:
            recovery_attempts += 1
            active_diag_shift = min(1.0e-1, 4.0 * active_diag_shift)
            damping_diagonal, result = solve_on_current_pool(active_diag_shift)
            (
                direction,
                iterations,
                residual_norm,
                relative_residual,
                converged,
                breakdown,
                metric_norm,
            ) = result
            iterations = int(iterations)
            residual_norm = float(residual_norm)
            relative_residual = float(relative_residual)
            converged = bool(converged)
            breakdown = bool(breakdown)
            metric_norm = float(metric_norm)
            total_iterations += iterations
            reliable = (
                converged and not breakdown and math.isfinite(relative_residual)
            )

        ess_fraction = float(jax.device_get(self._ess_fraction(vstate, weights)))
        self._active_diag_shift, damping_action = self._next_automatic_diag_shift(
            active_diag_shift, reliable
        )

        raw_direction = direction
        if not reliable:
            # Never feed a non-converged or numerically invalid direction to
            # the optimizer after the same-pool recovery has been exhausted.
            raw_direction = _tree_zeros_like(raw_direction)
            direction = raw_direction
            metric_norm = 0.0
        scale = 1.0
        current_learning_rate = self._learning_rate_at(step)
        if self.trust_region is not None and metric_norm > 0.0:
            outer_norm = current_learning_rate * metric_norm
            if outer_norm > self.trust_region:
                scale = self.trust_region / outer_norm
                direction = _tree_scale(direction, scale)

        self._last_damping_diagonal = damping_diagonal
        if not reliable:
            self._last_solver_direction = None
            self._last_solver_diag_shift = None
            self._last_solver_metric_diagonal = None
            self._last_momentum_direction = None
        else:
            self._last_solver_direction = raw_direction
            self._last_solver_diag_shift = active_diag_shift
            self._last_solver_metric_diagonal = metric_diagonal
            self._last_momentum_direction = raw_direction

        self.last_info = {
            "SRMethod": "weighted_minsr",
            "CGIterations": total_iterations,
            "CGResidualNorm": residual_norm,
            "CGRelativeResidual": relative_residual,
            "CGConverged": converged,
            "CGResidualTarget": self.residual_target,
            "CGResidualTargetReached": converged,
            "CGOnDevice": True,
            "CGExtensions": recovery_attempts,
            "CGRequestedMaxiter": self.maxiter,
            "CGCurvatureRetries": 0,
            "CGRecoveryAttempts": recovery_attempts,
            "CGWarmStarted": False,
            "SRMomentum": momentum,
            "SRMomentumEnabled": momentum_enabled,
            "SRMomentumActive": use_momentum,
            "ProjectionRegularization": self.proj_reg,
            "ProjectionRegularizationEnabled": (
                self.proj_reg is not None and self.proj_reg > 0.0
            ),
            "DiagonalPreconditioner": False,
            "DiagonalRefreshed": diagonal_refreshed,
            "DiagonalProbes": self.diagonal_probes,
            "DiagonalMode": self.diagonal_mode,
            "KernelDiagonalPreconditioner": False,
            "KernelDiagonalProbes": 0,
            "KernelDiagonalRefreshed": False,
            "RelativeDamping": True,
            "ScaleInvariantDamping": True,
            "ESSFraction": ess_fraction,
            "SolverReliable": reliable,
            "UpdateSkipped": not reliable,
            # This is not the legacy adaptive weighted-SR controller.  The
            # compact backend may nevertheless adjust scale-invariant damping after
            # an unsuccessful solve or a rejected paired-energy update.
            "AdaptiveEnabled": False,
            "AutomaticDamping": True,
            "InitialDiagShift": initial_diag_shift,
            "DiagShift": active_diag_shift,
            "NextDiagShift": self._active_diag_shift,
            "DampingControllerAction": damping_action,
            "QGTDirectionNorm": metric_norm,
            "LearningRate": current_learning_rate,
            "EffectiveTrustRegion": self.trust_region,
            "AdaptiveTrustMultiplier": 1.0,
            "TrustRegionScale": scale,
            "MinSRSampleSpaceDimension": 2 * int(vstate.n_samples),
        }
        return direction
