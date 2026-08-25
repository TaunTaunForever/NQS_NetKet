"""Exact small-system validation of the matrix-free weighted SR solver."""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk

from optim.weighted_sr import WeightedMinSR, WeightedSR, weighted_qgt
from samplers.neural_importance_sampling import NeuralImportanceSampler
from samplers.proposal_wrappers import AutoregressiveProposalWrapper
from vqs import WeightedNISState


class _ComplexTarget(nn.Module):
    @nn.compact
    def __call__(self, sigma):
        sigma = jnp.asarray(sigma, jnp.float64)
        real = nn.Dense(1, dtype=jnp.float64, name="real")(sigma).squeeze(-1)
        imag = nn.Dense(1, dtype=jnp.float64, name="imag")(sigma).squeeze(-1)
        return real + 1j * imag


class _UniformProposal(nn.Module):
    n_sites: int

    @nn.compact
    def __call__(self, sigma):
        bias = self.param("bias", nn.initializers.zeros, (self.n_sites, 2), jnp.float64)
        return jnp.broadcast_to(bias, (sigma.shape[0], self.n_sites, 2))


def _make_exact_state():
    hilbert = nk.hilbert.Spin(s=0.5, N=2)
    target, proposal = _ComplexTarget(), _UniformProposal(hilbert.size)
    key = jax.random.PRNGKey(12)
    key, target_key, proposal_key = jax.random.split(key, 3)
    sigma = jnp.ones((1, hilbert.size))
    target_variables = target.init(target_key, sigma)
    proposal_variables = proposal.init(proposal_key, sigma)
    wrapper = AutoregressiveProposalWrapper(hilbert, proposal)
    sampler = NeuralImportanceSampler(
        hilbert, proposal, proposal_variables, wrapper, n_proposals=4, resample_size=4
    )
    state = WeightedNISState(
        hilbert,
        target,
        sampler,
        variables=target_variables,
        proposal_variables=proposal_variables,
        n_samples=4,
        chunk_size=2,
    )
    hamiltonian = nk.operator.spin.sigmax(hilbert, 0) + 0.37 * nk.operator.spin.sigmaz(hilbert, 1)
    states = hilbert.all_states()
    logpsi = target.apply(state.variables, states)
    weights = jax.nn.softmax(2.0 * jnp.real(logpsi))
    local_energies = state.local_energy(hamiltonian, states, log_psi=logpsi)
    batch = {"sigma_prop": states, "weight_info": {"weights_normalized": weights}}
    state._last_batch = batch
    state._last_local_energies = local_energies
    return state, batch, local_energies


def _explicit_log_derivatives(state, sigma):
    # The target has real parameters and complex log-amplitudes.  Build the
    # complex Jacobian from two real-output reverse-mode Jacobians; ``jacrev``
    # deliberately rejects a complex output in its non-holomorphic mode.
    apply = lambda parameters: state.model.apply({"params": parameters}, sigma)
    jacobian_real = jax.jacrev(lambda parameters: jnp.real(apply(parameters)))(state.parameters)
    jacobian_imag = jax.jacrev(lambda parameters: jnp.imag(apply(parameters)))(state.parameters)
    jacobian = jax.tree_util.tree_map(
        lambda real, imag: real + 1j * imag, jacobian_real, jacobian_imag
    )
    return jnp.concatenate(
        [leaf.reshape(sigma.shape[0], -1) for leaf in jax.tree_util.tree_leaves(jacobian)],
        axis=1,
    )


def _explicit_weighted_minsr_terms(state, batch, local_energies):
    """Return the realified weighted MinSR kernel factors for a tiny pool."""
    weights = batch["weight_info"]["weights_normalized"]
    log_derivatives = _explicit_log_derivatives(state, batch["sigma_prop"])
    centered = log_derivatives - jnp.einsum("n,np->p", weights, log_derivatives)
    sample_jacobian = (
        jnp.sqrt(weights)[:, None, None]
        * jnp.stack((jnp.real(centered), jnp.imag(centered)), axis=1)
    ).reshape(2 * weights.size, -1)
    energy = jnp.sum(weights * local_energies)
    rhs_complex = 2.0 * jnp.sqrt(weights) * (local_energies - energy)
    rhs = jnp.stack((jnp.real(rhs_complex), jnp.imag(rhs_complex)), axis=1).reshape(-1)
    return sample_jacobian, rhs, weights


def test_matrix_free_weighted_sr_matches_dense_weighted_qgt_solve():
    state, batch, local_energies = _make_exact_state()
    gradient = state._target_gradient(batch, local_energies)
    shift = 0.1
    sr = WeightedSR(
        diag_shift=shift,
        maxiter=32,
        tol=1.0e-10,
        chunk_size=2,
        trust_region=None,
    )
    matrix_free_direction = sr(state, gradient)

    log_derivatives = _explicit_log_derivatives(state, batch["sigma_prop"])
    dense_qgt = jnp.real(weighted_qgt(log_derivatives, batch["weight_info"]["weights_normalized"], diag_shift=shift))
    flat_gradient, _ = jax.flatten_util.ravel_pytree(gradient)
    dense_direction = jnp.linalg.solve(dense_qgt, flat_gradient)
    flat_matrix_free, _ = jax.flatten_util.ravel_pytree(matrix_free_direction)

    assert sr.last_info is not None
    assert sr.last_info["CGOnDevice"]
    assert sr.last_info["CGConverged"]
    assert jnp.allclose(flat_matrix_free, dense_direction, rtol=1.0e-7, atol=1.0e-8)


def test_netket_style_weighted_minsr_matches_dense_weighted_qgt_solve():
    """The weighted sample-space solve is algebraically the same SR step."""
    state, batch, local_energies = _make_exact_state()
    gradient = state._target_gradient(batch, local_energies)
    shift = 0.1
    minsr = WeightedMinSR(
        diag_shift=shift,
        maxiter=64,
        tol=1.0e-10,
        chunk_size=2,
        trust_region=None,
    )
    # This test isolates the scalable, matrix-free route. The direct kernel
    # Cholesky route is exercised separately below.
    minsr._direct_cholesky_max_sample_dimension = 0
    sample_space_direction = minsr(state, gradient)

    log_derivatives = _explicit_log_derivatives(state, batch["sigma_prop"])
    dense_qgt = jnp.real(
        weighted_qgt(
            log_derivatives,
            batch["weight_info"]["weights_normalized"],
            diag_shift=0.0,
        )
    )
    assert minsr._last_damping_diagonal is not None
    flat_damping, _ = jax.flatten_util.ravel_pytree(minsr._last_damping_diagonal)
    flat_gradient, _ = jax.flatten_util.ravel_pytree(gradient)
    dense_direction = jnp.linalg.solve(
        dense_qgt + jnp.diag(flat_damping), flat_gradient
    )
    flat_sample_space, _ = jax.flatten_util.ravel_pytree(sample_space_direction)

    assert minsr.last_info is not None
    assert minsr.last_info["SRMethod"] == "weighted_minsr"
    assert minsr.last_info["CGOnDevice"]
    assert minsr.last_info["CGConverged"]
    assert minsr.last_info["ScaleInvariantDamping"]
    assert not minsr.last_info["DiagonalPreconditioner"]
    assert not minsr.last_info["KernelDiagonalPreconditioner"]
    assert minsr.last_info["KernelDiagonalProbes"] == 0
    # The production sample-space solve follows JAX's enabled precision
    # (float32 on the default CPU/GPU configuration), while the explicit
    # reference is evaluated in float64. This tolerance tests the algebraic
    # equivalence without assuming x64 is enabled globally.
    assert jnp.allclose(flat_sample_space, dense_direction, rtol=2.0e-7, atol=2.0e-7)


def test_weighted_minsr_reliable_solves_relax_relative_damping():
    """A reliable solve restores the established gentle damping relaxation."""
    minsr = WeightedMinSR(diag_shift=0.01, maxiter=100, trust_region=None)

    relaxed_shift, relaxed_action = minsr._next_automatic_diag_shift(0.02, True)
    failed_shift, failed_action = minsr._next_automatic_diag_shift(0.02, False)

    assert relaxed_shift == 0.016
    assert relaxed_action == "successful_relax"
    assert failed_shift == 0.04
    assert failed_action == "unreliable_increase"


def test_weighted_minsr_cholesky_kernel_matches_dense_weighted_qgt_solve():
    """The small-pool direct route matches NetKet's kernel-Cholesky solve."""
    state, batch, local_energies = _make_exact_state()
    gradient = state._target_gradient(batch, local_energies)
    shift = 0.1
    minsr = WeightedMinSR(
        diag_shift=shift,
        maxiter=64,
        tol=1.0e-10,
        chunk_size=2,
        trust_region=None,
    )
    assert minsr._direct_cholesky_is_safe(state)
    direction = minsr(state, gradient)

    log_derivatives = _explicit_log_derivatives(state, batch["sigma_prop"])
    dense_qgt = jnp.real(
        weighted_qgt(
            log_derivatives,
            batch["weight_info"]["weights_normalized"],
            diag_shift=shift,
        )
    )
    flat_gradient, _ = jax.flatten_util.ravel_pytree(gradient)
    dense_direction = jnp.linalg.solve(dense_qgt, flat_gradient)
    flat_direction, _ = jax.flatten_util.ravel_pytree(direction)

    assert minsr.last_info is not None
    assert minsr.last_info["SRMethod"] == "weighted_minsr_cholesky"
    assert minsr.last_info["LinearSolver"] == "cholesky"
    assert minsr.last_info["SolveOnDevice"]
    assert jnp.allclose(flat_direction, dense_direction, rtol=2.0e-7, atol=2.0e-7)


def test_weighted_minsr_direct_solver_policy_is_explicit():
    """The dense Cholesky ablation must never change the default policy."""
    state, _, _ = _make_exact_state()

    assert WeightedMinSR(direct_solver="auto")._direct_cholesky_is_safe(state)
    assert not WeightedMinSR(
        direct_solver="matrix_free"
    )._direct_cholesky_is_safe(state)
    assert WeightedMinSR(direct_solver="cholesky")._direct_cholesky_is_safe(state)


def test_weighted_minsr_projection_and_spring_match_explicit_weighted_kernel():
    """Projection and SPRING retain the NIS-weighted sample-space algebra."""
    state, batch, local_energies = _make_exact_state()
    gradient = state._target_gradient(batch, local_energies)
    momentum, proj_reg = 0.35, 0.17
    minsr = WeightedMinSR(
        diag_shift=0.1,
        maxiter=64,
        tol=1.0e-10,
        chunk_size=2,
        trust_region=None,
        momentum=momentum,
        proj_reg=proj_reg,
    )
    minsr._direct_cholesky_max_sample_dimension = 0
    previous_direction = jax.tree_util.tree_map(
        lambda leaf: jnp.full_like(leaf, 0.03125), state.parameters
    )
    minsr._last_momentum_direction = previous_direction

    direction = minsr(state, gradient)
    sample_jacobian, rhs, weights = _explicit_weighted_minsr_terms(
        state, batch, local_energies
    )
    flat_damping, _ = jax.flatten_util.ravel_pytree(minsr._last_damping_diagonal)
    previous_flat, _ = jax.flatten_util.ravel_pytree(previous_direction)
    # In the weighted extension ``sqrt(w)`` is the centering null vector.
    # This reduces to NetKet's proj_reg / N * 11.T for uniform samples.
    projection_vector = jnp.repeat(jnp.sqrt(weights), 2)
    kernel = (
        jnp.eye(rhs.size)
        + (sample_jacobian / flat_damping[None, :]) @ sample_jacobian.T
        + proj_reg * jnp.outer(projection_vector, projection_vector)
    )
    auxiliary = jnp.linalg.solve(
        kernel, rhs - momentum * (sample_jacobian @ previous_flat)
    )
    expected = (
        sample_jacobian.T @ auxiliary / flat_damping
        + momentum * previous_flat
    )
    flat_direction, _ = jax.flatten_util.ravel_pytree(direction)

    assert minsr.last_info is not None
    assert minsr.last_info["SRMomentum"] == momentum
    assert minsr.last_info["SRMomentumEnabled"]
    assert minsr.last_info["ProjectionRegularization"] == proj_reg
    assert minsr.last_info["ProjectionRegularizationEnabled"]
    assert jnp.allclose(flat_direction, expected, rtol=2.0e-7, atol=2.0e-7)


def test_weighted_minsr_cholesky_projection_and_spring_match_explicit_kernel():
    """The small-pool direct route applies the same weighted extensions."""
    state, batch, local_energies = _make_exact_state()
    gradient = state._target_gradient(batch, local_energies)
    momentum, proj_reg, shift = 0.25, 0.13, 0.1
    minsr = WeightedMinSR(
        diag_shift=shift,
        maxiter=64,
        tol=1.0e-10,
        chunk_size=2,
        trust_region=None,
        momentum=momentum,
        proj_reg=proj_reg,
    )
    previous_direction = jax.tree_util.tree_map(
        lambda leaf: jnp.full_like(leaf, -0.015625), state.parameters
    )
    minsr._last_momentum_direction = previous_direction

    direction = minsr(state, gradient)
    sample_jacobian, rhs, weights = _explicit_weighted_minsr_terms(
        state, batch, local_energies
    )
    previous_flat, _ = jax.flatten_util.ravel_pytree(previous_direction)
    projection_vector = jnp.repeat(jnp.sqrt(weights), 2)
    kernel = (
        sample_jacobian @ sample_jacobian.T
        + shift * jnp.eye(rhs.size)
        + proj_reg * jnp.outer(projection_vector, projection_vector)
    )
    auxiliary = jnp.linalg.solve(
        kernel, rhs - momentum * (sample_jacobian @ previous_flat)
    )
    expected = sample_jacobian.T @ auxiliary + momentum * previous_flat
    flat_direction, _ = jax.flatten_util.ravel_pytree(direction)

    assert minsr.last_info is not None
    assert minsr.last_info["SRMethod"] == "weighted_minsr_cholesky"
    assert minsr.last_info["SRMomentum"] == momentum
    assert minsr.last_info["ProjectionRegularization"] == proj_reg
    assert jnp.allclose(flat_direction, expected, rtol=2.0e-7, atol=2.0e-7)


def test_weighted_minsr_rejection_clears_spring_momentum_state():
    """A paired-energy rollback must not contaminate the next SR solve."""
    state, _, _ = _make_exact_state()
    minsr = WeightedMinSR(momentum=0.8)
    minsr._last_momentum_direction = state.parameters

    minsr.notify_rejected_update()

    assert minsr._last_momentum_direction is None


def test_weighted_minsr_recovers_once_on_the_same_pool_with_more_damping():
    """An unreliable first solve is retried without drawing another pool."""
    state, _, local_energies = _make_exact_state()
    gradient = state._target_gradient(state.last_batch, local_energies)
    minsr = WeightedMinSR(
        diag_shift=0.01,
        maxiter=8,
        chunk_size=2,
        trust_region=None,
    )
    minsr._direct_cholesky_max_sample_dimension = 0
    # Avoid spending probe products in this control-flow test. The estimator
    # itself is exercised by the dense-direction test above.
    minsr._metric_diagonal = jax.tree_util.tree_map(jnp.ones_like, state.parameters)
    minsr._metric_diagonal_step = 0
    calls = []

    def scripted_kernel_factory(*args, **kwargs):
        del args, kwargs

        def kernel(
            parameters,
            sigma,
            weights,
            energies,
            damping_diagonal,
            previous_direction,
            momentum,
            proj_reg,
            maxiter,
            residual_target,
        ):
            del sigma, weights, energies, previous_direction, momentum, proj_reg
            del maxiter, residual_target
            calls.append(damping_diagonal)
            if len(calls) == 1:
                return (
                    jax.tree_util.tree_map(jnp.zeros_like, parameters),
                    jnp.asarray(1, dtype=jnp.int32),
                    jnp.asarray(jnp.inf),
                    jnp.asarray(jnp.inf),
                    jnp.asarray(False),
                    jnp.asarray(True),
                    jnp.asarray(0.0),
                )
            return (
                gradient,
                jnp.asarray(2, dtype=jnp.int32),
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.asarray(True),
                jnp.asarray(False),
                jnp.asarray(1.0),
            )

        return kernel

    minsr._device_minsr_kernel = scripted_kernel_factory
    direction = minsr(state, gradient, step=0)

    assert len(calls) == 2
    assert minsr.last_info is not None
    assert minsr.last_info["CGRecoveryAttempts"] == 1
    assert minsr.last_info["CGIterations"] == 3
    assert minsr.last_info["SolverReliable"]
    assert all(
        jnp.allclose(recovered, 4.0 * initial)
        for initial, recovered in zip(
            jax.tree_util.tree_leaves(calls[0]),
            jax.tree_util.tree_leaves(calls[1]),
        )
    )
    assert all(
        jnp.allclose(actual, expected)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(direction),
            jax.tree_util.tree_leaves(gradient),
        )
    )


def test_weighted_minsr_skips_an_update_after_exhausting_same_pool_recovery():
    """A second failed solve must not leak an unreliable target update."""
    state, _, local_energies = _make_exact_state()
    gradient = state._target_gradient(state.last_batch, local_energies)
    minsr = WeightedMinSR(
        diag_shift=0.01,
        maxiter=8,
        chunk_size=2,
        trust_region=None,
    )
    minsr._direct_cholesky_max_sample_dimension = 0
    minsr._metric_diagonal = jax.tree_util.tree_map(jnp.ones_like, state.parameters)
    minsr._metric_diagonal_step = 0
    calls = 0

    def always_breaks_down(*args, **kwargs):
        del args, kwargs

        def kernel(
            parameters,
            sigma,
            weights,
            energies,
            damping_diagonal,
            previous_direction,
            momentum,
            proj_reg,
            maxiter,
            residual_target,
        ):
            nonlocal calls
            del sigma, weights, energies, damping_diagonal, previous_direction
            del momentum, proj_reg, maxiter, residual_target
            calls += 1
            return (
                jax.tree_util.tree_map(jnp.zeros_like, parameters),
                jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(jnp.inf),
                jnp.asarray(jnp.inf),
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(0.0),
            )

        return kernel

    minsr._device_minsr_kernel = always_breaks_down
    direction = minsr(state, gradient, step=0)

    assert calls == 2
    assert minsr.last_info is not None
    assert minsr.last_info["CGRecoveryAttempts"] == 1
    assert minsr.last_info["UpdateSkipped"]
    assert not minsr.last_info["SolverReliable"]
    assert all(
        jnp.allclose(leaf, 0.0) for leaf in jax.tree_util.tree_leaves(direction)
    )


def test_weighted_sr_trust_region_bounds_outer_sgd_step():
    state, batch, local_energies = _make_exact_state()
    gradient = state._target_gradient(batch, local_energies)
    learning_rate, trust_region = 0.2, 0.01
    sr = WeightedSR(
        diag_shift=0.1,
        maxiter=32,
        tol=1.0e-10,
        chunk_size=2,
        trust_region=trust_region,
        learning_rate=learning_rate,
    )
    direction = sr(state, gradient)
    log_derivatives = _explicit_log_derivatives(state, batch["sigma_prop"])
    dense_qgt = jnp.real(weighted_qgt(log_derivatives, batch["weight_info"]["weights_normalized"], diag_shift=0.0))
    flat_direction, _ = jax.flatten_util.ravel_pytree(direction)
    outer_step = learning_rate * flat_direction
    outer_metric_norm = jnp.sqrt(outer_step @ dense_qgt @ outer_step)

    assert outer_metric_norm <= trust_region * (1.0 + 1.0e-6)


def test_adaptive_weighted_sr_extends_a_useful_cg_solve():
    state, batch, local_energies = _make_exact_state()
    gradient = state._target_gradient(batch, local_energies)
    sr = WeightedSR(
        diag_shift=0.1,
        maxiter=1,
        tol=1.0e-10,
        chunk_size=2,
        trust_region=None,
        adaptive=True,
        adaptive_maxiter=32,
        adaptive_ess_threshold=0.0,
    )

    sr(state, gradient)

    assert sr.last_info is not None
    assert sr.last_info["CGConverged"]
    assert sr.last_info["CGIterations"] > 1
    assert sr.last_info["CGExtensions"] > 0
    assert sr.last_info["CGRelativeResidual"] <= sr.tol


def test_diagonal_preconditioned_relative_damping_matches_dense_reference():
    state, batch, local_energies = _make_exact_state()
    gradient = state._target_gradient(batch, local_energies)
    shift = 0.1
    sr = WeightedSR(
        diag_shift=shift,
        maxiter=64,
        tol=1.0e-9,
        chunk_size=2,
        trust_region=None,
        adaptive=False,
        adaptive_maxiter=64,
        diagonal_preconditioner=True,
        relative_damping=True,
        diagonal_probes=32,
        diagonal_update_interval=1,
        diagonal_ema=0.0,
        residual_replacement_interval=1,
    )
    direction = sr(state, gradient)

    assert sr._metric_diagonal is not None
    log_derivatives = _explicit_log_derivatives(state, batch["sigma_prop"])
    dense_qgt = jnp.real(
        weighted_qgt(log_derivatives, batch["weight_info"]["weights_normalized"], diag_shift=0.0)
    )
    flat_gradient, _ = jax.flatten_util.ravel_pytree(gradient)
    flat_diagonal, _ = jax.flatten_util.ravel_pytree(sr._metric_diagonal)
    flat_direction, _ = jax.flatten_util.ravel_pytree(direction)
    dense_direction = jnp.linalg.solve(
        dense_qgt + shift * jnp.diag(flat_diagonal), flat_gradient
    )

    assert sr.last_info is not None
    assert sr.last_info["DiagonalPreconditioner"]
    assert sr.last_info["RelativeDamping"]
    assert sr.last_info["CGResidualReplacements"] > 0
    assert sr.last_info["CGConverged"]
    assert jnp.allclose(flat_direction, dense_direction, rtol=1.0e-7, atol=1.0e-8)


def test_adaptive_weighted_sr_stabilises_then_relaxes_its_cap():
    state, batch, local_energies = _make_exact_state()
    gradient = state._target_gradient(batch, local_energies)
    sr = WeightedSR(
        diag_shift=0.01,
        maxiter=1,
        tol=1.0e-20,
        chunk_size=2,
        trust_region=0.1,
        adaptive=True,
        adaptive_maxiter=1,
        adaptive_ess_threshold=0.0,
        adaptive_diag_shift_factor=10.0,
        adaptive_max_diag_shift=1.0,
        adaptive_trust_region_shrink=0.5,
        adaptive_trust_region_growth=2.0,
    )

    sr(state, gradient)
    assert sr.last_info is not None
    assert not sr.last_info["SolverReliable"]
    assert sr.last_info["NextDiagShift"] == 0.1
    assert sr.last_info["AdaptiveTrustMultiplier"] == 0.5

    sr.maxiter = 32
    sr.adaptive_maxiter = 32
    sr.tol = 1.0e-10
    sr(state, gradient)
    assert sr.last_info is not None
    assert sr.last_info["SolverReliable"]
    assert sr.last_info["DiagShift"] == 0.1
    assert sr.last_info["NextDiagShift"] < sr.last_info["DiagShift"]
    assert sr.last_info["AdaptiveTrustMultiplier"] == 1.0


def test_adaptive_weighted_sr_skips_an_irrecoverable_curvature_failure():
    state, batch, local_energies = _make_exact_state()
    gradient = state._target_gradient(batch, local_energies)
    sr = WeightedSR(
        diag_shift=0.01,
        maxiter=2,
        chunk_size=2,
        trust_region=0.1,
        adaptive=True,
        adaptive_max_diag_shift=0.1,
        adaptive_diag_shift_factor=10.0,
    )

    def always_break_down(*args, **kwargs):
        del args, kwargs

        def kernel(
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
            del parameters, sigma, weights, initial_solution, use_warm_start
            del diag_shift, metric_diagonal, preconditioner_diagonal
            del use_relative_damping, maxiter, residual_target, residual_replacement_interval
            zeros = jax.tree_util.tree_map(jnp.zeros_like, rhs)
            return (
                zeros,
                rhs,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(jnp.inf),
                jnp.asarray(jnp.inf),
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
            )

        return kernel

    sr._device_cg_kernel = always_break_down
    direction = sr(state, gradient)

    assert sr.last_info is not None
    assert sr.last_info["CGCurvatureRetries"] == 1
    assert sr.last_info["UpdateSkipped"]
    assert sr.last_info["NextDiagShift"] == 0.1
    assert all(
        bool(jnp.allclose(leaf, 0.0))
        for leaf in jax.tree_util.tree_leaves(direction)
    )


def test_weighted_sr_uses_the_current_scheduled_learning_rate():
    state, batch, local_energies = _make_exact_state()
    gradient = state._target_gradient(batch, local_energies)
    sr = WeightedSR(
        diag_shift=0.1,
        maxiter=32,
        tol=1.0e-10,
        chunk_size=2,
        trust_region=0.01,
        learning_rate=lambda step: 0.2 / (step + 1),
    )

    sr(state, gradient, step=3)

    assert sr.last_info is not None
    assert sr.last_info["LearningRate"] == 0.05


def test_rejected_update_discards_the_warm_start_and_tightens_adaptive_controls():
    """A post-update rollback must make the next SR proposal conservative."""
    sr = WeightedSR(
        diag_shift=0.01,
        maxiter=4,
        chunk_size=2,
        trust_region=0.2,
        adaptive=True,
        adaptive_diag_shift_factor=4.0,
        adaptive_max_diag_shift=0.1,
        adaptive_trust_region_shrink=0.5,
    )
    sr._last_solver_direction = {"leaf": jnp.asarray(1.0)}
    sr._last_solver_diag_shift = 0.01
    sr._last_solver_metric_diagonal = {"leaf": jnp.asarray(1.0)}
    sr.last_info = {}

    sr.notify_rejected_update()

    assert sr._last_solver_direction is None
    assert sr._last_solver_diag_shift is None
    assert sr._last_solver_metric_diagonal is None
    assert sr._active_diag_shift == 0.04
    assert sr._trust_region_multiplier == 0.5
    assert sr.last_info["PostUpdateRejected"]
    assert sr.last_info["NextDiagShift"] == 0.04
    assert sr.last_info["NextTrustRegionMultiplier"] == 0.5
