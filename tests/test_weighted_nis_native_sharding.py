"""Three-GPU regression tests for NetKet-native weighted-NIS sharding."""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
import optax
import pytest
from netket.utils import config as netket_config
from netket.jax.sharding import gather

from optim import WeightedSR
from optim.weighted_sr import WeightedMinSR
from optim.weighted_sr import weighted_qgt
from samplers.neural_importance_sampling import NeuralImportanceSampler
from samplers.proposal_wrappers import AutoregressiveProposalWrapper
from vqs import WeightedNISState, WeightedNISVMC


pytestmark = pytest.mark.skipif(
    jax.device_count() < 2 or not netket_config.netket_experimental_sharding,
    reason="requires at least two GPUs with NETKET_EXPERIMENTAL_SHARDING=1",
)


class _Target(nn.Module):
    @nn.compact
    def __call__(self, sigma):
        sigma = jnp.asarray(sigma, jnp.float64)
        real = nn.Dense(1, dtype=jnp.float64, name="real")(sigma).squeeze(-1)
        imag = nn.Dense(1, dtype=jnp.float64, name="imag")(sigma).squeeze(-1)
        return real + 1j * imag


class _AutoregressiveProposal(nn.Module):
    n_sites: int

    @nn.compact
    def __call__(self, sigma):
        bias = self.param("bias", nn.initializers.zeros, (self.n_sites, 2), jnp.float64)
        return jnp.broadcast_to(bias, (sigma.shape[0], self.n_sites, 2))


def _make_sharded_state(seed=0):
    hilbert = nk.hilbert.Spin(s=0.5, N=2)
    target, proposal = _Target(), _AutoregressiveProposal(hilbert.size)
    key = jax.random.PRNGKey(seed)
    key, target_key, proposal_key = jax.random.split(key, 3)
    sigma = jnp.ones((1, hilbert.size))
    target_variables = target.init(target_key, sigma)
    proposal_variables = proposal.init(proposal_key, sigma)
    wrapper = AutoregressiveProposalWrapper(hilbert, proposal, probability_floor=1.0e-6)
    n_proposals = 4 * jax.device_count()
    sampler = NeuralImportanceSampler(
        hilbert,
        proposal,
        proposal_variables,
        wrapper,
        n_proposals=n_proposals,
        resample_size=4,
    )
    state = WeightedNISState(
        hilbert,
        target,
        sampler,
        variables=target_variables,
        n_samples=n_proposals,
        chunk_size=2,
        seed=seed,
        use_sharding=True,
    )
    return nk.operator.spin.sigmax(hilbert, 0), state


def test_native_sharding_distributes_one_global_weighted_pool():
    operator, state = _make_sharded_state()
    batch = state.sample_proposals()
    weights = batch["weight_info"]["weights_normalized"]

    assert state.uses_native_sharding
    assert len(batch["sigma_prop"].addressable_shards) == jax.device_count()
    assert all(
        shard.data.shape == (state.n_samples_per_device, state.hilbert.size)
        for shard in batch["sigma_prop"].addressable_shards
    )
    assert jnp.isclose(state.global_sum(weights), 1.0)

    stats, gradient = state.expect_and_grad(operator)
    assert isinstance(stats, nk.stats.Stats)
    assert jnp.isfinite(stats.mean)
    assert state.last_diagnostics["ESS"] > 0
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))

    # Compare the native mesh's globally reduced force against the same
    # weighted score-function force on the gathered global pool.
    sigma_global = gather(state.last_batch["sigma_prop"])
    weights_global = gather(state.last_batch["weight_info"]["weights_normalized"])
    energies_global = gather(state._last_local_energies)
    energy_global = jnp.sum(weights_global * energies_global)
    explicit_gradient = jax.tree.map(jnp.zeros_like, state.parameters)
    for start in range(0, sigma_global.shape[0], state.chunk_size):
        stop = min(start + state.chunk_size, sigma_global.shape[0])
        coefficients = weights_global[start:stop] * jax.lax.stop_gradient(
            energies_global[start:stop] - energy_global
        )
        partial = state._chunk_force(
            state.parameters, state.model_state, sigma_global[start:stop], coefficients
        )
        explicit_gradient = jax.tree.map(
            lambda total, contribution: total + 2.0 * contribution,
            explicit_gradient,
            partial,
        )
    assert all(
        jnp.allclose(distributed, explicit, rtol=1.0e-6, atol=1.0e-7)
        for distributed, explicit in zip(
            jax.tree.leaves(gradient), jax.tree.leaves(explicit_gradient)
        )
    )


def test_native_sharding_runs_weighted_matrix_free_sr():
    operator, state = _make_sharded_state(seed=1)
    proposal_before = jax.tree.map(lambda leaf: leaf.copy(), state.proposal_parameters)
    preconditioner = WeightedSR(
        diag_shift=0.1,
        maxiter=4,
        tol=1.0e-5,
        chunk_size=2,
        trust_region=0.01,
        learning_rate=1.0e-2,
    )
    driver = WeightedNISVMC(
        operator,
        optax.sgd(learning_rate=1.0e-2),
        variational_state=state,
        preconditioner=preconditioner,
        proposal_optimizer=optax.sgd(learning_rate=1.0e-2),
        proposal_train_steps=1,
    )
    driver.advance(1)

    assert preconditioner.last_info is not None
    assert preconditioner.last_info["CGIterations"] >= 1
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(state.parameters))
    assert driver._proposal_loss is not None
    assert any(
        not jnp.allclose(before, after)
        for before, after in zip(
            jax.tree.leaves(proposal_before), jax.tree.leaves(state.proposal_parameters)
        )
    )


def test_native_sharding_runs_netket_style_weighted_minsr():
    """Native pools support either safe dense or scalable kernel MinSR solves."""
    operator, state = _make_sharded_state(seed=5)
    before = jax.tree.map(lambda leaf: leaf.copy(), state.parameters)
    preconditioner = WeightedMinSR(
        diag_shift=0.1,
        maxiter=32,
        tol=1.0e-5,
        chunk_size=2,
        trust_region=0.01,
        learning_rate=1.0e-2,
        momentum=0.4,
        proj_reg=0.1,
    )
    # Exercise the scalable, sharded kernel where the weighted projection
    # reduction and SPRING JVP must use the global pool correctly.
    preconditioner._direct_cholesky_max_sample_dimension = 0
    driver = WeightedNISVMC(
        operator,
        optax.sgd(learning_rate=1.0e-2),
        variational_state=state,
        preconditioner=preconditioner,
        proposal_optimizer=optax.sgd(learning_rate=1.0e-2),
        proposal_train_steps=1,
    )
    # The first update seeds SPRING memory; the second exercises it.
    driver.advance(2)

    assert preconditioner.last_info is not None
    assert preconditioner.last_info["SRMethod"] == "weighted_minsr"
    assert preconditioner.last_info["SRMomentumEnabled"]
    assert preconditioner.last_info["SRMomentumActive"]
    assert preconditioner.last_info["ProjectionRegularizationEnabled"]
    assert (
        preconditioner.last_info["CGOnDevice"]
        or preconditioner.last_info["SolveOnDevice"]
    )
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(state.parameters))
    assert any(
        not jnp.allclose(old, new)
        for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(state.parameters))
    )


def test_native_sharding_constructs_weighted_dense_kernel_for_cholesky():
    """Jacobian rows stay distributed while the tiny kernel is factorized."""
    operator, state = _make_sharded_state(seed=7)
    before = jax.tree.map(lambda leaf: leaf.copy(), state.parameters)
    preconditioner = WeightedMinSR(
        diag_shift=0.1,
        trust_region=0.01,
        learning_rate=1.0e-2,
        direct_solver="distributed_cholesky",
        momentum=0.4,
        proj_reg=0.1,
    )
    driver = WeightedNISVMC(
        operator,
        optax.sgd(learning_rate=1.0e-2),
        variational_state=state,
        preconditioner=preconditioner,
        proposal_optimizer=optax.sgd(learning_rate=1.0e-2),
        proposal_train_steps=1,
    )
    # The second update activates the retained SPRING direction.
    driver.advance(2)

    assert preconditioner.last_info is not None
    assert preconditioner.last_info["SRMethod"] == "weighted_minsr_distributed_cholesky"
    assert preconditioner.last_info["KernelConstruction"] == "distributed_local_jacobian_ring"
    assert preconditioner.last_info["KernelConstructionDevices"] == jax.local_device_count()
    assert preconditioner.last_info["SRMomentumActive"]
    assert preconditioner.last_info["ProjectionRegularizationEnabled"]
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(state.parameters))
    assert any(
        not jnp.allclose(old, new)
        for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(state.parameters))
    )


def test_distributed_weighted_dense_kernel_matches_direct_cholesky():
    """The ring-assembled weighted kernel has the direct-solver update."""
    operator, state = _make_sharded_state(seed=8)
    _, gradient = state.expect_and_grad(operator)
    kwargs = dict(
        diag_shift=0.1,
        trust_region=None,
        learning_rate=1.0,
        proj_reg=0.1,
    )
    direct = WeightedMinSR(direct_solver="cholesky", **kwargs)
    distributed = WeightedMinSR(direct_solver="distributed_cholesky", **kwargs)

    direct_direction = direct(state, gradient)
    distributed_direction = distributed(state, gradient)
    direct_flat, _ = jax.flatten_util.ravel_pytree(direct_direction)
    distributed_flat, _ = jax.flatten_util.ravel_pytree(distributed_direction)

    assert distributed.last_info is not None
    assert distributed.last_info["KernelConstructionDevices"] == jax.local_device_count()
    assert jnp.allclose(
        distributed_flat, direct_flat, rtol=2.0e-7, atol=2.0e-7
    )


def test_native_sharding_sr_matches_gathered_dense_qgt_reference():
    operator, state = _make_sharded_state(seed=2)
    _, gradient = state.expect_and_grad(operator)
    sr = WeightedSR(
        diag_shift=0.1,
        maxiter=32,
        tol=1.0e-8,
        chunk_size=2,
        trust_region=None,
    )
    distributed_direction = sr(state, gradient)

    sigma = gather(state.last_batch["sigma_prop"])
    weights = gather(state.last_batch["weight_info"]["weights_normalized"])
    apply = lambda parameters: state.model.apply({"params": parameters}, sigma)
    jacobian_real = jax.jacrev(lambda parameters: jnp.real(apply(parameters)))(state.parameters)
    jacobian_imag = jax.jacrev(lambda parameters: jnp.imag(apply(parameters)))(state.parameters)
    log_derivatives = jnp.concatenate(
        [
            (real + 1j * imag).reshape(sigma.shape[0], -1)
            for real, imag in zip(
                jax.tree.leaves(jacobian_real), jax.tree.leaves(jacobian_imag)
            )
        ],
        axis=1,
    )
    dense_qgt = jnp.real(weighted_qgt(log_derivatives, weights, diag_shift=0.1))
    flat_gradient, _ = jax.flatten_util.ravel_pytree(gradient)
    dense_direction = jnp.linalg.solve(dense_qgt, flat_gradient)
    flat_distributed, _ = jax.flatten_util.ravel_pytree(distributed_direction)

    assert sr.last_info is not None
    assert sr.last_info["CGConverged"]
    assert jnp.allclose(flat_distributed, dense_direction, rtol=1.0e-5, atol=1.0e-6)
