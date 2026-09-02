import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
import optax

from optim.weighted_sr import WeightedMinSR, WeightedSR
from samplers.neural_importance_sampling import NeuralImportanceSampler
from samplers.proposal_wrappers import AutoregressiveProposalWrapper
from vqs import WeightedNISState, WeightedNISVMC


class _Target(nn.Module):
    @nn.compact
    def __call__(self, sigma):
        return nn.Dense(1, dtype=jnp.float64)(jnp.asarray(sigma, jnp.float64)).squeeze(-1)


class _AutoregressiveProposal(nn.Module):
    n_sites: int

    @nn.compact
    def __call__(self, sigma):
        bias = self.param("bias", nn.initializers.zeros, (self.n_sites, 2), jnp.float64)
        return jnp.broadcast_to(bias, (sigma.shape[0], self.n_sites, 2))


def _make_state(seed=0):
    hilbert = nk.hilbert.Spin(s=0.5, N=2)
    target = _Target()
    proposal = _AutoregressiveProposal(n_sites=hilbert.size)
    key = jax.random.PRNGKey(seed)
    key, target_key, proposal_key = jax.random.split(key, 3)
    sigma = jnp.ones((1, hilbert.size))
    target_variables = target.init(target_key, sigma)
    proposal_variables = proposal.init(proposal_key, sigma)
    wrapper = AutoregressiveProposalWrapper(hilbert, proposal, probability_floor=1e-6)
    sampler = NeuralImportanceSampler(
        hilbert,
        proposal,
        proposal_variables,
        wrapper,
        n_proposals=16,
        resample_size=8,
    )
    return nk.operator.spin.sigmax(hilbert, 0), WeightedNISState(
        hilbert,
        target,
        sampler,
        variables=target_variables,
        n_samples=16,
        chunk_size=8,
        seed=seed,
    )


def test_weighted_nis_state_returns_netket_stats_and_parameter_tree_gradient():
    operator, state = _make_state()
    stats, gradient = state.expect_and_grad(operator)

    assert isinstance(stats, nk.stats.Stats)
    assert jnp.isfinite(stats.mean)
    assert jnp.isfinite(stats.variance)
    assert jnp.isfinite(stats.error_of_mean)
    assert state.last_diagnostics["ESS"] > 0
    assert jax.tree.structure(gradient) == jax.tree.structure(state.parameters)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))


def test_post_update_evaluation_reuses_pool_with_current_importance_weights():
    """A no-op candidate must reproduce the pre-update weighted estimate."""
    operator, state = _make_state(seed=14)
    before, _ = state.expect_and_grad(operator)
    retained_batch = state.last_batch

    candidate, candidate_batch, diagnostics, finite = (
        state.evaluate_current_target_on_last_batch(operator)
    )

    assert finite
    assert state.last_batch is retained_batch
    assert jnp.allclose(candidate.mean, before.mean, rtol=1.0e-12, atol=1.0e-12)
    assert jnp.allclose(candidate.variance, before.variance, rtol=1.0e-12, atol=1.0e-12)
    assert jnp.allclose(candidate.error_of_mean, before.error_of_mean, rtol=1.0e-12, atol=1.0e-12)
    assert diagnostics["ESS"] > 0
    assert jnp.allclose(
        candidate_batch["weight_info"]["weights_normalized"],
        retained_batch["weight_info"]["weights_normalized"],
    )


def test_weighted_nis_state_runs_with_standard_netket_vmc():
    operator, state = _make_state(seed=1)
    before = jax.tree.map(lambda leaf: leaf.copy(), state.parameters)
    driver = nk.driver.VMC(
        operator,
        optax.sgd(learning_rate=1.0e-3),
        variational_state=state,
    )
    # NetKet 3.22 replaced the old ``advance`` helper with ``run``.
    driver.run(1, out=None, show_progress=False)

    assert isinstance(driver.energy, nk.stats.Stats)
    # NetKet resets a VQS after applying parameters; the completed weighted
    # proposal pool must remain available for the driver's iteration logging.
    assert state.last_batch is not None
    assert state.last_diagnostics["ESS"] > 0
    assert any(
        not jnp.allclose(old, new)
        for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(state.parameters))
    )


def test_weighted_nis_driver_updates_target_and_proposal():
    operator, state = _make_state(seed=2)
    before = jax.tree.map(lambda leaf: leaf.copy(), state.proposal_parameters)
    driver = WeightedNISVMC(
        operator,
        optax.sgd(learning_rate=1.0e-3),
        variational_state=state,
        proposal_optimizer=optax.sgd(learning_rate=1.0e-2),
        proposal_train_steps=1,
        proposal_train_batch_size=8,
    )
    driver.advance(1)

    assert isinstance(driver.energy, nk.stats.Stats)
    assert driver._proposal_loss is not None
    assert any(
        not jnp.allclose(old, new)
        for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(state.proposal_parameters))
    )


def test_weighted_nis_driver_runs_matrix_free_weighted_sr_update():
    operator, state = _make_state(seed=3)
    before = jax.tree.map(lambda leaf: leaf.copy(), state.parameters)
    preconditioner = WeightedSR(
        diag_shift=0.1,
        maxiter=16,
        tol=1.0e-6,
        chunk_size=8,
        trust_region=0.01,
        learning_rate=1.0e-2,
    )
    driver = WeightedNISVMC(
        operator,
        optax.sgd(learning_rate=1.0e-2),
        variational_state=state,
        preconditioner=preconditioner,
    )
    driver.advance(1)

    assert isinstance(driver.energy, nk.stats.Stats)
    assert preconditioner.last_info is not None
    assert preconditioner.last_info["CGIterations"] >= 1
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(state.parameters))
    assert any(
        not jnp.allclose(old, new)
        for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(state.parameters))
    )


def test_weighted_nis_driver_runs_netket_style_weighted_minsr_update():
    """The new backend uses the ordinary NetKet driver/preconditioner hook."""
    operator, state = _make_state(seed=4)
    before = jax.tree.map(lambda leaf: leaf.copy(), state.parameters)
    preconditioner = WeightedMinSR(
        diag_shift=0.1,
        maxiter=32,
        tol=1.0e-6,
        chunk_size=8,
        trust_region=0.01,
        learning_rate=1.0e-2,
    )
    driver = WeightedNISVMC(
        operator,
        optax.sgd(learning_rate=1.0e-2),
        variational_state=state,
        preconditioner=preconditioner,
    )
    driver.advance(1)

    assert isinstance(driver.energy, nk.stats.Stats)
    assert preconditioner.last_info is not None
    assert preconditioner.last_info["SRMethod"] in {
        "weighted_minsr",
        "weighted_minsr_cholesky",
    }
    assert (
        preconditioner.last_info["CGOnDevice"]
        or preconditioner.last_info["SolveOnDevice"]
    )
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(state.parameters))
    assert any(
        not jnp.allclose(old, new)
        for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(state.parameters))
    )
