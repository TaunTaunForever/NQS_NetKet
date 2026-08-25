import jax.numpy as jnp

from diagnostics.nis_metrics import weighted_variational_statistics


def test_weighted_variational_statistics_matches_manual_expectation():
    local_energies = jnp.array([1.0, 3.0])
    weights = jnp.array([0.25, 0.75])
    stats = weighted_variational_statistics(local_energies, weights)
    assert jnp.allclose(stats["Mean"], 2.5)
    assert jnp.allclose(stats["Variance"], 0.75)
    assert jnp.allclose(stats["ErrorOfMean"], jnp.sqrt(0.75 / 1.6))
