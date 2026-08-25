"""Regression coverage for every direct Gamma weighted-NIS target variant."""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest


_NIS_DIR = Path(__file__).resolve().parents[1] / "Gamma_Honeycomb" / "8-site" / "nis_experiments"
if str(_NIS_DIR) not in sys.path:
    sys.path.insert(0, str(_NIS_DIR))

from run_gamma_nis import NISRunConfig, build_target_model
from hamiltonian import gamma_hamiltonian


@pytest.mark.parametrize("variant", ("plain", "site_relation", "inputproj", "symmproj"))
def test_every_gamma_nis_target_variant_initializes_and_evaluates(variant):
    graph, symmetry_group, _hilbert, _hamiltonian = gamma_hamiltonian(8)
    config = NISRunConfig(
        variant=variant,
        embed_dim=4,
        num_heads=2,
        num_layers=1,
        mlp_hidden_dim=8,
        patch_size=1,
    )
    model = build_target_model(config, graph, symmetry_group)
    sigma = jnp.ones((2, graph.n_nodes), dtype=jnp.float64)
    variables = model.init(jax.random.PRNGKey(0), sigma)
    log_psi = model.apply(variables, sigma)

    assert log_psi.shape == (2,)
    assert jnp.all(jnp.isfinite(log_psi))
