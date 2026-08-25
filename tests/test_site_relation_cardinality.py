"""Regression tests for the extended site-relation attention cardinality."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest


def _bond_attention_class():
    path = (
        Path(__file__).resolve().parents[1]
        / "Gamma_Honeycomb"
        / "18-site"
        / "vit_model.py"
    )
    spec = importlib.util.spec_from_file_location("_test_bond_aware_vit_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.BondAwareSelfAttention


def test_bond_attention_infers_extended_relation_cardinality():
    attention = _bond_attention_class()(
        embed_dim=4,
        num_heads=1,
        relation_matrix=((0, 5), (5, 0)),
        data_type=jnp.float64,
    )
    variables = attention.init(jax.random.PRNGKey(0), jnp.ones((2, 2, 4)))

    assert variables["params"]["bond_attention_bias"].shape == (1, 6)
    assert attention.apply(variables, jnp.ones((2, 2, 4))).shape == (2, 2, 4)


def test_bond_attention_rejects_too_small_explicit_relation_table():
    attention = _bond_attention_class()(
        embed_dim=4,
        num_heads=1,
        relation_matrix=((0, 5), (5, 0)),
        num_relation_types=5,
        data_type=jnp.float64,
    )

    with pytest.raises(ValueError, match="requires 6 relation types"):
        attention.init(jax.random.PRNGKey(1), jnp.ones((1, 2, 4)))
