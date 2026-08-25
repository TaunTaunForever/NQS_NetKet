"""Small pure tests for Gamma weighted-NIS learning-rate controls."""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import pytest


NIS_DIR = Path(__file__).resolve().parents[1] / "Gamma_Honeycomb/8-site/nis_experiments"
if str(NIS_DIR) not in sys.path:
    sys.path.insert(0, str(NIS_DIR))

from run_gamma_nis import NISRunConfig, make_target_learning_rate


def test_exponential_target_learning_rate_hits_configured_endpoints():
    config = NISRunConfig(
        target_lr=1.0e-3,
        target_lr_final=2.0e-5,
        target_lr_decay_steps=100,
    )
    schedule = make_target_learning_rate(config)

    assert float(jax.device_get(schedule(0))) == pytest.approx(1.0e-3)
    assert float(jax.device_get(schedule(100))) == pytest.approx(2.0e-5)
    assert float(jax.device_get(schedule(250))) == pytest.approx(2.0e-5)


def test_constant_target_learning_rate_remains_a_scalar():
    config = NISRunConfig(target_lr=3.0e-3)

    assert make_target_learning_rate(config) == 3.0e-3
