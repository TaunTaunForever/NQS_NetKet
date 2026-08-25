"""Integration benchmark placeholder: select a project-specific Heisenberg launcher.

The reusable NIS layer is intentionally model-agnostic; this test is enabled by
downstream experiments that provide a honeycomb construction fixture.
"""
import pytest
pytestmark = pytest.mark.skip(reason="requires a project-specific 8-site honeycomb fixture/checkpoint")
def test_nis_8site_exact_benchmark(): pass
