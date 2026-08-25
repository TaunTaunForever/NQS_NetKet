import jax.numpy as jnp
from optim.weighted_sr import weighted_force_vector, weighted_qgt, solve_weighted_sr_step
def test_weighted_force_matches_fullsum_smallN():
 o=jnp.array([[0.],[2.]]); e=jnp.array([1.,3.]); w=jnp.array([.25,.75])
 assert jnp.allclose(weighted_force_vector(o,e,w),jnp.array([.75]))
def test_weighted_qgt_is_psd_up_to_numerical_tolerance():
 q=weighted_qgt(jnp.array([[1.,0.],[0.,1.],[1.,1.]]),jnp.ones(3)/3)
 assert jnp.min(jnp.linalg.eigvalsh(q)) > -1e-10
def test_weighted_sr_step_reduces_energy_on_small_problem():
 q=jnp.eye(1); force=jnp.array([1.]); assert solve_weighted_sr_step(q,force)[0] > 0
