import jax.numpy as jnp
from samplers.importance_weights import stable_log_weights
def test_ess_is_maximal_for_matching_proposal_and_target():
 d=stable_log_weights(jnp.log(jnp.array([.2,.3,.5])),jnp.log(jnp.array([.2,.3,.5])))
 assert jnp.allclose(d['ess'],3)
def test_ess_decreases_under_controlled_mismatch():
 d=stable_log_weights(jnp.log(jnp.array([.98,.01,.01])),jnp.log(jnp.array([1/3,1/3,1/3])))
 assert d['ess_frac'] < .5
