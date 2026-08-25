import jax.numpy as jnp
from samplers.importance_weights import stable_log_weights,effective_sample_size
def test_stable_log_weights_handles_large_positive_and_negative_values():
 d=stable_log_weights(jnp.array([1000.,-1000.]),jnp.array([0.,0.])); assert jnp.all(jnp.isfinite(d['weights_normalized']))
def test_weights_sum_to_one_in_snis_mode(): assert jnp.allclose(jnp.sum(stable_log_weights(jnp.array([1.,2.]),jnp.zeros(2))['weights_normalized']),1)
def test_effective_sample_size_known_cases():
 assert jnp.allclose(effective_sample_size(jnp.array([.5,.5])),2); assert jnp.allclose(effective_sample_size(jnp.array([1.,0.])),1)
