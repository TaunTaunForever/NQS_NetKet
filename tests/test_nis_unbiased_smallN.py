import jax, jax.numpy as jnp
from samplers.importance_weights import stable_log_weights
def test_ordinary_is_energy_estimator_is_unbiased_when_exact_logZ_is_known():
 p=jnp.array([.2,.8]); q=jnp.array([.5,.5]); e=jnp.array([1.,3.]); key=jax.random.PRNGKey(2)
 draws=jax.random.choice(key,2,(50000,),p=q); d=stable_log_weights(jnp.log(p[draws]),jnp.log(q[draws]),exact_logZ=0.)
 assert abs(jnp.mean(d['weights']*e[draws])-jnp.dot(p,e)) < .03
def test_snis_energy_estimator_converges_to_exact_energy_smallN():
 p=jnp.array([.2,.8]); q=jnp.array([.5,.5]); e=jnp.array([1.,3.]); key=jax.random.PRNGKey(3)
 draws=jax.random.choice(key,2,(50000,),p=q); d=stable_log_weights(jnp.log(p[draws]),jnp.log(q[draws]))
 assert abs(jnp.sum(d['weights_normalized']*e[draws])-jnp.dot(p,e)) < .03
