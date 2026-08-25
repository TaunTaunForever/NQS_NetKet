import jax,jax.numpy as jnp
from samplers.resampling import systematic_resample,stratified_resample
def test_resampling_indices_are_valid():
 w=jnp.array([.1,.2,.7]);
 for f in (systematic_resample,stratified_resample):
  i=f(jax.random.PRNGKey(0),w,100); assert jnp.all((i>=0)&(i<3))
def test_resampling_degeneracy_detected_when_single_weight_dominates():
 assert jnp.all(systematic_resample(jax.random.PRNGKey(0),jnp.array([0.,1.]),8)==1)
