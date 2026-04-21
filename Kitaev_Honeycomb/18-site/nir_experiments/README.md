# NIR Experiments

This directory is an isolated sandbox for trying ideas from:

`Neural Importance Resampling: A Practical Sampling Strategy for Neural Quantum States`
`arXiv:2507.20510v2`

The goal is to keep the production 18-site launchers untouched while we prototype a
new sampling pipeline around the current symmetry-projected ViT.

## Paper ideas we want to test here

1. Use a separate sampling neural network (SNN) as an autoregressive proposal model.
2. Sample a large proposal batch from the SNN.
3. Compute importance weights using the target NQS and the proposal distribution.
4. Monitor effective sample size (ESS) and sampling efficiency.
5. Resample from the proposal batch to obtain approximate NQS-distributed samples.
6. Train the proposal network with a forward-KL objective on the resampled states.

## Suggested development order

1. Keep the current ViT wavefunction as the target model.
2. Add a minimal proposal model for spin configurations.
3. Replace `MetropolisLocal` in the amplitude-only stage with an NIR sampling loop.
4. Compare ESS, wall-clock cost, and final energy against the baseline sampler.

## Files in this directory

- `nir_utils.py`
  Core importance-resampling utilities: log-weight normalization, ESS, efficiency,
  and multinomial resampling.

- `vit_nir_symmproj_singlegpu.py`
  A starter launcher for 18-site single-GPU symmetry-projected ViT experiments.
  It currently scaffolds the Hamiltonian/model setup and imports the NIR utilities,
  but does not yet replace the full NetKet sampling loop.
