# Debug-first Neural Importance Sampling

`samplers/importance_weights.py` is the authoritative core: it computes
`log |psi_theta(s)|^2 - log q_phi(s)` in float64 and exposes OIS (when an
exact `logZ` is supplied) and SNIS weights. `WeightedNISState` consumes those
weights directly for energy, force and SR estimates; it does not falsely treat
resampled configurations as independent target samples.

`WeightedNISState` now subclasses NetKet's `VariationalState`. It returns a
native `nk.stats.Stats` value and a target-parameter gradient tree, so it can
run with `nk.driver.VMC` and first-order Optax optimizers. `WeightedNISVMC`
adds the weighted proposal maximum-likelihood update while retaining NetKet's
driver/logging/callback interface. `NeuralImportanceSampler` remains a
proposal-container compatibility layer; its resampled configurations must not
be injected into `MCState` for weighted energy or gradient estimation.

This first NetKet integration deliberately supports first-order target updates.
Dense weighted SR/QGT construction is not suitable for the intended 128-site
systems; a matrix-free weighted QGT is a separate next step. The retained
explicit-JAX pmap backend is currently the experimental route for multi-GPU
NIS pools.

For small systems enumerate `hilbert.all_states()`, compare normalized target
and proposal probabilities with `compare_distributions_smallN`, then compare
OIS/SNIS energies against FullSumState before attempting 18-site runs. Inspect
ESS fraction, top-weight mass, and log-weight range; low ESS means the proposal
does not cover the target and resampling cannot repair it.

```
python scripts/run_nis_debug_smallN.py --config configs/nis/debug_8site.yaml --diagnostics_dir results/nis/8site
python scripts/run_nis_benchmark.py --config configs/nis/benchmark_honeycomb.yaml --Lx 3 --Ly 3 --j2 .22
```

Each front-end stores the resolved configuration and `NISLogger` writes JSONL
iteration metrics plus a CSV summary. Checkpointing/model construction remains
in the existing honeycomb launchers, whose model-specific architecture is not
duplicated by this reusable core.
