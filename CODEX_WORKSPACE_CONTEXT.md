# Codex Workspace Context

Snapshot date: 2026-04-21

This file is a handoff note for future Codex threads. It captures the current practical understanding of the workspace, especially the active Gamma honeycomb NIR work, so a new thread can get oriented quickly without rebuilding all context from scratch.

## Repo Summary

This repository is a research workspace for neural quantum state experiments built around NetKet and JAX. It is not organized like a polished Python package; most work happens through standalone scripts.

Main project areas:

- `Gamma_Honeycomb/`
- `Kitaev_Honeycomb/`
- `Heisenberg_Honeycomb/`
- `J1_J2_Heisenberg_Triangular/`
- `Results/`

The top-level README is a good high-level orientation document:

- `README.md`

## Current Active Area

The most active area appears to be:

- `Gamma_Honeycomb/18-site/nir_experiments/`

The user’s current active file was:

- `Gamma_Honeycomb/18-site/nir_experiments/vit_nir_multigpu.py`

This script is the main entry point worth reading first when resuming Gamma 18-site NIR work.

## Gamma 18-Site NIR Stack

The main execution path for the current Gamma 18-site NIR workflow is:

- Launcher: `Gamma_Honeycomb/18-site/nir_experiments/vit_nir_multigpu.py`
- Gamma Hamiltonian builder: `Gamma_Honeycomb/18-site/hamiltonian.py`
- ViT target ansatz: `Gamma_Honeycomb/18-site/vit_model.py`
- NIR proposal model: `Gamma_Honeycomb/18-site/nir_experiments/proposal_network.py`
- NIR utilities: `Gamma_Honeycomb/18-site/nir_experiments/nir_utils.py`
- Continuation helpers: `Gamma_Honeycomb/18-site/vit_continue_utils.py`

Related Gamma 18-site variants in the same folder:

- `vit_nir_continue.py`
- `vit_nir_continue_inputproj.py`
- `vit_nir_continue_symmproj.py`
- `vit_nir_inputproj.py`
- `vit_nir_symmproj.py`

## What The Active Script Does

`vit_nir_multigpu.py` currently does the following at a high level:

- Builds the 18-site Gamma honeycomb Hamiltonian.
- Computes the exact ground-state energy from the sparse Hamiltonian with `eigsh`.
- Builds a ViT-style target wavefunction model.
- Builds an autoregressive proposal network for Neural Importance Resampling (NIR).
- Runs an adaptive NIR loop:
  - draw proposal samples
  - compute target and proposal log-probabilities
  - measure ESS and sampling efficiency
  - importance-resample a target batch
  - train the proposal model on the resampled batch
  - apply a target-model update only if the efficiency gate is met
- Logs metrics and saves checkpoints, a plot, and a summary JSON under a dated `runs/` folder.

Important implementation detail:

- The target update path is currently set to `TARGET_PRECONDITIONER = "minsr"`.
- The script switches `learn_phase` and sample count across stages.
- It injects external NIR samples directly into `nk.vqs.MCState` by setting `vstate._samples`.

## Important Caveats

These are worth keeping in mind before assuming the current code or filenames mean exactly what they suggest.

1. The filename says `multigpu`, but no explicit multi-GPU sharding logic was found in the current script.
   The script prints `jax.devices()`, but there was no obvious `pmap`, `pjit`, mesh setup, or other explicit multi-device partitioning in the inspected file. It may still rely on runtime/device defaults, but it does not currently read like a clearly sharded training script.

2. The current script constants do not obviously match the most recent saved run names.
   The current file contains expressions like `NUM_SAMPLES_STAGE_1 = 3**8` and `NUM_SAMPLES_STAGE_3 = 3**11`, while the recent run folder from `2026-04-20` is named `768to6144_samples`.
   This likely means at least one of the following is true:
   - the script was edited after the run was created
   - the run was produced by an earlier local version
   - the current file is mid-edit

3. The workspace is source-first and experiment-first.
   There is no obvious automated test suite, and the safest workflow is to inspect the top of each launcher script before running it.

## Recent Run Artifacts

The Gamma 18-site NIR runs folder already contains real runtime artifacts:

- `Gamma_Honeycomb/18-site/nir_experiments/runs/2026-04-19/`
- `Gamma_Honeycomb/18-site/nir_experiments/runs/2026-04-20/`

Observed state:

- `2026-04-19` contains at least one completed summary JSON for an input-projected Gamma 18-site NIR run.
- `2026-04-20` contains a single-stage run directory with log and checkpoint files, but no final summary JSON was found.

Useful completed run snapshot from `2026-04-19`:

- Run type: `Gamma_ViT_NIR_inputproj`
- Exact ground-state energy recorded there: `-25.632707769644185`
- Final energy: `-25.09305777174716`
- Best energy seen: `-25.261343930444035`
- Best absolute distance to exact: `0.3713638392001499`
- Best iteration: `964`

Useful incomplete run snapshot from `2026-04-20` log:

- Logged iterations: `1600`
- Best logged energy in the log blob: about `-25.092581560493816`
- Last logged energy: about `-24.933521671613597`
- The run appears to have produced checkpoints and a log, but not the final summary JSON.

## Git Snapshot

At the time this note was written:

- Branch: `main`
- Tracking: `origin/main`

Local modified files that were already present and were not touched during orientation:

- `README.md`
- `Kitaev_Honeycomb/18-site/vit_symmproj_singlegpu_continue.py`

Future Codex threads should treat those as user-owned in-progress edits unless told otherwise.

## Environment Notes

- The repo includes local virtual environment directories:
  - `NetKet_venv/`
  - `NetKet_Updated_venv/`
- The shell had `python3` available on `PATH`.
- Plain `python` was not available on `PATH` during orientation.
- Runtime artifacts are intentionally ignored by git through `.gitignore`.

## Recommended Re-Orientation Path For A New Thread

If a new Codex thread needs to get back up to speed quickly, the fastest path is:

1. Read `CODEX_WORKSPACE_CONTEXT.md`.
2. Read `README.md`.
3. Inspect `Gamma_Honeycomb/18-site/nir_experiments/vit_nir_multigpu.py`.
4. Read these support files:
   - `Gamma_Honeycomb/18-site/hamiltonian.py`
   - `Gamma_Honeycomb/18-site/vit_model.py`
   - `Gamma_Honeycomb/18-site/nir_experiments/proposal_network.py`
   - `Gamma_Honeycomb/18-site/nir_experiments/nir_utils.py`
5. Check the newest contents under `Gamma_Honeycomb/18-site/nir_experiments/runs/`.
6. Compare current script hyperparameters with the names and summaries of any recent run directories before launching anything.

## Practical Working Assumptions

Until the user says otherwise, these are reasonable assumptions to carry forward:

- The Gamma 18-site NIR workflow is the current main focus.
- The repo is actively evolving, so code and saved artifacts may not match perfectly.
- Continuation, comparison, and cleanup work should be done carefully without overwriting existing local edits.
- The safest next step before any run is to reconcile current hyperparameters with the intended experimental target and with the latest saved run directories.
