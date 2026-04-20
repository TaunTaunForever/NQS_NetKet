# NQS_NeKet

Research code for variational neural-network studies of frustrated spin models using [NetKet](https://www.netket.org/) as the optimization and sampling backbone.

## Author

Daniel Sepulveda  
Academic email: `sepulved@mcmaster.ca`

## Original Project Aim

The original intention of this project was to study **GCNN applications to the Kitaev and Gamma honeycomb models**. In particular, the early focus was on using graph-based neural quantum states to approximate ground states, compare convergence across system sizes, and evaluate how well symmetry-aware neural architectures perform on these frustrated spin systems.

Over time, that initial GCNN-centered program expanded into a broader research workspace that now also includes:

- ViT-based neural quantum states
- SR / minSR / SRt optimization studies
- symmetry-projected and input-projected ansatzes
- Neural Importance Resampling (NIR) workflows
- continuation and refinement pipelines for long-running experiments

The move toward **ViT-based ansatzes** and later **NIR-based training** was largely precipitated by persistent convergence difficulties with the earlier **GCNN + local spin-flip sampling regime**. That struggle motivated a broader search for architectures and sampling/update strategies that could behave more robustly on the Kitaev and Gamma systems, especially once the project began pushing beyond the easiest small-system settings.

This workspace collects the author’s extensions on top of the original NetKet API, with a strong focus on:

- Kitaev honeycomb models
- Gamma honeycomb models
- Heisenberg honeycomb models
- `J1-J2` Heisenberg models on the triangular lattice
- Transformer-based neural quantum states (ViT-style ansatzes)
- GCNN / residual-GCNN style ansatzes
- SR / minSR / SRt optimization workflows
- Neural Importance Resampling (NIR) with learned proposal networks
- symmetry-aware ansatz design, including full projection and input-level canonicalization

The current emphasis on **8-site** and **18-site** support reflects the project’s practical workflow: smaller systems were used heavily for local prototyping, debugging, and rapid iteration, while larger system studies were intended to be pushed to HPC clusters once model architecture and optimization behavior had been validated on those smaller instances.

The repository is organized as an experimental research workspace rather than a polished Python package. Most scripts are intended to be run directly.

## What This Repo Contains

This codebase extends NetKet with several layers of custom research infrastructure:

- Hamiltonian builders for multiple lattice models
- custom ViT ansatzes for honeycomb systems
- symmetry-projected ViT variants
- input-projected / canonical-representative ViT variants
- NIR training pipelines with learned proposal models
- continuation scripts for resuming/refining previous NIR runs
- benchmark and comparison scripts for SR solver and warm-start studies
- analysis and plotting scripts used to inspect energies, expectations, and scaling trends

In short, this repo captures the progression from conventional SR/minSR studies toward more aggressive ViT+NIR experiments across multiple lattice Hamiltonians.

## Main Project Areas

### `Kitaev_Honeycomb/`

Code for the Kitaev honeycomb model, including:

- standard SR / minSR workflows
- 8-site and 18-site ViT ansatzes
- symmetry-projected ViT models
- input-level symmetry/canonicalization variants
- 8-site and 18-site NIR experiments
- continuation scripts and benchmark studies

Important subdirectories:

- `Kitaev_Honeycomb/8-site/`
  - core 8-site Hamiltonian and ViT model definitions
- `Kitaev_Honeycomb/8-site/nir_experiments/`
  - plain ViT, full symmetry-projected ViT, and input-projected ViT NIR runs
- `Kitaev_Honeycomb/18-site/`
  - 18-site ViT, symmetry, continuation, and benchmark scripts
- `Kitaev_Honeycomb/18-site/nir_experiments/`
  - 18-site NIR workflows, proposal models, continuation utilities

### `Gamma_Honeycomb/`

Code for the Gamma honeycomb model, mirroring much of the Kitaev workflow:

- Gamma Hamiltonian definitions
- SR / minSR studies
- ViT ansatzes
- 8-site and 18-site NIR experiments
- symmetry-projected and input-projected ViT variants
- continuation utilities for long NIR runs

Important subdirectories:

- `Gamma_Honeycomb/8-site/`
- `Gamma_Honeycomb/8-site/nir_experiments/`
- `Gamma_Honeycomb/18-site/`
- `Gamma_Honeycomb/18-site/nir_experiments/`

### `Heisenberg_Honeycomb/`

Earlier honeycomb Heisenberg experiments using SRt/SR-style workflows and expectation-value scripts.

### `J1_J2_Heisenberg_Triangular/`

Triangular-lattice `J1-J2` experiments, primarily focused on GCNN-style ansatzes and SR/SRt workflows, plus a ViT experiment script.

### `Results/`

Analysis and plotting scripts used to turn raw experiment outputs into paper-style figures and comparison plots.

This folder intentionally contains plotting code only in the repository snapshot. Large runtime artifacts such as plots, logs, and checkpoints are excluded from version control.

## ViT Model Variants

Across the honeycomb projects, the following ViT-style ansatz families appear:

- **Plain ViT**
  - a direct transformer ansatz on the spin configuration
- **Full symmetry-projected ViT**
  - evaluates symmetry images and combines them at the wavefunction level
- **Input-projected / canonical-representative ViT**
  - maps each configuration to a canonical symmetry representative before evaluation

These variants were used to study the tradeoff between:

- raw expressivity
- inductive bias from lattice symmetries
- memory cost
- evaluation speed
- NIR convergence behavior

## NIR Infrastructure

One of the central contributions in this workspace is a family of **Neural Importance Resampling (NIR)** pipelines for honeycomb ViT models.

These scripts typically combine:

- a target NQS (plain / symmetry-projected / input-projected ViT)
- a learned proposal network
- adaptive proposal batching
- ESS / efficiency-based update gating
- SR / minSR target directions
- checkpointed continuation workflows
- stage-based learning schedules

Shared NIR support code lives in files such as:

- `proposal_network.py`
- `nir_utils.py`
- `vit_continue_utils.py`

The NIR scripts support:

- plain training runs
- continuation from saved checkpoints
- best-checkpoint continuation
- proposal-state persistence
- late-stage sample-count changes
- tail-statistics reporting

## Optimization Workflows

The repo includes several related optimization approaches:

- **SR / Stochastic Reconfiguration**
- **minSR / projected regularized SR**
- **SRt-style updates**
- **warm starts and continuation**
- **best-of-k / best-checkpoint refinement**
- **NIR with learned proposal adaptation**

These workflows were used to compare ansatz families, convergence quality, symmetry strategies, and scaling behavior across system sizes.

## Typical File Roles

The naming conventions are fairly consistent:

- `define_*_Hamiltonian.py`
  - lattice and Hamiltonian construction
- `*_ViT*.py`
  - model training or experiment launchers
- `*_symm*.py`
  - symmetry-aware ansatz variants
- `*_inputproj*.py`
  - canonical-input symmetry handling
- `*_continue*.py`
  - continuation/refinement runs
- `proposal_network.py`
  - learned proposal model for NIR
- `nir_utils.py`
  - shared NIR bookkeeping, ESS/efficiency logic, proposal pooling, and utilities
- `expectations*.py`
  - observable estimation scripts
- `plot_*.py`
  - analysis/plotting scripts for finished runs

## Running the Code

This repository assumes an already configured Python virtual environment with an up-to-date NetKet/JAX installation.

That environment is **not** part of the committed source snapshot.

In practice, most scripts are launched directly, for example:

```bash
python Kitaev_Honeycomb/8-site/nir_experiments/kitaev_honeycomb_ViT_8_nir.py
python Gamma_Honeycomb/18-site/nir_experiments/gamma_honeycomb_ViT_18_nir_multigpu.py
```

Because this is a research workspace, it is best to inspect the top of a script before running it to confirm:

- sample count
- system size
- device assumptions
- optimizer schedule
- checkpoint paths
- continuation settings

## Notes on Repository Scope

This Git snapshot is intentionally **source-only**.

Excluded from version control:

- runtime checkpoints
- run directories
- logs
- energy trace text files
- generated plots and PDFs
- local virtual environments
- shell scripts containing machine- or user-specific information

The goal is to preserve the research code itself without bundling bulky or sensitive runtime artifacts.

## Research Themes Captured Here

Taken together, this repository documents work on:

- building neural quantum states for frustrated lattice spin models
- exploring transformer ansatzes for honeycomb systems
- adding symmetry handling at both output and input levels
- testing NIR as an alternative to standard MCMC-driven optimization
- comparing convergence behavior across model classes and system sizes
- refining continuation workflows for long-running variational experiments

## Status

This is an active research codebase. Expect:

- script-oriented workflows
- some duplicated experimental scaffolding across directories
- historical prototypes alongside newer NIR pipelines
- parameter settings embedded directly in scripts

That is normal for the way this workspace evolved: it is meant to preserve the actual experimental path, not just a final cleaned abstraction layer.
