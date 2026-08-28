# NIR Paper Reproduction

This directory is for reproducing the baseline results from:

Ledinauskas and Anisimovas, "Neural Importance Resampling: A Practical Sampling Strategy for Neural Quantum States", arXiv:2507.20510.

The goal is to isolate the original NIR method from our honeycomb ViT experiments. If the paper-like reproduction works, then later failures are more likely to come from our model/system choices. If it does not work, then we should debug the NIR implementation before drawing conclusions about the physics.

## What This Directory Is

This is a paper-faithful reproduction workspace using our local Python/JAX/NetKet environment where possible. The arXiv source does not include an official runnable codebase, so this directory records the paper settings and provides a scaffold for recreating them cleanly.

## Reproduction Targets

1. TFI sampler benchmark:
   - 2D transverse-field Ising model.
   - Paper compares NIR against MCMC on 4x4, 6x6, and 8x8 lattices.
   - Paper also uses a 2x3 small-system JSD diagnostic against exact probabilities.
   - Target NQS is a multi-state MLP with Adam plus MinSR.

2. Square-lattice J1-J2 benchmark:
   - 10x10 square lattice at J2/J1 = 0.5.
   - Target NQS is a residual CNN, not a honeycomb ViT.
   - Proposal network is the same autoregressive Transformer family used for TFI.
   - Paper reports final energy per site near -0.4966 to -0.4969 depending on proposal size.

## Important Separation From Our Current Runs

This reproduction should not use:

- Honeycomb lattices.
- Site-aware ViT / relation-gated / attention-pooling models.
- Exact or full-sum late-stage refinement.
- Proposal fixed-magnetization constraints unless explicitly testing a variant.
- Our staged learning-rate and staged sample-count schedules unless we add them as a non-paper ablation.

## First Commands

Print the current paper-derived reproduction plan:

```bash
./NetKet_Updated_venv/bin/python NIR_Paper_Reproduction/scripts/print_reproduction_plan.py
```

Run a small single-state TFI NIR smoke test:

```bash
NIR_PAPER_NUM_STEPS=20 \
NIR_PAPER_BATCH_SIZE=128 \
NIR_PAPER_PROPOSAL_BATCH=128 \
./NetKet_Updated_venv/bin/python NIR_Paper_Reproduction/scripts/run_tfi_single_state_nir.py
```

Run a small square-lattice J1-J2 ResNet smoke test:

```bash
NIR_PAPER_J1J2_LENGTH=4 \
NIR_PAPER_NUM_STEPS=20 \
NIR_PAPER_BATCH_SIZE=128 \
NIR_PAPER_PROPOSAL_BATCH=128 \
./NetKet_Updated_venv/bin/python NIR_Paper_Reproduction/scripts/run_j1j2_square_resnet_nir.py
```

## Ground-State Runs

Small TFI ground-state run with ED comparison:

```bash
MPLCONFIGDIR=/tmp/mpl \
JAX_PLATFORMS=cpu \
NIR_PAPER_TFI_LX=3 \
NIR_PAPER_TFI_LY=3 \
NIR_PAPER_TFI_G=0.01 \
NIR_PAPER_NUM_STEPS=10000 \
NIR_PAPER_BATCH_SIZE=512 \
NIR_PAPER_PROPOSAL_BATCH=512 \
NIR_PAPER_TARGET_PRECONDITIONER=minsr \
NIR_PAPER_RUN_NAME=tfi_3x3_g=0.01_ground_state_paper_nir \
./NetKet_Updated_venv/bin/python NIR_Paper_Reproduction/scripts/run_tfi_single_state_nir.py
```

Paper-comparable square J1-J2 ground-state run:

```bash
MPLCONFIGDIR=/tmp/mpl \
JAX_PLATFORMS=gpu \
NIR_PAPER_J1J2_LENGTH=10 \
NIR_PAPER_J1=1.0 \
NIR_PAPER_J2=0.5 \
NIR_PAPER_NUM_STEPS=10000 \
NIR_PAPER_BATCH_SIZE=512 \
NIR_PAPER_PROPOSAL_BATCH=512 \
NIR_PAPER_MAX_PROPOSAL_BATCHES=64 \
NIR_PAPER_MAX_ADAPTIVE_ROUNDS=8 \
NIR_PAPER_ALPHA_ESS=2.0 \
NIR_PAPER_ALPHA_EFF=0.1 \
NIR_PAPER_TARGET_LR=1e-3 \
NIR_PAPER_PROPOSAL_LR=1e-3 \
NIR_PAPER_TARGET_PRECONDITIONER=minsr \
NIR_PAPER_PROPOSAL_EMBED_DIM=32 \
NIR_PAPER_PROPOSAL_HEADS=4 \
NIR_PAPER_PROPOSAL_LAYERS=4 \
NIR_PAPER_RUN_NAME=j1j2_square_10x10_j2=0.5_ground_state_paper_nir \
./NetKet_Updated_venv/bin/python NIR_Paper_Reproduction/scripts/run_j1j2_square_resnet_nir.py
```

If a long run reaches good energy and then ends with NaNs, first rerun with the
paper-scale batch size instead of the smoke-test batch size. The smoke tests use
128 samples for turnaround time, but the paper settings use 512 samples and a
larger proposal pool. The reproduction loop also records target/proposal update
norms and rejects non-finite parameter updates so one unstable MinSR/Adam step
does not poison the rest of the run.

Conservative 4x4 diagnostic rerun:

```bash
MPLCONFIGDIR=/tmp/mpl \
JAX_PLATFORMS=cpu \
NIR_PAPER_J1J2_LENGTH=4 \
NIR_PAPER_J1=1.0 \
NIR_PAPER_J2=0.5 \
NIR_PAPER_NUM_STEPS=1000 \
NIR_PAPER_BATCH_SIZE=512 \
NIR_PAPER_PROPOSAL_BATCH=512 \
NIR_PAPER_MAX_PROPOSAL_BATCHES=64 \
NIR_PAPER_MAX_ADAPTIVE_ROUNDS=8 \
NIR_PAPER_TARGET_LR=3e-4 \
NIR_PAPER_SR_DIAG_SHIFT=1e-2 \
NIR_PAPER_REJECT_NONFINITE_UPDATES=true \
NIR_PAPER_TARGET_UPDATE_NORM_CLIP=10.0 \
NIR_PAPER_RUN_NAME=j1j2_square_4x4_j2=0.5_guarded_paper_nir \
./NetKet_Updated_venv/bin/python NIR_Paper_Reproduction/scripts/run_j1j2_square_resnet_nir.py
```

## Current Runnable Scope

The implemented code now includes:

- Paper-style ferromagnetic TFI Hamiltonian builder.
- Paper-style square-lattice J1-J2 Hamiltonian builder.
- Single-state MLP target for TFI.
- Residual CNN target for square J1-J2.
- Autoregressive Transformer proposal network.
- Importance resampling with forward-KL proposal updates.
- ESS accumulation and efficiency-gated adaptive retraining.
- Adam target/proposal optimizers, with optional MinSR for the target.

The TFI script is currently a single-state reproduction harness. The paper's published TFI benchmark uses a multi-state determinant-style NQS for the three lowest states, so that remains the next required step before claiming a full TFI paper reproduction.
