import os
import json
from datetime import date
from pathlib import Path
from typing import List
from collections import deque

SHOW_PLOTS = False
if not SHOW_PLOTS:
    os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import netket as nk
from netket.optimizer.solver import cholesky
import optax

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

from hamiltonian import gamma_hamiltonian
from vit_model import (
    KitaevBondAwareHoneycombViT,
    build_kitaev_relation_matrix,
    site_relation_to_patch_relation,
)

print(jax.devices())

# ============================================================
# JAX configuration
# ============================================================
jax.config.update("jax_enable_x64", True)

# ============================================================
# USER OPTIONS
# ============================================================
LEARN_PHASE_WARMUP = False
LEARN_PHASE_MAIN = True

NUM_SITES = 18
NUM_SAMPLES_WARMUP = 3 * 2**9
NUM_SAMPLES = 3 * 2**9

# ---------- Amplitude-only / main training ----------
# Keep SR-only optimisation here; in this project the Adam -> SR handoff
# has been less reliable than staying in SR throughout.
NUM_STARTS = 1
NUM_ITERS_WARM = 100      # amplitude-only SR iterations
NUM_ITERS_MAIN = 1000     # main SR continuation after amplitude-only training
NUM_ITERS_SR = 2000       # final PT-SR refinement

# ---------- Model ----------
EMBED_DIM = 18
NUM_HEADS = 9
NUM_LAYERS = 2
PATCH_SIZE = 1

MLP_HIDDEN_DIM = 2 * EMBED_DIM
CHUNK_SIZE = None
CHUNK_SIZE_BWD = None

# ---------- Runtime / logging ----------
NETKET_DEBUG = False
PROFILE_TIME = False
LOG_STEP_SIZE = 1
WRITE_EVERY = 1
SAVE_PARAMS_EVERY = 25

# ---------- SR implementation ----------
# The stable VMC_SR driver also supports the NTK/on-the-fly path, which is a
# sensible default to test for attention-like models such as this ViT.
USE_EXPERIMENTAL_VMC_SR = False
EXPERIMENTAL_USE_NTK = True
EXPERIMENTAL_ON_THE_FLY = True

# ---------- Amplitude-only SR ----------
WARM_SR_LR = 1e-3
WARM_SR_MOMENTUM = 0.7
WARM_SR_DIAGSHIFT = 1e-3
WARM_SR_SOLVER = cholesky

# ---------- Main SR ----------
MAIN_SR_LR = 1e-3
MAIN_SR_MOMENTUM = 0.9
MAIN_SR_DIAGSHIFT = 1e-4
MAIN_SR_SOLVER = cholesky

# ---------- Final PT-SR ----------
SR_LR = 1e-4
SR_MOMENTUM = 0.0
SR_SOLVER = cholesky
SR_DIAGSHIFT_SCHEDULE = [
    (0.30, 1e-2, "sr_shift1e-2"),
    (0.40, 5e-4, "sr_shift5e-4"),
    (0.30, 2e-4, "sr_shift2e-4"),
]

# ---------- Sampling ----------
# Use sample counts divisible by 3 so work can be split evenly across
# the three CUDA devices available on this machine.
N_DISCARD_PER_CHAIN = 4
TARGET_CHAIN_LENGTH = 64
PT_SWEEP_SIZE = NUM_SITES * 2

TODAY = date.today().isoformat()

JOB_BASE = (
    f"{NUM_SITES}-site_"
    f"{NUM_LAYERS}_layers_"
    f"{NUM_HEADS}_heads_"
    f"{PATCH_SIZE}_patches_"
    f"{NUM_SAMPLES}_samples_"
    f"{TODAY}_Gamma_ViT_SRonly_bondaware"
)

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs" / TODAY
RUN_DIR = RUNS_DIR / JOB_BASE
RUN_DIR.mkdir(parents=True, exist_ok=True)

os.environ["NETKET_DEBUG"] = "1" if NETKET_DEBUG else "0"

# ============================================================
# Hamiltonian + lattice
# ============================================================
graph, symm_group, hi, ha = gamma_hamiltonian(NUM_SITES)

if NUM_SITES <= 18:
    sp_h = ha.to_sparse()
    eig_vals, _ = eigsh(sp_h, k=2, which="SA")
    print("Exact ground-state energy:", eig_vals[0])
    print()


perm = tuple(range(graph.n_nodes))

# ============================================================
# Bond-aware relation matrix
# ============================================================
# The bond-aware ViT needs a static token-token relation matrix.
# For PATCH_SIZE=1 this is a site-site matrix:
#   0 = self
#   1 = graph edge color 0
#   2 = graph edge color 1
#   3 = graph edge color 2
#   4 = non-neighbor
#
# For PATCH_SIZE>1 we convert the site-level matrix into a patch-level
# relation matrix, so the attention bias matches the number of ViT tokens.
site_relation_matrix = build_kitaev_relation_matrix(graph, permutation=perm)

if PATCH_SIZE == 1:
    relation_matrix = site_relation_matrix
else:
    relation_matrix = site_relation_to_patch_relation(
        site_relation_matrix,
        PATCH_SIZE,
    )

NUM_RELATION_TYPES = max(max(row) for row in relation_matrix) + 1

print("Permutation:", perm)
print("Relation matrix shape:", (len(relation_matrix), len(relation_matrix[0])))
print("Number of relation types:", NUM_RELATION_TYPES)
print("Run directory:", RUN_DIR)

# ============================================================
# Sampler helpers
# ============================================================
def make_metropolis_local(hilbert, n_samples):
    n_chains = max(3, n_samples // TARGET_CHAIN_LENGTH)
    n_chains -= n_chains % 3
    n_chains = max(3, n_chains)
    return nk.sampler.MetropolisLocal(
        hilbert=hilbert,
        n_chains=n_chains,
    )

def make_parallel_tempering_local(hilbert, n_samples):
    n_chains = max(3, n_samples // TARGET_CHAIN_LENGTH)
    n_chains -= n_chains % 3
    n_chains = max(3, n_chains)
    return nk.sampler.ParallelTemperingLocal(
        hilbert=hilbert,
        n_chains=n_chains,
        sweep_size=PT_SWEEP_SIZE,
    )

def make_parallel_tempering_hamiltonian(hilbert, n_samples):
    # PT hamiltonian alternative
    n_chains = max(3, n_samples // TARGET_CHAIN_LENGTH)
    n_chains -= n_chains % 3
    n_chains = max(3, n_chains)
    return nk.sampler.ParallelTemperingHamiltonian(
        hilbert,
        hamiltonian=ha,
        n_chains=n_chains,
        sweep_size=PT_SWEEP_SIZE,
    )
# ============================================================
# Model factory
# ============================================================
def build_model():
    return build_model_with_phase(learn_phase=LEARN_PHASE_MAIN)


def build_model_with_phase(*, learn_phase: bool):
    return KitaevBondAwareHoneycombViT(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        patch_size=PATCH_SIZE,
        learn_phase=learn_phase,
        relation_matrix=relation_matrix,
        permutation=perm,
    )


def build_vmc_sr_driver(
    *,
    variational_state,
    optimizer,
    momentum,
    linear_solver,
    diag_shift,
):
    if USE_EXPERIMENTAL_VMC_SR:
        return nk.experimental.driver.VMC_SR(
            hamiltonian=ha,
            optimizer=optimizer,
            variational_state=variational_state,
            momentum=momentum,
            linear_solver_fn=linear_solver,
            mode="complex",
            diag_shift=diag_shift,
            use_ntk=EXPERIMENTAL_USE_NTK,
            on_the_fly=EXPERIMENTAL_ON_THE_FLY,
            chunk_size_bwd=CHUNK_SIZE_BWD,
        )

    return nk.driver.VMC_SR(
        hamiltonian=ha,
        optimizer=optimizer,
        variational_state=variational_state,
        momentum=momentum,
        linear_solver=linear_solver,
        mode="complex",
        diag_shift=diag_shift,
        use_ntk=EXPERIMENTAL_USE_NTK,
        on_the_fly=EXPERIMENTAL_ON_THE_FLY,
        chunk_size_bwd=CHUNK_SIZE_BWD,
    )

# ============================================================
# Utility
# ============================================================
def load_best_energy(logfile):
    with open(logfile) as f:
        data = json.load(f)
    energies = np.array(data["Energy"]["Mean"]["real"], dtype=float)
    tail = energies[-min(20, len(energies)):]
    return float(np.mean(tail)), list(tail)


def run_path(stem: str) -> str:
    return str(RUN_DIR / stem)

all_outs: List[str] = []
print(
    f"\n=== Multi-start amplitude-only SR: {NUM_STARTS} starts, "
    f"{NUM_ITERS_WARM} iterations each ===\n"
)

warmup_summaries = []

for start_idx in range(NUM_STARTS):
    print(f"--- Amplitude-only start {start_idx + 1}/{NUM_STARTS} ---")

    model_warm = build_model_with_phase(learn_phase=LEARN_PHASE_WARMUP)
    sampler_warm = make_metropolis_local(hi, NUM_SAMPLES_WARMUP)

    vstate_warm = nk.vqs.MCState(
        sampler=sampler_warm,
        model=model_warm,
        n_samples=NUM_SAMPLES_WARMUP,
        chunk_size=CHUNK_SIZE,
        n_discard_per_chain=N_DISCARD_PER_CHAIN,
        seed=1234 + start_idx,
    )

    print("Parameters:", vstate_warm.n_parameters)
    print("n_chains:", vstate_warm.sampler.n_chains)
    print("chain_length:", vstate_warm.chain_length)

    warm_opt = optax.sgd(learning_rate=WARM_SR_LR)

    driver_warm = build_vmc_sr_driver(
        optimizer=warm_opt,
        variational_state=vstate_warm,
        momentum=WARM_SR_MOMENTUM,
        linear_solver=WARM_SR_SOLVER,
        diag_shift=WARM_SR_DIAGSHIFT,
    )

    out_warm = run_path(f"out_{JOB_BASE}_warm_start_{start_idx}")

    driver_warm.run(
        n_iter=NUM_ITERS_WARM,
        out=out_warm,
        step_size=LOG_STEP_SIZE,
        write_every=WRITE_EVERY,
        save_params_every=SAVE_PARAMS_EVERY,
        timeit=PROFILE_TIME,
    )
    all_outs.append(out_warm)

    warmup_energy, _ = load_best_energy(f"{out_warm}.log")
    print("Amplitude-only energy:", warmup_energy)
    if NUM_SITES <= 18:
        print("Amplitude-only gap to exact:", warmup_energy - float(eig_vals[0]))
    print()

    warmup_summaries.append(
        {
            "start": start_idx,
            "warmup_energy": warmup_energy,
            "variables": vstate_warm.variables,
            "out": out_warm,
        }
    )

best_warmup = min(warmup_summaries, key=lambda row: row["warmup_energy"])

print("Selected amplitude-only start:", best_warmup["start"])
print("Selected amplitude-only energy:", best_warmup["warmup_energy"])
print()

# ============================================================
# Main SR continuation from amplitude-only training
# ============================================================
print(f"\n=== Main SR continuation for {NUM_ITERS_MAIN} iterations ===")

model_main = build_model_with_phase(learn_phase=LEARN_PHASE_MAIN)
sampler_main = make_metropolis_local(hi, NUM_SAMPLES)

vstate_main = nk.vqs.MCState(
    sampler=sampler_main,
    model=model_main,
    n_samples=NUM_SAMPLES,
    variables=best_warmup["variables"],
    chunk_size=CHUNK_SIZE,
    n_discard_per_chain=N_DISCARD_PER_CHAIN,
)

print("Parameters:", vstate_main.n_parameters)
print("n_chains:", vstate_main.sampler.n_chains)
print("chain_length:", vstate_main.chain_length)

main_sr_opt = optax.sgd(learning_rate=MAIN_SR_LR)

driver_main = build_vmc_sr_driver(
    optimizer=main_sr_opt,
    variational_state=vstate_main,
    momentum=MAIN_SR_MOMENTUM,
    linear_solver=MAIN_SR_SOLVER,
    diag_shift=MAIN_SR_DIAGSHIFT,
)

OUT_MAIN = run_path(f"out_{JOB_BASE}_main_sr")
driver_main.run(
    n_iter=NUM_ITERS_MAIN,
    out=OUT_MAIN,
    step_size=LOG_STEP_SIZE,
    write_every=WRITE_EVERY,
    save_params_every=SAVE_PARAMS_EVERY,
    timeit=PROFILE_TIME,
)
all_outs.append(OUT_MAIN)

# ============================================================
# Final PT-SR refinement
# ============================================================
print(f"\n=== Final PT-SR refinement for {NUM_ITERS_SR} iterations ===")

sampler_sr = make_parallel_tempering_local(hi, NUM_SAMPLES)

vstate_sr = nk.vqs.MCState(
    sampler=sampler_sr,
    model=build_model_with_phase(learn_phase=LEARN_PHASE_MAIN),
    n_samples=NUM_SAMPLES,
    variables=vstate_main.variables,
    chunk_size=CHUNK_SIZE,
    n_discard_per_chain=N_DISCARD_PER_CHAIN,
)

print("Parameters:", vstate_sr.n_parameters)
print("n_chains:", vstate_sr.sampler.n_chains)
print("chain_length:", vstate_sr.chain_length)

sr_opt = optax.sgd(learning_rate=SR_LR)

def run_sr_segment(n_iter, diag_shift, tag):
    driver_sr = build_vmc_sr_driver(
        optimizer=sr_opt,
        variational_state=vstate_sr,
        momentum=SR_MOMENTUM,
        linear_solver=SR_SOLVER,
        diag_shift=diag_shift,
    )
    out = run_path(f"out_{JOB_BASE}_{tag}")
    driver_sr.run(
        n_iter=n_iter,
        out=out,
        step_size=LOG_STEP_SIZE,
        write_every=WRITE_EVERY,
        save_params_every=SAVE_PARAMS_EVERY,
        timeit=PROFILE_TIME,
    )
    return out

remaining = NUM_ITERS_SR
for frac, shift, tag in SR_DIAGSHIFT_SCHEDULE:
    n_seg = int(round(frac * NUM_ITERS_SR))
    n_seg = max(1, min(n_seg, remaining))
    remaining -= n_seg
    all_outs.append(run_sr_segment(n_seg, shift, tag))
    if remaining <= 0:
        break

if remaining > 0:
    last_shift = SR_DIAGSHIFT_SCHEDULE[-1][1]
    all_outs.append(run_sr_segment(remaining, last_shift, "sr_tail"))

# ============================================================
# Collect all energies and summarize
# ============================================================
energy = []
for out in all_outs:
    logf = f"{out}.log"
    if os.path.exists(logf):
        with open(logf) as f:
            data = json.load(f)
            energy.extend(data["Energy"]["Mean"]["real"])

mean_energy_file = RUN_DIR / f"mean_energy_run_{JOB_BASE}.txt"
with open(mean_energy_file, "w") as f:
    for e in energy:
        f.write(f"{e}\n")

plt.figure(figsize=(10, 6))
plt.plot(energy)
plt.xscale("log")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title(f"{NUM_SITES}-site Gamma bond-aware ViT (Amplitude-only SR → SR → PT-SR)")
plt.tight_layout()
plot_file = RUN_DIR / f"kitaev_vit_{JOB_BASE}.png"
plt.savefig(plot_file)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

summary = {
    "job_base": JOB_BASE,
    "run_dir": str(RUN_DIR),
    "num_sites": NUM_SITES,
    "num_starts": NUM_STARTS,
    "num_samples_warmup": NUM_SAMPLES_WARMUP,
    "num_samples": NUM_SAMPLES,
    "num_iters_warm": NUM_ITERS_WARM,
    "num_iters_main": NUM_ITERS_MAIN,
    "num_iters_sr": NUM_ITERS_SR,
    "learn_phase_warmup": LEARN_PHASE_WARMUP,
    "learn_phase_main": LEARN_PHASE_MAIN,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "mlp_hidden_dim": MLP_HIDDEN_DIM,
    "patch_size": PATCH_SIZE,
    "sr_solver_name": "cholesky",
    "netket_debug": NETKET_DEBUG,
    "profile_time": PROFILE_TIME,
    "log_step_size": LOG_STEP_SIZE,
    "write_every": WRITE_EVERY,
    "save_params_every": SAVE_PARAMS_EVERY,
    "use_experimental_vmc_sr": USE_EXPERIMENTAL_VMC_SR,
    "experimental_use_ntk": EXPERIMENTAL_USE_NTK,
    "experimental_on_the_fly": EXPERIMENTAL_ON_THE_FLY,
    "permutation": list(perm),
    "relation_matrix": [list(row) for row in relation_matrix],
    "num_relation_types": NUM_RELATION_TYPES,
    "bond_aware": True,
    "outputs": all_outs,
    "mean_energy_file": str(mean_energy_file),
    "plot_file": str(plot_file),
    "selected_amplitude_only_start": int(best_warmup["start"]),
    "amplitude_only_energy": float(best_warmup["warmup_energy"]),
    "selected_warmup_start": int(best_warmup["start"]),
    "warmup_energy": float(best_warmup["warmup_energy"]),
}

if NUM_SITES <= 18:
    summary["exact_ground_state_energy"] = float(eig_vals[0])
if energy:
    summary["final_energy"] = float(energy[-1])
    summary["best_energy_seen"] = float(np.min(np.asarray(energy, dtype=float)))

summary_file = RUN_DIR / f"summary_{JOB_BASE}.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

if NUM_SITES <= 18 and energy:
    print("Final gap to exact:", float(energy[-1]) - float(eig_vals[0]))
