from __future__ import annotations

import json
import os

SHOW_PLOTS = False
if not SHOW_PLOTS:
    os.environ.setdefault("MPLBACKEND", "Agg")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
import optax
from netket.optimizer.solver import cholesky, pinv_smooth
from scipy.sparse.linalg import eigsh

from define_Kitaev_Hamiltonian import kitaev_hamiltonian
from kitaev_honeycomb_vit_symm_model import SymmetryProjectedHoneycombViT
from vit_continue_utils import (
    collect_previous_energy,
    find_latest_checkpoint,
    load_summary,
    make_continuation_dir,
    require_summary,
    resolve_base_summary,
    resolve_source_run_dir,
)

print(jax.devices())
jax.config.update("jax_enable_x64", True)

# ============================================================
# USER OPTIONS
# ============================================================
SOURCE_SUMMARY_FILE = "runs/2026-04-03/18-site_2_layers_12_heads_2_patches_512_samples_2026-04-03_Kitaev_ViT_SRonly_symmproj_singlegpu/continuation_symm_singlegpu_2026-04-05_14-32-51/summary_continue_18-site_2_layers_12_heads_2_patches_512_samples_2026-04-03_Kitaev_ViT_SRonly_symmproj_singlegpu.json"

NUM_ITERS_CONTINUE = 8000
CONTINUE_DIAGSHIFT = 1e-3
CONTINUE_LR = 1e-2
CONTINUE_MOMENTUM = 0.0
CONTINUE_TAG = "sr_continue"
RUN_SAMPLER_MODES = ("pt",)

N_DISCARD_PER_CHAIN = 4
TARGET_CHAIN_LENGTH = 32
PT_SWEEP_SIZE_FACTOR = 2
SOURCE_SAMPLER_MODE = "pt"

PROFILE_TIME = False
LOG_STEP_SIZE = 1
WRITE_EVERY = 1
SAVE_PARAMS_EVERY = 25


def solver_from_name(name: str):
    mapping = {
        "pinv_smooth": pinv_smooth,
        "cholesky": cholesky,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported solver in summary: {name}")
    return mapping[name]


def normalise_sampler_modes(modes):
    if isinstance(modes, str):
        modes = [modes]
    out = []
    seen = set()
    for mode in modes:
        mode = mode.lower()
        if mode not in {"local", "pt"}:
            raise ValueError(f"Unsupported sampler mode: {mode}")
        if mode not in seen:
            seen.add(mode)
            out.append(mode)
    if not out:
        raise ValueError("RUN_SAMPLER_MODES must contain at least one of 'local' or 'pt'.")
    return tuple(out)


summary_path = require_summary(SOURCE_SUMMARY_FILE)
source_summary = load_summary(summary_path)
base_summary, _base_summary_path = resolve_base_summary(source_summary, summary_path)
source_run_dir = resolve_source_run_dir(source_summary, summary_path)
checkpoint_path = find_latest_checkpoint(
    source_summary, source_run_dir, preferred_mode=SOURCE_SAMPLER_MODE
)
continue_run_dir = make_continuation_dir(source_run_dir, "continuation_symm_singlegpu")
SAMPLER_MODES = normalise_sampler_modes(RUN_SAMPLER_MODES)

NUM_SITES = int(base_summary["num_sites"])
NUM_SAMPLES = int(base_summary["num_samples"])
EMBED_DIM = int(base_summary["embed_dim"])
NUM_HEADS = int(base_summary["num_heads"])
NUM_LAYERS = int(base_summary["num_layers"])
MLP_HIDDEN_DIM = int(base_summary["mlp_hidden_dim"])
PATCH_SIZE = int(base_summary["patch_size"])
PERM = tuple(base_summary["permutation"])
LEARN_PHASE = bool(base_summary.get("learn_phase_main", True))
SR_SOLVER = solver_from_name(base_summary.get("sr_solver_name", "cholesky"))
CHUNK_SIZE = 2**8
CHUNK_SIZE_BWD = 2**8

print("Source summary:", summary_path)
print("Source run dir:", source_run_dir)
print("Checkpoint:", checkpoint_path)
print("Continuation run dir:", continue_run_dir)
print("Sampler modes:", SAMPLER_MODES)
if SOURCE_SAMPLER_MODE is not None:
    print("Source sampler mode:", SOURCE_SAMPLER_MODE)

graph, symm_group, hi, ha = kitaev_hamiltonian(NUM_SITES)
symmetry_perms = tuple(tuple(g.inverse_permutation_array.tolist()) for g in symm_group)

if NUM_SITES <= 18:
    sp_h = ha.to_sparse()
    eig_vals, _ = eigsh(sp_h, k=2, which="SA")
    print("Exact ground-state energy:", eig_vals[0])
    print()


def make_metropolis_local(hilbert, n_samples):
    n_chains = max(1, n_samples // TARGET_CHAIN_LENGTH)
    return nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=n_chains)


def make_parallel_tempering_local(hilbert, n_samples):
    n_chains = max(1, n_samples // TARGET_CHAIN_LENGTH)
    return nk.sampler.ParallelTemperingLocal(
        hilbert=hilbert,
        n_chains=n_chains,
        sweep_size=NUM_SITES * PT_SWEEP_SIZE_FACTOR,
    )


def build_model():
    return SymmetryProjectedHoneycombViT(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        patch_size=PATCH_SIZE,
        learn_phase=LEARN_PHASE,
        symmetries=symmetry_perms,
        permutation=PERM,
    )


def build_vmc_sr_driver(*, variational_state, optimizer, momentum, linear_solver, diag_shift):
    return nk.driver.VMC_SR(
        hamiltonian=ha,
        optimizer=optimizer,
        variational_state=variational_state,
        momentum=momentum,
        linear_solver=linear_solver,
        mode="complex",
        diag_shift=diag_shift,
        use_ntk=bool(base_summary.get("experimental_use_ntk", True)),
        on_the_fly=bool(base_summary.get("experimental_on_the_fly", True)),
        chunk_size_bwd=CHUNK_SIZE_BWD,
    )


def sampler_from_mode(mode: str):
    if mode == "local":
        return make_metropolis_local(hi, NUM_SAMPLES)
    return make_parallel_tempering_local(hi, NUM_SAMPLES)


previous_energy = collect_previous_energy(
    source_summary, summary_path=summary_path, preferred_mode=SOURCE_SAMPLER_MODE
)
continuation_runs = {}

for mode in SAMPLER_MODES:
    print(f"\n=== Continuation with {mode} sampler for {NUM_ITERS_CONTINUE} iterations ===")
    sampler = sampler_from_mode(mode)
    vstate_sr = nk.vqs.MCState(
        sampler=sampler,
        model=build_model(),
        n_samples=NUM_SAMPLES,
        chunk_size=CHUNK_SIZE,
        n_discard_per_chain=N_DISCARD_PER_CHAIN,
    )
    vstate_sr.variables = nk.experimental.vqs.variables_from_file(
        str(checkpoint_path), vstate_sr.variables
    )

    print("Parameters:", vstate_sr.n_parameters)
    print("n_chains:", vstate_sr.sampler.n_chains)
    print("chain_length:", vstate_sr.chain_length)

    driver_sr = build_vmc_sr_driver(
        optimizer=optax.sgd(learning_rate=CONTINUE_LR),
        variational_state=vstate_sr,
        momentum=CONTINUE_MOMENTUM,
        linear_solver=SR_SOLVER,
        diag_shift=CONTINUE_DIAGSHIFT,
    )

    out_continue = continue_run_dir / f"out_{source_run_dir.name}_{CONTINUE_TAG}_{mode}"
    driver_sr.run(
        n_iter=NUM_ITERS_CONTINUE,
        out=str(out_continue),
        step_size=LOG_STEP_SIZE,
        write_every=WRITE_EVERY,
        save_params_every=SAVE_PARAMS_EVERY,
        timeit=PROFILE_TIME,
    )

    with open(out_continue.with_suffix(".log")) as f:
        data = json.load(f)
        new_energy = list(np.asarray(data["Energy"]["Mean"]["real"], dtype=float))

    combined_energy = previous_energy + new_energy
    mean_energy_file = continue_run_dir / f"mean_energy_continue_{source_run_dir.name}_{mode}.txt"
    with open(mean_energy_file, "w") as f:
        for e in combined_energy:
            f.write(f"{e}\n")

    plt.figure(figsize=(10, 6))
    plt.plot(combined_energy)
    plt.xscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"{NUM_SITES}-site Kitaev continuation (symmetry-projected ViT single GPU, {mode})")
    plt.tight_layout()
    plot_file = continue_run_dir / f"kitaev_vit_continue_{source_run_dir.name}_{mode}.png"
    plt.savefig(plot_file)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    run_summary = {
        "sampler_mode": mode,
        "output": str(out_continue),
        "mean_energy_file": str(mean_energy_file),
        "plot_file": str(plot_file),
    }
    if combined_energy:
        run_summary["final_energy"] = float(combined_energy[-1])
        run_summary["best_energy_seen"] = float(np.min(np.asarray(combined_energy)))
    continuation_runs[mode] = run_summary

continuation_summary = {
    "source_summary_file": str(summary_path),
    "source_run_dir": str(source_run_dir),
    "checkpoint_file": str(checkpoint_path),
    "continuation_run_dir": str(continue_run_dir),
    "num_iters_continue": NUM_ITERS_CONTINUE,
    "continue_diagshift": CONTINUE_DIAGSHIFT,
    "continue_lr": CONTINUE_LR,
    "continue_momentum": CONTINUE_MOMENTUM,
    "sr_solver_name": base_summary.get("sr_solver_name", "cholesky"),
    "sampler_modes": list(SAMPLER_MODES),
    "source_sampler_mode": SOURCE_SAMPLER_MODE,
    "runs": continuation_runs,
}

if NUM_SITES <= 18:
    continuation_summary["exact_ground_state_energy"] = float(eig_vals[0])
    for mode, run_summary in continuation_runs.items():
        if "final_energy" in run_summary:
            print(f"Final gap to exact ({mode}):", run_summary["final_energy"] - float(eig_vals[0]))

summary_file = continue_run_dir / f"summary_continue_{source_run_dir.name}.json"
with open(summary_file, "w") as f:
    json.dump(continuation_summary, f, indent=2)
