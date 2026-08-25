import os
import json
from pathlib import Path

from flax import serialization

SHOW_PLOTS = False
if not SHOW_PLOTS:
    os.environ.setdefault("MPLBACKEND", "Agg")

# Default to CPU for continuation runs because the archived 18-site bond-aware
# checkpoint can exceed available GPU memory on this machine.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import netket as nk
import optax
from netket._src.ngd.sr_srt_common import srt as compute_minsr_direction

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

THIS_DIR = Path(__file__).resolve().parent
PARENT_18 = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PARENT_18) not in sys.path:
    sys.path.insert(0, str(PARENT_18))

from hamiltonian import gamma_hamiltonian
from vit_continue_utils import (
    collect_previous_energy,
    find_best_training_state,
    find_latest_checkpoint,
    find_latest_training_state,
    load_summary,
    make_continuation_dir,
    require_summary,
    resolve_base_summary,
    resolve_source_run_dir,
)
from vit_model import (
    KitaevBondAwareHoneycombViT,
    build_kitaev_relation_matrix,
    site_relation_to_patch_relation,
)

print(jax.devices())
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def resolve_artifact_path(candidate, run_dir):
    path = Path(candidate).expanduser().resolve()
    if path.is_file():
        return path
    local_candidate = (run_dir / path.name).resolve()
    if local_candidate.is_file():
        return local_candidate
    return path


# ============================================================
# USER SETTINGS
# Edit this directly. No environment flags are required.
# ============================================================
# Main input: point this at the source summary JSON you want to continue from.
SOURCE_SUMMARY_PATH = str(
    PARENT_18
    / "18-site_6_layers_4_heads_1_patches_identity_1024to8192_samples_2026-04-28_Gamma_ViT_NIR_bondaware_multiphase"
    / "summary_18-site_6_layers_4_heads_1_patches_identity_1024to8192_samples_2026-04-28_Gamma_ViT_NIR_bondaware_multiphase.json"
)

USE_TRAINING_STATE = False
USE_BEST_CHECKPOINT = True
RESET_TARGET_OPT_STATE = True
RESET_TARGET_UPDATE_STATE = True

CONTINUE_ITERS = 300
CONTINUE_TAG = "mcmc_refinement_from_nir_bondaware_summary"
CONTINUE_LR = 2e-5
TARGET_PRECONDITIONER = "minsr"
TARGET_SR_DIAG_SHIFT_OVERRIDE = "1e-5"
TARGET_SR_MOMENTUM = 0.0
TARGET_SR_MODE = "complex"
NUM_SAMPLES_OVERRIDE = "8192"
SAMPLER_MODE = "pt_local"
TARGET_CHAIN_LENGTH = 128
N_DISCARD_PER_CHAIN = 16
CHUNK_SIZE = 512
LOG_STEP_SIZE = 1
WRITE_EVERY = 1
SAVE_PARAMS_EVERY = 25


if not SOURCE_SUMMARY_PATH.strip():
    raise FileNotFoundError(
        "Set SOURCE_SUMMARY_PATH near the top of this file before launching."
    )

source_summary_path = require_summary(SOURCE_SUMMARY_PATH)

source_summary = load_summary(source_summary_path)
base_summary, base_summary_path = resolve_base_summary(source_summary, source_summary_path)
source_run_dir = resolve_source_run_dir(source_summary, source_summary_path)
continue_dir = make_continuation_dir(source_run_dir, CONTINUE_TAG)

training_state_path = None
if USE_TRAINING_STATE:
    if USE_BEST_CHECKPOINT:
        training_state_path = find_best_training_state(source_summary, source_run_dir)
    else:
        training_state_path = find_latest_training_state(source_summary, source_run_dir)

if training_state_path is None and USE_BEST_CHECKPOINT:
    best_params = source_summary.get("best_params_file") or base_summary.get("best_params_file")
    if not best_params:
        raise FileNotFoundError("No best_params_file found in source summary.")
    checkpoint_path = resolve_artifact_path(best_params, source_run_dir)
else:
    checkpoint_path = find_latest_checkpoint(source_summary, source_run_dir)

NUM_SITES = int(base_summary["num_sites"])
LEARN_PHASE = bool(
    base_summary.get(
        "learn_phase_stage_3",
        base_summary.get("learn_phase_stage_2", base_summary.get("learn_phase", True)),
    )
)
EMBED_DIM = int(base_summary["embed_dim"])
NUM_HEADS = int(base_summary["num_heads"])
NUM_LAYERS = int(base_summary["num_layers"])
MLP_HIDDEN_DIM = int(base_summary["mlp_hidden_dim"])
PATCH_SIZE = int(base_summary["patch_size"])
TARGET_SR_DIAG_SHIFT = float(
    TARGET_SR_DIAG_SHIFT_OVERRIDE
    or base_summary.get(
        "target_sr_diag_shift_stage_3",
        base_summary.get("target_sr_diag_shift", 1e-5),
    )
)
CONTINUE_NUM_SAMPLES = int(
    NUM_SAMPLES_OVERRIDE
    or base_summary.get("num_samples_stage_3", base_summary.get("num_samples", 4096))
)

print("Source summary:", source_summary_path)
print("Base summary:", base_summary_path)
print("Source run dir:", source_run_dir)
print("Continuation dir:", continue_dir)
print("Training state:", training_state_path if training_state_path is not None else "none")
print("Checkpoint:", checkpoint_path)

graph, _symm_group, hi, ha = gamma_hamiltonian(NUM_SITES)
sp_h = ha.to_sparse()
eig_vals, _ = eigsh(sp_h, k=2, which="SA")
EXACT_GROUND_STATE_ENERGY = float(eig_vals[0])
print("Exact ground-state energy:", EXACT_GROUND_STATE_ENERGY)
print()


def summary_permutation():
    perm = base_summary.get("permutation")
    if perm is not None:
        return tuple(int(x) for x in perm)
    return tuple(range(graph.n_nodes))


def summary_relation_matrix(perm):
    rel = base_summary.get("relation_matrix")
    if rel is not None:
        return tuple(tuple(int(x) for x in row) for row in rel)
    site_relation_matrix = build_kitaev_relation_matrix(graph, permutation=perm)
    if PATCH_SIZE == 1:
        return tuple(tuple(int(x) for x in row) for row in site_relation_matrix)
    return tuple(
        tuple(int(x) for x in row)
        for row in site_relation_to_patch_relation(site_relation_matrix, PATCH_SIZE)
    )


perm = summary_permutation()
relation_matrix = summary_relation_matrix(perm)

print("Permutation:", perm)
print("Relation matrix shape:", (len(relation_matrix), len(relation_matrix[0])))
print("Run learn_phase:", LEARN_PHASE)
print("Sampler mode:", SAMPLER_MODE)
print("Continuation samples:", CONTINUE_NUM_SAMPLES)
print("Continuation lr:", CONTINUE_LR)
print("Target preconditioner:", TARGET_PRECONDITIONER)
print("Target SR diag shift:", TARGET_SR_DIAG_SHIFT)
print("Target SR momentum:", TARGET_SR_MOMENTUM)
print()


def make_metropolis_local(hilbert, n_samples):
    n_chains = max(1, n_samples // TARGET_CHAIN_LENGTH)
    return nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=n_chains)


def make_parallel_tempering_local(hilbert, n_samples):
    n_chains = max(3, n_samples // TARGET_CHAIN_LENGTH)
    return nk.sampler.ParallelTemperingLocal(
        hilbert=hilbert,
        n_chains=n_chains,
        sweep_size=NUM_SITES * 2,
    )


def make_parallel_tempering_hamiltonian(hilbert, n_samples):
    n_chains = max(3, n_samples // TARGET_CHAIN_LENGTH)
    return nk.sampler.ParallelTemperingHamiltonian(
        hilbert,
        hamiltonian=ha,
        n_chains=n_chains,
        sweep_size=NUM_SITES * 2,
    )


def make_sampler(hilbert, n_samples):
    if SAMPLER_MODE == "metropolis":
        return make_metropolis_local(hilbert, n_samples)
    if SAMPLER_MODE == "pt_local":
        return make_parallel_tempering_local(hilbert, n_samples)
    if SAMPLER_MODE == "pt_hamiltonian":
        return make_parallel_tempering_hamiltonian(hilbert, n_samples)
    raise ValueError(
        "NIR_CONTINUE_SAMPLER must be one of: metropolis, pt_local, pt_hamiltonian"
    )


def fresh_key():
    seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    return jax.random.PRNGKey(seed)


def build_model():
    return KitaevBondAwareHoneycombViT(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        patch_size=PATCH_SIZE,
        learn_phase=LEARN_PHASE,
        relation_matrix=relation_matrix,
        permutation=perm,
    )


def run_path(stem: str) -> str:
    return str(continue_dir / stem)


def load_checkpoint_variables(path, variables_template):
    with open(path, "rb") as f:
        return serialization.from_bytes(variables_template, f.read())


def load_raw_state(path):
    with open(path, "rb") as f:
        return serialization.msgpack_restore(f.read())


def save_continue_state(
    path,
    *,
    variables,
    target_opt_state,
    target_update_state,
    rng,
    completed_iterations,
    best_energy,
    best_energy_distance_to_exact,
    best_iteration,
):
    state = {
        "variables": variables,
        "target_opt_state": target_opt_state,
        "target_update_state": target_update_state,
        "rng": rng,
        "completed_iterations": int(completed_iterations),
        "best_energy": best_energy,
        "best_energy_distance_to_exact": best_energy_distance_to_exact,
        "best_iteration": best_iteration,
    }
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def clear_cached_samples(vstate):
    if hasattr(vstate, "reset"):
        vstate.reset()
        return
    if hasattr(vstate, "_samples"):
        vstate._samples = None


def _schedule_value(value, step):
    return value(step) if callable(value) else value


def build_target_update_state():
    kind = TARGET_PRECONDITIONER.lower()
    if kind == "sr":
        return {
            "kind": kind,
            "preconditioner": nk.optimizer.SR(diag_shift=TARGET_SR_DIAG_SHIFT),
            "old_updates": None,
            "info": None,
        }
    if kind == "minsr":
        return {
            "kind": kind,
            "preconditioner": None,
            "old_updates": None,
            "info": None,
        }
    raise ValueError(f"Unsupported TARGET_PRECONDITIONER: {TARGET_PRECONDITIONER}")


def compute_target_direction(vstate, hamiltonian, target_update_state, step):
    kind = target_update_state["kind"]
    if kind == "sr":
        stats, grad = vstate.expect_and_grad(hamiltonian)
        dp = target_update_state["preconditioner"](vstate, grad, step)
        info = getattr(target_update_state["preconditioner"], "info", None)
        target_update_state["info"] = info
        return stats, dp, target_update_state

    if kind == "minsr":
        local_energies = vstate.local_estimators(hamiltonian, chunk_size=CHUNK_SIZE)
        stats = nk.stats.statistics(local_energies)
        samples = jax.lax.collapse(vstate.samples, 0, vstate.samples.ndim - 1)
        dp, old_updates, info = compute_minsr_direction(
            vstate._apply_fun,
            local_energies,
            vstate.parameters,
            vstate.model_state,
            samples,
            diag_shift=_schedule_value(TARGET_SR_DIAG_SHIFT, step),
            solver_fn=nk.optimizer.solver.cholesky,
            mode=TARGET_SR_MODE,
            proj_reg=None,
            momentum=_schedule_value(TARGET_SR_MOMENTUM, step),
            old_updates=target_update_state["old_updates"],
            chunk_size=CHUNK_SIZE,
        )
        target_update_state["old_updates"] = old_updates
        target_update_state["info"] = info
        return stats, dp, target_update_state

    raise ValueError(f"Unsupported TARGET_PRECONDITIONER state: {kind}")


def run_mcmc_continue(
    *,
    out_prefix,
    vstate,
    hamiltonian,
    n_iter,
    optimizer,
    start_iteration=0,
    initial_target_opt_state=None,
    initial_target_update_state=None,
    initial_rng=None,
    initial_best_energy=None,
    initial_best_energy_distance=None,
    initial_best_iteration=None,
):
    params = vstate.parameters
    opt_state = (
        optimizer.init(params) if initial_target_opt_state is None else initial_target_opt_state
    )
    target_update_state = (
        build_target_update_state()
        if initial_target_update_state is None
        else initial_target_update_state
    )
    rng = fresh_key() if initial_rng is None else initial_rng
    history = []
    best_energy = initial_best_energy
    best_energy_distance = initial_best_energy_distance
    best_iteration = initial_best_iteration
    best_params_file = f"{out_prefix}_best.mpack"
    training_state_file = f"{out_prefix}_training_state.mpack"
    best_training_state_file = f"{out_prefix}_training_state_best.mpack"
    logger = nk.logging.JsonLog(
        out_prefix,
        write_every=WRITE_EVERY,
        save_params_every=SAVE_PARAMS_EVERY,
        save_params=True,
    )

    for it in range(n_iter):
        global_it = start_iteration + it
        clear_cached_samples(vstate)
        stats, dp, target_update_state = compute_target_direction(
            vstate, hamiltonian, target_update_state, global_it
        )
        dp = jax.tree.map(
            lambda update, param: jnp.asarray(update, dtype=param.dtype), dp, params
        )
        updates, opt_state = optimizer.update(dp, opt_state, params)
        params = optax.apply_updates(params, updates)
        vstate.parameters = params

        energy = float(np.real(np.asarray(stats.mean)))
        energy_distance = abs(energy - EXACT_GROUND_STATE_ENERGY)
        history.append(
            {
                "iteration": global_it + 1,
                "sampling_mode": SAMPLER_MODE,
                "energy": energy,
                "energy_distance_to_exact": energy_distance,
                "target_update_applied": True,
            }
        )
        if best_energy_distance is None or energy_distance < best_energy_distance:
            best_energy = energy
            best_energy_distance = energy_distance
            best_iteration = global_it + 1
            with open(best_params_file, "wb") as f:
                f.write(serialization.to_bytes(vstate.variables))
            save_continue_state(
                best_training_state_file,
                variables=vstate.variables,
                target_opt_state=opt_state,
                target_update_state=target_update_state,
                rng=rng,
                completed_iterations=global_it + 1,
                best_energy=best_energy,
                best_energy_distance_to_exact=best_energy_distance,
                best_iteration=best_iteration,
            )
        logger(
            global_it,
            {
                "Energy": stats,
                "MCMC": {
                    "TargetSamples": int(vstate.n_samples),
                    "TargetUpdateApplied": 1,
                },
            },
            variational_state=vstate,
        )
        if ((it + 1) % SAVE_PARAMS_EVERY == 0) or (it + 1 == n_iter):
            save_continue_state(
                training_state_file,
                variables=vstate.variables,
                target_opt_state=opt_state,
                target_update_state=target_update_state,
                rng=rng,
                completed_iterations=global_it + 1,
                best_energy=best_energy,
                best_energy_distance_to_exact=best_energy_distance,
                best_iteration=best_iteration,
            )
        if (it + 1) % max(1, LOG_STEP_SIZE) == 0:
            print(
                f"it={global_it + 1:5d} "
                f"Mode={SAMPLER_MODE} "
                f"Energy={energy:.8f} "
                f"Samples={vstate.n_samples:d}"
            )

    logger.flush(vstate)
    return {
        "history": history,
        "final_energy": history[-1]["energy"] if history else None,
        "best_energy": best_energy,
        "best_energy_distance_to_exact": best_energy_distance,
        "best_iteration": best_iteration,
        "best_params_file": best_params_file,
        "training_state_file": training_state_file,
        "best_training_state_file": best_training_state_file,
        "log_file": out_prefix + ".log",
        "completed_iterations": start_iteration + n_iter,
    }


print(f"\n=== MCMC continuation: {CONTINUE_ITERS} iterations ===\n")

model = build_model()
sampler = make_sampler(hi, CONTINUE_NUM_SAMPLES)
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    n_samples=CONTINUE_NUM_SAMPLES,
    chunk_size=CHUNK_SIZE,
    n_discard_per_chain=N_DISCARD_PER_CHAIN,
)

target_optimizer = optax.sgd(learning_rate=CONTINUE_LR)

start_iteration = 0
initial_target_opt_state = None
initial_target_update_state = None
initial_rng = None
initial_best_energy = None
initial_best_energy_distance = None
initial_best_iteration = None
resume_source_training_state = None

if training_state_path is not None:
    resume_state = load_raw_state(training_state_path)
    vstate.variables = serialization.from_state_dict(vstate.variables, resume_state["variables"])
    if not RESET_TARGET_OPT_STATE and resume_state.get("target_opt_state") is not None:
        initial_target_opt_state = serialization.from_state_dict(
            target_optimizer.init(vstate.parameters),
            resume_state["target_opt_state"],
        )
    if not RESET_TARGET_UPDATE_STATE and resume_state.get("target_update_state") is not None:
        initial_target_update_state = serialization.from_state_dict(
            build_target_update_state(),
            resume_state["target_update_state"],
        )
    initial_rng = resume_state.get("rng")
    start_iteration = int(resume_state.get("completed_iterations", 0))
    initial_best_energy = resume_state.get("best_energy")
    initial_best_energy_distance = resume_state.get("best_energy_distance_to_exact")
    initial_best_iteration = resume_state.get("best_iteration")
    resume_source_training_state = training_state_path
    print("Loaded training state:", training_state_path)
    print("Resuming from completed iteration:", start_iteration)
else:
    vstate.variables = load_checkpoint_variables(checkpoint_path, vstate.variables)
    if USE_BEST_CHECKPOINT:
        start_iteration = int(
            source_summary.get(
                "best_energy_iteration", base_summary.get("best_energy_iteration", 0)
            )
            or 0
        )
    else:
        start_iteration = len(
            collect_previous_energy(source_summary, summary_path=source_summary_path)
        )
    initial_best_energy = source_summary.get(
        "best_energy_seen", base_summary.get("best_energy_seen")
    )
    initial_best_energy_distance = source_summary.get(
        "best_energy_distance_to_exact", base_summary.get("best_energy_distance_to_exact")
    )
    initial_best_iteration = source_summary.get(
        "best_energy_iteration", base_summary.get("best_energy_iteration")
    )
    print("Loaded checkpoint-only state:", checkpoint_path)
    print("Recovered iteration count from prior logs:", start_iteration)

print("Parameters:", vstate.n_parameters)

out_prefix = run_path(
    f"out_continue_{NUM_SITES}-site_{NUM_LAYERS}_layers_{NUM_HEADS}_heads_"
    f"{PATCH_SIZE}_patches_bondaware"
)
result = run_mcmc_continue(
    out_prefix=out_prefix,
    vstate=vstate,
    hamiltonian=ha,
    n_iter=CONTINUE_ITERS,
    optimizer=target_optimizer,
    start_iteration=start_iteration,
    initial_target_opt_state=initial_target_opt_state,
    initial_target_update_state=initial_target_update_state,
    initial_rng=initial_rng,
    initial_best_energy=initial_best_energy,
    initial_best_energy_distance=initial_best_energy_distance,
    initial_best_iteration=initial_best_iteration,
)

history_file = Path(f"{out_prefix}.json")
with open(history_file, "w") as f:
    json.dump(result["history"], f, indent=2)

training_state_file = Path(result["training_state_file"])

print("Final MCMC energy:", float(result["final_energy"]))
print("Final MCMC gap to exact:", float(result["final_energy"]) - EXACT_GROUND_STATE_ENERGY)
print("Best MCMC energy:", float(result["best_energy"]))
print("Best MCMC abs distance to exact:", float(result["best_energy_distance_to_exact"]))
print("Best MCMC iteration:", int(result["best_iteration"]))
print("Best params file:", result["best_params_file"])
print("Training state file:", training_state_file)
print("Best training state file:", result["best_training_state_file"])
print()

energy = [row["energy"] for row in result["history"]]
tail_energy_window = min(100, len(energy))
tail_energy_mean = None
tail_energy_std = None
tail_energy_stderr = None
if tail_energy_window:
    tail_energy = np.asarray(energy[-tail_energy_window:], dtype=float)
    tail_energy_mean = float(np.mean(tail_energy))
    tail_energy_std = float(np.std(tail_energy, ddof=1)) if tail_energy_window > 1 else 0.0
    tail_energy_stderr = float(tail_energy_std / np.sqrt(tail_energy_window))
    print(
        f"Last {tail_energy_window} mean energy: "
        f"{tail_energy_mean:.12f} ± {tail_energy_stderr:.12f} "
        f"(std={tail_energy_std:.12f})"
    )
    print()

mean_energy_file = continue_dir / "mean_energy_continue_mcmc.txt"
with open(mean_energy_file, "w") as f:
    for e in energy:
        f.write(f"{e}\n")

plt.figure(figsize=(10, 6))
if energy:
    plt.plot(energy)
plt.xscale("log")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title(f"{NUM_SITES}-site Gamma (bond-aware MCMC continuation)")
plt.tight_layout()
plot_file = continue_dir / "energy_continue_mcmc.png"
plt.savefig(plot_file)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

summary = {
    "source_summary_file": str(source_summary_path),
    "base_summary_file": str(base_summary_path),
    "source_run_dir": str(source_run_dir),
    "continuation_run_dir": str(continue_dir),
    "continue_tag": CONTINUE_TAG,
    "continue_iters": CONTINUE_ITERS,
    "continue_lr": CONTINUE_LR,
    "continue_num_samples": CONTINUE_NUM_SAMPLES,
    "continue_sampler": SAMPLER_MODE,
    "continue_n_discard_per_chain": N_DISCARD_PER_CHAIN,
    "continue_use_training_state": USE_TRAINING_STATE,
    "continue_use_best_checkpoint": USE_BEST_CHECKPOINT,
    "continue_reset_target_opt_state": RESET_TARGET_OPT_STATE,
    "continue_reset_target_update_state": RESET_TARGET_UPDATE_STATE,
    "sampler_modes": [SAMPLER_MODE],
    "runs": {
        SAMPLER_MODE: {
            "output": out_prefix,
            "checkpoint_file": str(Path(f"{out_prefix}.mpack")),
            "history_file": str(history_file),
            "training_state_file": str(training_state_file),
            "best_training_state_file": str(result["best_training_state_file"]),
            "best_params_file": str(result["best_params_file"]),
            "best_energy": float(result["best_energy"]),
            "best_iteration": int(result["best_iteration"]),
        }
    },
    "num_sites": NUM_SITES,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "mlp_hidden_dim": MLP_HIDDEN_DIM,
    "patch_size": PATCH_SIZE,
    "learn_phase": LEARN_PHASE,
    "permutation": list(perm),
    "relation_matrix": [list(row) for row in relation_matrix],
    "bond_aware": True,
    "target_preconditioner": TARGET_PRECONDITIONER,
    "target_sr_diag_shift": TARGET_SR_DIAG_SHIFT,
    "target_sr_momentum": TARGET_SR_MOMENTUM,
    "exact_ground_state_energy": EXACT_GROUND_STATE_ENERGY,
    "mean_energy_file": str(mean_energy_file),
    "plot_file": str(plot_file),
    "completed_iterations": int(result["completed_iterations"]),
}

if resume_source_training_state is not None:
    summary["resume_source_training_state_file"] = str(resume_source_training_state)
else:
    summary["resume_source_checkpoint_file"] = str(checkpoint_path)

if energy:
    summary["final_energy"] = float(energy[-1])
    summary["best_energy_seen"] = float(result["best_energy"])
    summary["best_energy_distance_to_exact"] = float(result["best_energy_distance_to_exact"])
    summary["best_energy_iteration"] = int(result["best_iteration"])
    summary["tail_energy_window"] = int(tail_energy_window)
    summary["tail_energy_mean"] = float(tail_energy_mean)
    summary["tail_energy_std"] = float(tail_energy_std)
    summary["tail_energy_stderr"] = float(tail_energy_stderr)
    summary["target_updates_applied"] = int(
        sum(row["target_update_applied"] for row in result["history"])
    )

summary_file = continue_dir / "summary_mcmc_continue.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print("Summary file:", summary_file)
