import os
import json
import sys
from datetime import date, datetime
from pathlib import Path

from flax import serialization

SHOW_PLOTS = False
if not SHOW_PLOTS:
    os.environ.setdefault("MPLBACKEND", "Agg")

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

import jax.numpy as jnp

from nir_utils import (
    effective_sample_size,
    importance_resample,
    normalised_importance_weights_from_log_probs,
    sampling_efficiency,
)
from proposal_network import (
    AutoregressiveProposalNet,
    proposal_log_prob,
    sample_from_proposal,
    train_proposal_step,
)

# ============================================================
# USER OPTIONS
# ============================================================
SOURCE_SUMMARY_FILE = os.environ.get("NIR_CONTINUE_SOURCE_SUMMARY", "")

USE_BEST_CHECKPOINT = True
CONTINUE_ITERS = 500
CONTINUE_TAG = "nir_continue_single_refinement_stage"
CONTINUE_NUM_SAMPLES = 3 * 2**11

CONTINUE_LR = 1e-3
CONTINUE_PROPOSAL_WARMUP_ITERS = 100

# Proposal network settings for continuation.
# These use a short proposal-recovery warmup followed by stricter refinement.
NIR_PROPOSAL_BATCH = 3 * 2**10
NIR_MAX_PROPOSAL_BATCHES = 8
NIR_MAX_ADAPTIVE_ROUNDS = 4
NIR_ESS_THRESHOLD_FRAC = 0.4
NIR_EFFICIENCY_THRESHOLD_WARMUP = 0.08
NIR_EFFICIENCY_THRESHOLD_REFINEMENT = 0.20
NIR_PROPOSAL_LR_WARMUP = 3e-3
NIR_PROPOSAL_LR_REFINEMENT = 3e-4
NIR_PROPOSAL_STEPS_WARMUP = 2
NIR_PROPOSAL_STEPS_REFINEMENT = 1
NIR_PROPOSAL_EMBED_DIM = 24
NIR_PROPOSAL_HEADS = 12
NIR_PROPOSAL_LAYERS = 2
NIR_PROPOSAL_MLP = 2 * NIR_PROPOSAL_EMBED_DIM
NIR_PROB_FLOOR = 1e-6

# ---------- NQS target update ----------
TARGET_PRECONDITIONER = "minsr"
TARGET_SR_DIAG_SHIFT = 1e-5
TARGET_SR_PROJ_REG = None
TARGET_SR_MOMENTUM = 0.0
TARGET_SR_MODE = "complex"

PROPOSAL_LR_SCHEDULE = optax.join_schedules(
    schedules=[
        optax.constant_schedule(NIR_PROPOSAL_LR_WARMUP),
        optax.constant_schedule(NIR_PROPOSAL_LR_REFINEMENT),
    ],
    boundaries=[CONTINUE_PROPOSAL_WARMUP_ITERS],
)

TARGET_CHAIN_LENGTH = 128
CHUNK_SIZE = None
LOG_STEP_SIZE = 1
WRITE_EVERY = 1
SAVE_PARAMS_EVERY = 25

source_summary_path = require_summary(SOURCE_SUMMARY_FILE)
source_summary = load_summary(source_summary_path)
base_summary, _base_summary_path = resolve_base_summary(source_summary, source_summary_path)
source_run_dir = resolve_source_run_dir(source_summary, source_summary_path)
continue_dir = make_continuation_dir(source_run_dir, "nir_continue")

if USE_BEST_CHECKPOINT:
    best_params = source_summary.get("best_params_file") or base_summary.get("best_params_file")
    if not best_params:
        raise FileNotFoundError("No best_params_file found in source summary.")
    checkpoint_path = Path(best_params).expanduser().resolve()
else:
    checkpoint_path = find_latest_checkpoint(source_summary, source_run_dir)

if USE_BEST_CHECKPOINT:
    proposal_state_path_str = source_summary.get("best_proposal_state_file") or base_summary.get(
        "best_proposal_state_file"
    )
else:
    proposal_state_path_str = source_summary.get("proposal_state_file") or base_summary.get(
        "proposal_state_file"
    )
proposal_state_path = (
    Path(proposal_state_path_str).expanduser().resolve()
    if proposal_state_path_str
    else None
)

NUM_SITES = int(base_summary["num_sites"])
NUM_SAMPLES = int(CONTINUE_NUM_SAMPLES)
LEARN_PHASE = bool(base_summary.get("learn_phase", True))
EMBED_DIM = int(base_summary["embed_dim"])
NUM_HEADS = int(base_summary["num_heads"])
NUM_LAYERS = int(base_summary["num_layers"])
MLP_HIDDEN_DIM = int(base_summary["mlp_hidden_dim"])
PATCH_SIZE = int(base_summary["patch_size"])

print("Source summary:", source_summary_path)
print("Source run dir:", source_run_dir)
print("Continuation dir:", continue_dir)
print("Checkpoint:", checkpoint_path)
print("Proposal state:", proposal_state_path if proposal_state_path is not None else "fresh init")

graph, _symm_group, hi, ha = kitaev_hamiltonian(NUM_SITES)
sp_h = ha.to_sparse()
eig_vals, _ = eigsh(sp_h, k=2, which="SA")
EXACT_GROUND_STATE_ENERGY = float(eig_vals[0])
print("Exact ground-state energy:", EXACT_GROUND_STATE_ENERGY)
print()


def symmetry_inverse_perm(g, n_sites):
    if hasattr(g, "inverse_permutation_array"):
        return tuple(np.asarray(g.inverse_permutation_array, dtype=int).tolist())
    if hasattr(g, "permutation_array"):
        perm = np.asarray(g.permutation_array, dtype=int)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        return tuple(inv.tolist())
    if callable(g):
        perm = np.asarray([g(i) for i in range(n_sites)], dtype=int)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        return tuple(inv.tolist())
    raise TypeError(f"Unsupported symmetry object type: {type(g)}")


symmetry_perms = tuple(symmetry_inverse_perm(g, graph.n_nodes) for g in _symm_group)


def make_metropolis_local(hilbert, n_samples):
    n_chains = max(1, n_samples // TARGET_CHAIN_LENGTH)
    return nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=n_chains)


def fresh_key():
    seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    return jax.random.PRNGKey(seed)


def build_model():
    return SymmetryProjectedHoneycombViT(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        patch_size=PATCH_SIZE,
        learn_phase=LEARN_PHASE,
        symmetries=symmetry_perms,
        permutation=tuple(range(graph.n_nodes)),
    )


def target_log_probs(vstate, sigma):
    log_psi = vstate.log_value(sigma)
    return 2.0 * jnp.real(log_psi)


def inject_external_samples(vstate, samples):
    samples = jnp.asarray(samples, dtype=jnp.float64)
    reshaped = samples.reshape(vstate.sampler.n_chains, vstate.chain_length, NUM_SITES)
    vstate._samples = reshaped
    return reshaped


def _schedule_value(value, step):
    return value(step) if callable(value) else value


def current_efficiency_threshold(step):
    if step < CONTINUE_PROPOSAL_WARMUP_ITERS:
        return NIR_EFFICIENCY_THRESHOLD_WARMUP
    return NIR_EFFICIENCY_THRESHOLD_REFINEMENT


def current_proposal_steps(step):
    if step < CONTINUE_PROPOSAL_WARMUP_ITERS:
        return NIR_PROPOSAL_STEPS_WARMUP
    return NIR_PROPOSAL_STEPS_REFINEMENT


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
            proj_reg=_schedule_value(TARGET_SR_PROJ_REG, step),
            momentum=_schedule_value(TARGET_SR_MOMENTUM, step),
            old_updates=target_update_state["old_updates"],
            chunk_size=CHUNK_SIZE,
        )
        target_update_state["old_updates"] = old_updates
        target_update_state["info"] = info
        return stats, dp, target_update_state

    raise ValueError(f"Unsupported TARGET_PRECONDITIONER state: {kind}")


def sample_until_ess(vstate, proposal_model, proposal_params, rng, *, target_n_samples):
    proposal_batches = []
    log_target_batches = []
    log_proposal_batches = []
    ess_threshold = NIR_ESS_THRESHOLD_FRAC * target_n_samples

    for _ in range(NIR_MAX_PROPOSAL_BATCHES):
        proposal_samples, rng = sample_from_proposal(
            proposal_model,
            proposal_params,
            rng,
            NIR_PROPOSAL_BATCH,
            NUM_SITES,
            prob_floor=NIR_PROB_FLOOR,
        )
        log_target = target_log_probs(vstate, proposal_samples)
        log_proposal = proposal_log_prob(
            proposal_model,
            proposal_params,
            proposal_samples,
            prob_floor=NIR_PROB_FLOOR,
        )

        proposal_batches.append(np.asarray(proposal_samples))
        log_target_batches.append(np.asarray(log_target))
        log_proposal_batches.append(np.asarray(log_proposal))

        stacked_target = np.concatenate(log_target_batches, axis=0)
        stacked_proposal = np.concatenate(log_proposal_batches, axis=0)
        weights = normalised_importance_weights_from_log_probs(
            stacked_target, stacked_proposal
        )
        ess = effective_sample_size(weights)
        if ess >= ess_threshold:
            break

    all_samples = np.concatenate(proposal_batches, axis=0)
    all_log_target = np.concatenate(log_target_batches, axis=0)
    all_log_proposal = np.concatenate(log_proposal_batches, axis=0)
    weights = normalised_importance_weights_from_log_probs(
        all_log_target, all_log_proposal
    )
    return all_samples, all_log_target, all_log_proposal, weights, rng


def run_adaptive_nir_round(
    vstate,
    proposal_model,
    proposal_params,
    proposal_opt_state,
    proposal_optimizer,
    rng,
    *,
    step,
):
    round_summaries = []
    params = proposal_params
    opt_state = proposal_opt_state
    final_resampled = None

    for round_idx in range(NIR_MAX_ADAPTIVE_ROUNDS):
        all_samples, all_log_target, all_log_proposal, weights, rng = sample_until_ess(
            vstate, proposal_model, params, rng, target_n_samples=vstate.n_samples
        )
        ess = effective_sample_size(weights)
        eff = sampling_efficiency(weights)
        resampled, _indices, _weights = importance_resample(
            all_samples,
            all_log_target,
            all_log_proposal,
            n_samples=vstate.n_samples,
        )
        final_resampled = resampled

        train_batch = jnp.asarray(resampled)
        last_loss = None
        proposal_steps = current_proposal_steps(step)
        efficiency_threshold = current_efficiency_threshold(step)
        for _ in range(proposal_steps):
            params, opt_state, last_loss = train_proposal_step(
                proposal_model,
                params,
                opt_state,
                proposal_optimizer,
                train_batch,
                prob_floor=NIR_PROB_FLOOR,
            )

        round_summaries.append(
            {
                "round": round_idx,
                "proposal_pool": int(len(all_samples)),
                "resampled_batch": int(len(resampled)),
                "ess": float(ess),
                "efficiency": float(eff),
                "meets_ess_threshold": bool(ess >= NIR_ESS_THRESHOLD_FRAC * vstate.n_samples),
                "meets_efficiency_threshold": bool(eff >= efficiency_threshold),
                "efficiency_threshold": float(efficiency_threshold),
                "proposal_steps": int(proposal_steps),
                "forward_kl_loss_after_steps": None if last_loss is None else float(last_loss),
            }
        )

        if eff >= efficiency_threshold:
            break

    return {
        "rounds": round_summaries,
        "final_resampled_shape": None if final_resampled is None else tuple(final_resampled.shape),
    }, params, opt_state, final_resampled, rng


def run_nir_stage(
    *,
    out_prefix,
    vstate,
    hamiltonian,
    n_iter,
    optimizer,
    proposal_model,
    proposal_params,
    proposal_opt_state,
    proposal_optimizer,
):
    params = vstate.parameters
    opt_state = optimizer.init(params)
    target_update_state = build_target_update_state()
    rng = fresh_key()
    history = []
    best_energy = None
    best_energy_distance = None
    best_iteration = None
    best_params_file = f"{out_prefix}_best.mpack"
    logger = nk.logging.JsonLog(
        out_prefix,
        write_every=WRITE_EVERY,
        save_params_every=SAVE_PARAMS_EVERY,
        save_params=True,
    )

    for it in range(n_iter):
        nir_summary, proposal_params, proposal_opt_state, resampled, rng = run_adaptive_nir_round(
            vstate,
            proposal_model,
            proposal_params,
            proposal_opt_state,
            proposal_optimizer,
            rng,
            step=it,
        )
        last_round = nir_summary["rounds"][-1]
        target_update_applied = bool(last_round["meets_efficiency_threshold"])
        inject_external_samples(vstate, resampled)
        if target_update_applied:
            stats, dp, target_update_state = compute_target_direction(
                vstate, hamiltonian, target_update_state, it
            )
            dp = jax.tree.map(
                lambda update, param: jnp.asarray(update, dtype=param.dtype), dp, params
            )
            updates, opt_state = optimizer.update(dp, opt_state, params)
            params = optax.apply_updates(params, updates)
            vstate.parameters = params
        else:
            stats = nk.stats.statistics(
                vstate.local_estimators(hamiltonian, chunk_size=CHUNK_SIZE)
            )

        energy = float(np.real(np.asarray(stats.mean)))
        energy_distance = abs(energy - EXACT_GROUND_STATE_ENERGY)
        history.append(
            {
                "iteration": it + 1,
                "energy": energy,
                "energy_distance_to_exact": energy_distance,
                "target_update_applied": target_update_applied,
                "nir": nir_summary,
            }
        )
        if best_energy_distance is None or energy_distance < best_energy_distance:
            best_energy = energy
            best_energy_distance = energy_distance
            best_iteration = it + 1
            with open(best_params_file, "wb") as f:
                f.write(serialization.to_bytes(vstate.variables))
        logger(
            it,
            {
                "Energy": stats,
                "NIR": {
                    "ESS": float(last_round["ess"]),
                    "Efficiency": float(last_round["efficiency"]),
                    "ProposalPool": int(last_round["proposal_pool"]),
                    "TargetUpdateApplied": int(target_update_applied),
                },
            },
            variational_state=vstate,
        )
        if (it + 1) % max(1, LOG_STEP_SIZE) == 0:
            print(
                f"it={it + 1:5d} "
                f"Energy={energy:.8f} "
                f"ESS={last_round['ess']:.2f} "
                f"Eff={last_round['efficiency']:.4f} "
                f"Update={'yes' if target_update_applied else 'no'}"
            )

    logger.flush(vstate)
    return {
        "history": history,
        "final_energy": history[-1]["energy"] if history else None,
        "best_energy": best_energy,
        "best_energy_distance_to_exact": best_energy_distance,
        "best_iteration": best_iteration,
        "best_params_file": best_params_file,
        "proposal_params": proposal_params,
        "proposal_opt_state": proposal_opt_state,
        "log_file": out_prefix + ".log",
    }


print(
    "\n=== NIR continuation: "
    f"{CONTINUE_ITERS} iterations at fixed refinement lr={CONTINUE_LR} "
    f"with {CONTINUE_PROPOSAL_WARMUP_ITERS} proposal-warmup iterations ===\n"
)

model = build_model()
sampler = make_metropolis_local(hi, NUM_SAMPLES)

vstate = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    n_samples=NUM_SAMPLES,
    chunk_size=CHUNK_SIZE,
)
vstate.variables = nk.experimental.vqs.variables_from_file(str(checkpoint_path), vstate.variables)

proposal_model = AutoregressiveProposalNet(
    n_sites=NUM_SITES,
    embed_dim=NIR_PROPOSAL_EMBED_DIM,
    num_heads=NIR_PROPOSAL_HEADS,
    num_layers=NIR_PROPOSAL_LAYERS,
    mlp_hidden_dim=NIR_PROPOSAL_MLP,
)
init_sigma = jnp.ones((4, NUM_SITES), dtype=jnp.float64)
proposal_params = proposal_model.init(fresh_key(), init_sigma)["params"]
proposal_optimizer = optax.adam(PROPOSAL_LR_SCHEDULE)
proposal_opt_state = proposal_optimizer.init(proposal_params)
if proposal_state_path is not None and proposal_state_path.exists():
    with open(proposal_state_path, "rb") as f:
        proposal_bytes = f.read()
    proposal_template = {
        "params": proposal_params,
        "opt_state": proposal_opt_state,
    }
    try:
        proposal_state = serialization.from_bytes(proposal_template, proposal_bytes)
        proposal_params = proposal_state["params"]
        proposal_opt_state = proposal_state["opt_state"]
        print("Loaded proposal params and optimizer state from source run.")
    except ValueError as err:
        print(
            "Proposal optimizer-state restore was incompatible with the continuation "
            f"optimizer shape ({err}). Loading proposal params only and reinitializing "
            "the proposal optimizer state."
        )
        raw_state = serialization.msgpack_restore(proposal_bytes)
        proposal_params = serialization.from_state_dict(proposal_params, raw_state["params"])
        proposal_opt_state = proposal_optimizer.init(proposal_params)
else:
    print("Proposal state file not found; using fresh proposal initialization.")

print("Parameters:", vstate.n_parameters)
print("n_chains:", vstate.sampler.n_chains)
print("chain_length:", vstate.chain_length)

out_prefix = str(continue_dir / f"out_{source_run_dir.name}_{CONTINUE_TAG}")
result = run_nir_stage(
    out_prefix=out_prefix,
    vstate=vstate,
    hamiltonian=ha,
    n_iter=CONTINUE_ITERS,
    optimizer=optax.sgd(learning_rate=CONTINUE_LR),
    proposal_model=proposal_model,
    proposal_params=proposal_params,
    proposal_opt_state=proposal_opt_state,
    proposal_optimizer=proposal_optimizer,
)

history_file = Path(f"{out_prefix}.json")
with open(history_file, "w") as f:
    json.dump(result["history"], f, indent=2)

previous_energy = collect_previous_energy(source_summary, summary_path=source_summary_path)
new_energy = [row["energy"] for row in result["history"]]
combined_energy = previous_energy + new_energy
tail_energy_window = min(100, len(new_energy))
tail_energy_mean = None
tail_energy_std = None
tail_energy_stderr = None
if tail_energy_window:
    tail_energy = np.asarray(new_energy[-tail_energy_window:], dtype=float)
    tail_energy_mean = float(np.mean(tail_energy))
    tail_energy_std = float(np.std(tail_energy, ddof=1)) if tail_energy_window > 1 else 0.0
    tail_energy_stderr = float(tail_energy_std / np.sqrt(tail_energy_window))

mean_energy_file = continue_dir / f"mean_energy_continue_{source_run_dir.name}.txt"
with open(mean_energy_file, "w") as f:
    for e in combined_energy:
        f.write(f"{e}\n")

plt.figure(figsize=(10, 6))
if combined_energy:
    plt.plot(combined_energy)
plt.xscale("log")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title(f"{NUM_SITES}-site Kitaev (NIR continuation)")
plt.tight_layout()
plot_file = continue_dir / f"kitaev_vit_continue_{source_run_dir.name}.png"
plt.savefig(plot_file)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

final_energy = float(result["final_energy"])
print("Final continuation energy:", final_energy)
print("Final continuation gap to exact:", final_energy - EXACT_GROUND_STATE_ENERGY)
print("Best continuation energy:", float(result["best_energy"]))
print(
    "Best continuation abs distance to exact:",
    float(result["best_energy_distance_to_exact"]),
)
print("Best continuation iteration:", int(result["best_iteration"]))
print("Best continuation params:", result["best_params_file"])
if tail_energy_window:
    print(
        f"Last {tail_energy_window} continuation mean energy: "
        f"{tail_energy_mean:.12f} ± {tail_energy_stderr:.12f} "
        f"(std={tail_energy_std:.12f})"
    )

summary = {
    "source_summary_file": str(source_summary_path),
    "source_run_dir": str(source_run_dir),
    "checkpoint_file": str(checkpoint_path),
    "proposal_state_file": None if proposal_state_path is None else str(proposal_state_path),
    "continuation_run_dir": str(continue_dir),
    "continue_iters": CONTINUE_ITERS,
    "continue_lr": CONTINUE_LR,
    "proposal_warmup_iters": CONTINUE_PROPOSAL_WARMUP_ITERS,
    "target_preconditioner": TARGET_PRECONDITIONER,
    "target_sr_diag_shift": TARGET_SR_DIAG_SHIFT,
    "target_sr_proj_reg": TARGET_SR_PROJ_REG,
    "target_sr_momentum": TARGET_SR_MOMENTUM,
    "target_efficiency_gate_warmup": NIR_EFFICIENCY_THRESHOLD_WARMUP,
    "target_efficiency_gate_refinement": NIR_EFFICIENCY_THRESHOLD_REFINEMENT,
    "use_best_checkpoint": USE_BEST_CHECKPOINT,
    "num_sites": NUM_SITES,
    "num_samples": NUM_SAMPLES,
    "source_num_samples": int(base_summary["num_samples"]),
    "continue_num_samples": NUM_SAMPLES,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "patch_size": PATCH_SIZE,
    "proposal_batch": NIR_PROPOSAL_BATCH,
    "proposal_lr_warmup": NIR_PROPOSAL_LR_WARMUP,
    "proposal_lr_refinement": NIR_PROPOSAL_LR_REFINEMENT,
    "proposal_steps_warmup": NIR_PROPOSAL_STEPS_WARMUP,
    "proposal_steps_refinement": NIR_PROPOSAL_STEPS_REFINEMENT,
    "outputs": [str(result["log_file"])],
    "history_file": str(history_file),
    "mean_energy_file": str(mean_energy_file),
    "plot_file": str(plot_file),
    "best_params_file": str(result["best_params_file"]),
    "exact_ground_state_energy": EXACT_GROUND_STATE_ENERGY,
}
if combined_energy:
    summary["final_energy"] = float(combined_energy[-1])
    combined = np.asarray(combined_energy, dtype=float)
    closest_idx = int(np.argmin(np.abs(combined - EXACT_GROUND_STATE_ENERGY)))
    summary["best_energy_seen"] = float(combined[closest_idx])
    summary["best_energy_distance_to_exact"] = float(
        abs(combined[closest_idx] - EXACT_GROUND_STATE_ENERGY)
    )
    summary["best_energy_iteration"] = int(closest_idx + 1)
if new_energy:
    summary["continuation_tail_energy_window"] = int(tail_energy_window)
    summary["continuation_tail_energy_mean"] = float(tail_energy_mean)
    summary["continuation_tail_energy_std"] = float(tail_energy_std)
    summary["continuation_tail_energy_stderr"] = float(tail_energy_stderr)
summary["target_updates_applied"] = int(
    sum(row["target_update_applied"] for row in result["history"])
)
summary["continuation_final_energy"] = float(result["final_energy"])
summary["continuation_best_energy"] = float(result["best_energy"])
summary["continuation_best_energy_distance_to_exact"] = float(
    result["best_energy_distance_to_exact"]
)
summary["continuation_best_iteration"] = int(result["best_iteration"])

summary_file = continue_dir / f"summary_continue_{source_run_dir.name}.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)
