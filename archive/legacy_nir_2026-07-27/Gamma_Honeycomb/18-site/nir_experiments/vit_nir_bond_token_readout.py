import os
import json
import sys
from datetime import date
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

from hamiltonian import gamma_hamiltonian
from nir_resume_utils import load_training_state, save_training_state
from vit_continue_utils import (
    collect_previous_energy,
    find_best_training_state,
    find_latest_checkpoint,
    find_latest_training_state,
    load_summary,
    make_continuation_dir,
    require_summary,
    resolve_source_run_dir,
)
from vit_model import build_kitaev_relation_matrix
from vit_bond_token_model import (
    BondAwareBondTokenReadoutHoneycombViT,
    build_edge_index_and_color,
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

LEARN_PHASE_STAGE_1 = False
LEARN_PHASE_STAGE_2 = True
LEARN_PHASE_STAGE_3 = True

NUM_SITES = 18

NUM_SAMPLES_STAGE_1 = 3 * 2**10
NUM_SAMPLES_STAGE_2 = NUM_SAMPLES_STAGE_1
NUM_SAMPLES_STAGE_3 = 3 * 2**11
NUM_ITERS_TOTAL = 2000

EMBED_DIM = 24
NUM_HEADS = 8
NUM_LAYERS = 2
PATCH_SIZE = 1
MLP_HIDDEN_DIM = 2 * EMBED_DIM
CHUNK_SIZE = 512

TRAIN_LR_STAGE_1 = 1e-2
TRAIN_LR_STAGE_2 = 1e-2
TRAIN_LR_STAGE_3 = 5e-3
TRAIN_LR_STAGE_1_ITERS = 50
TRAIN_LR_STAGE_2_ITERS = 600
LOG_STEP_SIZE = 1
WRITE_EVERY = 1
SAVE_PARAMS_EVERY = 25

# ---------- NIR proposal network ----------
NIR_PROPOSAL_BATCH = 3*2**10
NIR_MAX_PROPOSAL_BATCHES = 8
NIR_MAX_ADAPTIVE_ROUNDS = 4
NIR_ESS_THRESHOLD_FRAC = 0.4
NIR_EFFICIENCY_THRESHOLD_STAGE_1 = 0.15
NIR_EFFICIENCY_THRESHOLD_STAGE_2 = 0.15
NIR_EFFICIENCY_THRESHOLD_STAGE_3 = 0.20
NIR_PROPOSAL_LR_STAGE_1 = 3e-3
NIR_PROPOSAL_LR_STAGE_2 = 3e-3
NIR_PROPOSAL_LR_STAGE_3 = 3e-4
NIR_PROPOSAL_STEPS_STAGE_1 = 4
NIR_PROPOSAL_STEPS_STAGE_2 = 2
NIR_PROPOSAL_STEPS_STAGE_3 = 1
NIR_PROPOSAL_EMBED_DIM = 24
NIR_PROPOSAL_HEADS = 12
NIR_PROPOSAL_LAYERS = 2
NIR_PROPOSAL_MLP = 2 * NIR_PROPOSAL_EMBED_DIM
NIR_PROB_FLOOR = 1e-6

# ---------- NQS target update ----------
TARGET_PRECONDITIONER = "minsr"
TARGET_SR_DIAG_SHIFT_STAGE_1 = 1e-3
TARGET_SR_DIAG_SHIFT_STAGE_2 = 1e-4
TARGET_SR_DIAG_SHIFT_STAGE_3 = 1e-5
TARGET_SR_PROJ_REG = None
TARGET_SR_MOMENTUM = 0.9
TARGET_SR_MODE = "complex"

TRAIN_LR_BOUNDARY_1 = TRAIN_LR_STAGE_1_ITERS
TRAIN_LR_BOUNDARY_2 = TRAIN_LR_STAGE_1_ITERS + TRAIN_LR_STAGE_2_ITERS

TRAIN_LR_SCHEDULE = optax.join_schedules(
    schedules=[
        optax.constant_schedule(TRAIN_LR_STAGE_1),
        optax.constant_schedule(TRAIN_LR_STAGE_2),
        optax.constant_schedule(TRAIN_LR_STAGE_3),
    ],
    boundaries=[
        TRAIN_LR_BOUNDARY_1,
        TRAIN_LR_BOUNDARY_2,
    ],
)

TARGET_SR_DIAG_SHIFT = optax.join_schedules(
    schedules=[
        optax.constant_schedule(TARGET_SR_DIAG_SHIFT_STAGE_1),
        optax.constant_schedule(TARGET_SR_DIAG_SHIFT_STAGE_2),
        optax.constant_schedule(TARGET_SR_DIAG_SHIFT_STAGE_3),
    ],
    boundaries=[
        TRAIN_LR_BOUNDARY_1,
        TRAIN_LR_BOUNDARY_2,
    ],
)

PROPOSAL_LR_SCHEDULE = optax.join_schedules(
    schedules=[
        optax.constant_schedule(NIR_PROPOSAL_LR_STAGE_1),
        optax.constant_schedule(NIR_PROPOSAL_LR_STAGE_2),
        optax.constant_schedule(NIR_PROPOSAL_LR_STAGE_3),
    ],
    boundaries=[
        TRAIN_LR_BOUNDARY_1,
        TRAIN_LR_BOUNDARY_2,
    ],
)

# Only used to choose a convenient MCState chain layout for injected NIR samples.
TARGET_CHAIN_LENGTH = 128

TODAY = date.today().isoformat()
RESUME_SOURCE_SUMMARY = os.environ.get("NIR_RESUME_SOURCE_SUMMARY", "").strip()
RESUME_MODE = os.environ.get("NIR_RESUME_MODE", "latest").strip().lower()
RESUME_ADDITIONAL_ITERS = int(
    os.environ.get("NIR_RESUME_ADDITIONAL_ITERS", str(NUM_ITERS_TOTAL))
)

JOB_BASE = (
    f"{NUM_SITES}-site_"
    f"{NUM_LAYERS}_layers_"
    f"{NUM_HEADS}_heads_"
    f"{PATCH_SIZE}_patches_"
    f"identity_"
    f"{NUM_SAMPLES_STAGE_1}to{NUM_SAMPLES_STAGE_3}_samples_"
    f"{TODAY}_Gamma_ViT_NIR_bond_token_readout_multiphase"
)
resume_summary_path = None
resume_summary = None
resume_source_run_dir = None
if RESUME_SOURCE_SUMMARY:
    if RESUME_MODE not in {"latest", "best"}:
        raise ValueError("NIR_RESUME_MODE must be either 'latest' or 'best'.")
    resume_summary_path = require_summary(RESUME_SOURCE_SUMMARY)
    resume_summary = load_summary(resume_summary_path)
    resume_source_run_dir = resolve_source_run_dir(resume_summary, resume_summary_path)
    RUN_DIR = make_continuation_dir(resume_source_run_dir, "nir_resume")
else:
    RUNS_DIR = THIS_DIR / "runs" / TODAY
    RUN_DIR = RUNS_DIR / JOB_BASE
    RUN_DIR.mkdir(parents=True, exist_ok=True)

os.environ["NETKET_DEBUG"] = "1"

graph, _symm_group, hi, ha = gamma_hamiltonian(NUM_SITES)

sp_h = ha.to_sparse()
eig_vals, _ = eigsh(sp_h, k=2, which="SA")
EXACT_GROUND_STATE_ENERGY = float(eig_vals[0])
print("Exact ground-state energy:", EXACT_GROUND_STATE_ENERGY)
print()

# ============================================================
# Bond-aware relation matrix + physical edge list
# ============================================================
# This model currently assumes PATCH_SIZE=1, so each token is one physical site.
# relation_matrix is used inside attention; edge_indices/edge_relations are used
# by the final bond-token readout.
if PATCH_SIZE != 1:
    raise ValueError("Bond-token readout NIR script currently requires PATCH_SIZE=1.")

perm = tuple(range(graph.n_nodes))
relation_matrix = build_kitaev_relation_matrix(graph, permutation=perm)
edge_indices, edge_relations = build_edge_index_and_color(graph, permutation=perm)
NUM_RELATION_TYPES = max(max(row) for row in relation_matrix) + 1

print("Permutation:", perm)
print("Relation matrix shape:", (len(relation_matrix), len(relation_matrix[0])))
print("Number of physical edges:", len(edge_indices))
print("Number of relation types:", NUM_RELATION_TYPES)
print("Run directory:", RUN_DIR)


def make_metropolis_local(hilbert, n_samples):
    n_chains = max(1, n_samples // TARGET_CHAIN_LENGTH)
    return nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=n_chains)


def fresh_key():
    seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    return jax.random.PRNGKey(seed)


def build_model():
    return BondAwareBondTokenReadoutHoneycombViT(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        patch_size=PATCH_SIZE,
        learn_phase=LEARN_PHASE_STAGE_1,
        relation_matrix=relation_matrix,
        edge_indices=edge_indices,
        edge_relations=edge_relations,
        num_relation_types=NUM_RELATION_TYPES,
        permutation=perm,
    )


def build_model_for_phase(learn_phase):
    return BondAwareBondTokenReadoutHoneycombViT(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        patch_size=PATCH_SIZE,
        learn_phase=learn_phase,
        relation_matrix=relation_matrix,
        edge_indices=edge_indices,
        edge_relations=edge_relations,
        num_relation_types=NUM_RELATION_TYPES,
        permutation=perm,
    )


def current_learn_phase(step):
    if step < TRAIN_LR_BOUNDARY_1:
        return LEARN_PHASE_STAGE_1
    if step < TRAIN_LR_BOUNDARY_2:
        return LEARN_PHASE_STAGE_2
    return LEARN_PHASE_STAGE_3


def current_num_samples(step):
    if step < TRAIN_LR_BOUNDARY_1:
        return NUM_SAMPLES_STAGE_1
    if step < TRAIN_LR_BOUNDARY_2:
        return NUM_SAMPLES_STAGE_2
    return NUM_SAMPLES_STAGE_3


def rebuild_vstate_for_stage(vstate, learn_phase, n_samples):
    sampler = make_metropolis_local(hi, n_samples)
    rebuilt = nk.vqs.MCState(
        sampler=sampler,
        model=build_model_for_phase(learn_phase),
        n_samples=n_samples,
        variables=vstate.variables,
        chunk_size=CHUNK_SIZE,
    )
    if (
        hasattr(vstate, "_samples")
        and getattr(vstate, "_samples", None) is not None
        and vstate.n_samples == n_samples
    ):
        rebuilt._samples = vstate._samples
    return rebuilt


def run_path(stem: str) -> str:
    return str(RUN_DIR / stem)


def save_proposal_state(path, proposal_params, proposal_opt_state):
    state = {
        "params": proposal_params,
        "opt_state": proposal_opt_state,
    }
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint_variables(path, variables_template):
    with open(path, "rb") as f:
        return serialization.from_bytes(variables_template, f.read())


def load_proposal_state(path, proposal_params, proposal_opt_state, proposal_optimizer):
    with open(path, "rb") as f:
        proposal_bytes = f.read()
    proposal_template = {
        "params": proposal_params,
        "opt_state": proposal_opt_state,
    }
    try:
        proposal_state = serialization.from_bytes(proposal_template, proposal_bytes)
        return proposal_state["params"], proposal_state["opt_state"]
    except ValueError:
        raw_state = serialization.msgpack_restore(proposal_bytes)
        proposal_params = serialization.from_state_dict(proposal_params, raw_state["params"])
        proposal_opt_state = proposal_optimizer.init(proposal_params)
        return proposal_params, proposal_opt_state


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
    if step < TRAIN_LR_BOUNDARY_1:
        return NIR_EFFICIENCY_THRESHOLD_STAGE_1
    if step < TRAIN_LR_BOUNDARY_2:
        return NIR_EFFICIENCY_THRESHOLD_STAGE_2
    return NIR_EFFICIENCY_THRESHOLD_STAGE_3


def current_proposal_steps(step):
    if step < TRAIN_LR_BOUNDARY_1:
        return NIR_PROPOSAL_STEPS_STAGE_1
    if step < TRAIN_LR_BOUNDARY_2:
        return NIR_PROPOSAL_STEPS_STAGE_2
    return NIR_PROPOSAL_STEPS_STAGE_3


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
        target_sr_mode = "complex" if current_learn_phase(step) else "real"
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
            mode=target_sr_mode,
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
    eff_threshold = current_efficiency_threshold(step)
    proposal_steps = current_proposal_steps(step)

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
                "efficiency_threshold": float(eff_threshold),
                "proposal_steps": int(proposal_steps),
                "meets_ess_threshold": bool(ess >= NIR_ESS_THRESHOLD_FRAC * vstate.n_samples),
                "meets_efficiency_threshold": bool(eff >= eff_threshold),
                "forward_kl_loss_after_steps": None if last_loss is None else float(last_loss),
            }
        )

        if eff >= eff_threshold:
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
    best_proposal_state_file = f"{out_prefix}_proposal_best.mpack"
    training_state_file = f"{out_prefix}_training_state.mpack"
    best_training_state_file = f"{out_prefix}_training_state_best.mpack"
    logger = nk.logging.JsonLog(
        out_prefix,
        write_every=WRITE_EVERY,
        save_params_every=SAVE_PARAMS_EVERY,
        save_params=True,
    )
    current_phase = current_learn_phase(start_iteration)
    current_sample_count = current_num_samples(start_iteration)
    if start_iteration != 0 or vstate.n_samples != current_sample_count:
        vstate = rebuild_vstate_for_stage(vstate, current_phase, current_sample_count)
        params = vstate.parameters

    for it in range(n_iter):
        global_it = start_iteration + it
        learn_phase = current_learn_phase(global_it)
        n_samples = current_num_samples(global_it)
        if learn_phase != current_phase or n_samples != current_sample_count:
            vstate = rebuild_vstate_for_stage(vstate, learn_phase, n_samples)
            params = vstate.parameters
            current_phase = learn_phase
            current_sample_count = n_samples
        nir_summary, proposal_params, proposal_opt_state, resampled, rng = run_adaptive_nir_round(
            vstate,
            proposal_model,
            proposal_params,
            proposal_opt_state,
            proposal_optimizer,
            rng,
            step=global_it,
        )
        last_round = nir_summary["rounds"][-1]
        target_update_applied = bool(last_round["meets_efficiency_threshold"])
        inject_external_samples(vstate, resampled)
        if target_update_applied:
            stats, dp, target_update_state = compute_target_direction(
                vstate, hamiltonian, target_update_state, global_it
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
                "iteration": global_it + 1,
                "energy": energy,
                "energy_distance_to_exact": energy_distance,
                "learn_phase": bool(current_phase),
                "target_update_applied": target_update_applied,
                "nir": nir_summary,
            }
        )
        if best_energy_distance is None or energy_distance < best_energy_distance:
            best_energy = energy
            best_energy_distance = energy_distance
            best_iteration = global_it + 1
            with open(best_params_file, "wb") as f:
                f.write(serialization.to_bytes(vstate.variables))
            save_proposal_state(best_proposal_state_file, proposal_params, proposal_opt_state)
            save_training_state(
                best_training_state_file,
                variables=vstate.variables,
                proposal_params=proposal_params,
                proposal_opt_state=proposal_opt_state,
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
                "NIR": {
                    "ESS": float(last_round["ess"]),
                    "Efficiency": float(last_round["efficiency"]),
                    "ProposalPool": int(last_round["proposal_pool"]),
                    "TargetUpdateApplied": int(target_update_applied),
                    "LearnPhase": int(current_phase),
                    "TargetSamples": int(vstate.n_samples),
                },
            },
            variational_state=vstate,
        )
        if ((it + 1) % SAVE_PARAMS_EVERY == 0) or (it + 1 == n_iter):
            save_training_state(
                training_state_file,
                variables=vstate.variables,
                proposal_params=proposal_params,
                proposal_opt_state=proposal_opt_state,
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
                f"Energy={energy:.8f} "
                f"ESS={last_round['ess']:.2f} "
                f"Eff={last_round['efficiency']:.4f} "
                f"Samples={vstate.n_samples:d} "
                f"LearnPhase={'yes' if current_phase else 'no'} "
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
        "best_proposal_state_file": best_proposal_state_file,
        "training_state_file": training_state_file,
        "best_training_state_file": best_training_state_file,
        "proposal_params": proposal_params,
        "proposal_opt_state": proposal_opt_state,
        "log_file": out_prefix + ".log",
        "completed_iterations": start_iteration + n_iter,
    }


run_iterations = RESUME_ADDITIONAL_ITERS if resume_summary_path is not None else NUM_ITERS_TOTAL
print(f"\n=== NIR training: {run_iterations} iterations ===\n")

model = build_model()
sampler = make_metropolis_local(hi, NUM_SAMPLES_STAGE_1)

vstate = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    n_samples=NUM_SAMPLES_STAGE_1,
    chunk_size=CHUNK_SIZE,
)

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
target_optimizer = optax.sgd(learning_rate=TRAIN_LR_SCHEDULE)

resume_training_state_path = None
resume_checkpoint_path = None
resume_proposal_state_path = None
start_iteration = 0
initial_target_opt_state = None
initial_target_update_state = None
initial_rng = None
initial_best_energy = None
initial_best_energy_distance = None
initial_best_iteration = None

if resume_summary_path is not None:
    print("Resume source summary:", resume_summary_path)
    print("Resume source run dir:", resume_source_run_dir)
    if RESUME_MODE == "best":
        resume_training_state_path = find_best_training_state(resume_summary, resume_source_run_dir)
    else:
        resume_training_state_path = find_latest_training_state(
            resume_summary, resume_source_run_dir
        )

    if resume_training_state_path is not None:
        resume_template = {
            "variables": vstate.variables,
            "proposal_params": proposal_params,
            "proposal_opt_state": proposal_opt_state,
            "target_opt_state": target_optimizer.init(vstate.parameters),
            "target_update_state": build_target_update_state(),
            "rng": fresh_key(),
            "completed_iterations": 0,
            "best_energy": None,
            "best_energy_distance_to_exact": None,
            "best_iteration": None,
        }
        resume_state = load_training_state(resume_training_state_path, resume_template)
        vstate.variables = resume_state["variables"]
        proposal_params = resume_state["proposal_params"]
        proposal_opt_state = resume_state["proposal_opt_state"]
        initial_target_opt_state = resume_state["target_opt_state"]
        initial_target_update_state = resume_state["target_update_state"]
        initial_rng = resume_state["rng"]
        start_iteration = int(resume_state["completed_iterations"])
        initial_best_energy = resume_state["best_energy"]
        initial_best_energy_distance = resume_state["best_energy_distance_to_exact"]
        initial_best_iteration = resume_state["best_iteration"]
        print("Loaded full training state:", resume_training_state_path)
        print("Resuming from completed iteration:", start_iteration)
    else:
        resume_checkpoint_path = find_latest_checkpoint(resume_summary, resume_source_run_dir)
        vstate.variables = load_checkpoint_variables(resume_checkpoint_path, vstate.variables)
        if RESUME_MODE == "best":
            proposal_state_path_str = resume_summary.get("best_proposal_state_file")
        else:
            proposal_state_path_str = resume_summary.get("proposal_state_file")
        if proposal_state_path_str:
            resume_proposal_state_path = Path(proposal_state_path_str).expanduser().resolve()
            if resume_proposal_state_path.is_file():
                proposal_params, proposal_opt_state = load_proposal_state(
                    resume_proposal_state_path,
                    proposal_params,
                    proposal_opt_state,
                    proposal_optimizer,
                )
        start_iteration = len(
            collect_previous_energy(resume_summary, summary_path=resume_summary_path)
        )
        initial_best_energy = resume_summary.get("best_energy_seen")
        initial_best_energy_distance = resume_summary.get("best_energy_distance_to_exact")
        initial_best_iteration = resume_summary.get("best_energy_iteration")
        print("Loaded checkpoint-only resume state:", resume_checkpoint_path)
        print("Recovered iteration count from prior logs:", start_iteration)

print("Parameters:", vstate.n_parameters)

out_prefix = run_path(f"out_{JOB_BASE}_nir")
result = run_nir_stage(
    out_prefix=out_prefix,
    vstate=vstate,
    hamiltonian=ha,
    n_iter=run_iterations,
    optimizer=target_optimizer,
    proposal_model=proposal_model,
    proposal_params=proposal_params,
    proposal_opt_state=proposal_opt_state,
    proposal_optimizer=proposal_optimizer,
    start_iteration=start_iteration,
    initial_target_opt_state=initial_target_opt_state,
    initial_target_update_state=initial_target_update_state,
    initial_rng=initial_rng,
    initial_best_energy=initial_best_energy,
    initial_best_energy_distance=initial_best_energy_distance,
    initial_best_iteration=initial_best_iteration,
)

proposal_state_file = Path(f"{out_prefix}_proposal_state.mpack")
save_proposal_state(proposal_state_file, result["proposal_params"], result["proposal_opt_state"])
training_state_file = Path(result["training_state_file"])

history_file = Path(f"{out_prefix}.json")
with open(history_file, "w") as f:
    json.dump(result["history"], f, indent=2)

final_energy = float(result["final_energy"])
print("Final NIR energy:", final_energy)
print("Final NIR gap to exact:", final_energy - EXACT_GROUND_STATE_ENERGY)
print("Best NIR energy:", float(result["best_energy"]))
print("Best NIR abs distance to exact:", float(result["best_energy_distance_to_exact"]))
print("Best NIR iteration:", int(result["best_iteration"]))
print("Best params file:", result["best_params_file"])
print("Proposal state file:", proposal_state_file)
print("Best proposal state file:", result["best_proposal_state_file"])
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

mean_energy_file = RUN_DIR / f"mean_energy_run_{JOB_BASE}.txt"
with open(mean_energy_file, "w") as f:
    for e in energy:
        f.write(f"{e}\n")

plt.figure(figsize=(10, 6))
if energy:
    plt.plot(energy)
plt.xscale("log")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title(f"{NUM_SITES}-site Gamma (NIR training, bond-token readout ViT)")
plt.tight_layout()
plot_file = RUN_DIR / f"gamma_vit_{JOB_BASE}.png"
plt.savefig(plot_file)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

summary = {
    "job_base": JOB_BASE,
    "run_dir": str(RUN_DIR),
    "num_sites": NUM_SITES,
    "learn_phase_stage_1": LEARN_PHASE_STAGE_1,
    "learn_phase_stage_2": LEARN_PHASE_STAGE_2,
    "learn_phase_stage_3": LEARN_PHASE_STAGE_3,
    "num_samples": NUM_SAMPLES_STAGE_1,
    "num_samples_stage_1": NUM_SAMPLES_STAGE_1,
    "num_samples_stage_2": NUM_SAMPLES_STAGE_2,
    "num_samples_stage_3": NUM_SAMPLES_STAGE_3,
    "num_iters_total": NUM_ITERS_TOTAL,
    "run_iterations": run_iterations,
    "completed_iterations": int(result["completed_iterations"]),
    "train_lr_stage_1": TRAIN_LR_STAGE_1,
    "train_lr_stage_2": TRAIN_LR_STAGE_2,
    "train_lr_stage_3": TRAIN_LR_STAGE_3,
    "train_lr_stage_1_iters": TRAIN_LR_STAGE_1_ITERS,
    "train_lr_stage_2_iters": TRAIN_LR_STAGE_2_ITERS,
    "proposal_lr_stage_1": NIR_PROPOSAL_LR_STAGE_1,
    "proposal_lr_stage_2": NIR_PROPOSAL_LR_STAGE_2,
    "proposal_lr_stage_3": NIR_PROPOSAL_LR_STAGE_3,
    "proposal_steps_stage_1": NIR_PROPOSAL_STEPS_STAGE_1,
    "proposal_steps_stage_2": NIR_PROPOSAL_STEPS_STAGE_2,
    "proposal_steps_stage_3": NIR_PROPOSAL_STEPS_STAGE_3,
    "target_preconditioner": TARGET_PRECONDITIONER,
    "target_sr_diag_shift_stage_1": TARGET_SR_DIAG_SHIFT_STAGE_1,
    "target_sr_diag_shift_stage_2": TARGET_SR_DIAG_SHIFT_STAGE_2,
    "target_sr_diag_shift_stage_3": TARGET_SR_DIAG_SHIFT_STAGE_3,
    "target_sr_proj_reg": TARGET_SR_PROJ_REG,
    "target_sr_momentum": TARGET_SR_MOMENTUM,
    "target_efficiency_gate_stage_1": NIR_EFFICIENCY_THRESHOLD_STAGE_1,
    "target_efficiency_gate_stage_2": NIR_EFFICIENCY_THRESHOLD_STAGE_2,
    "target_efficiency_gate_stage_3": NIR_EFFICIENCY_THRESHOLD_STAGE_3,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "mlp_hidden_dim": MLP_HIDDEN_DIM,
    "patch_size": PATCH_SIZE,
    "permutation": list(perm),
    "relation_matrix": [list(row) for row in relation_matrix],
    "edge_indices": [list(edge) for edge in edge_indices],
    "edge_relations": list(edge_relations),
    "num_relation_types": NUM_RELATION_TYPES,
    "bond_token_readout": True,
    "outputs": [str(result["log_file"])],
    "history_file": str(history_file),
    "mean_energy_file": str(mean_energy_file),
    "plot_file": str(plot_file),
    "best_params_file": str(result["best_params_file"]),
    "training_state_file": str(training_state_file),
    "best_training_state_file": str(result["best_training_state_file"]),
    "proposal_state_file": str(proposal_state_file),
    "best_proposal_state_file": str(result["best_proposal_state_file"]),
    "exact_ground_state_energy": EXACT_GROUND_STATE_ENERGY,
}

if resume_summary_path is not None:
    summary["source_summary_file"] = str(resume_summary_path)
    summary["resume_mode"] = RESUME_MODE
    summary["resume_start_iteration"] = int(start_iteration)
    if resume_training_state_path is not None:
        summary["resume_source_training_state_file"] = str(resume_training_state_path)
    if resume_checkpoint_path is not None:
        summary["resume_source_checkpoint_file"] = str(resume_checkpoint_path)
    if resume_proposal_state_path is not None:
        summary["resume_source_proposal_state_file"] = str(resume_proposal_state_path)

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

summary_file = RUN_DIR / f"summary_{JOB_BASE}.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

if energy:
    print("Final gap to exact:", float(energy[-1]) - EXACT_GROUND_STATE_ENERGY)
