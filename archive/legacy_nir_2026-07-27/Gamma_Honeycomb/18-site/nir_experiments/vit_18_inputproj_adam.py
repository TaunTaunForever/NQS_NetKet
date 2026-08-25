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
from vit_symm_model import CanonicalRepresentativeHoneycombViT

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

NUM_SAMPLES_STAGE_1 = 3 * 2**9
NUM_SAMPLES_STAGE_2 = NUM_SAMPLES_STAGE_1
NUM_SAMPLES_STAGE_3 = 3 * 2**10
NUM_ITERS_TOTAL = 2000

EMBED_DIM = 24
NUM_HEADS = 8
NUM_LAYERS = 2
PATCH_SIZE = 2
MLP_HIDDEN_DIM = 2 * EMBED_DIM
CHUNK_SIZE = 512

TRAIN_LR_STAGE_1 = 1e-2
TRAIN_LR_STAGE_2 = 1e-2
TRAIN_LR_STAGE_3 = 5e-3
TRAIN_LR_STAGE_1_ITERS = 50
TRAIN_LR_STAGE_2_ITERS = 600
ADAM_CLIP_NORM = 1.0

LOG_STEP_SIZE = 1
WRITE_EVERY = 1
SAVE_PARAMS_EVERY = 25

# ---------- NIR proposal network ----------
NIR_PROPOSAL_BATCH = 3 * 2**10
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

TARGET_CHAIN_LENGTH = 128

TODAY = date.today().isoformat()

JOB_BASE = (
    f"{NUM_SITES}-site_"
    f"{NUM_LAYERS}_layers_"
    f"{NUM_HEADS}_heads_"
    f"{PATCH_SIZE}_patches_"
    f"identity_"
    f"{NUM_SAMPLES_STAGE_1}to{NUM_SAMPLES_STAGE_3}_samples_"
    f"{TODAY}_Gamma_ViT_NIR_inputproj_Adam"
)

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

print("Symmetry count:", len(symmetry_perms))
print("Run directory:", RUN_DIR)


def make_metropolis_local(hilbert, n_samples):
    n_chains = max(1, n_samples // TARGET_CHAIN_LENGTH)
    return nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=n_chains)


def fresh_key():
    seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    return jax.random.PRNGKey(seed)


def build_model():
    return CanonicalRepresentativeHoneycombViT(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        patch_size=PATCH_SIZE,
        learn_phase=LEARN_PHASE_STAGE_1,
        symmetries=symmetry_perms,
        permutation=tuple(range(graph.n_nodes)),
    )


def build_model_for_phase(learn_phase):
    return CanonicalRepresentativeHoneycombViT(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        patch_size=PATCH_SIZE,
        learn_phase=learn_phase,
        symmetries=symmetry_perms,
        permutation=tuple(range(graph.n_nodes)),
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


def target_log_probs(vstate, sigma):
    log_psi = vstate.log_value(sigma)
    return 2.0 * jnp.real(log_psi)


def inject_external_samples(vstate, samples):
    samples = jnp.asarray(samples, dtype=jnp.float64)
    reshaped = samples.reshape(vstate.sampler.n_chains, vstate.chain_length, NUM_SITES)
    vstate._samples = reshaped
    return reshaped


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


def compute_target_grad(vstate, hamiltonian):
    stats, grad = vstate.expect_and_grad(hamiltonian)
    return stats, grad


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


def run_nir_adam_stage(
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
    rng = fresh_key()
    history = []
    best_energy = None
    best_energy_distance = None
    best_iteration = None
    best_params_file = f"{out_prefix}_best.mpack"
    best_proposal_state_file = f"{out_prefix}_proposal_best.mpack"
    logger = nk.logging.JsonLog(
        out_prefix,
        write_every=WRITE_EVERY,
        save_params_every=SAVE_PARAMS_EVERY,
        save_params=True,
    )
    current_phase = current_learn_phase(0)
    current_sample_count = current_num_samples(0)
    if vstate.n_samples != current_sample_count:
        vstate = rebuild_vstate_for_stage(vstate, current_phase, current_sample_count)
        params = vstate.parameters

    for it in range(n_iter):
        learn_phase = current_learn_phase(it)
        n_samples = current_num_samples(it)
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
            step=it,
        )
        last_round = nir_summary["rounds"][-1]
        target_update_applied = bool(last_round["meets_efficiency_threshold"])
        inject_external_samples(vstate, resampled)

        if target_update_applied:
            stats, grad = compute_target_grad(vstate, hamiltonian)
            grad = jax.tree.map(
                lambda update, param: jnp.asarray(update, dtype=param.dtype), grad, params
            )
            updates, opt_state = optimizer.update(grad, opt_state, params)
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
                "learn_phase": bool(current_phase),
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
            save_proposal_state(best_proposal_state_file, proposal_params, proposal_opt_state)

        logger(
            it,
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
                "Adam": {
                    "TargetUpdateApplied": int(target_update_applied),
                    "LearnPhase": int(current_phase),
                    "TargetSamples": int(vstate.n_samples),
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
        "proposal_params": proposal_params,
        "proposal_opt_state": proposal_opt_state,
        "log_file": out_prefix + ".log",
    }


print(f"\n=== NIR + Adam training: {NUM_ITERS_TOTAL} iterations ===\n")

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
target_optimizer = optax.chain(
    optax.clip_by_global_norm(ADAM_CLIP_NORM),
    optax.adam(learning_rate=TRAIN_LR_SCHEDULE),
)

print("Parameters:", vstate.n_parameters)

out_prefix = run_path(f"out_{JOB_BASE}")
result = run_nir_adam_stage(
    out_prefix=out_prefix,
    vstate=vstate,
    hamiltonian=ha,
    n_iter=NUM_ITERS_TOTAL,
    optimizer=target_optimizer,
    proposal_model=proposal_model,
    proposal_params=proposal_params,
    proposal_opt_state=proposal_opt_state,
    proposal_optimizer=proposal_optimizer,
)

proposal_state_file = Path(f"{out_prefix}_proposal_state.mpack")
save_proposal_state(proposal_state_file, result["proposal_params"], result["proposal_opt_state"])

history_file = Path(f"{out_prefix}.json")
with open(history_file, "w") as f:
    json.dump(result["history"], f, indent=2)

final_energy = float(result["final_energy"])
print("Final NIR+Adam energy:", final_energy)
print("Final NIR+Adam gap to exact:", final_energy - EXACT_GROUND_STATE_ENERGY)
print("Best NIR+Adam energy:", float(result["best_energy"]))
print("Best NIR+Adam abs distance to exact:", float(result["best_energy_distance_to_exact"]))
print("Best NIR+Adam iteration:", int(result["best_iteration"]))
print("Best params file:", result["best_params_file"])
print("Proposal state file:", proposal_state_file)
print("Best proposal state file:", result["best_proposal_state_file"])
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
plt.title(f"{NUM_SITES}-site Gamma (NIR training, input-projected, Adam)")
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
    "num_samples_stage_1": NUM_SAMPLES_STAGE_1,
    "num_samples_stage_2": NUM_SAMPLES_STAGE_2,
    "num_samples_stage_3": NUM_SAMPLES_STAGE_3,
    "num_iters_total": NUM_ITERS_TOTAL,
    "optimizer": "adam",
    "train_lr_stage_1": TRAIN_LR_STAGE_1,
    "train_lr_stage_2": TRAIN_LR_STAGE_2,
    "train_lr_stage_3": TRAIN_LR_STAGE_3,
    "train_lr_stage_1_iters": TRAIN_LR_STAGE_1_ITERS,
    "train_lr_stage_2_iters": TRAIN_LR_STAGE_2_ITERS,
    "adam_clip_norm": ADAM_CLIP_NORM,
    "proposal_lr_stage_1": NIR_PROPOSAL_LR_STAGE_1,
    "proposal_lr_stage_2": NIR_PROPOSAL_LR_STAGE_2,
    "proposal_lr_stage_3": NIR_PROPOSAL_LR_STAGE_3,
    "proposal_steps_stage_1": NIR_PROPOSAL_STEPS_STAGE_1,
    "proposal_steps_stage_2": NIR_PROPOSAL_STEPS_STAGE_2,
    "proposal_steps_stage_3": NIR_PROPOSAL_STEPS_STAGE_3,
    "target_efficiency_gate_stage_1": NIR_EFFICIENCY_THRESHOLD_STAGE_1,
    "target_efficiency_gate_stage_2": NIR_EFFICIENCY_THRESHOLD_STAGE_2,
    "target_efficiency_gate_stage_3": NIR_EFFICIENCY_THRESHOLD_STAGE_3,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "mlp_hidden_dim": MLP_HIDDEN_DIM,
    "patch_size": PATCH_SIZE,
    "permutation": list(range(graph.n_nodes)),
    "outputs": [str(result["log_file"])],
    "history_file": str(history_file),
    "mean_energy_file": str(mean_energy_file),
    "plot_file": str(plot_file),
    "best_params_file": str(result["best_params_file"]),
    "proposal_state_file": str(proposal_state_file),
    "best_proposal_state_file": str(result["best_proposal_state_file"]),
    "exact_ground_state_energy": EXACT_GROUND_STATE_ENERGY,
}

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
    print("Best NIR+Adam gap to exact:", float(result["best_energy"]) - EXACT_GROUND_STATE_ENERGY)
