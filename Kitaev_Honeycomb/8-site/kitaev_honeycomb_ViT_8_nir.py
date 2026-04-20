import os
import json
import sys
from datetime import date
from pathlib import Path

SHOW_PLOTS = False
if not SHOW_PLOTS:
    os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import netket as nk
import optax

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

from define_Kitaev_Hamiltonian import kitaev_hamiltonian
from kitaev_honeycomb_vit_model import HoneycombPatchViT

print(jax.devices())
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

THIS_DIR = Path(__file__).resolve().parent
NIR_DIR = THIS_DIR / "nir_experiments"
if str(NIR_DIR) not in sys.path:
    sys.path.insert(0, str(NIR_DIR))

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

NUM_SITES = 8
NUM_SAMPLES = 3 * 2**8
NUM_ITERS_TOTAL = 1000
N_DISCARD_PER_CHAIN = 8
TARGET_CHAIN_LENGTH = 16

LEARN_PHASE = True

EMBED_DIM = 8
NUM_HEADS = 4
NUM_LAYERS = 2
PATCH_SIZE = 2
MLP_HIDDEN_DIM = 2 * EMBED_DIM
CHUNK_SIZE = None

TRAIN_LR = 1e-2
LOG_STEP_SIZE = 1

# ---------- NIR proposal network ----------
NIR_PROPOSAL_BATCH = 256
NIR_MAX_PROPOSAL_BATCHES = 8
NIR_MAX_ADAPTIVE_ROUNDS = 3
NIR_ESS_THRESHOLD_FRAC = 0.50
NIR_EFFICIENCY_THRESHOLD = 0.10
NIR_PROPOSAL_LR = 1e-3
NIR_PROPOSAL_STEPS = 5
NIR_PROPOSAL_EMBED_DIM = 16
NIR_PROPOSAL_HEADS = 2
NIR_PROPOSAL_LAYERS = 2
NIR_PROPOSAL_MLP = 32
NIR_PROB_FLOOR = 1e-6

TODAY = date.today().isoformat()
JOB_BASE = (
    f"{NUM_SITES}-site_"
    f"{NUM_LAYERS}_layers_"
    f"{NUM_HEADS}_heads_"
    f"{PATCH_SIZE}_patches_"
    f"{NUM_SAMPLES}_samples_"
    f"{TODAY}_Kitaev_ViT_NIR_single_stage"
)

RUNS_DIR = NIR_DIR / "runs" / TODAY
RUN_DIR = RUNS_DIR / JOB_BASE
RUN_DIR.mkdir(parents=True, exist_ok=True)


def make_metropolis_local(hilbert, n_samples):
    n_chains = max(1, n_samples // TARGET_CHAIN_LENGTH)
    return nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=n_chains)


def fresh_key():
    seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    return jax.random.PRNGKey(seed)


def build_model(permutation):
    return HoneycombPatchViT(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        patch_size=PATCH_SIZE,
        learn_phase=LEARN_PHASE,
        permutation=permutation,
    )


def target_log_probs(vstate, sigma):
    log_psi = vstate.log_value(sigma)
    return 2.0 * jnp.real(log_psi)


def inject_external_samples(vstate, samples):
    samples = jnp.asarray(samples, dtype=jnp.float64)
    reshaped = samples.reshape(vstate.sampler.n_chains, vstate.chain_length, NUM_SITES)
    vstate._samples = reshaped
    return reshaped


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
        for _ in range(NIR_PROPOSAL_STEPS):
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
                "meets_efficiency_threshold": bool(eff >= NIR_EFFICIENCY_THRESHOLD),
                "forward_kl_loss_after_steps": None if last_loss is None else float(last_loss),
            }
        )

        if eff >= NIR_EFFICIENCY_THRESHOLD:
            break

    return {
        "rounds": round_summaries,
        "final_resampled_shape": None if final_resampled is None else tuple(final_resampled.shape),
    }, params, opt_state, final_resampled, rng


def run_nir_vmc_stage(
    *,
    stage_name,
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

    for it in range(n_iter):
        nir_summary, proposal_params, proposal_opt_state, resampled, rng = run_adaptive_nir_round(
            vstate,
            proposal_model,
            proposal_params,
            proposal_opt_state,
            proposal_optimizer,
            rng,
        )
        inject_external_samples(vstate, resampled)
        stats, grad = vstate.expect_and_grad(hamiltonian)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        vstate.parameters = params

        energy = float(np.real(np.asarray(stats.mean)))
        history.append(
            {
                "iteration": it + 1,
                "energy": energy,
                "nir": nir_summary,
            }
        )
        if (it + 1) % max(1, LOG_STEP_SIZE) == 0:
            last_round = nir_summary["rounds"][-1]
            print(
                f"[{stage_name}] iter={it + 1} "
                f"energy={energy:.8f} "
                f"ess={last_round['ess']:.2f} "
                f"eff={last_round['efficiency']:.4f}"
            )

    return {
        "history": history,
        "final_energy": history[-1]["energy"] if history else None,
        "proposal_params": proposal_params,
        "proposal_opt_state": proposal_opt_state,
    }


def main():
    graph, _symm_group, hi, ha = kitaev_hamiltonian(NUM_SITES)
    sp_h = ha.to_sparse()
    eig_vals, _ = eigsh(sp_h, k=2, which="SA")
    perm = tuple(range(graph.n_nodes))

    print("Permutation:", perm)
    print("Run directory:", RUN_DIR)
    print("Exact ground-state energy:", eig_vals[0])
    print()

    model = build_model(perm)
    sampler = make_metropolis_local(hi, NUM_SAMPLES)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=NUM_SAMPLES,
        chunk_size=CHUNK_SIZE,
        n_discard_per_chain=N_DISCARD_PER_CHAIN,
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
    proposal_optimizer = optax.adam(NIR_PROPOSAL_LR)
    proposal_opt_state = proposal_optimizer.init(proposal_params)

    print("Parameters:", vstate.n_parameters)
    print("n_chains:", vstate.sampler.n_chains)
    print("chain_length:", vstate.chain_length)

    result = run_nir_vmc_stage(
        stage_name="kitaev_8_nir",
        vstate=vstate,
        hamiltonian=ha,
        n_iter=NUM_ITERS_TOTAL,
        optimizer=optax.sgd(learning_rate=TRAIN_LR),
        proposal_model=proposal_model,
        proposal_params=proposal_params,
        proposal_opt_state=proposal_opt_state,
        proposal_optimizer=proposal_optimizer,
    )

    log_file = RUN_DIR / f"nir_{JOB_BASE}.json"
    with open(log_file, "w") as f:
        json.dump(result["history"], f, indent=2)

    print("Saved NIR log:", log_file)
    print("Final NIR energy:", result["final_energy"])


if __name__ == "__main__":
    main()
