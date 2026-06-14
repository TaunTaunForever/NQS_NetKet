from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

from flax import serialization

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("NETKET_DEBUG", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
import optax
from netket._src.ngd.sr_srt_common import srt as compute_minsr_direction
from scipy.sparse.linalg import eigsh

from common import make_heisenberg_hamiltonian
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
from vit_model import PatchViT
from vit_site_type_relation_gated_pool_model import (
    HoneycombSiteTypeRelationGatedPoolViT,
    build_honeycomb_bond_oriented_relation_matrix,
)
from vit_site_type_relation_model import (
    HoneycombSiteTypeRelationViT,
    build_bipartite_site_type_ids,
    build_honeycomb_relation_matrix,
    site_relation_to_patch_relation_expanded,
    site_type_ids_to_patch_type_ids,
)


jax.config.update("jax_enable_x64", True)


def _count_params(tree) -> int:
    leaves = jax.tree.leaves(tree)
    return int(sum(np.asarray(leaf).size for leaf in leaves))


def run_nir_experiment(
    *,
    num_sites: int,
    j1: float,
    j2: float,
    num_samples_stage_1: int = 3*2**8,
    num_samples_stage_2: int = 3*2**9,
    num_samples_stage_3: int = 3*2**10,
    num_samples_stage_4: int | None = None,
    num_iters_total: int = 1000,
    patch_size: int = 1,
    embed_dim: int = 18,
    num_heads: int = 2,
    num_layers: int = 4,
    mlp_hidden: int | None = None,
    chunk_size: int | None = 2**9,
    train_lr_stage_1: float = 1e-2,
    train_lr_stage_2: float = 1e-2,
    train_lr_stage_3: float = 1e-3,
    train_lr_stage_4: float | None = None,
    train_lr_stage_1_iters: int | None = None,
    train_lr_stage_2_iters: int | None = None,
    train_lr_stage_3_iters: int | None = None,
    learn_phase_stage_1: bool = False,
    learn_phase_stage_2: bool = True,
    learn_phase_stage_3: bool = True,
    learn_phase_stage_4: bool | None = None,
    target_sampler_name: str = "local",
    target_optimizer_name: str = "adam",
    target_sgd_momentum: float = 0.0,
    target_preconditioner: str = "none",
    target_sr_diag_shift: float = 1e-2,
    target_sr_diag_shift_stage_1: float | None = None,
    target_sr_diag_shift_stage_2: float | None = None,
    target_sr_diag_shift_stage_3: float | None = None,
    target_sr_diag_shift_stage_4: float | None = None,
    target_sr_proj_reg: float | None = None,
    target_sr_momentum: float | None = None,
    target_sr_mode: str = "complex",
    nir_proposal_batch: int = 2**9,
    nir_max_proposal_batches: int = 12,
    nir_max_adaptive_rounds: int = 6,
    nir_min_adaptive_rounds: int = 1,
    nir_force_adapt_until_iter: int = 0,
    nir_ess_threshold_frac: float = 0.4,
    nir_efficiency_threshold_stage_1: float = 0.10,
    nir_efficiency_threshold_stage_2: float = 0.10,
    nir_efficiency_threshold_stage_3: float = 0.10,
    nir_efficiency_threshold_stage_4: float | None = None,
    nir_proposal_lr_stage_1: float = 1e-3,
    nir_proposal_lr_stage_2: float = 1e-3,
    nir_proposal_lr_stage_3: float = 1e-3,
    nir_proposal_lr_stage_4: float | None = None,
    nir_proposal_steps_stage_1: int = 1,
    nir_proposal_steps_stage_2: int = 1,
    nir_proposal_steps_stage_3: int = 1,
    nir_proposal_steps_stage_4: int | None = None,
    nir_proposal_embed_dim: int = 32,
    nir_proposal_heads: int = 4,
    nir_proposal_layers: int = 4,
    nir_proposal_mlp: int | None = None,
    nir_prob_floor: float = 1e-6,
    model_type: str = "site_type_relation",
    run_tag: str | None = None,
):
    today = date.today().isoformat()
    mlp_hidden = 2 * embed_dim if mlp_hidden is None else mlp_hidden
    train_lr_stage_1_iters = max(1, num_iters_total // 20) if train_lr_stage_1_iters is None else train_lr_stage_1_iters
    train_lr_stage_2_iters = max(1, num_iters_total // 2) if train_lr_stage_2_iters is None else train_lr_stage_2_iters
    use_stage_4 = num_samples_stage_4 is not None or train_lr_stage_4 is not None
    if use_stage_4 and train_lr_stage_3_iters is None:
        remaining = num_iters_total - train_lr_stage_1_iters - train_lr_stage_2_iters
        train_lr_stage_3_iters = max(1, remaining // 2)
    nir_proposal_mlp = 2 * nir_proposal_embed_dim if nir_proposal_mlp is None else nir_proposal_mlp
    num_samples_stage_4 = num_samples_stage_3 if num_samples_stage_4 is None else num_samples_stage_4
    train_lr_stage_4 = train_lr_stage_3 if train_lr_stage_4 is None else train_lr_stage_4
    learn_phase_stage_4 = learn_phase_stage_3 if learn_phase_stage_4 is None else learn_phase_stage_4
    target_sr_diag_shift_stage_1 = (
        target_sr_diag_shift if target_sr_diag_shift_stage_1 is None else target_sr_diag_shift_stage_1
    )
    target_sr_diag_shift_stage_2 = (
        target_sr_diag_shift if target_sr_diag_shift_stage_2 is None else target_sr_diag_shift_stage_2
    )
    target_sr_diag_shift_stage_3 = (
        target_sr_diag_shift if target_sr_diag_shift_stage_3 is None else target_sr_diag_shift_stage_3
    )
    target_sr_diag_shift_stage_4 = (
        target_sr_diag_shift_stage_3 if target_sr_diag_shift_stage_4 is None else target_sr_diag_shift_stage_4
    )
    nir_efficiency_threshold_stage_4 = (
        nir_efficiency_threshold_stage_3 if nir_efficiency_threshold_stage_4 is None else nir_efficiency_threshold_stage_4
    )
    nir_proposal_lr_stage_4 = nir_proposal_lr_stage_3 if nir_proposal_lr_stage_4 is None else nir_proposal_lr_stage_4
    nir_proposal_steps_stage_4 = (
        nir_proposal_steps_stage_3 if nir_proposal_steps_stage_4 is None else nir_proposal_steps_stage_4
    )

    train_lr_boundary_1 = train_lr_stage_1_iters
    train_lr_boundary_2 = train_lr_stage_1_iters + train_lr_stage_2_iters
    train_lr_boundary_3 = (
        train_lr_stage_1_iters + train_lr_stage_2_iters + train_lr_stage_3_iters
        if use_stage_4
        else None
    )

    train_lr_schedules = [
        optax.constant_schedule(train_lr_stage_1),
        optax.constant_schedule(train_lr_stage_2),
        optax.constant_schedule(train_lr_stage_3),
    ]
    proposal_lr_schedules = [
        optax.constant_schedule(nir_proposal_lr_stage_1),
        optax.constant_schedule(nir_proposal_lr_stage_2),
        optax.constant_schedule(nir_proposal_lr_stage_3),
    ]
    target_sr_schedules = [
        optax.constant_schedule(target_sr_diag_shift_stage_1),
        optax.constant_schedule(target_sr_diag_shift_stage_2),
        optax.constant_schedule(target_sr_diag_shift_stage_3),
    ]
    boundaries = [train_lr_boundary_1, train_lr_boundary_2]
    if use_stage_4:
        train_lr_schedules.append(optax.constant_schedule(train_lr_stage_4))
        proposal_lr_schedules.append(optax.constant_schedule(nir_proposal_lr_stage_4))
        target_sr_schedules.append(optax.constant_schedule(target_sr_diag_shift_stage_4))
        boundaries.append(train_lr_boundary_3)

    train_lr_schedule = optax.join_schedules(
        schedules=train_lr_schedules,
        boundaries=boundaries,
    )
    proposal_lr_schedule = optax.join_schedules(
        schedules=proposal_lr_schedules,
        boundaries=boundaries,
    )
    target_sr_diag_shift_schedule = optax.join_schedules(
        schedules=target_sr_schedules,
        boundaries=boundaries,
    )

    job_base = (
        f"J1={j1}_J2={j2}_{num_sites}-site_"
        f"{num_layers}L_{num_heads}H_{patch_size}p_"
        f"{num_samples_stage_1}to{num_samples_stage_4 if use_stage_4 else num_samples_stage_3}_samples_{today}_"
        f"J1_J2_Honeycomb_ViT_NIR_site_type_relation"
    )
    if run_tag:
        job_base = f"{job_base}_{run_tag}"
    run_dir = Path("runs") / today / job_base
    run_dir.mkdir(parents=True, exist_ok=True)

    graph, extent, hi, ha = make_heisenberg_hamiltonian(num_sites, j1=j1, j2=j2)
    perm = tuple(range(graph.n_nodes))
    site_type_ids = build_bipartite_site_type_ids(graph, permutation=perm)
    token_site_type_ids = (
        site_type_ids
        if patch_size == 1
        else site_type_ids_to_patch_type_ids(site_type_ids, patch_size)
    )
    site_relation_matrix = build_honeycomb_relation_matrix(graph, permutation=perm)
    relation_matrix = (
        site_relation_matrix
        if patch_size == 1
        else site_relation_to_patch_relation_expanded(site_relation_matrix, patch_size)
    )
    bond_oriented_site_relation_matrix = build_honeycomb_bond_oriented_relation_matrix(
        graph, permutation=perm
    )
    bond_oriented_relation_matrix = (
        bond_oriented_site_relation_matrix
        if patch_size == 1
        else site_relation_to_patch_relation_expanded(
            bond_oriented_site_relation_matrix, patch_size
        )
    )
    num_relation_types = max(max(row) for row in relation_matrix) + 1
    num_bond_oriented_relation_types = (
        max(max(row) for row in bond_oriented_relation_matrix) + 1
    )
    num_site_types = max(token_site_type_ids) + 1
    target_head_dim = embed_dim // num_heads
    proposal_head_dim = nir_proposal_embed_dim // nir_proposal_heads
    exact_gs = None
    if num_sites <= 24:
        sp_h = ha.to_sparse()
        eig_vals, _ = eigsh(sp_h, k=2, which="SA")
        exact_gs = float(eig_vals[0])
        print("Exact ground-state energy:", exact_gs)
        print(
            "Target architecture:",
            f"embed_dim={embed_dim}",
            f"head_dim={target_head_dim}",
            f"num_heads={num_heads}",
            f"num_layers={num_layers}",
        )
        print(
            "Proposal architecture:",
            f"embed_dim={nir_proposal_embed_dim}",
            f"head_dim={proposal_head_dim}",
            f"num_heads={nir_proposal_heads}",
            f"num_layers={nir_proposal_layers}",
        )

    print("Run directory:", run_dir)
    print("Model type:", model_type)
    print("Site type ids:", token_site_type_ids)
    print("Number of site types:", num_site_types)
    print("Relation matrix shape:", (len(relation_matrix), len(relation_matrix[0])))
    print("Number of relation types:", num_relation_types)
    print("Bond-oriented relation types:", num_bond_oriented_relation_types)

    def fresh_key():
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        return jax.random.PRNGKey(seed)

    def current_learn_phase(step):
        if step < train_lr_boundary_1:
            return learn_phase_stage_1
        if step < train_lr_boundary_2:
            return learn_phase_stage_2
        if use_stage_4 and step < train_lr_boundary_3:
            return learn_phase_stage_3
        if use_stage_4:
            return learn_phase_stage_4
        return learn_phase_stage_3

    def current_num_samples(step):
        if step < train_lr_boundary_1:
            return num_samples_stage_1
        if step < train_lr_boundary_2:
            return num_samples_stage_2
        if use_stage_4 and step < train_lr_boundary_3:
            return num_samples_stage_3
        if use_stage_4:
            return num_samples_stage_4
        return num_samples_stage_3

    def current_efficiency_threshold(step):
        if step < train_lr_boundary_1:
            return nir_efficiency_threshold_stage_1
        if step < train_lr_boundary_2:
            return nir_efficiency_threshold_stage_2
        if use_stage_4 and step < train_lr_boundary_3:
            return nir_efficiency_threshold_stage_3
        if use_stage_4:
            return nir_efficiency_threshold_stage_4
        return nir_efficiency_threshold_stage_3

    def current_proposal_steps(step):
        if step < train_lr_boundary_1:
            return nir_proposal_steps_stage_1
        if step < train_lr_boundary_2:
            return nir_proposal_steps_stage_2
        if use_stage_4 and step < train_lr_boundary_3:
            return nir_proposal_steps_stage_3
        if use_stage_4:
            return nir_proposal_steps_stage_4
        return nir_proposal_steps_stage_3

    def current_target_sr_diag_shift(step):
        return float(target_sr_diag_shift_schedule(step))

    def build_model(learn_phase):
        if model_type == "plain":
            return PatchViT(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_hidden=mlp_hidden,
                patch_size=patch_size,
                learn_phase=learn_phase,
            )
        if model_type == "site_type_relation_gated_pool_bond":
            return HoneycombSiteTypeRelationGatedPoolViT(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_hidden_dim=mlp_hidden,
                patch_size=patch_size,
                learn_phase=learn_phase,
                relation_matrix=bond_oriented_relation_matrix,
                site_type_ids=token_site_type_ids,
                permutation=perm,
            )
        if model_type != "site_type_relation":
            raise ValueError(f"Unsupported model_type={model_type!r}")
        return HoneycombSiteTypeRelationViT(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_hidden_dim=mlp_hidden,
            patch_size=patch_size,
            learn_phase=learn_phase,
            relation_matrix=relation_matrix,
            site_type_ids=token_site_type_ids,
            permutation=perm,
        )

    def make_sampler(n_samples):
        if target_sampler_name == "exact":
            return nk.sampler.ExactSampler(hi)
        if target_sampler_name == "local":
            n_chains = max(1, n_samples // 128)
            return nk.sampler.MetropolisLocal(hilbert=hi, n_chains=n_chains)
        raise ValueError(f"Unsupported target_sampler_name={target_sampler_name!r}")

    def make_target_optimizer():
        if target_optimizer_name == "adam":
            return optax.adam(learning_rate=train_lr_schedule)
        if target_optimizer_name == "sgd":
            return optax.sgd(learning_rate=train_lr_schedule, momentum=target_sgd_momentum)
        raise ValueError(f"Unsupported target_optimizer_name={target_optimizer_name!r}")

    def make_target_preconditioner():
        if target_preconditioner == "none":
            return {"kind": "none", "preconditioner": None, "old_updates": None, "info": None}
        if target_preconditioner == "sr":
            return {
                "kind": "sr",
                "preconditioner": None,
                "old_updates": None,
                "info": None,
            }
        if target_preconditioner == "minsr":
            return {"kind": "minsr", "preconditioner": None, "old_updates": None, "info": None}
        raise ValueError(f"Unsupported target_preconditioner={target_preconditioner!r}")

    def compute_target_direction(vstate, hamiltonian, target_update_state, step, learn_phase):
        kind = target_update_state["kind"]
        if kind == "none":
            stats, grad = vstate.expect_and_grad(hamiltonian)
            return stats, grad, target_update_state

        if kind == "sr":
            stats, grad = vstate.expect_and_grad(hamiltonian)
            preconditioner = nk.optimizer.SR(diag_shift=current_target_sr_diag_shift(step))
            dp = preconditioner(vstate, grad)
            info = getattr(preconditioner, "info", None)
            target_update_state["info"] = info
            return stats, dp, target_update_state

        if kind == "minsr":
            local_energies = vstate.local_estimators(hamiltonian, chunk_size=chunk_size)
            stats = nk.stats.statistics(local_energies)
            samples = jax.lax.collapse(vstate.samples, 0, vstate.samples.ndim - 1)
            effective_mode = target_sr_mode if learn_phase else "real"
            dp, old_updates, info = compute_minsr_direction(
                vstate._apply_fun,
                local_energies,
                vstate.parameters,
                vstate.model_state,
                samples,
                diag_shift=current_target_sr_diag_shift(step),
                solver_fn=nk.optimizer.solver.cholesky,
                mode=effective_mode,
                proj_reg=target_sr_proj_reg,
                momentum=target_sr_momentum,
                old_updates=target_update_state["old_updates"],
                chunk_size=chunk_size,
            )
            target_update_state["old_updates"] = old_updates
            target_update_state["info"] = info
            return stats, dp, target_update_state

        raise ValueError(f"Unsupported target_preconditioner state: {kind}")

    def rebuild_vstate(vstate, learn_phase, n_samples):
        sampler = make_sampler(n_samples)
        rebuilt = nk.vqs.MCState(
            sampler=sampler,
            model=build_model(learn_phase),
            n_samples=n_samples,
            variables=vstate.variables,
            chunk_size=chunk_size,
        )
        return rebuilt

    def target_log_probs(vstate, sigma):
        log_psi = vstate.log_value(sigma)
        return 2.0 * jnp.real(log_psi)

    def inject_external_samples(vstate, samples):
        samples = jnp.asarray(samples, dtype=jnp.float64)
        reshaped = samples.reshape(vstate.sampler.n_chains, vstate.chain_length, num_sites)
        vstate._samples = reshaped
        return reshaped

    def sample_until_ess(vstate, proposal_model, proposal_params, rng, *, target_n_samples):
        proposal_batches = []
        log_target_batches = []
        log_proposal_batches = []
        ess_threshold = nir_ess_threshold_frac * target_n_samples

        for _ in range(nir_max_proposal_batches):
            proposal_samples, rng = sample_from_proposal(
                proposal_model,
                proposal_params,
                rng,
                nir_proposal_batch,
                num_sites,
                prob_floor=nir_prob_floor,
            )
            log_target = target_log_probs(vstate, proposal_samples)
            log_proposal = proposal_log_prob(
                proposal_model,
                proposal_params,
                proposal_samples,
                prob_floor=nir_prob_floor,
            )

            proposal_batches.append(np.asarray(proposal_samples))
            log_target_batches.append(np.asarray(log_target))
            log_proposal_batches.append(np.asarray(log_proposal))

            stacked_target = np.concatenate(log_target_batches, axis=0)
            stacked_proposal = np.concatenate(log_proposal_batches, axis=0)
            weights = normalised_importance_weights_from_log_probs(stacked_target, stacked_proposal)
            ess = effective_sample_size(weights)
            if ess >= ess_threshold:
                break

        all_samples = np.concatenate(proposal_batches, axis=0)
        all_log_target = np.concatenate(log_target_batches, axis=0)
        all_log_proposal = np.concatenate(log_proposal_batches, axis=0)
        weights = normalised_importance_weights_from_log_probs(all_log_target, all_log_proposal)
        return all_samples, all_log_target, all_log_proposal, weights, rng

    def run_adaptive_nir_round(vstate, proposal_model, proposal_params, proposal_opt_state, proposal_optimizer, rng, *, step):
        round_summaries = []
        params = proposal_params
        opt_state = proposal_opt_state
        final_resampled = None
        eff_threshold = current_efficiency_threshold(step)
        proposal_steps = current_proposal_steps(step)
        force_adapt = step < nir_force_adapt_until_iter

        for round_idx in range(nir_max_adaptive_rounds):
            all_samples, all_log_target, all_log_proposal, weights, rng = sample_until_ess(
                vstate,
                proposal_model,
                params,
                rng,
                target_n_samples=vstate.n_samples,
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
                    prob_floor=nir_prob_floor,
                )

            round_summaries.append(
                {
                    "round": round_idx,
                    "proposal_pool": int(len(all_samples)),
                    "ess": float(ess),
                    "efficiency": float(eff),
                    "proposal_steps": int(proposal_steps),
                    "forward_kl_loss_after_steps": None if last_loss is None else float(last_loss),
                }
            )
            if round_idx + 1 < nir_min_adaptive_rounds:
                continue
            if (not force_adapt) and eff >= eff_threshold:
                break

        return {"rounds": round_summaries}, params, opt_state, final_resampled, rng

    model = build_model(learn_phase_stage_1)
    sampler = make_sampler(num_samples_stage_1)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=num_samples_stage_1,
        chunk_size=chunk_size,
    )

    proposal_model = AutoregressiveProposalNet(
        n_sites=num_sites,
        embed_dim=nir_proposal_embed_dim,
        num_heads=nir_proposal_heads,
        num_layers=nir_proposal_layers,
        mlp_hidden_dim=nir_proposal_mlp,
    )
    init_sigma = jnp.ones((4, num_sites), dtype=jnp.float64)
    proposal_params = proposal_model.init(fresh_key(), init_sigma)["params"]
    target_param_count = _count_params(vstate.parameters)
    proposal_param_count = _count_params(proposal_params)
    print("Target parameter count:", target_param_count)
    print("Proposal parameter count:", proposal_param_count)
    print(
        "Target model summary:",
        f"params={target_param_count}",
        f"embed_dim={embed_dim}",
        f"head_dim={target_head_dim}",
    )
    print(
        "Proposal model summary:",
        f"params={proposal_param_count}",
        f"embed_dim={nir_proposal_embed_dim}",
        f"head_dim={proposal_head_dim}",
    )
    proposal_optimizer = optax.adam(proposal_lr_schedule)
    proposal_opt_state = proposal_optimizer.init(proposal_params)
    target_optimizer = make_target_optimizer()
    target_opt_state = target_optimizer.init(vstate.parameters)
    target_preconditioner_state = make_target_preconditioner()

    history = []
    rng = fresh_key()
    best_energy = None
    best_vars = None

    for step in range(num_iters_total):
        learn_phase = current_learn_phase(step)
        n_samples = current_num_samples(step)
        if vstate.n_samples != n_samples or getattr(vstate.model, "learn_phase", None) != learn_phase:
            vstate = rebuild_vstate(vstate, learn_phase, n_samples)

        nir_summary, proposal_params, proposal_opt_state, resampled, rng = run_adaptive_nir_round(
            vstate,
            proposal_model,
            proposal_params,
            proposal_opt_state,
            proposal_optimizer,
            rng,
            step=step,
        )
        last_round = nir_summary["rounds"][-1]
        target_update_due = True
        target_update_applied = bool(last_round["efficiency"] >= current_efficiency_threshold(step))

        inject_external_samples(vstate, resampled)
        if target_update_applied:
            stats, direction, target_preconditioner_state = compute_target_direction(
                vstate, ha, target_preconditioner_state, step, learn_phase
            )
            direction = jax.tree.map(
                lambda update, param: jnp.asarray(update, dtype=param.dtype),
                direction,
                vstate.parameters,
            )
            updates, target_opt_state = target_optimizer.update(direction, target_opt_state, vstate.parameters)
            vstate.parameters = optax.apply_updates(vstate.parameters, updates)
        else:
            stats = nk.stats.statistics(vstate.local_estimators(ha, chunk_size=chunk_size))

        energy = float(np.real(np.asarray(stats.mean)))
        history.append(
            {
                "iteration": step + 1,
                "energy": energy,
                "learn_phase": bool(learn_phase),
                "due": target_update_due,
                "target_update_applied": target_update_applied,
                "nir": nir_summary,
            }
        )
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_vars = vstate.variables

        print(
            f"it={step + 1:5d} "
            f"Energy={energy:.8f} "
            f"ESS={last_round['ess']:.2f} "
            f"Eff={last_round['efficiency']:.4f} "
            f"Samples={vstate.n_samples:d} "
            f"LearnPhase={'yes' if learn_phase else 'no'} "
            f"Due={'yes' if target_update_due else 'no'} "
            f"Update={'yes' if target_update_applied else 'no'}"
        )

    out_prefix = run_dir / f"out_{job_base}"
    final_ckpt = run_dir / f"{job_base}.mpack"
    best_ckpt = run_dir / f"{job_base}_best.mpack"
    final_ckpt.write_bytes(serialization.to_bytes(vstate.variables))
    if best_vars is not None:
        best_ckpt.write_bytes(serialization.to_bytes(best_vars))

    history_file = run_dir / f"{job_base}.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    energy = [row["energy"] for row in history]
    with open(run_dir / f"mean_energy_run_{job_base}.txt", "w") as f:
        for item in energy:
            f.write(f"{item}\n")

    plt.figure(figsize=(12, 8))
    plt.plot(energy)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"J1-J2 Honeycomb {num_sites}-site (ViT NIR)")
    plt.tight_layout()
    plot_file = run_dir / f"energy_{job_base}.png"
    plt.savefig(plot_file)
    plt.close()

    tail_window = min(100, len(energy))
    tail_mean = float(np.mean(energy[-tail_window:])) if tail_window else None
    tail_std = float(np.std(energy[-tail_window:], ddof=1)) if tail_window > 1 else 0.0

    summary = {
        "job_base": job_base,
        "run_dir": str(run_dir),
        "num_sites": num_sites,
        "extent": extent,
        "j1": j1,
        "j2": j2,
        "num_samples_stage_1": num_samples_stage_1,
        "num_samples_stage_2": num_samples_stage_2,
        "num_samples_stage_3": num_samples_stage_3,
        "num_samples_stage_4": num_samples_stage_4 if use_stage_4 else None,
        "num_iters_total": num_iters_total,
        "patch_size": patch_size,
        "permutation": list(perm),
        "site_type_ids": list(token_site_type_ids),
        "num_site_types": num_site_types,
        "relation_matrix": [list(row) for row in relation_matrix],
        "num_relation_types": num_relation_types,
        "site_type_relation_model": model_type == "site_type_relation",
        "model_type": model_type,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "mlp_hidden": mlp_hidden,
        "nir_strategy": "paper_inspired",
        "target_optimizer": "adam",
        "target_sampler_name": target_sampler_name,
        "target_optimizer_name": target_optimizer_name,
        "target_sgd_momentum": target_sgd_momentum,
        "target_preconditioner": target_preconditioner,
        "target_sr_diag_shift": target_sr_diag_shift,
        "target_sr_diag_shift_stage_1": target_sr_diag_shift_stage_1,
        "target_sr_diag_shift_stage_2": target_sr_diag_shift_stage_2,
        "target_sr_diag_shift_stage_3": target_sr_diag_shift_stage_3,
        "target_sr_diag_shift_stage_4": target_sr_diag_shift_stage_4 if use_stage_4 else None,
        "target_sr_proj_reg": target_sr_proj_reg,
        "target_sr_momentum": target_sr_momentum,
        "target_sr_mode": target_sr_mode,
        "train_lr_stage_1": train_lr_stage_1,
        "train_lr_stage_2": train_lr_stage_2,
        "train_lr_stage_3": train_lr_stage_3,
        "train_lr_stage_4": train_lr_stage_4 if use_stage_4 else None,
        "train_lr_stage_1_iters": train_lr_stage_1_iters,
        "train_lr_stage_2_iters": train_lr_stage_2_iters,
        "train_lr_stage_3_iters": train_lr_stage_3_iters if use_stage_4 else None,
        "learn_phase_stage_1": learn_phase_stage_1,
        "learn_phase_stage_2": learn_phase_stage_2,
        "learn_phase_stage_3": learn_phase_stage_3,
        "learn_phase_stage_4": learn_phase_stage_4 if use_stage_4 else None,
        "nir_proposal_batch": nir_proposal_batch,
        "nir_max_proposal_batches": nir_max_proposal_batches,
        "nir_max_adaptive_rounds": nir_max_adaptive_rounds,
        "nir_min_adaptive_rounds": nir_min_adaptive_rounds,
        "nir_force_adapt_until_iter": nir_force_adapt_until_iter,
        "nir_ess_threshold_frac": nir_ess_threshold_frac,
        "nir_efficiency_threshold_stage_1": nir_efficiency_threshold_stage_1,
        "nir_efficiency_threshold_stage_2": nir_efficiency_threshold_stage_2,
        "nir_efficiency_threshold_stage_3": nir_efficiency_threshold_stage_3,
        "nir_efficiency_threshold_stage_4": nir_efficiency_threshold_stage_4 if use_stage_4 else None,
        "nir_proposal_lr_stage_1": nir_proposal_lr_stage_1,
        "nir_proposal_lr_stage_2": nir_proposal_lr_stage_2,
        "nir_proposal_lr_stage_3": nir_proposal_lr_stage_3,
        "nir_proposal_lr_stage_4": nir_proposal_lr_stage_4 if use_stage_4 else None,
        "nir_proposal_steps_stage_1": nir_proposal_steps_stage_1,
        "nir_proposal_steps_stage_2": nir_proposal_steps_stage_2,
        "nir_proposal_steps_stage_3": nir_proposal_steps_stage_3,
        "nir_proposal_steps_stage_4": nir_proposal_steps_stage_4 if use_stage_4 else None,
        "nir_proposal_embed_dim": nir_proposal_embed_dim,
        "nir_proposal_heads": nir_proposal_heads,
        "nir_proposal_layers": nir_proposal_layers,
        "nir_proposal_mlp": nir_proposal_mlp,
        "nir_prob_floor": nir_prob_floor,
        "run_tag": run_tag,
        "final_energy": float(energy[-1]),
        "best_energy_seen": float(best_energy),
        "tail_energy_window": tail_window,
        "tail_energy_mean": tail_mean,
        "tail_energy_std": tail_std,
        "tail_mean_last_20": float(np.mean(energy[-20:])) if len(energy) >= 20 else None,
        "tail_mean_last_50": float(np.mean(energy[-50:])) if len(energy) >= 50 else None,
        "tail_mean_last_100": float(np.mean(energy[-100:])) if len(energy) >= 100 else None,
        "tail_std_last_50": float(np.std(energy[-50:], ddof=1)) if len(energy) >= 50 else None,
        "exact_ground_state_energy": exact_gs,
        "history_file": str(history_file),
        "mean_energy_file": str(run_dir / f"mean_energy_run_{job_base}.txt"),
        "plot_file": str(plot_file),
        "final_checkpoint_file": str(final_ckpt),
        "best_checkpoint_file": str(best_ckpt),
    }

    with open(run_dir / f"summary_{job_base}.json", "w") as f:
        json.dump(summary, f, indent=2)
