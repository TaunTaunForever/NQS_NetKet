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
GAMMA_MODEL_ROOT = ROOT / "18-site"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GAMMA_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(GAMMA_MODEL_ROOT))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import netket.jax as nkjax
import numpy as np
import optax
from netket._src.ngd.sr_srt_common import sr as compute_sr_direction
from netket._src.ngd.sr_srt_common import srt as compute_minsr_direction
from netket.stats import Stats
from netket.vqs.mc import get_local_kernel
from netket.vqs.mc.common import force_to_grad
from netket.operator import DiscreteJaxOperator
from scipy.sparse.linalg import eigsh

from hamiltonian import gamma_hamiltonian
from nir_utils import (
    effective_sample_size,
    importance_resample,
    normalised_importance_weights_from_log_probs,
    sampling_efficiency,
)
from proposal_network import (
    AutoregressiveProposalNet,
    proposal_log_prob,
    proposal_log_prob_fixed_magnetization,
    sample_from_proposal,
    train_proposal_step,
)
from vit_model import HoneycombPatchViT
from vit_site_type_relation_gated_pool_model import (
    KitaevSiteTypeRelationGatedPoolViT,
)
from vit_site_type_relation_model import (
    KitaevSiteTypeRelationHoneycombViT,
    build_bipartite_site_type_ids,
    build_extended_kitaev_relation_matrix,
    site_relation_to_patch_relation_expanded,
    site_type_ids_to_patch_type_ids,
)


jax.config.update("jax_enable_x64", True)


def _count_params(tree) -> int:
    leaves = jax.tree.leaves(tree)
    return int(sum(np.asarray(leaf).size for leaf in leaves))


def _gamma_extent(num_sites: int) -> str:
    return {
        4: "[2,1]",
        8: "[2,2]",
        12: "[2,3]",
        18: "[3,3]",
        24: "[3,4]",
        32: "[4,4]",
        50: "[5,5]",
        72: "[6,6]",
        128: "[8,8]",
    }.get(num_sites, "custom")


def run_nir_experiment(
    *,
    num_sites: int,
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
    nir_proposal_constrain_total_sz: bool = True,
    nir_proposal_training_mode: str = "resampled",
    nir_proposal_weight_power: float = 1.0,
    nir_proposal_weight_clip_factor: float | None = None,
    nir_prob_floor: float = 1e-6,
    nir_resampling_method: str = "systematic",
    nir_weighted_pool_cap_factor: float | None = 2.0,
    nir_adapt_metric: str = "efficiency",
    nir_log_ratio_std_threshold_stage_1: float | None = None,
    nir_log_ratio_std_threshold_stage_2: float | None = None,
    nir_log_ratio_std_threshold_stage_3: float | None = None,
    nir_log_ratio_std_threshold_stage_4: float | None = None,
    nir_target_update_mode: str = "resample",
    nir_exact_refine_after_iter: int | None = None,
    nir_exact_refine_max_states: int = 4096,
    nir_sampler_refine_after_iter: int | None = None,
    nir_reset_state_on_refine: bool = True,
    resume_checkpoint_path: str | None = None,
    resume_proposal_checkpoint_path: str | None = None,
    rng_seed: int | None = None,
    model_type: str = "site_type_relation_gated_pool_bond",
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
    if nir_proposal_training_mode not in {"resampled", "weighted_pool"}:
        raise ValueError(
            "nir_proposal_training_mode must be one of "
            "{'resampled', 'weighted_pool'}."
        )
    if nir_proposal_weight_power <= 0:
        raise ValueError("nir_proposal_weight_power must be positive.")
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
    if nir_adapt_metric not in {"efficiency", "log_ratio_std"}:
        raise ValueError(
            f"Unsupported nir_adapt_metric={nir_adapt_metric!r}; expected 'efficiency' or 'log_ratio_std'."
        )
    if nir_target_update_mode not in {
        "resample",
        "weighted",
        "unique_weighted",
        "exact",
        "sampler",
    }:
        raise ValueError(
            f"Unsupported nir_target_update_mode={nir_target_update_mode!r}; "
            "expected 'resample', 'weighted', 'unique_weighted', 'exact', or 'sampler'."
        )
    if nir_target_update_mode == "exact" and nir_exact_refine_after_iter is None:
        nir_exact_refine_after_iter = 0
    if nir_target_update_mode == "sampler" and nir_sampler_refine_after_iter is None:
        nir_sampler_refine_after_iter = 0

    def _log_ratio_std_from_efficiency_threshold(efficiency_threshold: float) -> float:
        clipped = float(np.clip(efficiency_threshold, 1.0e-12, 1.0))
        return float(np.sqrt(max(0.0, -np.log(clipped))))

    nir_log_ratio_std_threshold_stage_1 = (
        _log_ratio_std_from_efficiency_threshold(nir_efficiency_threshold_stage_1)
        if nir_log_ratio_std_threshold_stage_1 is None
        else nir_log_ratio_std_threshold_stage_1
    )
    nir_log_ratio_std_threshold_stage_2 = (
        _log_ratio_std_from_efficiency_threshold(nir_efficiency_threshold_stage_2)
        if nir_log_ratio_std_threshold_stage_2 is None
        else nir_log_ratio_std_threshold_stage_2
    )
    nir_log_ratio_std_threshold_stage_3 = (
        _log_ratio_std_from_efficiency_threshold(nir_efficiency_threshold_stage_3)
        if nir_log_ratio_std_threshold_stage_3 is None
        else nir_log_ratio_std_threshold_stage_3
    )
    nir_log_ratio_std_threshold_stage_4 = (
        nir_log_ratio_std_threshold_stage_3
        if nir_log_ratio_std_threshold_stage_4 is None
        else nir_log_ratio_std_threshold_stage_4
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
        f"{num_sites}-site_"
        f"{num_layers}L_{num_heads}H_{patch_size}p_"
        f"{num_samples_stage_1}to{num_samples_stage_4 if use_stage_4 else num_samples_stage_3}_samples_{today}_"
        f"Gamma_Honeycomb_ViT_NIR_{model_type}"
    )
    if run_tag:
        job_base = f"{job_base}_{run_tag}"
    run_dir = Path("runs") / today / job_base
    run_dir.mkdir(parents=True, exist_ok=True)

    graph, _symm_group, hi, ha = gamma_hamiltonian(num_sites)
    extent = _gamma_extent(num_sites)
    exact_hilbert_size = int(hi.n_states)
    proposal_n_up = num_sites // 2 if nir_proposal_constrain_total_sz else None
    if (
        nir_exact_refine_after_iter is not None
        and exact_hilbert_size > nir_exact_refine_max_states
    ):
        raise ValueError(
            "Exact NIR refinement was requested, but the constrained Hilbert space has "
            f"{exact_hilbert_size} states, above GAMMA_NIR_EXACT_REFINE_MAX_STATES="
            f"{nir_exact_refine_max_states}. Increase the limit only if the exact "
            "SR/Jacobian memory cost is acceptable."
        )
    perm = tuple(range(graph.n_nodes))
    site_type_ids = build_bipartite_site_type_ids(graph, permutation=perm)
    token_site_type_ids = (
        site_type_ids
        if patch_size == 1
        else site_type_ids_to_patch_type_ids(site_type_ids, patch_size)
    )
    site_relation_matrix = build_extended_kitaev_relation_matrix(graph, permutation=perm)
    relation_matrix = (
        site_relation_matrix
        if patch_size == 1
        else site_relation_to_patch_relation_expanded(site_relation_matrix, patch_size)
    )
    bond_oriented_site_relation_matrix = site_relation_matrix
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
    print("NIR adaptive metric:", nir_adapt_metric)
    print("NIR target update mode:", nir_target_update_mode)
    print("NIR resampling method:", nir_resampling_method)
    print("NIR weighted pool cap factor:", nir_weighted_pool_cap_factor)
    print("NIR proposal fixed total Sz constraint:", nir_proposal_constrain_total_sz)
    if proposal_n_up is not None:
        print("NIR proposal fixed up-spin count:", proposal_n_up)
    print("NIR exact refine after iter:", nir_exact_refine_after_iter)
    print("NIR sampler refine after iter:", nir_sampler_refine_after_iter)
    print("NIR reset state on refine:", nir_reset_state_on_refine)
    print("Exact Hilbert states:", exact_hilbert_size)
    print("Resume checkpoint:", resume_checkpoint_path)
    print("Resume proposal checkpoint:", resume_proposal_checkpoint_path)

    if rng_seed is None:
        def fresh_key():
            seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
            return jax.random.PRNGKey(seed)
    else:
        base_key = jax.random.PRNGKey(int(rng_seed))
        key_counter = {"count": 0}

        def fresh_key():
            key = jax.random.fold_in(base_key, key_counter["count"])
            key_counter["count"] += 1
            return key

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

    def current_log_ratio_std_threshold(step):
        if step < train_lr_boundary_1:
            return nir_log_ratio_std_threshold_stage_1
        if step < train_lr_boundary_2:
            return nir_log_ratio_std_threshold_stage_2
        if use_stage_4 and step < train_lr_boundary_3:
            return nir_log_ratio_std_threshold_stage_3
        if use_stage_4:
            return nir_log_ratio_std_threshold_stage_4
        return nir_log_ratio_std_threshold_stage_3

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
            return HoneycombPatchViT(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_hidden_dim=mlp_hidden,
                patch_size=patch_size,
                learn_phase=learn_phase,
            )
        if model_type == "site_type_relation_gated_pool_bond":
            return KitaevSiteTypeRelationGatedPoolViT(
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
        return KitaevSiteTypeRelationHoneycombViT(
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

    def compute_exact_target_direction(vstate, hamiltonian, target_update_state, step, learn_phase):
        exact_vstate = nk.vqs.MCState(
            sampler=nk.sampler.ExactSampler(hi),
            model=build_model(learn_phase),
            n_samples=exact_hilbert_size,
            variables=vstate.variables,
            chunk_size=chunk_size,
        )
        return compute_target_direction(
            exact_vstate,
            hamiltonian,
            target_update_state,
            step,
            learn_phase,
        )

    def compute_local_estimators_on_samples(vstate, hamiltonian, samples):
        samples = jnp.asarray(samples, dtype=jnp.float64)
        if chunk_size is None:
            kernel = get_local_kernel(vstate, hamiltonian)
        else:
            kernel = get_local_kernel(vstate, hamiltonian, chunk_size)

        def logpsi(params, sigma):
            return vstate._apply_fun({"params": params, **vstate.model_state}, sigma)

        if isinstance(hamiltonian, DiscreteJaxOperator):
            if chunk_size is None:
                return kernel(logpsi, vstate.parameters, samples, hamiltonian)
            return kernel(
                logpsi,
                vstate.parameters,
                samples,
                hamiltonian,
                chunk_size=chunk_size,
            )

        sigma_p, mels = hamiltonian.get_conn_padded(samples)
        if chunk_size is None:
            return kernel(logpsi, vstate.parameters, samples, (sigma_p, mels))
        return kernel(
            logpsi,
            vstate.parameters,
            samples,
            (sigma_p, mels),
            chunk_size=chunk_size,
        )

    def weighted_stats(values, weights):
        values = jnp.asarray(values)
        weights = jnp.asarray(weights, dtype=jnp.float64)
        mean = jnp.sum(weights * values)
        centered = values - mean
        variance = jnp.real(jnp.sum(weights * jnp.abs(centered) ** 2))
        ess = effective_sample_size(weights)
        error_of_mean = jnp.sqrt(jnp.maximum(variance, 0.0) / jnp.maximum(ess, 1.0))
        return Stats(
            mean=mean,
            error_of_mean=error_of_mean,
            variance=variance,
            tau_corr=jnp.nan,
            R_hat=jnp.nan,
            tau_corr_max=jnp.nan,
        )

    def compute_weighted_target_direction(
        vstate,
        hamiltonian,
        target_update_state,
        step,
        learn_phase,
        samples,
        weights,
    ):
        local_energies = compute_local_estimators_on_samples(vstate, hamiltonian, samples)
        stats = weighted_stats(local_energies, weights)
        kind = target_update_state["kind"]
        effective_mode = target_sr_mode if learn_phase else "real"

        if kind == "none":
            centered = local_energies - stats.mean
            _, vjp_fun = nkjax.vjp(
                lambda params: vstate._apply_fun(
                    {"params": params, **vstate.model_state}, samples
                ),
                vstate.parameters,
                conjugate=True,
            )
            forces = vjp_fun(jnp.conjugate(centered) * weights)[0]
            grad = force_to_grad(forces, vstate.parameters)
            return stats, grad, target_update_state, "weighted_gradient"

        if kind == "sr":
            dp, old_updates, info = compute_sr_direction(
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
                weights=weights,
            )
            target_update_state["old_updates"] = old_updates
            target_update_state["info"] = info
            return stats, dp, target_update_state, "weighted_sr"

        if kind == "minsr":
            dp, old_updates, info = compute_sr_direction(
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
                weights=weights,
            )
            target_update_state["old_updates"] = old_updates
            target_update_state["info"] = {
                "weighted_target_update_fallback": "sr",
                "original_preconditioner": "minsr",
                "sr_info": info,
            }
            return stats, dp, target_update_state, "weighted_sr_fallback_from_minsr"

        raise ValueError(f"Unsupported target_preconditioner state: {kind}")

    def unique_weighted_samples(samples, weights):
        samples_np = np.asarray(jax.device_get(samples))
        weights_np = np.asarray(jax.device_get(weights), dtype=np.float64)
        unique_np, inverse = np.unique(samples_np, axis=0, return_inverse=True)
        unique_weights_np = np.zeros(unique_np.shape[0], dtype=np.float64)
        np.add.at(unique_weights_np, inverse, weights_np)
        unique_weights_np = unique_weights_np / np.sum(unique_weights_np)
        return (
            jnp.asarray(unique_np, dtype=jnp.float64),
            jnp.asarray(unique_weights_np, dtype=jnp.float64),
        )

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
        weighted_pool_cap = None
        if (
            nir_target_update_mode in {"weighted", "unique_weighted"}
            and nir_weighted_pool_cap_factor is not None
        ):
            weighted_pool_cap = max(
                target_n_samples,
                int(np.ceil(float(nir_weighted_pool_cap_factor) * target_n_samples)),
            )

        for _ in range(nir_max_proposal_batches):
            batch_size = nir_proposal_batch
            if weighted_pool_cap is not None:
                current_pool = sum(batch.shape[0] for batch in proposal_batches)
                remaining = weighted_pool_cap - current_pool
                if remaining <= 0:
                    break
                batch_size = min(batch_size, remaining)

            proposal_samples, rng = sample_from_proposal(
                proposal_model,
                proposal_params,
                rng,
                batch_size,
                num_sites,
                prob_floor=nir_prob_floor,
                n_up=proposal_n_up,
            )
            log_target = target_log_probs(vstate, proposal_samples)
            if proposal_n_up is None:
                log_proposal = proposal_log_prob(
                    proposal_model,
                    proposal_params,
                    proposal_samples,
                    prob_floor=nir_prob_floor,
                )
            else:
                log_proposal = proposal_log_prob_fixed_magnetization(
                    proposal_model,
                    proposal_params,
                    proposal_samples,
                    prob_floor=nir_prob_floor,
                    n_up=proposal_n_up,
                )

            proposal_batches.append(proposal_samples)
            log_target_batches.append(log_target)
            log_proposal_batches.append(log_proposal)

            stacked_target = jnp.concatenate(log_target_batches, axis=0)
            stacked_proposal = jnp.concatenate(log_proposal_batches, axis=0)
            weights = normalised_importance_weights_from_log_probs(stacked_target, stacked_proposal)
            ess = effective_sample_size(weights)
            if float(ess) >= ess_threshold:
                break
            if weighted_pool_cap is not None and stacked_target.shape[0] >= weighted_pool_cap:
                break

        all_samples = jnp.concatenate(proposal_batches, axis=0)
        all_log_target = jnp.concatenate(log_target_batches, axis=0)
        all_log_proposal = jnp.concatenate(log_proposal_batches, axis=0)
        weights = normalised_importance_weights_from_log_probs(all_log_target, all_log_proposal)
        return all_samples, all_log_target, all_log_proposal, weights, rng

    def evaluate_adapt_metric(log_target_probs, log_proposal_probs, weights, step):
        log_ratio = jnp.asarray(log_target_probs - log_proposal_probs, dtype=jnp.float64)
        ess = effective_sample_size(weights)
        efficiency = sampling_efficiency(weights)
        log_ratio_std = jnp.std(log_ratio)
        if nir_adapt_metric == "efficiency":
            threshold = current_efficiency_threshold(step)
            satisfied = efficiency >= threshold
        else:
            threshold = current_log_ratio_std_threshold(step)
            satisfied = log_ratio_std <= threshold
        return {
            "ess": float(ess),
            "efficiency": float(efficiency),
            "log_ratio_std": float(log_ratio_std),
            "selected_metric": nir_adapt_metric,
            "selected_threshold": float(threshold),
            "selected_satisfied": bool(satisfied),
        }

    def proposal_training_batch(resampled, all_samples, weights):
        if nir_proposal_training_mode == "resampled":
            return resampled, None

        train_weights = jnp.asarray(weights, dtype=jnp.float64)
        if nir_proposal_weight_power != 1.0:
            train_weights = jnp.power(
                train_weights,
                jnp.asarray(nir_proposal_weight_power, dtype=jnp.float64),
            )
        if nir_proposal_weight_clip_factor is not None:
            cap = jnp.asarray(nir_proposal_weight_clip_factor, dtype=jnp.float64)
            cap = cap / jnp.asarray(train_weights.shape[0], dtype=jnp.float64)
            train_weights = jnp.minimum(train_weights, cap)
        train_weights = train_weights / jnp.maximum(
            jnp.sum(train_weights),
            jnp.finfo(train_weights.dtype).tiny,
        )
        return all_samples, train_weights

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
        final_all_samples = None
        final_weights = None
        proposal_steps = current_proposal_steps(step)
        force_adapt = step < nir_force_adapt_until_iter
        final_metrics = None

        for round_idx in range(nir_max_adaptive_rounds):
            all_samples, all_log_target, all_log_proposal, weights, rng = sample_until_ess(
                vstate,
                proposal_model,
                params,
                rng,
                target_n_samples=vstate.n_samples,
            )
            metrics = evaluate_adapt_metric(
                all_log_target,
                all_log_proposal,
                weights,
                step,
            )
            unique_count = None
            if nir_target_update_mode == "unique_weighted":
                unique_count = int(
                    np.unique(np.asarray(jax.device_get(all_samples)), axis=0).shape[0]
                )
            resampled, _indices, _weights, rng = importance_resample(
                all_samples,
                all_log_target,
                all_log_proposal,
                n_samples=vstate.n_samples,
                method=nir_resampling_method,
                rng=rng,
            )
            final_resampled = resampled
            final_all_samples = all_samples
            final_weights = weights
            final_metrics = metrics

            train_batch, train_weights = proposal_training_batch(
                resampled,
                all_samples,
                weights,
            )
            last_loss = None
            for _ in range(proposal_steps):
                params, opt_state, last_loss = train_proposal_step(
                    proposal_model,
                    params,
                    opt_state,
                    proposal_optimizer,
                    train_batch,
                    prob_floor=nir_prob_floor,
                    n_up=proposal_n_up,
                    sample_weights=train_weights,
                )

            round_summaries.append(
                {
                    "round": round_idx,
                    "proposal_pool": int(all_samples.shape[0]),
                    "ess": metrics["ess"],
                    "efficiency": metrics["efficiency"],
                    "log_ratio_std": metrics["log_ratio_std"],
                    "selected_metric": metrics["selected_metric"],
                    "selected_threshold": metrics["selected_threshold"],
                    "selected_satisfied": metrics["selected_satisfied"],
                    "unique_count": unique_count,
                    "proposal_steps": int(proposal_steps),
                    "proposal_training_mode": nir_proposal_training_mode,
                    "forward_kl_loss_after_steps": None if last_loss is None else float(last_loss),
                }
            )
            if round_idx + 1 < nir_min_adaptive_rounds:
                continue
            if (not force_adapt) and metrics["selected_satisfied"]:
                break

        return {
            "rounds": round_summaries,
            "target_samples": final_all_samples,
            "target_weights": final_weights,
            "final_metrics": final_metrics,
        }, params, opt_state, final_resampled, rng

    model = build_model(learn_phase_stage_1)
    sampler = make_sampler(num_samples_stage_1)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=num_samples_stage_1,
        chunk_size=chunk_size,
    )
    if resume_checkpoint_path:
        checkpoint_path = Path(resume_checkpoint_path).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        print("Loading target checkpoint:", checkpoint_path)
        loaded_variables = serialization.from_bytes(
            vstate.variables,
            checkpoint_path.read_bytes(),
        )
        vstate = nk.vqs.MCState(
            sampler=sampler,
            model=model,
            n_samples=num_samples_stage_1,
            variables=loaded_variables,
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
    if resume_proposal_checkpoint_path:
        proposal_checkpoint_path = Path(resume_proposal_checkpoint_path).expanduser()
        if not proposal_checkpoint_path.is_absolute():
            proposal_checkpoint_path = Path.cwd() / proposal_checkpoint_path
        print("Loading proposal checkpoint:", proposal_checkpoint_path)
        proposal_params = serialization.from_bytes(
            proposal_params,
            proposal_checkpoint_path.read_bytes(),
        )
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
    refinement_has_started = False

    for step in range(num_iters_total):
        learn_phase = current_learn_phase(step)
        n_samples = current_num_samples(step)
        if vstate.n_samples != n_samples or getattr(vstate.model, "learn_phase", None) != learn_phase:
            vstate = rebuild_vstate(vstate, learn_phase, n_samples)

        exact_refine_active = (
            nir_exact_refine_after_iter is not None
            and step >= nir_exact_refine_after_iter
        )
        sampler_refine_active = (
            (not exact_refine_active)
            and nir_sampler_refine_after_iter is not None
            and step >= nir_sampler_refine_after_iter
        )
        refine_active = exact_refine_active or sampler_refine_active
        if refine_active and not refinement_has_started:
            if nir_reset_state_on_refine:
                target_opt_state = target_optimizer.init(vstate.parameters)
                target_preconditioner_state = make_target_preconditioner()
            refinement_has_started = True

        if exact_refine_active or sampler_refine_active:
            selected_metric = "exact_refine" if exact_refine_active else "sampler_refine"
            proposal_pool = exact_hilbert_size if exact_refine_active else vstate.n_samples
            last_round = {
                "round": 0,
                "proposal_pool": int(proposal_pool),
                "ess": float(proposal_pool),
                "efficiency": 1.0,
                "log_ratio_std": 0.0,
                "selected_metric": selected_metric,
                "selected_threshold": 0.0,
                "selected_satisfied": True,
                "proposal_steps": 0,
                "forward_kl_loss_after_steps": None,
            }
            nir_summary = {"rounds": [last_round]}
            resampled = None
        else:
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
        target_update_applied = bool(last_round["selected_satisfied"]) or refine_active
        if exact_refine_active:
            target_update_strategy = "exact_refine"
        elif sampler_refine_active:
            target_update_strategy = "sampler_refine"
        else:
            target_update_strategy = "resample"

        if target_update_applied:
            if exact_refine_active:
                stats, direction, target_preconditioner_state = compute_exact_target_direction(
                    vstate,
                    ha,
                    target_preconditioner_state,
                    step,
                    learn_phase,
                )
            elif sampler_refine_active:
                vstate.reset()
                stats, direction, target_preconditioner_state = compute_target_direction(
                    vstate,
                    ha,
                    target_preconditioner_state,
                    step,
                    learn_phase,
                )
            elif nir_target_update_mode == "weighted":
                stats, direction, target_preconditioner_state, target_update_strategy = (
                    compute_weighted_target_direction(
                        vstate,
                        ha,
                        target_preconditioner_state,
                        step,
                        learn_phase,
                        nir_summary["target_samples"],
                        nir_summary["target_weights"],
                    )
                )
            elif nir_target_update_mode == "unique_weighted":
                unique_samples, unique_weights = unique_weighted_samples(
                    nir_summary["target_samples"],
                    nir_summary["target_weights"],
                )
                stats, direction, target_preconditioner_state, target_update_strategy = (
                    compute_weighted_target_direction(
                        vstate,
                        ha,
                        target_preconditioner_state,
                        step,
                        learn_phase,
                        unique_samples,
                        unique_weights,
                    )
                )
                target_update_strategy = f"unique_{target_update_strategy}"
                last_round["unique_count"] = int(unique_samples.shape[0])
            else:
                inject_external_samples(vstate, resampled)
                stats, direction, target_preconditioner_state = compute_target_direction(
                    vstate,
                    ha,
                    target_preconditioner_state,
                    step,
                    learn_phase,
                )
            direction = jax.tree.map(
                lambda update, param: jnp.asarray(update, dtype=param.dtype),
                direction,
                vstate.parameters,
            )
            updates, target_opt_state = target_optimizer.update(
                direction,
                target_opt_state,
                vstate.parameters,
            )
            vstate.parameters = optax.apply_updates(vstate.parameters, updates)
        else:
            if resampled is not None:
                inject_external_samples(vstate, resampled)
                stats = nk.stats.statistics(vstate.local_estimators(ha, chunk_size=chunk_size))
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
                "target_update_strategy": target_update_strategy,
                "nir": {"rounds": nir_summary["rounds"]},
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
            f"StdLogPQ={last_round['log_ratio_std']:.4f} "
            f"Samples={(exact_hilbert_size if exact_refine_active else vstate.n_samples):d} "
            f"LearnPhase={'yes' if learn_phase else 'no'} "
            f"Due={'yes' if target_update_due else 'no'} "
            f"Update={'yes' if target_update_applied else 'no'} "
            f"Strategy={target_update_strategy}"
        )

    out_prefix = run_dir / f"out_{job_base}"
    final_ckpt = run_dir / f"{job_base}.mpack"
    best_ckpt = run_dir / f"{job_base}_best.mpack"
    final_proposal_ckpt = run_dir / f"{job_base}_proposal.mpack"
    final_ckpt.write_bytes(serialization.to_bytes(vstate.variables))
    final_proposal_ckpt.write_bytes(serialization.to_bytes(proposal_params))
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
    plt.title(f"Gamma Honeycomb {num_sites}-site (ViT NIR)")
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
        "hamiltonian": "Gamma",
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
        "target_optimizer": target_optimizer_name,
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
        "nir_adapt_metric": nir_adapt_metric,
        "nir_log_ratio_std_threshold_stage_1": nir_log_ratio_std_threshold_stage_1,
        "nir_log_ratio_std_threshold_stage_2": nir_log_ratio_std_threshold_stage_2,
        "nir_log_ratio_std_threshold_stage_3": nir_log_ratio_std_threshold_stage_3,
        "nir_log_ratio_std_threshold_stage_4": nir_log_ratio_std_threshold_stage_4 if use_stage_4 else None,
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
        "nir_proposal_constrain_total_sz": nir_proposal_constrain_total_sz,
        "nir_proposal_n_up": proposal_n_up,
        "nir_proposal_training_mode": nir_proposal_training_mode,
        "nir_proposal_weight_power": nir_proposal_weight_power,
        "nir_proposal_weight_clip_factor": nir_proposal_weight_clip_factor,
        "nir_prob_floor": nir_prob_floor,
        "nir_weighted_pool_cap_factor": nir_weighted_pool_cap_factor,
        "nir_resampling_method": nir_resampling_method,
        "nir_target_update_mode": nir_target_update_mode,
        "nir_exact_refine_after_iter": nir_exact_refine_after_iter,
        "nir_exact_refine_max_states": nir_exact_refine_max_states,
        "nir_sampler_refine_after_iter": nir_sampler_refine_after_iter,
        "nir_reset_state_on_refine": nir_reset_state_on_refine,
        "resume_checkpoint_path": resume_checkpoint_path,
        "resume_proposal_checkpoint_path": resume_proposal_checkpoint_path,
        "exact_hilbert_size": exact_hilbert_size,
        "rng_seed": rng_seed,
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
        "final_proposal_checkpoint_file": str(final_proposal_ckpt),
        "best_checkpoint_file": str(best_ckpt),
    }

    with open(run_dir / f"summary_{job_base}.json", "w") as f:
        json.dump(summary, f, indent=2)
