from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
import math

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("NETKET_DEBUG", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import jax
import netket as nk
import optax
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

import expectations
from common import make_heisenberg_hamiltonian
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


def _extract_real_series(log_data: dict, observable: str = "Energy") -> list[float]:
    series = log_data[observable]["Mean"]
    if isinstance(series, dict):
        if "real" in series:
            return [float(x) for x in series["real"]]
        if "value" in series:
            return [float(x) for x in series["value"]]
    return [float(x) for x in series]


def _json_safe(value):
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _compatible_chunk_size(requested: int | None, n_samples: int) -> int | None:
    if requested is None or requested <= 0:
        return None
    if n_samples % requested == 0:
        return requested
    return math.gcd(requested, n_samples) or None


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def run_srt_experiment(
    *,
    num_sites: int,
    j1: float,
    j2: float,
    num_samples: int = 3*2**9,
    num_iters_warm: int = 100,
    num_iters: int = 1000,
    chunk_size: int | None = 2**8,
    patch_size: int = 1,
    embed_dim: int = 24,
    num_heads: int = 4,
    num_layers: int = 4,
    mlp_hidden: int | None = None,
    learn_phase_main: bool = True,
    learning_rate: float = 1e-2,
    learning_rate_warm: float | None = None,
    learning_rate_main: float | None = None,
    learning_rate_refine: float | None = None,
    momentum: float = 0.9,
    diag_shift_warm: float = 1e-4,
    diag_shift_main: float = 1e-3,
    diag_shift_refine: float | None = None,
    sampler_name: str = "pt_exchange",
    sampler_name_refine: str | None = None,
    d_max: int = 2,
    d_max_refine: int | None = None,
    observable_num_samples: int | None = None,
    observable_chunk_size: int | None = None,
    observable_sampler_name: str | None = None,
    observable_d_max: int | None = None,
    optimizer_name: str = "sgd",
    num_samples_refine: int | None = None,
    num_iters_refine: int = 0,
    resume_checkpoint_path: str | None = None,
    run_tag: str | None = None,
    model_type: str = "site_type_relation",
):
    today = date.today().isoformat()
    mlp_hidden = 2 * embed_dim if mlp_hidden is None else mlp_hidden
    learning_rate_warm = learning_rate if learning_rate_warm is None else learning_rate_warm
    learning_rate_main = learning_rate if learning_rate_main is None else learning_rate_main
    learning_rate_refine = learning_rate_main if learning_rate_refine is None else learning_rate_refine
    diag_shift_refine = diag_shift_main if diag_shift_refine is None else diag_shift_refine
    sampler_name_refine = sampler_name if sampler_name_refine is None else sampler_name_refine
    d_max_refine = d_max if d_max_refine is None else d_max_refine
    num_samples_refine = num_samples if num_samples_refine is None else num_samples_refine
    observable_sampler_name = (
        sampler_name_refine if observable_sampler_name is None else observable_sampler_name
    )
    observable_d_max = d_max_refine if observable_d_max is None else observable_d_max
    if observable_num_samples is None:
        observable_num_samples = 3 * 2**12
    if observable_chunk_size is None:
        observable_chunk_size = min(observable_num_samples, 2048)
    chunk_size_main = _compatible_chunk_size(chunk_size, num_samples)
    chunk_size_refine = _compatible_chunk_size(chunk_size, num_samples_refine)
    chunk_size_observable = _compatible_chunk_size(observable_chunk_size, observable_num_samples)
    model_tag = {
        "site_type_relation": "site_type_relation",
        "site_type_relation_gated_pool_bond": "site_type_relation_gated_pool_bond",
    }.get(model_type)
    if model_tag is None:
        raise ValueError(f"Unsupported model_type={model_type!r}")

    job_name = (
        f"J1={j1}_J2={j2}_{num_sites}-site_"
        f"{num_layers}L_{num_heads}H_{patch_size}p_"
        f"{num_samples}_samples_{today}_J1_J2_Honeycomb_ViT_SRt_{model_tag}"
    )
    if run_tag:
        job_name = f"{job_name}_{run_tag}"

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

    active_relation_matrix = (
        bond_oriented_relation_matrix
        if model_type == "site_type_relation_gated_pool_bond"
        else relation_matrix
    )
    active_num_relation_types = (
        num_bond_oriented_relation_types
        if model_type == "site_type_relation_gated_pool_bond"
        else num_relation_types
    )

    print(f"Defining {num_sites}-site J1-J2 Honeycomb lattice")
    print("___________________________________________________")
    print(f"Honeycomb lattice extent: {extent}")
    print("Sites:", graph.n_nodes)
    print("Edges:", graph.n_edges)
    print("Neighbor orders:", sorted(set(color for *_ij, color in graph.edges(return_color=True))))
    print("Model type:", model_type)
    print("Site type ids:", token_site_type_ids)
    print("Number of site types:", num_site_types)
    print(
        "Relation matrix shape:",
        (len(active_relation_matrix), len(active_relation_matrix[0])),
    )
    print("Number of relation types:", active_num_relation_types)
    print("Bond-oriented relation types:", num_bond_oriented_relation_types)
    print()

    exact_gs = None
    if num_sites <= 24:
        sp_h = ha.to_sparse()
        eig_vals, _ = eigsh(sp_h, k=2, which="SA")
        exact_gs = float(eig_vals[0])
        print("Exact ground-state energy:", exact_gs)
        print()

    def build_sampler(name: str, n_samples: int, d_max_value: int):
        if name == "pt_exchange":
            return nk.sampler.ParallelTemperingExchange(hi, graph=graph, d_max=d_max_value)
        if name == "pt_local":
            return nk.sampler.ParallelTemperingLocal(hi)
        if name == "local":
            n_chains = max(1, n_samples // 32)
            return nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
        if name == "exact":
            return nk.sampler.ExactSampler(hi)
        raise ValueError(f"Unsupported sampler_name={name!r}")

    sampler = build_sampler(sampler_name, num_samples, d_max)

    def build_optimizer(lr: float):
        if optimizer_name == "sgd":
            return optax.sgd(learning_rate=lr, momentum=momentum)
        if optimizer_name == "adam":
            return optax.adam(learning_rate=lr)
        raise ValueError(f"Unsupported optimizer_name={optimizer_name!r}")

    def build_model(*, learn_phase: bool):
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

    warm_log_file = None
    if num_iters_warm > 0:
        print("\n=== Stage 1: amplitude-only warm start ===\n")

        model_warm = build_model(learn_phase=False)

        vstate_1 = nk.vqs.MCState(
            sampler=sampler,
            model=model_warm,
            n_samples=num_samples,
            chunk_size=chunk_size_main,
        )

        print("num_samples:", num_samples)
        print("num params:", vstate_1.n_parameters)
        print("chain_length:", vstate_1.chain_length)
        print("n_discard_per_chain:", vstate_1.n_discard_per_chain)

        driver_1 = nk.driver.VMC_SR(
            hamiltonian=ha,
            optimizer=build_optimizer(learning_rate_warm),
            variational_state=vstate_1,
            diag_shift=diag_shift_warm,
        )
        warm_log_file = f"out_warm_{job_name}.log"
        driver_1.run(
            n_iter=num_iters_warm,
            out=f"out_warm_{job_name}",
            save_params_every=10,
        )
        initial_variables = vstate_1.variables
    else:
        print("\n=== Stage 1: skipped amplitude-only warm start ===\n")
        model_init = build_model(learn_phase=learn_phase_main)
        vstate_init = nk.vqs.MCState(
            sampler=sampler,
            model=model_init,
            n_samples=num_samples,
            chunk_size=chunk_size_main,
        )
        print("num_samples:", num_samples)
        print("num params:", vstate_init.n_parameters)
        print("chain_length:", vstate_init.chain_length)
        print("n_discard_per_chain:", vstate_init.n_discard_per_chain)
        initial_variables = vstate_init.variables

    if resume_checkpoint_path:
        print("\n=== Loading continuation checkpoint ===\n")
        print("Checkpoint:", resume_checkpoint_path)
        initial_variables = nk.experimental.vqs.variables_from_file(
            resume_checkpoint_path,
            initial_variables,
        )

    print("\n=== Stage 2: full complex optimization ===\n")

    model_full = build_model(learn_phase=learn_phase_main)

    vstate_2 = nk.vqs.MCState(
        sampler=sampler,
        model=model_full,
        n_samples=num_samples,
        variables=initial_variables,
        chunk_size=chunk_size_main,
    )

    obs = {
        "<X>": sum(nk.operator.spin.sigmax(hi, i) for i in range(hi.size)) / (2 * num_sites),
        "<Y>": sum(nk.operator.spin.sigmay(hi, i) for i in range(hi.size)) / (2 * num_sites),
        "<Z>": sum(nk.operator.spin.sigmaz(hi, i) for i in range(hi.size)) / (2 * num_sites),
    }

    driver_2 = nk.driver.VMC_SR(
        hamiltonian=ha,
        optimizer=build_optimizer(learning_rate_main),
        variational_state=vstate_2,
        diag_shift=diag_shift_main,
    )
    stage2_out = f"out_{job_name}_stage2"
    driver_2.run(
        n_iter=num_iters,
        obs=obs,
        out=stage2_out,
        save_params_every=10,
    )

    with open(f"{stage2_out}.log") as f:
        data = json.load(f)
    energy = _extract_real_series(data, "Energy")

    final_vstate = vstate_2
    refine_log_file = None
    if num_iters_refine > 0:
        print("\n=== Stage 3: local refinement ===\n")
        sampler_refine = build_sampler(sampler_name_refine, num_samples_refine, d_max_refine)
        vstate_3 = nk.vqs.MCState(
            sampler=sampler_refine,
            model=model_full,
            n_samples=num_samples_refine,
            variables=vstate_2.variables,
            chunk_size=chunk_size_refine,
        )
        driver_3 = nk.driver.VMC_SR(
            hamiltonian=ha,
            optimizer=build_optimizer(learning_rate_refine),
            variational_state=vstate_3,
            diag_shift=diag_shift_refine,
        )
        stage3_out = f"out_{job_name}_stage3"
        driver_3.run(
            n_iter=num_iters_refine,
            obs=obs,
            out=stage3_out,
            save_params_every=10,
        )
        with open(f"{stage3_out}.log") as f:
            data3 = json.load(f)
        energy.extend(_extract_real_series(data3, "Energy"))
        final_vstate = vstate_3
        refine_log_file = f"{stage3_out}.log"

    with open(f"mean_energy_run_{job_name}.txt", "w") as f:
        for item in energy:
            f.write(f"{item}\n")

    plt.figure(figsize=(12, 8))
    plt.plot(energy)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"J1-J2 Honeycomb {num_sites}-site (ViT SRt)")
    plt.tight_layout()
    plt.savefig(f"energy_{job_name}.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(energy)
    plt.xscale("log")
    plt.xlabel("Iteration (log)")
    plt.ylabel("Energy")
    plt.title(f"J1-J2 Honeycomb {num_sites}-site (ViT SRt)")
    plt.tight_layout()
    plt.savefig(f"energy_log_{job_name}.png")
    plt.close()

    print("\n=== Observable evaluation pass ===\n")
    print("observable_sampler:", observable_sampler_name)
    print("observable_num_samples:", observable_num_samples)
    print("observable_chunk_size:", chunk_size_observable)

    observable_sampler = build_sampler(
        observable_sampler_name,
        observable_num_samples,
        observable_d_max,
    )
    observable_vstate = nk.vqs.MCState(
        sampler=observable_sampler,
        model=model_full,
        n_samples=observable_num_samples,
        variables=final_vstate.variables,
        chunk_size=chunk_size_observable,
    )

    observables = expectations.define_observables(num_sites, hi, graph)
    observable_results = expectations.calculate_expectations(observable_vstate, ha, observables)
    observables_file = f"observables_{job_name}.json"
    with open(observables_file, "w") as f:
        json.dump(_json_safe(observable_results), f, indent=2)

    summary = {
        "job_name": job_name,
        "num_sites": num_sites,
        "extent": extent,
        "num_samples": num_samples,
        "num_iters_warm": num_iters_warm,
        "num_iters": num_iters,
        "chunk_size": chunk_size,
        "chunk_size_main": chunk_size_main,
        "chunk_size_refine": chunk_size_refine,
        "observable_num_samples": observable_num_samples,
        "observable_chunk_size": observable_chunk_size,
        "chunk_size_observable": chunk_size_observable,
        "j1": j1,
        "j2": j2,
        "patch_size": patch_size,
        "permutation": list(perm),
        "site_type_ids": list(token_site_type_ids),
        "num_site_types": num_site_types,
        "relation_matrix": [list(row) for row in active_relation_matrix],
        "num_relation_types": active_num_relation_types,
        "site_type_relation_model": model_type == "site_type_relation",
        "model_type": model_type,
        "bond_oriented_relation_matrix": [list(row) for row in bond_oriented_relation_matrix],
        "num_bond_oriented_relation_types": num_bond_oriented_relation_types,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "mlp_hidden": mlp_hidden,
        "learning_rate": learning_rate,
        "learning_rate_warm": learning_rate_warm,
        "learning_rate_main": learning_rate_main,
        "momentum": momentum,
        "learn_phase_main": learn_phase_main,
        "diag_shift_warm": diag_shift_warm,
        "diag_shift_main": diag_shift_main,
        "learning_rate_refine": learning_rate_refine,
        "diag_shift_refine": diag_shift_refine,
        "sampler_name": sampler_name,
        "sampler_name_refine": sampler_name_refine,
        "observable_sampler_name": observable_sampler_name,
        "d_max": d_max,
        "d_max_refine": d_max_refine,
        "observable_d_max": observable_d_max,
        "optimizer_name": optimizer_name,
        "num_samples_refine": num_samples_refine,
        "num_iters_refine": num_iters_refine,
        "resume_checkpoint_path": resume_checkpoint_path,
        "run_tag": run_tag,
        "final_energy": float(energy[-1]),
        "best_energy_seen": float(min(energy)),
        "exact_ground_state_energy": exact_gs,
        "mean_energy_file": f"mean_energy_run_{job_name}.txt",
        "log_file": f"{stage2_out}.log",
        "refine_log_file": refine_log_file,
        "warm_log_file": warm_log_file,
        "plot_file": f"energy_{job_name}.png",
        "plot_log_file": f"energy_log_{job_name}.png",
        "observables_file": observables_file,
        "tail_mean_last_20": float(np.mean(energy[-20:])) if len(energy) >= 20 else None,
        "tail_mean_last_50": float(np.mean(energy[-50:])) if len(energy) >= 50 else None,
        "tail_mean_last_100": float(np.mean(energy[-100:])) if len(energy) >= 100 else None,
        "tail_std_last_50": float(np.std(energy[-50:])) if len(energy) >= 50 else None,
    }

    with open(f"summary_{job_name}.json", "w") as f:
        json.dump(summary, f, indent=2)
