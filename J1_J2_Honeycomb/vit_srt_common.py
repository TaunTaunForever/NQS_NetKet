from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("NETKET_DEBUG", "1")

import jax
import netket as nk
import optax
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

import expectations
from common import make_heisenberg_hamiltonian
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
    momentum: float = 0.9,
    diag_shift_warm: float = 1e-4,
    diag_shift_main: float = 1e-3,
    sampler_name: str = "pt_exchange",
    d_max: int = 2,
    optimizer_name: str = "sgd",
):
    today = date.today().isoformat()
    mlp_hidden = 2 * embed_dim if mlp_hidden is None else mlp_hidden
    learning_rate_warm = learning_rate if learning_rate_warm is None else learning_rate_warm
    learning_rate_main = learning_rate if learning_rate_main is None else learning_rate_main
    job_name = (
        f"J1={j1}_J2={j2}_{num_sites}-site_"
        f"{num_layers}L_{num_heads}H_{patch_size}p_"
        f"{num_samples}_samples_{today}_J1_J2_Honeycomb_ViT_SRt_site_type_relation"
    )

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
    num_relation_types = max(max(row) for row in relation_matrix) + 1
    num_site_types = max(token_site_type_ids) + 1

    print(f"Defining {num_sites}-site J1-J2 Honeycomb lattice")
    print("___________________________________________________")
    print(f"Honeycomb lattice extent: {extent}")
    print("Sites:", graph.n_nodes)
    print("Edges:", graph.n_edges)
    print("Neighbor orders:", sorted(set(color for *_ij, color in graph.edges(return_color=True))))
    print("Site type ids:", token_site_type_ids)
    print("Number of site types:", num_site_types)
    print("Relation matrix shape:", (len(relation_matrix), len(relation_matrix[0])))
    print("Number of relation types:", num_relation_types)
    print()

    exact_gs = None
    if num_sites <= 24:
        sp_h = ha.to_sparse()
        eig_vals, _ = eigsh(sp_h, k=2, which="SA")
        exact_gs = float(eig_vals[0])
        print("Exact ground-state energy:", exact_gs)
        print()

    if sampler_name == "pt_exchange":
        sampler = nk.sampler.ParallelTemperingExchange(hi, graph=graph, d_max=d_max)
    elif sampler_name == "local":
        n_chains = max(1, num_samples // 32)
        sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
    elif sampler_name == "exact":
        sampler = nk.sampler.ExactSampler(hi)
    else:
        raise ValueError(f"Unsupported sampler_name={sampler_name!r}")

    def build_optimizer(lr: float):
        if optimizer_name == "sgd":
            return optax.sgd(learning_rate=lr, momentum=momentum)
        if optimizer_name == "adam":
            return optax.adam(learning_rate=lr)
        raise ValueError(f"Unsupported optimizer_name={optimizer_name!r}")

    warm_log_file = None
    if num_iters_warm > 0:
        print("\n=== Stage 1: amplitude-only warm start ===\n")

        model_warm = HoneycombSiteTypeRelationViT(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_hidden_dim=mlp_hidden,
            patch_size=patch_size,
            learn_phase=False,
            relation_matrix=relation_matrix,
            site_type_ids=token_site_type_ids,
            permutation=perm,
        )

        vstate_1 = nk.vqs.MCState(
            sampler=sampler,
            model=model_warm,
            n_samples=num_samples,
            chunk_size=chunk_size,
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
        model_init = HoneycombSiteTypeRelationViT(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_hidden_dim=mlp_hidden,
            patch_size=patch_size,
            learn_phase=learn_phase_main,
            relation_matrix=relation_matrix,
            site_type_ids=token_site_type_ids,
            permutation=perm,
        )
        vstate_init = nk.vqs.MCState(
            sampler=sampler,
            model=model_init,
            n_samples=num_samples,
            chunk_size=chunk_size,
        )
        print("num_samples:", num_samples)
        print("num params:", vstate_init.n_parameters)
        print("chain_length:", vstate_init.chain_length)
        print("n_discard_per_chain:", vstate_init.n_discard_per_chain)
        initial_variables = vstate_init.variables

    print("\n=== Stage 2: full complex optimization ===\n")

    model_full = HoneycombSiteTypeRelationViT(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_hidden_dim=mlp_hidden,
        patch_size=patch_size,
        learn_phase=learn_phase_main,
        relation_matrix=relation_matrix,
        site_type_ids=token_site_type_ids,
        permutation=perm,
    )

    vstate_2 = nk.vqs.MCState(
        sampler=sampler,
        model=model_full,
        n_samples=num_samples,
        variables=initial_variables,
        chunk_size=chunk_size,
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
    driver_2.run(
        n_iter=num_iters,
        obs=obs,
        out=f"out_{job_name}",
        save_params_every=10,
    )

    with open(f"out_{job_name}.log") as f:
        data = json.load(f)
    energy = _extract_real_series(data, "Energy")

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

    observables = expectations.define_observables(num_sites, hi)
    expectations.calculate_expectations(vstate_2, ha, observables)

    summary = {
        "job_name": job_name,
        "num_sites": num_sites,
        "extent": extent,
        "num_samples": num_samples,
        "num_iters_warm": num_iters_warm,
        "num_iters": num_iters,
        "chunk_size": chunk_size,
        "j1": j1,
        "j2": j2,
        "patch_size": patch_size,
        "permutation": list(perm),
        "site_type_ids": list(token_site_type_ids),
        "num_site_types": num_site_types,
        "relation_matrix": [list(row) for row in relation_matrix],
        "num_relation_types": num_relation_types,
        "site_type_relation_model": True,
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
        "sampler_name": sampler_name,
        "d_max": d_max,
        "optimizer_name": optimizer_name,
        "final_energy": float(energy[-1]),
        "best_energy_seen": float(min(energy)),
        "exact_ground_state_energy": exact_gs,
        "mean_energy_file": f"mean_energy_run_{job_name}.txt",
        "log_file": f"out_{job_name}.log",
        "warm_log_file": warm_log_file,
        "plot_file": f"energy_{job_name}.png",
        "plot_log_file": f"energy_log_{job_name}.png",
    }

    with open(f"summary_{job_name}.json", "w") as f:
        json.dump(summary, f, indent=2)
