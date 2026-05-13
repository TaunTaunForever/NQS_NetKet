from __future__ import annotations

import json
import os
from datetime import date

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
    num_samples: int = 2**10,
    num_iters_warm: int = 200,
    num_iters: int = 10_000,
    chunk_size: int | None = 2**9,
    patch_size: int = 1,
    embed_dim: int = 32,
    num_heads: int = 4,
    num_layers: int = 4,
    mlp_hidden: int | None = None,
):
    today = date.today().isoformat()
    mlp_hidden = 2 * embed_dim if mlp_hidden is None else mlp_hidden
    job_name = (
        f"Heisenberg_{num_sites}-site_"
        f"{num_layers}L_{num_heads}H_{patch_size}p_"
        f"{num_samples}_samples_{today}_Honeycomb_ViT_SRt_site_type_relation"
    )

    graph, extent, hi, ha = make_heisenberg_hamiltonian(num_sites)
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

    print(f"Defining {num_sites}-site Heisenberg Honeycomb lattice")
    print("___________________________________________________")
    print(f"Honeycomb lattice extent: {extent}")
    print("Sites:", graph.n_nodes)
    print("Edges:", graph.n_edges)
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

    sampler = nk.sampler.ParallelTemperingExchange(hi, graph=graph, d_max=1)
    optimizer = optax.sgd(learning_rate=1e-2, momentum=0.9)

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
        optimizer=optimizer,
        variational_state=vstate_1,
        diag_shift=1e-3,
    )
    driver_1.run(
        n_iter=num_iters_warm,
        out=f"out_warm_{job_name}",
        save_params_every=10,
    )

    print("\n=== Stage 2: full complex optimization ===\n")

    model_full = HoneycombSiteTypeRelationViT(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_hidden_dim=mlp_hidden,
        patch_size=patch_size,
        learn_phase=True,
        relation_matrix=relation_matrix,
        site_type_ids=token_site_type_ids,
        permutation=perm,
    )

    vstate_2 = nk.vqs.MCState(
        sampler=sampler,
        model=model_full,
        n_samples=num_samples,
        variables=vstate_1.variables,
        chunk_size=chunk_size,
    )

    obs = {
        "<X>": sum(nk.operator.spin.sigmax(hi, i) for i in range(hi.size)) / (2 * num_sites),
        "<Y>": sum(nk.operator.spin.sigmay(hi, i) for i in range(hi.size)) / (2 * num_sites),
        "<Z>": sum(nk.operator.spin.sigmaz(hi, i) for i in range(hi.size)) / (2 * num_sites),
    }

    driver_2 = nk.driver.VMC_SR(
        hamiltonian=ha,
        optimizer=optimizer,
        variational_state=vstate_2,
        diag_shift=1e-3,
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
    plt.title(f"Heisenberg Honeycomb {num_sites}-site (ViT SRt)")
    plt.tight_layout()
    plt.savefig(f"energy_{job_name}.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(energy)
    plt.xscale("log")
    plt.xlabel("Iteration (log)")
    plt.ylabel("Energy")
    plt.title(f"Heisenberg Honeycomb {num_sites}-site (ViT SRt)")
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
        "final_energy": float(energy[-1]),
        "best_energy_seen": float(min(energy)),
        "exact_ground_state_energy": exact_gs,
        "mean_energy_file": f"mean_energy_run_{job_name}.txt",
        "log_file": f"out_{job_name}.log",
        "warm_log_file": f"out_warm_{job_name}.log",
        "plot_file": f"energy_{job_name}.png",
        "plot_log_file": f"energy_log_{job_name}.png",
    }

    with open(f"summary_{job_name}.json", "w") as f:
        json.dump(summary, f, indent=2)
