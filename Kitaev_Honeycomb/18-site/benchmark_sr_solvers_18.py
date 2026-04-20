import json
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import jax
import netket as nk
from netket.optimizer.solver import pinv_smooth
import numpy as np
import optax
from scipy.sparse.linalg import eigsh

from define_Kitaev_Hamiltonian import kitaev_hamiltonian
from kitaev_honeycomb_vit_model import HoneycombPatchViT


jax.config.update("jax_enable_x64", True)


NUM_SITES = 18
LEARN_PHASE = True

EMBED_DIM = 20
NUM_HEADS = 4
NUM_LAYERS = 2
PATCH_SIZE = 3
MLP_HIDDEN_DIM = 2 * EMBED_DIM

NUM_SAMPLES_WARMUP = 3 * 2**10
NUM_SAMPLES_PT = 3 * 2**10
TARGET_CHAIN_LENGTH = 32
N_DISCARD_PER_CHAIN = 8
PT_SWEEP_SIZE = NUM_SITES * 3

NUM_ITERS_WARM = 40
NUM_ITERS_PT = 30
LOG_STEP_SIZE = 1
WRITE_EVERY = 1
SAVE_PARAMS_EVERY = 25

CHUNK_SIZE = 2**6
CHUNK_SIZE_BWD = 2**6

WARM_SR_LR = 1.5e-2
WARM_SR_MOMENTUM = 0.9
WARM_SR_DIAGSHIFT = 1e-2

PT_SR_LR = 5e-3
PT_SR_MOMENTUM = 0.9
PT_SR_DIAGSHIFT = 1e-3

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR / "benchmark_runs"
BENCH_DIR.mkdir(parents=True, exist_ok=True)
RUN_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = BENCH_DIR / f"solver_benchmark_{RUN_STAMP}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("NETKET_DEBUG", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


def bfs_ordering_kitaev(graph):
    adj = [[] for _ in range(graph.n_nodes)]
    for i, j, color in graph.edges(return_color=True):
        adj[i].append((j, color))
        adj[j].append((i, color))

    for neighbors in adj:
        neighbors.sort(key=lambda item: (item[1], item[0]))

    visited = [False] * graph.n_nodes
    order = []
    queue = [0]
    visited[0] = True

    while queue:
        u = queue.pop(0)
        order.append(u)
        for v, _color in adj[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)

    for node in range(graph.n_nodes):
        if not visited[node]:
            order.append(node)

    return tuple(order)


def make_n_chains(n_samples):
    n_chains = max(3, n_samples // TARGET_CHAIN_LENGTH)
    n_chains -= n_chains % 3
    return max(3, n_chains)


def make_metropolis_local(hilbert, n_samples):
    return nk.sampler.MetropolisLocal(
        hilbert=hilbert,
        n_chains=make_n_chains(n_samples),
    )


def make_parallel_tempering_local(hilbert, n_samples):
    return nk.sampler.ParallelTemperingLocal(
        hilbert=hilbert,
        n_chains=make_n_chains(n_samples),
        sweep_size=PT_SWEEP_SIZE,
    )


def build_model(perm):
    return HoneycombPatchViT(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        patch_size=PATCH_SIZE,
        learn_phase=LEARN_PHASE,
        permutation=perm,
    )


def build_driver(ha, variational_state, optimizer, momentum, linear_solver, diag_shift):
    return nk.driver.VMC_SR(
        hamiltonian=ha,
        optimizer=optimizer,
        variational_state=variational_state,
        momentum=momentum,
        linear_solver=linear_solver,
        mode="complex",
        diag_shift=diag_shift,
        use_ntk=True,
        on_the_fly=True,
        chunk_size_bwd=CHUNK_SIZE_BWD,
    )


def block_until_ready_tree(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def read_tail_metrics(log_path):
    with open(log_path) as f:
        data = json.load(f)
    energies = np.asarray(data["Energy"]["Mean"]["real"], dtype=float)
    tail = energies[-min(5, len(energies)) :]
    return {
        "n_logged": int(len(energies)),
        "final_energy": float(energies[-1]),
        "tail_mean_5": float(np.mean(tail)),
        "best_energy": float(np.min(energies)),
    }


def benchmark_solver(
    *,
    solver_name,
    linear_solver,
    ha,
    hi,
    perm,
    exact_energy,
    base_variables,
):
    run_stats = {"solver_name": solver_name}

    warm_sampler = make_metropolis_local(hi, NUM_SAMPLES_WARMUP)
    warm_vstate = nk.vqs.MCState(
        sampler=warm_sampler,
        model=build_model(perm),
        n_samples=NUM_SAMPLES_WARMUP,
        variables=deepcopy(base_variables),
        chunk_size=CHUNK_SIZE,
        n_discard_per_chain=N_DISCARD_PER_CHAIN,
    )
    block_until_ready_tree(warm_vstate.variables)

    warm_driver = build_driver(
        ha=ha,
        variational_state=warm_vstate,
        optimizer=optax.sgd(learning_rate=WARM_SR_LR),
        momentum=WARM_SR_MOMENTUM,
        linear_solver=linear_solver,
        diag_shift=WARM_SR_DIAGSHIFT,
    )
    warm_out = RUN_DIR / f"{solver_name}_warm"
    t0 = time.perf_counter()
    warm_driver.run(
        n_iter=NUM_ITERS_WARM,
        out=str(warm_out),
        step_size=LOG_STEP_SIZE,
        write_every=WRITE_EVERY,
        save_params_every=SAVE_PARAMS_EVERY,
        timeit=False,
    )
    block_until_ready_tree(warm_vstate.variables)
    warm_elapsed = time.perf_counter() - t0
    warm_metrics = read_tail_metrics(warm_out.with_suffix(".log"))
    warm_metrics["wall_time_s"] = warm_elapsed
    warm_metrics["sec_per_iter"] = warm_elapsed / NUM_ITERS_WARM
    warm_metrics["gap_to_exact"] = warm_metrics["final_energy"] - exact_energy
    warm_metrics["n_chains"] = int(warm_vstate.sampler.n_chains)
    warm_metrics["chain_length"] = int(warm_vstate.chain_length)
    run_stats["warmup"] = warm_metrics

    pt_sampler = make_parallel_tempering_local(hi, NUM_SAMPLES_PT)
    pt_vstate = nk.vqs.MCState(
        sampler=pt_sampler,
        model=build_model(perm),
        n_samples=NUM_SAMPLES_PT,
        variables=deepcopy(base_variables),
        chunk_size=CHUNK_SIZE,
        n_discard_per_chain=N_DISCARD_PER_CHAIN,
    )
    block_until_ready_tree(pt_vstate.variables)

    pt_driver = build_driver(
        ha=ha,
        variational_state=pt_vstate,
        optimizer=optax.sgd(learning_rate=PT_SR_LR),
        momentum=PT_SR_MOMENTUM,
        linear_solver=linear_solver,
        diag_shift=PT_SR_DIAGSHIFT,
    )
    pt_out = RUN_DIR / f"{solver_name}_pt"
    t0 = time.perf_counter()
    pt_driver.run(
        n_iter=NUM_ITERS_PT,
        out=str(pt_out),
        step_size=LOG_STEP_SIZE,
        write_every=WRITE_EVERY,
        save_params_every=SAVE_PARAMS_EVERY,
        timeit=False,
    )
    block_until_ready_tree(pt_vstate.variables)
    pt_elapsed = time.perf_counter() - t0
    pt_metrics = read_tail_metrics(pt_out.with_suffix(".log"))
    pt_metrics["wall_time_s"] = pt_elapsed
    pt_metrics["sec_per_iter"] = pt_elapsed / NUM_ITERS_PT
    pt_metrics["gap_to_exact"] = pt_metrics["final_energy"] - exact_energy
    pt_metrics["n_chains"] = int(pt_vstate.sampler.n_chains)
    pt_metrics["chain_length"] = int(pt_vstate.chain_length)
    run_stats["parallel_tempering"] = pt_metrics

    return run_stats


def main():
    print("devices", jax.devices())
    print("benchmark_run_dir", RUN_DIR)

    graph, _symm_group, hi, ha = kitaev_hamiltonian(NUM_SITES)
    exact_energy = None
    sp_h = ha.to_sparse()
    eig_vals, _ = eigsh(sp_h, k=2, which="SA")
    exact_energy = float(eig_vals[0])
    print("exact_ground_state_energy", exact_energy)

    perm = bfs_ordering_kitaev(graph)
    print("permutation", perm)

    init_vstate = nk.vqs.MCState(
        sampler=make_metropolis_local(hi, NUM_SAMPLES_WARMUP),
        model=build_model(perm),
        n_samples=NUM_SAMPLES_WARMUP,
        chunk_size=CHUNK_SIZE,
        n_discard_per_chain=N_DISCARD_PER_CHAIN,
        seed=1234,
    )
    base_variables = deepcopy(init_vstate.variables)
    print("n_parameters", init_vstate.n_parameters)

    results = []
    for solver_name, linear_solver in (("pinv_smooth", pinv_smooth),):
        print(f"=== Benchmarking {solver_name} ===")
        results.append(
            benchmark_solver(
                solver_name=solver_name,
                linear_solver=linear_solver,
                ha=ha,
                hi=hi,
                perm=perm,
                exact_energy=exact_energy,
                base_variables=base_variables,
            )
        )

    summary = {
        "devices": [str(device) for device in jax.devices()],
        "run_dir": str(RUN_DIR),
        "num_sites": NUM_SITES,
        "num_samples_warmup": NUM_SAMPLES_WARMUP,
        "num_samples_pt": NUM_SAMPLES_PT,
        "num_iters_warm": NUM_ITERS_WARM,
        "num_iters_pt": NUM_ITERS_PT,
        "chunk_size": CHUNK_SIZE,
        "chunk_size_bwd": CHUNK_SIZE_BWD,
        "target_chain_length": TARGET_CHAIN_LENGTH,
        "n_discard_per_chain": N_DISCARD_PER_CHAIN,
        "exact_ground_state_energy": exact_energy,
        "results": results,
    }

    summary_path = RUN_DIR / "summary_solver_benchmark.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
