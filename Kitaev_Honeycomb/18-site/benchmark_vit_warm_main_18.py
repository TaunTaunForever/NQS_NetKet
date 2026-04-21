import json
import os
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import netket as nk
import numpy as np
import optax
from netket.optimizer.solver import pinv_smooth
from scipy.sparse.linalg import eigsh

from hamiltonian import kitaev_hamiltonian
from vit_model import HoneycombPatchViT


jax.config.update("jax_enable_x64", True)
os.environ.setdefault("NETKET_DEBUG", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


LEARN_PHASE = True
NUM_SITES = 18
NUM_HEADS = 4
NUM_LAYERS = 2
PATCH_SIZE = 3
USE_NTK = True
ON_THE_FLY = True
CHUNK_SIZE = 2**6
CHUNK_SIZE_BWD = 2**6
N_DISCARD_PER_CHAIN = 8
TARGET_CHAIN_LENGTH = 32
LOG_STEP_SIZE = 1
WRITE_EVERY = 1
SAVE_PARAMS_EVERY = 25
WARM_SELECTION_TAIL_POINTS = 5

WARM_SR_LR = 1.5e-2
WARM_SR_MOMENTUM = 0.9
WARM_SR_DIAGSHIFT = 1e-2
MAIN_SR_LR = 5e-3
MAIN_SR_MOMENTUM = 0.9
MAIN_SR_DIAGSHIFT = 1e-3

MAIN_SEGMENT_ITERS = 20
PLATEAU_MIN_ITERS = 120
PLATEAU_WINDOW = 60
PLATEAU_MIN_IMPROVEMENT = 0.03
PLATEAU_MAX_STD = 0.03
PLATEAU_MAX_GAP = 0.9


CONFIGS = [
    {
        "name": "e24_w768_m1536_s8",
        "embed_dim": 24,
        "num_samples_warmup": 768,
        "num_samples_main": 1536,
        "num_starts": 8,
        "num_iters_warm": 150,
        "num_iters_main": 240,
    },
    {
        "name": "e24_w1536_m1536_s6",
        "embed_dim": 24,
        "num_samples_warmup": 1536,
        "num_samples_main": 1536,
        "num_starts": 6,
        "num_iters_warm": 180,
        "num_iters_main": 240,
    },
    {
        "name": "e32_w768_m1536_s6",
        "embed_dim": 32,
        "num_samples_warmup": 768,
        "num_samples_main": 1536,
        "num_starts": 6,
        "num_iters_warm": 150,
        "num_iters_main": 240,
    },
]


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR / "benchmark_runs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_warm_main")
RUN_ROOT.mkdir(parents=True, exist_ok=True)


def bfs_ordering_kitaev(graph):
    adj = [[] for _ in range(graph.n_nodes)]
    for i, j, color in graph.edges(return_color=True):
        adj[i].append((j, color))
        adj[j].append((i, color))

    for neighbors in adj:
        neighbors.sort(key=lambda item: (item[1], item[0]))

    visited = [False] * graph.n_nodes
    order = []
    queue = deque([0])
    visited[0] = True

    while queue:
        u = queue.popleft()
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


def make_parallel_tempering_local(hilbert, n_samples, sweep_size):
    return nk.sampler.ParallelTemperingLocal(
        hilbert=hilbert,
        n_chains=make_n_chains(n_samples),
        sweep_size=sweep_size,
    )


def build_model(perm, embed_dim):
    return HoneycombPatchViT(
        embed_dim=embed_dim,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_hidden_dim=2 * embed_dim,
        patch_size=PATCH_SIZE,
        learn_phase=LEARN_PHASE,
        permutation=perm,
    )


def build_driver(ha, variational_state, optimizer, momentum, diag_shift):
    return nk.driver.VMC_SR(
        hamiltonian=ha,
        optimizer=optimizer,
        variational_state=variational_state,
        momentum=momentum,
        linear_solver=pinv_smooth,
        mode="complex",
        diag_shift=diag_shift,
        use_ntk=USE_NTK,
        on_the_fly=ON_THE_FLY,
        chunk_size_bwd=CHUNK_SIZE_BWD,
    )


def load_energy_trace(log_path):
    with open(log_path) as f:
        data = json.load(f)
    return np.asarray(data["Energy"]["Mean"]["real"], dtype=float)


def should_stop_for_plateau(energies, exact_energy):
    if len(energies) < PLATEAU_MIN_ITERS:
        return False, {}

    recent = np.asarray(energies[-PLATEAU_WINDOW:], dtype=float)
    best_recent = float(np.min(recent))
    best_previous = float(np.min(np.asarray(energies[:-PLATEAU_WINDOW], dtype=float)))
    recent_std = float(np.std(recent))
    improvement = best_previous - best_recent
    gap = recent[-1] - exact_energy

    stop = (
        improvement < PLATEAU_MIN_IMPROVEMENT
        and recent_std < PLATEAU_MAX_STD
        and gap > PLATEAU_MAX_GAP
    )
    return stop, {
        "best_previous": best_previous,
        "best_recent": best_recent,
        "improvement": improvement,
        "recent_std": recent_std,
        "gap_to_exact": gap,
    }


def run_config(cfg, hi, ha, perm, exact_energy):
    run_dir = RUN_ROOT / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    sweep_size = NUM_SITES * 3

    print(f"\n=== Config {cfg['name']} ===")
    print(json.dumps(cfg, indent=2))

    warm_summaries = []
    for start_idx in range(cfg["num_starts"]):
        print(f"--- Warmup start {start_idx + 1}/{cfg['num_starts']} ---")
        vstate_warm = nk.vqs.MCState(
            sampler=make_metropolis_local(hi, cfg["num_samples_warmup"]),
            model=build_model(perm, cfg["embed_dim"]),
            n_samples=cfg["num_samples_warmup"],
            chunk_size=CHUNK_SIZE,
            n_discard_per_chain=N_DISCARD_PER_CHAIN,
            seed=1234 + start_idx,
        )
        print("Parameters:", vstate_warm.n_parameters)
        print("n_chains:", vstate_warm.sampler.n_chains)
        print("chain_length:", vstate_warm.chain_length)

        driver_warm = build_driver(
            ha=ha,
            variational_state=vstate_warm,
            optimizer=optax.sgd(learning_rate=WARM_SR_LR),
            momentum=WARM_SR_MOMENTUM,
            diag_shift=WARM_SR_DIAGSHIFT,
        )
        out_warm = run_dir / f"out_{cfg['name']}_warm_start_{start_idx}"
        driver_warm.run(
            n_iter=cfg["num_iters_warm"],
            out=str(out_warm),
            step_size=LOG_STEP_SIZE,
            write_every=WRITE_EVERY,
            save_params_every=SAVE_PARAMS_EVERY,
            timeit=False,
        )
        energies = load_energy_trace(out_warm.with_suffix(".log"))
        tail = energies[-min(WARM_SELECTION_TAIL_POINTS, len(energies)) :]
        tail_mean = float(np.mean(tail))
        warm_summary = {
            "start_idx": start_idx,
            "tail_mean": tail_mean,
            "final_energy": float(energies[-1]),
            "best_energy": float(np.min(energies)),
            "gap_to_exact": float(energies[-1] - exact_energy),
            "variables": deepcopy(vstate_warm.variables),
            "log_path": str(out_warm.with_suffix(".log")),
        }
        warm_summaries.append(warm_summary)
        print(
            "Warmup tail mean:",
            warm_summary["tail_mean"],
            "best:",
            warm_summary["best_energy"],
            "gap:",
            warm_summary["gap_to_exact"],
        )

    best_warm = min(warm_summaries, key=lambda row: row["tail_mean"])
    print("Selected warm start:", best_warm["start_idx"], "tail mean:", best_warm["tail_mean"])

    vstate_main = nk.vqs.MCState(
        sampler=make_parallel_tempering_local(hi, cfg["num_samples_main"], sweep_size),
        model=build_model(perm, cfg["embed_dim"]),
        n_samples=cfg["num_samples_main"],
        variables=deepcopy(best_warm["variables"]),
        chunk_size=CHUNK_SIZE,
        n_discard_per_chain=N_DISCARD_PER_CHAIN,
    )
    print("Main parameters:", vstate_main.n_parameters)
    print("Main n_chains:", vstate_main.sampler.n_chains)
    print("Main chain_length:", vstate_main.chain_length)

    driver_main = build_driver(
        ha=ha,
        variational_state=vstate_main,
        optimizer=optax.sgd(learning_rate=MAIN_SR_LR),
        momentum=MAIN_SR_MOMENTUM,
        diag_shift=MAIN_SR_DIAGSHIFT,
    )

    main_energies = []
    chunk_paths = []
    plateau_info = None
    iters_done = 0
    chunk_idx = 0

    while iters_done < cfg["num_iters_main"]:
        n_chunk = min(MAIN_SEGMENT_ITERS, cfg["num_iters_main"] - iters_done)
        out_main = run_dir / f"out_{cfg['name']}_main_chunk_{chunk_idx:03d}"
        driver_main.run(
            n_iter=n_chunk,
            out=str(out_main),
            step_size=LOG_STEP_SIZE,
            write_every=WRITE_EVERY,
            save_params_every=SAVE_PARAMS_EVERY,
            timeit=False,
        )
        energies = load_energy_trace(out_main.with_suffix(".log"))
        main_energies.extend(energies.tolist())
        chunk_paths.append(str(out_main.with_suffix(".log")))
        iters_done += n_chunk
        chunk_idx += 1

        best_seen = float(np.min(np.asarray(main_energies, dtype=float)))
        print(
            f"Main iters {iters_done}/{cfg['num_iters_main']}:",
            "latest =", float(main_energies[-1]),
            "best =", best_seen,
            "gap =", float(main_energies[-1] - exact_energy),
        )

        stop, info = should_stop_for_plateau(main_energies, exact_energy)
        if stop:
            plateau_info = info
            print("Stopping early for plateau:", json.dumps(info, indent=2))
            break

    result = {
        "config": cfg,
        "selected_warm_start": best_warm["start_idx"],
        "selected_warm_tail_mean": best_warm["tail_mean"],
        "selected_warm_best_energy": best_warm["best_energy"],
        "selected_warm_final_energy": best_warm["final_energy"],
        "main_iters_completed": iters_done,
        "main_final_energy": float(main_energies[-1]),
        "main_best_energy": float(np.min(np.asarray(main_energies, dtype=float))),
        "main_gap_to_exact": float(main_energies[-1] - exact_energy),
        "plateau_stopped": plateau_info is not None,
        "plateau_info": plateau_info,
        "warmup_summaries": [
            {k: v for k, v in row.items() if k != "variables"} for row in warm_summaries
        ],
        "main_chunk_logs": chunk_paths,
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    print("devices", jax.devices())
    print("run_root", RUN_ROOT)

    graph, _symm_group, hi, ha = kitaev_hamiltonian(NUM_SITES)
    sp_h = ha.to_sparse()
    eig_vals, _ = eigsh(sp_h, k=2, which="SA")
    exact_energy = float(eig_vals[0])
    print("exact_ground_state_energy", exact_energy)

    perm = bfs_ordering_kitaev(graph)
    print("permutation", perm)

    results = []
    for cfg in CONFIGS:
        results.append(run_config(cfg, hi, ha, perm, exact_energy))

    with open(RUN_ROOT / "all_results.json", "w") as f:
        json.dump(
            {
                "exact_ground_state_energy": exact_energy,
                "results": results,
            },
            f,
            indent=2,
        )

    print("\n=== Sweep Summary ===")
    for row in results:
        print(
            row["config"]["name"],
            "warm_tail =", row["selected_warm_tail_mean"],
            "main_best =", row["main_best_energy"],
            "main_final =", row["main_final_energy"],
            "plateau =", row["plateau_stopped"],
        )


if __name__ == "__main__":
    main()
