import os
import json
from datetime import date
from pathlib import Path
from typing import List
from collections import deque

SHOW_PLOTS = False
if not SHOW_PLOTS:
    os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import netket as nk
from netket.optimizer.solver import pinv_smooth
import optax

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

from define_Kitaev_Hamiltonian import kitaev_hamiltonian
from kitaev_honeycomb_vit_symm_model import SymmetryProjectedHoneycombViT

print(jax.devices())

jax.config.update("jax_enable_x64", True)

LEARN_PHASE = True

NUM_SITES = 8
USE_BFS_ORDERING = os.environ.get("USE_BFS_ORDERING", "0") == "1"

NUM_SAMPLES_WARMUP = 3 * 2**10
NUM_SAMPLES = 3 * 2**10

# Keep SR-only optimisation here; in this project the Adam -> SR handoff
# has been less reliable than staying in SR throughout.
NUM_STARTS = 1
NUM_ITERS_WARM = 1000
NUM_ITERS_MAIN = 0
NUM_ITERS_SR = 1000

EMBED_DIM = 10
NUM_HEADS = 5
NUM_LAYERS = 2
PATCH_SIZE = 1
MLP_HIDDEN_DIM = 2 * EMBED_DIM
CHUNK_SIZE = None

WARM_SR_LR = 1.5e-2
WARM_SR_MOMENTUM = 0.9
WARM_SR_DIAGSHIFT = 1e-3
WARM_SR_SOLVER = pinv_smooth

MAIN_SR_LR = 1e-2
MAIN_SR_MOMENTUM = 0.9
MAIN_SR_DIAGSHIFT = 1e-4
MAIN_SR_SOLVER = pinv_smooth

SR_LR = 1e-3
SR_MOMENTUM = 0.9
SR_SOLVER = pinv_smooth
SR_DIAGSHIFT_SCHEDULE = [
    (0.30, 1e-2, "sr_shift1e-2"),
    (0.40, 5e-4, "sr_shift5e-4"),
    (0.30, 2e-4, "sr_shift2e-4"),
]

# Use sample counts divisible by 3 so work can be split evenly across
# the three CUDA devices available on this machine.
N_DISCARD_PER_CHAIN = 64

TODAY = date.today().isoformat()

ordering_tag = "bfs" if USE_BFS_ORDERING else "identity"
JOB_BASE = (
    f"{NUM_SITES}-site_"
    f"{NUM_LAYERS}_layers_"
    f"{NUM_HEADS}_heads_"
    f"{PATCH_SIZE}_patches_"
    f"{ordering_tag}_"
    f"{NUM_SAMPLES}_samples_"
    f"{TODAY}_Kitaev_ViT_SRonly_symmproj_local"
)

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs" / TODAY
RUN_DIR = RUNS_DIR / JOB_BASE
RUN_DIR.mkdir(parents=True, exist_ok=True)

os.environ["NETKET_DEBUG"] = "1"

graph, symm_group, hi, ha = kitaev_hamiltonian(NUM_SITES)
point_group = graph.point_group

sp_h = ha.to_sparse()
eig_vals, _ = eigsh(sp_h, k=2, which="SA")
print("Exact ground-state energy:", eig_vals[0])
print()


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


if USE_BFS_ORDERING:
    perm = bfs_ordering_kitaev(graph)
else:
    perm = tuple(range(graph.n_nodes))
symmetry_perms = tuple(
    tuple(g.inverse_permutation_array.tolist()) for g in symm_group
)

print("Permutation:", perm)
print("Using BFS ordering:", USE_BFS_ORDERING)
print("Full automorphism symmetry count:", len(symmetry_perms))
print("Run directory:", RUN_DIR)


def make_metropolis_local(hilbert, n_samples):
    n_chains = max(1, n_samples // 64)
    return nk.sampler.MetropolisLocal(
        hilbert=hilbert,
        n_chains=n_chains,
    )

def make_parallel_tempering_local(hilbert, n_samples):
    n_chains = max(1, n_samples // 64)
    return nk.sampler.ParallelTemperingLocal(
        hilbert=hilbert,
        n_chains=n_chains,
        sweep_size=36,
    )


def build_model():
    return SymmetryProjectedHoneycombViT(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        patch_size=PATCH_SIZE,
        learn_phase=LEARN_PHASE,
        symmetries=symmetry_perms,
        permutation=perm,
    )


def load_best_energy(logfile):
    with open(logfile) as f:
        data = json.load(f)
    energies = np.array(data["Energy"]["Mean"]["real"], dtype=float)
    tail = energies[-min(20, len(energies)):]
    return float(np.mean(tail)), list(tail)


def run_path(stem: str) -> str:
    return str(RUN_DIR / stem)


all_outs: List[str] = []
print(
    f"\n=== Multi-start SR warmup: {NUM_STARTS} starts, "
    f"{NUM_ITERS_WARM} iterations each ===\n"
)

warmup_summaries = []

for start_idx in range(NUM_STARTS):
    print(f"--- Warmup start {start_idx + 1}/{NUM_STARTS} ---")

    model_warm = build_model()
    sampler_warm = make_metropolis_local(hi, NUM_SAMPLES_WARMUP)

    vstate_warm = nk.vqs.MCState(
        sampler=sampler_warm,
        model=model_warm,
        n_samples=NUM_SAMPLES_WARMUP,
        chunk_size=CHUNK_SIZE,
        n_discard_per_chain=N_DISCARD_PER_CHAIN,
        seed=1234 + start_idx,
    )

    print("Parameters:", vstate_warm.n_parameters)
    print("n_chains:", vstate_warm.sampler.n_chains)
    print("chain_length:", vstate_warm.chain_length)

    warm_opt = optax.sgd(learning_rate=WARM_SR_LR)

    driver_warm = nk.driver.VMC_SR(
        hamiltonian=ha,
        optimizer=warm_opt,
        variational_state=vstate_warm,
        momentum=WARM_SR_MOMENTUM,
        linear_solver=WARM_SR_SOLVER,
        mode="complex",
        diag_shift=WARM_SR_DIAGSHIFT,
    )

    out_warm = run_path(f"out_{JOB_BASE}_warm_start_{start_idx}")

    driver_warm.run(
        n_iter=NUM_ITERS_WARM,
        out=out_warm,
        save_params_every=10,
    )
    all_outs.append(out_warm)

    warmup_energy, _ = load_best_energy(f"{out_warm}.log")
    print("Warmup energy:", warmup_energy)
    print("Warmup gap to exact:", warmup_energy - float(eig_vals[0]))
    print()

    warmup_summaries.append(
        {
            "start": start_idx,
            "warmup_energy": warmup_energy,
            "variables": vstate_warm.variables,
            "out": out_warm,
        }
    )

best_warmup = min(warmup_summaries, key=lambda row: row["warmup_energy"])

print("Selected warmup start:", best_warmup["start"])
print("Selected warmup energy:", best_warmup["warmup_energy"])
print()

print(f"\n=== Final SR refinement for {NUM_ITERS_SR} iterations ===")

sampler_sr = make_parallel_tempering_local(hi, NUM_SAMPLES)

vstate_sr = nk.vqs.MCState(
    sampler=sampler_sr,
    model=build_model(),
    n_samples=NUM_SAMPLES,
    variables=best_warmup["variables"],
    chunk_size=CHUNK_SIZE,
    n_discard_per_chain=N_DISCARD_PER_CHAIN,
)

print("Parameters:", vstate_sr.n_parameters)
print("n_chains:", vstate_sr.sampler.n_chains)
print("chain_length:", vstate_sr.chain_length)

sr_opt = optax.sgd(learning_rate=SR_LR)


def run_sr_segment(n_iter, diag_shift, tag):
    driver_sr = nk.driver.VMC_SR(
        hamiltonian=ha,
        optimizer=sr_opt,
        variational_state=vstate_sr,
        momentum=SR_MOMENTUM,
        linear_solver=SR_SOLVER,
        mode="complex",
        diag_shift=diag_shift,
    )
    out = run_path(f"out_{JOB_BASE}_{tag}")
    driver_sr.run(n_iter=n_iter, out=out, save_params_every=10)
    return out


remaining = NUM_ITERS_SR
for frac, shift, tag in SR_DIAGSHIFT_SCHEDULE:
    n_seg = int(round(frac * NUM_ITERS_SR))
    n_seg = max(1, min(n_seg, remaining))
    remaining -= n_seg
    all_outs.append(run_sr_segment(n_seg, shift, tag))
    if remaining <= 0:
        break

if remaining > 0:
    last_shift = SR_DIAGSHIFT_SCHEDULE[-1][1]
    all_outs.append(run_sr_segment(remaining, last_shift, "sr_tail"))

energy = []
for out in all_outs:
    logf = f"{out}.log"
    if os.path.exists(logf):
        with open(logf) as f:
            data = json.load(f)
            energy.extend(data["Energy"]["Mean"]["real"])

mean_energy_file = RUN_DIR / f"mean_energy_run_{JOB_BASE}.txt"
with open(mean_energy_file, "w") as f:
    for e in energy:
        f.write(f"{e}\n")

plt.figure(figsize=(10, 6))
plt.plot(energy)
plt.xscale("log")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title(f"{NUM_SITES}-site Kitaev (SR warmup -> PT-SR)")
plt.tight_layout()
plot_file = RUN_DIR / f"kitaev_vit_{JOB_BASE}.png"
plt.savefig(plot_file)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

summary = {
    "job_base": JOB_BASE,
    "run_dir": str(RUN_DIR),
    "num_sites": NUM_SITES,
    "use_bfs_ordering": USE_BFS_ORDERING,
    "num_starts": NUM_STARTS,
    "num_samples_warmup": NUM_SAMPLES_WARMUP,
    "num_samples": NUM_SAMPLES,
    "num_iters_warm": NUM_ITERS_WARM,
    "num_iters_main": NUM_ITERS_MAIN,
    "num_iters_sr": NUM_ITERS_SR,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "mlp_hidden_dim": MLP_HIDDEN_DIM,
    "patch_size": PATCH_SIZE,
    "sr_solver_name": "pinv_smooth",
    "permutation": list(perm),
    "outputs": all_outs,
    "mean_energy_file": str(mean_energy_file),
    "plot_file": str(plot_file),
    "selected_warmup_start": int(best_warmup["start"]),
    "warmup_energy": float(best_warmup["warmup_energy"]),
    "exact_ground_state_energy": float(eig_vals[0]),
}

if energy:
    summary["final_energy"] = float(energy[-1])
    summary["best_energy_seen"] = float(np.min(np.asarray(energy, dtype=float)))

summary_file = RUN_DIR / f"summary_{JOB_BASE}.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

if energy:
    print("Final gap to exact:", float(energy[-1]) - float(eig_vals[0]))
