import json
import os
from datetime import date

os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["NETKET_DEBUG"] = "1"

import jax
import jax.numpy as jnp
import netket as nk
import optax

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

import expectations
from common import build_honeycomb_graph
from vit_model import PatchViT


jax.config.update("jax_enable_x64", True)

TODAY = date.today().isoformat()

NUM_SITES = 18
NUM_SAMPLES = 2**10
NUM_ITERS_WARM = 200
NUM_ITERS = 10_000
CHUNK_SIZE = 2**9

J1 = 1.0
J2 = 0.2

PATCH_SIZE = 1
EMBED_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 4
MLP_HIDDEN = 2 * EMBED_DIM

JOB_NAME = (
    f"J1={J1}_J2={J2}_{NUM_SITES}-site_"
    f"{NUM_LAYERS}L_{NUM_HEADS}H_{PATCH_SIZE}p_"
    f"{NUM_SAMPLES}_samples_{TODAY}_J1_J2_Honeycomb_ViT"
)

print(f"Defining {NUM_SITES}-site J1-J2 Honeycomb lattice")
print("___________________________________________________")
graph, extent = build_honeycomb_graph(NUM_SITES)
print(f"Honeycomb lattice extent: {extent}")
print("Sites:", graph.n_nodes)
print("Edges:", graph.n_edges)
print("Neighbor orders:", sorted(set(color for *_ij, color in graph.edges(return_color=True))))
print()

hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes, total_sz=0)
ha = nk.operator.Heisenberg(hilbert=hi, graph=graph, J=[J1, J2])

if NUM_SITES <= 18:
    sp_h = ha.to_sparse()
    eig_vals, _ = eigsh(sp_h, k=2, which="SA")
    exact_gs = float(eig_vals[0])
    print("Exact ground-state energy:", exact_gs)
    print()
else:
    exact_gs = None

sampler = nk.sampler.ParallelTemperingExchange(hi, graph=graph, d_max=2)

optimizer = optax.sgd(learning_rate=1e-2, momentum=0.9)

print("\n=== Stage 1: amplitude-only warm start ===\n")

model_warm = PatchViT(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    mlp_hidden=MLP_HIDDEN,
    patch_size=PATCH_SIZE,
    learn_phase=False,
)

vstate_1 = nk.vqs.MCState(
    sampler=sampler,
    model=model_warm,
    n_samples=NUM_SAMPLES,
    chunk_size=CHUNK_SIZE,
)

print("num_samples:", NUM_SAMPLES)
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
    n_iter=NUM_ITERS_WARM,
    out=f"out_warm_{JOB_NAME}",
    save_params_every=10,
)

print("\n=== Stage 2: full complex optimization ===\n")

model_full = PatchViT(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    mlp_hidden=MLP_HIDDEN,
    patch_size=PATCH_SIZE,
    learn_phase=True,
)

vstate_2 = nk.vqs.MCState(
    sampler=sampler,
    model=model_full,
    n_samples=NUM_SAMPLES,
    variables=vstate_1.variables,
    chunk_size=CHUNK_SIZE,
)

driver_2 = nk.driver.VMC_SR(
    hamiltonian=ha,
    optimizer=optimizer,
    variational_state=vstate_2,
    diag_shift=1e-3,
)

obs = {
    "<X>": sum(nk.operator.spin.sigmax(hi, i) for i in range(hi.size)) / (2 * NUM_SITES),
    "<Y>": sum(nk.operator.spin.sigmay(hi, i) for i in range(hi.size)) / (2 * NUM_SITES),
    "<Z>": sum(nk.operator.spin.sigmaz(hi, i) for i in range(hi.size)) / (2 * NUM_SITES),
}

driver_2.run(
    n_iter=NUM_ITERS,
    obs=obs,
    out=f"out_{JOB_NAME}",
    save_params_every=10,
)

with open(f"out_{JOB_NAME}.log") as f:
    data = json.load(f)
energy_mean = data["Energy"]["Mean"]
if isinstance(energy_mean, dict):
    energy = energy_mean["real"]
else:
    energy = energy_mean

with open(f"mean_energy_run_{JOB_NAME}.txt", "w") as f:
    for item in energy:
        f.write(f"{item}\n")

plt.figure(figsize=(12, 8))
plt.plot(energy)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title(f"J1-J2 Honeycomb {NUM_SITES}-site (ViT)")
plt.tight_layout()
plt.savefig(f"energy_{JOB_NAME}.png")

plt.figure(figsize=(12, 8))
plt.plot(energy)
plt.xscale("log")
plt.xlabel("Iteration (log)")
plt.ylabel("Energy")
plt.title(f"J1-J2 Honeycomb {NUM_SITES}-site (ViT)")
plt.tight_layout()
plt.savefig(f"energy_log_{JOB_NAME}.png")

observables = expectations.define_observables(NUM_SITES, hi)
expectations.calculate_expectations(vstate_2, ha, observables)

summary = {
    "job_name": JOB_NAME,
    "num_sites": NUM_SITES,
    "extent": extent,
    "num_samples": NUM_SAMPLES,
    "num_iters_warm": NUM_ITERS_WARM,
    "num_iters": NUM_ITERS,
    "chunk_size": CHUNK_SIZE,
    "j1": J1,
    "j2": J2,
    "patch_size": PATCH_SIZE,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "mlp_hidden": MLP_HIDDEN,
    "final_energy": float(energy[-1]),
    "best_energy_seen": float(min(energy)),
    "exact_ground_state_energy": exact_gs,
    "mean_energy_file": f"mean_energy_run_{JOB_NAME}.txt",
    "log_file": f"out_{JOB_NAME}.log",
    "warm_log_file": f"out_warm_{JOB_NAME}.log",
    "plot_file": f"energy_{JOB_NAME}.png",
    "plot_log_file": f"energy_log_{JOB_NAME}.png",
}

with open(f"summary_{JOB_NAME}.json", "w") as f:
    json.dump(summary, f, indent=2)
