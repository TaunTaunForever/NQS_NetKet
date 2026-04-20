import os
import json
from datetime import date

os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["NETKET_DEBUG"] = "1"

import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import optax

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

import flax.linen as nn
from flax.linen.initializers import normal

# ============================================================
# JAX config
# ============================================================
jax.config.update("jax_enable_x64", True)

# ============================================================
# Run configuration
# ============================================================
TODAY = date.today().isoformat()

NUM_SITES = 64
NUM_SAMPLES = 2**10
NUM_ITERS = 10_000
CHUNK_SIZE = 2**9

J1 = 1.0
J2 = 0.02

# --- ViT hyperparameters ---
PATCH_SIZE = 4               # 2×2 real-space patches
EMBED_DIM = 4
NUM_HEADS = 4
NUM_LAYERS = 6
MLP_HIDDEN = 2 * EMBED_DIM

JOB_NAME = (
    f"J1={J1}_J2={J2}_{NUM_SITES}-site_"
    f"{NUM_LAYERS}L_{NUM_HEADS}H_ViT_{TODAY}"
)

# ============================================================
# Triangular lattice + symmetries
# ============================================================
print(f"Defining {NUM_SITES}-site triangular lattice")

graph = nk.graph.Triangular(
    extent=[2, 2],
    max_neighbor_order=2,
    pbc=True,
)

hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes, total_sz=0)
ha = nk.operator.Heisenberg(hilbert=hi, graph=graph, J=[J1, J2])

# ============================================================
# Exact diagonalization check (small only)
# ============================================================
if NUM_SITES <= 18:
    sp_h = ha.to_sparse()
    eig_vals, _ = eigsh(sp_h, k=2, which="SA")
    print("Exact ground-state energy:", eig_vals[0])

# ============================================================
# ViT model
# ============================================================
class MLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_hidden: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            deterministic=True,
        )(y)
        x = x + y

        y = nn.LayerNorm()(x)
        y = MLP(self.mlp_hidden, self.embed_dim)(y)
        return x + y


class PatchViT(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden: int
    patch_size: int
    learn_phase: bool

    @nn.compact
    def __call__(self, σ):
        B, N = σ.shape
        assert N % self.patch_size == 0

        n_patches = N // self.patch_size

        x = σ.reshape(B, n_patches, self.patch_size).astype(jnp.float64)
        x = nn.Dense(self.embed_dim)(x)

        pos_emb = self.param(
            "pos_emb",
            normal(stddev=0.02),
            (n_patches, self.embed_dim),
        )
        x = x + pos_emb

        for _ in range(self.num_layers):
            x = TransformerBlock(
                self.embed_dim,
                self.num_heads,
                self.mlp_hidden,
            )(x)

        x = jnp.mean(x, axis=1)

        log_amp = nn.Dense(1, name="amp_head")(x).squeeze(-1)
        log_phase = nn.Dense(1, name="phase_head")(x).squeeze(-1)

        if self.learn_phase:
            return log_amp + 1j * log_phase
        else:
            return log_amp


# ============================================================
# Sampler
# ============================================================
sampler = nk.sampler.ParallelTemperingExchange(
    hi,
    graph=graph,
    d_max=2,
)

# ============================================================
# -------- Stage 1: Phase-only (|ψ| = 1) --------
# ============================================================
print("\n=== Stage 1: Phase pretraining ===\n")

model_phase = PatchViT(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    mlp_hidden=MLP_HIDDEN,
    patch_size=PATCH_SIZE,
    learn_phase=False,
)

vstate_1 = nk.vqs.MCState(
    sampler=sampler,
    model=model_phase,
    n_samples=NUM_SAMPLES,
    chunk_size=CHUNK_SIZE,
)

optimizer = optax.sgd(learning_rate=1e-2, momentum=0.9)

driver_1 = nk.driver.VMC_SR(
    hamiltonian=ha,
    optimizer=optimizer,
    variational_state=vstate_1,
    diag_shift=1e-3,
)

driver_1.run(
    n_iter=200,
    out=f"out_phase_{JOB_NAME}",
    save_params_every=10,
)

# ============================================================
# -------- Stage 2: Full complex optimization --------
# ============================================================
print("\n=== Stage 2: Full amplitude + phase ===\n")

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

driver_2.run(
    n_iter=NUM_ITERS,
    out=f"out_{JOB_NAME}",
    save_params_every=10,
)

# ============================================================
# Plot energy
# ============================================================
data = json.load(open(f"out_{JOB_NAME}.log"))
energy = data["Energy"]["Mean"]["real"]

plt.figure(figsize=(12, 8))
plt.plot(energy)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title(f"J1–J2 Triangular {NUM_SITES}-site (ViT)")
plt.savefig(f"energy_{JOB_NAME}.png")

plt.figure(figsize=(12, 8))
plt.plot(energy)
plt.xscale("log")
plt.xlabel("Iteration (log)")
plt.ylabel("Energy")
plt.title(f"J1–J2 Triangular {NUM_SITES}-site (ViT)")
plt.savefig(f"energy_log_{JOB_NAME}.png")
