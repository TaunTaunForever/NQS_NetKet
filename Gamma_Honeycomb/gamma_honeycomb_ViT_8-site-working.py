import os
import json
from datetime import date
from collections import deque
from typing import Optional, Tuple, List

import jax
import jax.numpy as jnp
import netket as nk
from netket.optimizer.solver import pinv_smooth
import optax

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

from define_Gamma_Hamiltonian import gamma_hamiltonian

import flax.linen as nn
from flax.linen.initializers import normal

# ============================================================
# JAX configuration
# ============================================================
jax.config.update("jax_enable_x64", True)

# ============================================================
# USER OPTIONS (requested changes)
# ============================================================
LEARN_PHASE = True

# Training split
ADAM_FRACTION = 0.80
# remaining 25% is SR

# Adam improvements
ADAM_LR_INIT = 1e-2                 # your previous value
ADAM_LR_FINAL = 1e-3                # end of Adam phase
ADAM_CLIP_NORM = 1.0                # gradient clipping threshold

# SR parameters
SR_LR = 5e-3
SR_MOMENTUM = 0.9
SR_SOLVER = pinv_smooth

SR_DIAGSHIFT_SCHEDULE = [
    (0.30, 2e-3, "sr_shift2e-3"),
    (0.40, 5e-4, "sr_shift5e-4"),
    (0.30, 2e-4, "sr_shift2e-4"),
]

# Samplers (requested):
# - ADAM phase uses regular local updates (MetropolisLocal)
# - SR phase uses ParallelTemperingLocal
N_DISCARD_PER_CHAIN = 64 

# ============================================================
# Geometry-aware ordering (BFS on colored adjacency)
# ============================================================
def bfs_ordering_kitaev(graph):
    N = graph.n_nodes
    adj = [[] for _ in range(N)]
    for i, j, c in graph.edges(return_color=True):
        adj[i].append((j, c))
        adj[j].append((i, c))
    for i in range(N):
        adj[i].sort(key=lambda t: (t[1], t[0]))

    visited = [False] * N
    order = []
    q = deque([0])
    visited[0] = True
    while q:
        u = q.popleft()
        order.append(u)
        for v, _c in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)

    for i in range(N):
        if not visited[i]:
            order.append(i)

    return tuple(order)

# ============================================================
# Run configuration
# ============================================================
TODAY = date.today().isoformat()

NUM_SITES = 8
NUM_SAMPLES = 3 * 2**10
NUM_ITERS = 10000

EMBED_DIM = 8
NUM_HEADS = 8
NUM_LAYERS = 2
MLP_HIDDEN_DIM = 2 * EMBED_DIM
PATCH_SIZE = 2

CHUNK_SIZE = 2**7

JOB_BASE = (
    f"{NUM_SITES}-site_"
    f"{NUM_LAYERS}_layers_"
    f"{NUM_HEADS}_heads_"
    f"{NUM_SAMPLES}_samples_"
    f"{TODAY}_Gamma_ViT_AdamLocal_to_SRPT"
)

os.environ["NETKET_DEBUG"] = "1"

# ============================================================
# Hamiltonian + lattice
# ============================================================
graph, symm_group, hi, ha = gamma_hamiltonian(NUM_SITES)

perm = bfs_ordering_kitaev(graph)
print("Geometry-aware permutation:", np.array(perm))
print()

if NUM_SITES <= 18:
    sp_h = ha.to_sparse()
    eig_vals, _ = eigsh(sp_h, k=2, which="SA")
    print("Exact ground-state energy:", eig_vals[0])
    print()

# ============================================================
# Patch-based ViT model (CLS + geometry-aware ordering)
# Requested: "CLS token to the combination with mean pooling"
# We implement a blended readout:
#   x_read = alpha * CLS + (1-alpha) * mean(tokens_without_cls)
# ============================================================
class MLP(nn.Module):
    hidden_dim: int
    out_dim: int
    data_type: jnp.dtype

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, dtype=self.data_type, param_dtype=self.data_type)(x)
        x = nn.silu(x)
        x = nn.Dense(self.out_dim, dtype=self.data_type, param_dtype=self.data_type)(x)
        return x


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_hidden_dim: int
    data_type: jnp.dtype
    residual_scale: float = 0.8

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(dtype=self.data_type, param_dtype=self.data_type)(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            deterministic=True,
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(y)
        x = x + self.residual_scale * y

        y = nn.LayerNorm(dtype=self.data_type, param_dtype=self.data_type)(x)
        y = MLP(self.mlp_hidden_dim, self.embed_dim, data_type=self.data_type)(y)
        x = x + self.residual_scale * y
        return x


class HoneycombPatchViT(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool

    data_type: jnp.dtype = jnp.float64

    use_cls_token: bool = True
    cls_init_std: float = 0.02

    # NEW: blend CLS with mean pooling (alpha in [0,1])
    cls_mean_alpha: float = 0.5

    permutation: Optional[Tuple[int, ...]] = None  # hashable

    @nn.compact
    def __call__(self, σ):
        B, N = σ.shape

        if self.permutation is not None:
            perm_arr = jnp.asarray(self.permutation, dtype=jnp.int32)
            σ = σ[:, perm_arr]

        assert N % self.patch_size == 0
        n_patches = N // self.patch_size

        x = σ.reshape(B, n_patches, self.patch_size).astype(self.data_type)
        x = nn.Dense(self.embed_dim, dtype=self.data_type, param_dtype=self.data_type)(x)

        if self.use_cls_token:
            cls = self.param(
                "cls_token",
                normal(stddev=self.cls_init_std),
                (1, 1, self.embed_dim),
                self.data_type,
            )
            x = jnp.concatenate([jnp.tile(cls, (B, 1, 1)), x], axis=1)
            seq_len = n_patches + 1
        else:
            seq_len = n_patches

        pos_emb = self.param(
            "pos_emb",
            normal(stddev=0.02),
            (seq_len, self.embed_dim),
            self.data_type,
        )
        x = x + pos_emb[None, :, :]

        for _ in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                data_type=self.data_type,
            )(x)

        if self.use_cls_token:
            cls_vec = x[:, 0, :]                 # (B, D)
            mean_vec = jnp.mean(x[:, 1:, :], axis=1)  # (B, D)
            a = jnp.asarray(self.cls_mean_alpha, dtype=self.data_type)
            x_read = a * cls_vec + (1.0 - a) * mean_vec
        else:
            x_read = jnp.mean(x, axis=1)

        log_amp = nn.Dense(1, dtype=self.data_type, param_dtype=self.data_type, name="amp_head")(x_read).squeeze(-1)
        log_phase = nn.Dense(1, dtype=self.data_type, param_dtype=self.data_type, name="phase_head")(x_read).squeeze(-1)

        if self.learn_phase:
            return log_amp + 1j * log_phase
        else:
            return jnp.real(log_amp)

# ============================================================
# Sampler helpers
# ============================================================
def make_metropolis_local(hilbert, n_samples):
    n_chains = max(1, n_samples // 16)
    return nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=n_chains)

def make_parallel_tempering_local(hilbert, n_samples):
    n_chains = max(1, n_samples // 16)
    return nk.sampler.ParallelTemperingLocal(hilbert=hilbert, n_chains=n_chains)

# ============================================================
# Training split
# ============================================================
n_adam = max(1, int(NUM_ITERS * ADAM_FRACTION))
n_sr_total = max(1, NUM_ITERS - n_adam)

print(f"Total iters: {NUM_ITERS} | Adam: {n_adam} | SR: {n_sr_total}\n")

# ============================================================
# Build model (shared)
# ============================================================
model = HoneycombPatchViT(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    mlp_hidden_dim=MLP_HIDDEN_DIM,
    patch_size=PATCH_SIZE,
    learn_phase=LEARN_PHASE,
    use_cls_token=True,
    cls_mean_alpha=0.5,   # blend CLS + mean pooling
    permutation=perm,
)

# ============================================================
# Phase 1: Adam + LR schedule + gradient clipping (MetropolisLocal)
# ============================================================
print(f"=== Phase 1: Adam (MetropolisLocal) for {n_adam} iterations ===")

sampler_adam = make_metropolis_local(hi, NUM_SAMPLES)

vstate_adam = nk.vqs.MCState(
    sampler=sampler_adam,
    model=model,
    n_samples=NUM_SAMPLES,
    chunk_size=CHUNK_SIZE,
    n_discard_per_chain=N_DISCARD_PER_CHAIN,
)

print("Parameters:", vstate_adam.n_parameters)
print("Chain length:", vstate_adam.chain_length)
print()

adam_lr_schedule = optax.join_schedules(
    schedules=[
        optax.linear_schedule(ADAM_LR_INIT, ADAM_LR_FINAL, n_adam),
    ],
    boundaries=[],
)

adam_opt = optax.chain(
    optax.clip_by_global_norm(ADAM_CLIP_NORM),
    optax.adam(learning_rate=adam_lr_schedule),
)

driver_adam = nk.driver.VMC(
    hamiltonian=ha,
    optimizer=adam_opt,
    variational_state=vstate_adam,
)

OUT_ADAM = f"out_{JOB_BASE}_adam"
driver_adam.run(n_iter=n_adam, out=OUT_ADAM, save_params_every=10)

# ============================================================
# Phase 2: SR (ParallelTemperingLocal) with diag_shift schedule
# IMPORTANT: we continue from Adam-trained parameters.
# ============================================================
print(f"\n=== Phase 2: SR (ParallelTemperingLocal) for {n_sr_total} iterations ===")

sampler_sr = make_parallel_tempering_local(hi, NUM_SAMPLES)

vstate_sr = nk.vqs.MCState(
    sampler=sampler_sr,
    model=model,
    n_samples=NUM_SAMPLES,
    variables=vstate_adam.variables,   # <-- carry over parameters
    chunk_size=CHUNK_SIZE,
    n_discard_per_chain=N_DISCARD_PER_CHAIN,
)

sr_opt = optax.sgd(learning_rate=SR_LR)

def run_sr_segment(n_iter, diag_shift, tag):
    driver_sr = nk.driver.VMC_SR(
        hamiltonian=ha,
        optimizer=sr_opt,
        variational_state=vstate_sr,
        momentum=SR_MOMENTUM,
        linear_solver=SR_SOLVER,
        diag_shift=diag_shift,
    )
    out = f"out_{JOB_BASE}_{tag}"
    driver_sr.run(n_iter=n_iter, out=out, save_params_every=10)
    return out

outs: List[str] = [OUT_ADAM]

remaining = n_sr_total
for frac, shift, tag in SR_DIAGSHIFT_SCHEDULE:
    n_seg = int(round(frac * n_sr_total))
    n_seg = max(1, min(n_seg, remaining))
    remaining -= n_seg
    outs.append(run_sr_segment(n_seg, shift, tag))
    if remaining <= 0:
        break

if remaining > 0:
    last_shift = SR_DIAGSHIFT_SCHEDULE[-1][1]
    outs.append(run_sr_segment(remaining, last_shift, "sr_tail"))

# ============================================================
# Collect energies from all outputs and plot
# ============================================================
energy = []
for out in outs:
    logf = f"{out}.log"
    if os.path.exists(logf):
        with open(logf) as f:
            data = json.load(f)
            energy.extend(data["Energy"]["Mean"]["real"])

with open(f"mean_energy_run_{JOB_BASE}_adam_to_sr.txt", "w") as f:
    for e in energy:
        f.write(f"{e}\n")

plt.figure(figsize=(10, 6))
plt.plot(energy)
plt.xscale("log")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title(f"{NUM_SITES}-site Gamma (ViT, Adam(Local) → SR(PT), CLS+Mean, clip, LR sched)")
plt.tight_layout()
plt.savefig(f"gamma_vit_{JOB_BASE}_adam_to_sr.png")
