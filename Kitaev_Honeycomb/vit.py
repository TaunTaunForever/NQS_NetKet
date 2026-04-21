import os
import json
from datetime import date

import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
from netket.optimizer.solver import pinv_smooth
import optax

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

from hamiltonian import kitaev_hamiltonian, build_flux_operators

import flax.linen as nn
from flax.linen.initializers import normal

# ============================================================
# JAX configuration
# ============================================================
jax.config.update("jax_enable_x64", True)

# ============================================================
# Run configuration
# ============================================================
TODAY = date.today().isoformat()

NUM_SITES = 8

# --- Stage 1 (amplitude only) ---
NUM_SAMPLES_COARSE = 3 * 2**7
NUM_ITERS_COARSE = 10000


EMBED_DIM = 18
NUM_HEADS = 3
NUM_LAYERS = 2
MLP_HIDDEN_DIM = 2 * EMBED_DIM
PATCH_SIZE = 2

CHUNK_SIZE = None

JOB_BASE = (
    f"{NUM_SITES}-site_"
    f"{NUM_LAYERS}_layers_"
    f"{NUM_HEADS}_heads_"
    f"{TODAY}_PatchViT_SR"
)

os.environ["NETKET_DEBUG"] = "1"

# ============================================================
# Hamiltonian + lattice
# ============================================================
graph, symm_group, hi, ha = kitaev_hamiltonian(NUM_SITES)

flux_ops = build_flux_operators(graph, hi)
flux_penalty = sum(flux_ops)

if NUM_SITES <= 18:
    sp_h = ha.to_sparse()
    eig_vals, _ = eigsh(sp_h, k=2, which="SA")
    print("Exact ground-state energy:", eig_vals[0])
    print()

# ============================================================
# Patch-based ViT model
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
    mlp_hidden_dim: int

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
        y = MLP(self.mlp_hidden_dim, self.embed_dim)(y)
        x = x + y
        return x


class HoneycombPatchViT(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
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
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
            )(x)

        x = jnp.mean(x, axis=1)

        log_amp = nn.Dense(1, name="amp_head")(x).squeeze(-1)
        log_phase = nn.Dense(1, name="phase_head")(x).squeeze(-1)

        if self.learn_phase:
            return log_amp + 1j * log_phase
        else:
            return log_amp


# ============================================================
# ---------------- STAGE 1: AMPLITUDE ONLY ----------------
# ============================================================
print("\n=== Stage 1: Amplitude-only optimization ===\n")

model_stage1 = HoneycombPatchViT(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    mlp_hidden_dim=MLP_HIDDEN_DIM,
    patch_size=PATCH_SIZE,
    learn_phase=True,
)

sampler_coarse = nk.sampler.ParallelTemperingLocal(
    hilbert=hi,
    n_chains=NUM_SAMPLES_COARSE // 36,
    reset_chains=True,
)

vstate_coarse = nk.vqs.MCState(
    sampler=sampler_coarse,
    model=model_stage1,
    n_samples=NUM_SAMPLES_COARSE,
    chunk_size=CHUNK_SIZE,
)

optimizer = optax.sgd(learning_rate=5e-3)


driver_coarse = nk.driver.VMC_SR(
    hamiltonian=ha,
    optimizer=optimizer,
    variational_state=vstate_coarse,
    momentum = 0.9,
    linear_solver = pinv_smooth,
    mode = 'complex',
    diag_shift=1e-5,
)

OUT_COARSE = f"out_{JOB_BASE}_amp"

print("Parameters:", vstate_coarse.n_parameters)

driver_coarse.run(
    n_iter=NUM_ITERS_COARSE,
    out=OUT_COARSE,
    save_params_every=10,
)



# ============================================================
# Plot results
# ============================================================
energy = []

for fname in [f"{OUT_COARSE}.log", f"{OUT_COARSE}.log"]:
    with open(fname) as f:
        data = json.load(f)
        energy.extend(data["Energy"]["Mean"]["real"])

with open("mean_energy_run_{}.txt".format(JOB_BASE), "w") as f:
    for item in energy:
        f.write("{}\n".format(item))
f.close()



plt.figure(figsize=(10, 6))
plt.plot(energy)
plt.xscale("log")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title(f"{NUM_SITES}-site Kitaev (Amplitude → Phase ViT)")
plt.tight_layout()
plt.savefig(f"kitaev_patch_vit_{JOB_BASE}_amp_to_phase.png")
plt.show()
