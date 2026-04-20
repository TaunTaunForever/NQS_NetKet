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

from define_Gamma_Hamiltonian import gamma_hamiltonian

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
NUM_SAMPLES_COARSE = 3*2**8
NUM_ITERS_COARSE = 10000


EMBED_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 4
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
graph, symm_group, hi, ha = gamma_hamiltonian(NUM_SITES)

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
    data_type: jnp.dtype

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, dtype=self.data_type, param_dtype=self.data_type)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, dtype=self.data_type, param_dtype=self.data_type)(x)
        return x


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_hidden_dim: int
    data_type: jnp.dtype

    @nn.compact
    def __call__(self, x):
        # Pre-norm attention
        y = nn.LayerNorm(dtype=self.data_type, param_dtype=self.data_type)(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            deterministic=True,
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(y)
        x = x + y

        # Pre-norm MLP
        y = nn.LayerNorm(dtype=self.data_type, param_dtype=self.data_type)(x)
        y = MLP(self.mlp_hidden_dim, self.embed_dim, data_type=self.data_type)(y)
        x = x + y
        return x


class HoneycombPatchViT(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool
    data_type: jnp.dtype = jnp.float64

    # NEW: CLS token support
    use_cls_token: bool = True
    cls_init_std: float = 0.02

    @nn.compact
    def __call__(self, σ):
        B, N = σ.shape
        assert N % self.patch_size == 0

        n_patches = N // self.patch_size

        # Patchify + cast
        x = σ.reshape(B, n_patches, self.patch_size).astype(self.data_type)

        # Patch embedding to (B, n_patches, embed_dim)
        x = nn.Dense(self.embed_dim, dtype=self.data_type, param_dtype=self.data_type)(x)

        # Optionally prepend CLS token
        if self.use_cls_token:
            cls = self.param(
                "cls_token",
                normal(stddev=self.cls_init_std),
                (1, 1, self.embed_dim),
                self.data_type,
            )
            cls_tokens = jnp.tile(cls, (B, 1, 1))  # (B, 1, embed_dim)
            x = jnp.concatenate([cls_tokens, x], axis=1)  # (B, 1+n_patches, embed_dim)
            seq_len = n_patches + 1
        else:
            seq_len = n_patches

        # Positional embedding matches sequence length
        pos_emb = self.param(
            "pos_emb",
            normal(stddev=0.02),
            (seq_len, self.embed_dim),
            self.data_type,
        )
        x = x + pos_emb  # broadcast over batch

        # Transformer stack
        for _ in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                data_type=self.data_type,
            )(x)

        # Readout: CLS token or mean pooling
        if self.use_cls_token:
            x_read = x[:, 0, :]          # (B, embed_dim)
        else:
            x_read = jnp.mean(x, axis=1) # (B, embed_dim)

        # Heads
        log_amp = nn.Dense(1, dtype=self.data_type, param_dtype=self.data_type, name="amp_head")(x_read).squeeze(-1)
        log_phase = nn.Dense(1, dtype=self.data_type, param_dtype=self.data_type, name="phase_head")(x_read).squeeze(-1)

        if self.learn_phase:
            return log_amp + 1j * log_phase
        else:
            return jnp.real(log_amp)

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
    use_cls_token=True
)

sampler_coarse = nk.sampler.ParallelTemperingLocal(
    hilbert=hi,
    n_chains=NUM_SAMPLES_COARSE // 32,
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
    diag_shift=1e-4,
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
