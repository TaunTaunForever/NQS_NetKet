import os
import json
from datetime import date
from typing import Optional, Tuple, List

import jax
import jax.numpy as jnp
import netket as nk
from netket.optimizer.solver import pinv_smooth
import optax

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

import flax
import flax.linen as nn
from flax.linen.initializers import normal

from hamiltonian import kitaev_hamiltonian

print(jax.devices())
jax.config.update("jax_enable_x64", True)
os.environ["NETKET_DEBUG"] = "1"

# ============================================================
# USER OPTIONS
# Point this to the summary produced by the best-of-K script.
# ============================================================
BESTOFK_SUMMARY_FILE = None
# Example:
# BESTOFK_SUMMARY_FILE = "bestofk_summary_18-site_2_layers_4_heads_2048_samples_2026-03-08_Kitaev_ViT_bestofK_only.json"

LEARN_PHASE = True

NUM_SITES = 18
NUM_SAMPLES = 2**11

# ---------- Main SR ----------
NUM_ITERS_MAIN = 2000
MAIN_SR_LR = 1e-2
MAIN_SR_MOMENTUM = 0.9
MAIN_SR_DIAGSHIFT = 5e-4
MAIN_SR_SOLVER = pinv_smooth

# ---------- Final PT-SR ----------
NUM_ITERS_SR = 10000
SR_LR = 1e-2
SR_MOMENTUM = 0.9
SR_SOLVER = pinv_smooth
SR_DIAGSHIFT_SCHEDULE = [
    (0.30, 1e-2, "sr_shift1e-2"),
    (0.40, 5e-4, "sr_shift5e-4"),
    (0.30, 2e-4, "sr_shift2e-4"),
]

# ---------- Model ----------
EMBED_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 2
MLP_HIDDEN_DIM = 2 * EMBED_DIM
PATCH_SIZE = 3
CHUNK_SIZE = None

# ---------- Sampling ----------
N_DISCARD_PER_CHAIN = 64

TODAY = date.today().isoformat()
JOB_BASE = (
    f"{NUM_SITES}-site_"
    f"{NUM_LAYERS}_layers_"
    f"{NUM_HEADS}_heads_"
    f"{NUM_SAMPLES}_samples_"
    f"{TODAY}_Kitaev_ViT_SR_from_bestofK"
)

# ============================================================
# Helpers for loading summary + variables
# ============================================================
def maybe_load_summary(path):
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


summary = maybe_load_summary(BESTOFK_SUMMARY_FILE)
if summary is not None:
    best_params_file = summary["best_params_file"]
    print("Loading best-of-K summary from:", BESTOFK_SUMMARY_FILE)
    print("Loading parameters from:", best_params_file)
else:
    raise ValueError("Set BESTOFK_SUMMARY_FILE to the JSON produced by the warmup script.")

# ============================================================
# Hamiltonian + lattice
# ============================================================
graph, symm_group, hi, ha = kitaev_hamiltonian(NUM_SITES)

if NUM_SITES <= 18:
    sp_h = ha.to_sparse()
    eig_vals, _ = eigsh(sp_h, k=2, which="SA")
    print("Exact ground-state energy:", eig_vals[0])
    print()

# ============================================================
# Model
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


def log_cosh(z):
    return jnp.log(jnp.cosh(z))


class OutputHead(nn.Module):
    d_model: int
    data_type: jnp.dtype

    @nn.compact
    def __call__(self, x):
        z = x.sum(axis=1)
        z = nn.LayerNorm(dtype=self.data_type, param_dtype=self.data_type, name="out_layer_norm")(z)

        out_real = nn.Dense(
            self.d_model,
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="output_layer_real",
        )(z)
        out_real = nn.LayerNorm(dtype=self.data_type, param_dtype=self.data_type, name="norm_real")(out_real)

        out_imag = nn.Dense(
            self.d_model,
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="output_layer_imag",
        )(z)
        out_imag = nn.LayerNorm(dtype=self.data_type, param_dtype=self.data_type, name="norm_imag")(out_imag)

        out = out_real + 1j * out_imag
        return jnp.sum(log_cosh(out), axis=-1)


class HoneycombPatchViT(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool
    data_type: jnp.dtype = jnp.float64
    permutation: Optional[Tuple[int, ...]] = None

    @nn.compact
    def __call__(self, sigma):
        x = jnp.atleast_2d(sigma)
        B, N = x.shape

        if self.permutation is not None:
            perm_arr = jnp.asarray(self.permutation, dtype=jnp.int32)
            x = x[:, perm_arr]

        if N % self.patch_size != 0:
            raise ValueError(f"N={N} must be divisible by patch_size={self.patch_size}")

        n_patches = N // self.patch_size
        x = x.reshape(B, n_patches, self.patch_size).astype(self.data_type)

        x = nn.Dense(
            self.embed_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
            kernel_init=nn.initializers.xavier_uniform(),
            name="patch_embed",
        )(x)

        pos_emb = self.param("pos_emb", normal(stddev=0.02), (n_patches, self.embed_dim), self.data_type)
        x = x + pos_emb[None, :, :]

        for layer in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                data_type=self.data_type,
                name=f"TransformerBlock_{layer}",
            )(x)

        log_psi = OutputHead(d_model=self.embed_dim, data_type=self.data_type, name="OutputHead")(x)
        return log_psi if self.learn_phase else jnp.real(log_psi)


if NUM_SITES == 8 and PATCH_SIZE == 2:
    perm = (0, 1, 2, 3, 4, 5, 6, 7)
else:
    perm = tuple(range(NUM_SITES))

print("Permutation:", perm)


def make_metropolis_local(hilbert, n_samples):
    n_chains = max(1, n_samples // 8)
    return nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=n_chains)


def make_parallel_tempering_local(hilbert, n_samples):
    n_chains = max(1, n_samples // 8)
    return nk.sampler.ParallelTemperingLocal(hilbert=hilbert, n_chains=n_chains)


def build_model():
    return HoneycombPatchViT(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        patch_size=PATCH_SIZE,
        learn_phase=LEARN_PHASE,
        permutation=perm,
    )


def load_variables_from_mpack(template_variables, filename):
    """Load NetKet/Flax variables from an .mpack file using NetKet's helper."""
    return nk.experimental.vqs.variables_from_file(filename, template_variables)


all_outs: List[str] = []

# ============================================================
# Initialize a state, then load the selected warmup parameters
# ============================================================
init_vstate = nk.vqs.MCState(
    sampler=make_metropolis_local(hi, NUM_SAMPLES),
    model=build_model(),
    n_samples=NUM_SAMPLES,
    chunk_size=CHUNK_SIZE,
    n_discard_per_chain=N_DISCARD_PER_CHAIN,
    seed=1234,
)
loaded_variables = load_variables_from_mpack(init_vstate.variables, best_params_file)

# ============================================================
# Main SR continuation
# ============================================================
print(f"=== Main SR continuation for {NUM_ITERS_MAIN} iterations ===")

vstate_main = nk.vqs.MCState(
    sampler=make_metropolis_local(hi, NUM_SAMPLES),
    model=build_model(),
    n_samples=NUM_SAMPLES,
    chunk_size=CHUNK_SIZE,
    n_discard_per_chain=N_DISCARD_PER_CHAIN,
)
vstate_main.variables = loaded_variables

print("Parameters:", vstate_main.n_parameters)
print("n_chains:", vstate_main.sampler.n_chains)
print("chain_length:", vstate_main.chain_length)

main_driver = nk.driver.VMC_SR(
    hamiltonian=ha,
    optimizer=optax.sgd(learning_rate=MAIN_SR_LR),
    variational_state=vstate_main,
    momentum=MAIN_SR_MOMENTUM,
    linear_solver=MAIN_SR_SOLVER,
    diag_shift=MAIN_SR_DIAGSHIFT,
)

OUT_MAIN = f"out_{JOB_BASE}_main_sr"
main_driver.run(n_iter=NUM_ITERS_MAIN, out=OUT_MAIN, save_params_every=10)
all_outs.append(OUT_MAIN)

# ============================================================
# Final PT-SR refinement
# ============================================================
print(f"\n=== Final PT-SR refinement for {NUM_ITERS_SR} iterations ===")

vstate_sr = nk.vqs.MCState(
    sampler=make_parallel_tempering_local(hi, NUM_SAMPLES),
    model=build_model(),
    n_samples=NUM_SAMPLES,
    variables=vstate_main.variables,
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
        diag_shift=diag_shift,
    )
    out = f"out_{JOB_BASE}_{tag}"
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

# Save final variables too
final_params_file = f"final_sr_params_{JOB_BASE}.mpack"
with open(final_params_file, "wb") as f:
    f.write(flax.serialization.to_bytes(vstate_sr.variables))

energy = []
for out in all_outs:
    logf = f"{out}.log"
    if os.path.exists(logf):
        with open(logf) as f:
            data = json.load(f)
            energy.extend(data["Energy"]["Mean"]["real"])

with open(f"mean_energy_run_{JOB_BASE}.txt", "w") as f:
    for e in energy:
        f.write(f"{e}\n")

plt.figure(figsize=(10, 6))
plt.plot(energy)
plt.xscale("log")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title(f"{NUM_SITES}-site Kitaev (SR from best-of-K warmup)")
plt.tight_layout()
plt.savefig(f"kitaev_vit_{JOB_BASE}.png")
plt.show()

print("\nSaved final parameters to:", final_params_file)