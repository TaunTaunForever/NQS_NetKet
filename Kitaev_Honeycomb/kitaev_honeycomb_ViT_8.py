import os
import json
from datetime import date
from typing import Optional, Tuple, List
from collections import deque


import jax
import jax.numpy as jnp
import netket as nk
from netket.optimizer.solver import pinv_smooth, cholesky
import optax

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

from define_Kitaev_Hamiltonian import kitaev_hamiltonian

import flax.linen as nn
from flax.linen.initializers import normal

print(jax.devices())

# ============================================================
# JAX configuration
# ============================================================
jax.config.update("jax_enable_x64", True)

# ============================================================
# USER OPTIONS
# ============================================================
LEARN_PHASE = True

NUM_SITES = 18
NUM_SAMPLES_WARMUP = 3*2**9
NUM_SAMPLES = 3*2**9

# ---------- Multi-start ----------
NUM_STARTS = 1
NUM_ITERS_WARM = 500      # max SR warmup iterations for each restart
NUM_ITERS_MAIN = 1000     # main SR continuation of best restart
NUM_ITERS_SR = 5000       # final PT-SR refinement

# ---------- Warmup early stopping ----------
WARMUP_ENERGY_TOL = 1e-3
WARMUP_PATIENCE = 40
WARMUP_WINDOW = 150
WARMUP_MIN_STEPS = 50

# ---------- Model ----------
EMBED_DIM = 20
NUM_HEADS = 20
NUM_LAYERS = 2
MLP_HIDDEN_DIM = 2 * EMBED_DIM
PATCH_SIZE = 1
CHUNK_SIZE = None

# ---------- Warmup SR ----------
WARM_SR_LR = 1e-2
WARM_SR_MOMENTUM = 0.9
WARM_SR_DIAGSHIFT = 1e-4
WARM_SR_SOLVER = cholesky

# ---------- Main SR ----------
MAIN_SR_LR = 1e-2
MAIN_SR_MOMENTUM = 0.9
MAIN_SR_DIAGSHIFT = 5e-4
MAIN_SR_SOLVER = cholesky

# ---------- Final PT-SR ----------
SR_LR = 5e-3
SR_MOMENTUM = 0.9
SR_SOLVER = cholesky
SR_DIAGSHIFT_SCHEDULE = [
    (0.30, 5e-4, "sr_shift5e-2"),
    (0.40, 2e-4, "sr_shift2e-4"),
    (0.30, 1e-4, "sr_shift1e-4"),
]

# ---------- Sampling ----------
N_DISCARD_PER_CHAIN = 8

TODAY = date.today().isoformat()

JOB_BASE = (
    f"{NUM_SITES}-site_"
    f"{NUM_LAYERS}_layers_"
    f"{NUM_HEADS}_heads_"
    f"{PATCH_SIZE}_patches_"
    f"{NUM_SAMPLES}_samples_"
    f"{TODAY}_Kitaev_ViT_multistart_SRonly"
)

os.environ["NETKET_DEBUG"] = "1"

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
# Simple model
# ============================================================
class MLP(nn.Module):
    hidden_dim: int
    out_dim: int
    data_type: jnp.dtype

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.hidden_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(x)
        x = nn.gelu(x)
        x = nn.Dense(
            self.out_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(x)
        return x


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_hidden_dim: int
    data_type: jnp.dtype
    residual_scale: float = 0.8

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            deterministic=True,
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(y)
        x = x + self.residual_scale * y

        y = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(x)
        y = MLP(
            hidden_dim=self.mlp_hidden_dim,
            out_dim=self.embed_dim,
            data_type=self.data_type,
        )(y)
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

        z = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="out_layer_norm",
        )(z)

        out_real = nn.Dense(
            self.d_model,
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="output_layer_real",
        )(z)
        out_real = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="norm_real",
        )(out_real)

        out_imag = nn.Dense(
            self.d_model,
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="output_layer_imag",
        )(z)
        out_imag = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="norm_imag",
        )(out_imag)

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
            raise ValueError(
                f"Number of sites N={N} must be divisible by patch_size={self.patch_size}"
            )

        n_patches = N // self.patch_size

        x = x.reshape(B, n_patches, self.patch_size).astype(self.data_type)

        x = nn.Dense(
            self.embed_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
            kernel_init=nn.initializers.xavier_uniform(),
            name="patch_embed",
        )(x)

        pos_emb = self.param(
            "pos_emb",
            normal(stddev=0.02),
            (n_patches, self.embed_dim),
            self.data_type,
        )
        x = x + pos_emb[None, :, :]

        for layer in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                data_type=self.data_type,
                name=f"TransformerBlock_{layer}",
            )(x)

        log_psi = OutputHead(
            d_model=self.embed_dim,
            data_type=self.data_type,
            name="OutputHead",
        )(x)

        if self.learn_phase:
            return log_psi
        else:
            return jnp.real(log_psi)

# ============================================================
# Simple permutation
# ============================================================
if NUM_SITES == 8 and PATCH_SIZE == 2:
    perm = (0, 1, 2, 3, 4, 5, 6, 7)
else:
    perm = tuple(range(NUM_SITES))

print("Permutation:", perm)

# ============================================================
# Sampler helpers
# ============================================================
def make_metropolis_local(hilbert, n_samples):
    n_chains = max(1, n_samples//64)
    return nk.sampler.MetropolisLocal(
        hilbert=hilbert,
        n_chains=n_chains,
    )

def make_parallel_tempering_local(hilbert, n_samples):
    n_chains = max(1, n_samples//64)
    return nk.sampler.ParallelTemperingLocal(
        hilbert=hilbert,
        n_chains=n_chains,
        sweep_size=36
    )

def make_parallel_tempering_hamiltonian(hilbert, n_samples):
    # PT hamiltonian alternative
    n_chains = max(1, n_samples//64)
    return nk.sampler.ParallelTemperingHamiltonian(
        hilbert,
        hamiltonian=ha,
        n_chains=n_chains,         # or 256
        sweep_size=36,        # then try 36
    )
# ============================================================
# Model factory
# ============================================================
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

# ============================================================
# Utility
# ============================================================
def load_best_energy(logfile):
    with open(logfile) as f:
        data = json.load(f)
    energies = np.array(data["Energy"]["Mean"]["real"], dtype=float)
    tail = energies[-min(20, len(energies)):]
    return float(np.mean(tail)), list(tail)

class EnergyPlateauStopping:
    def __init__(self, tol=1e-4, patience=20, window=20, min_steps=50):
        self.tol = tol
        self.patience = patience
        self.window = window
        self.min_steps = min_steps
        self.energies = deque(maxlen=2 * window)
        self.small_change_count = 0

    def __call__(self, step, log_data, driver):
        energy_stats = log_data.get("Energy", None)
        if energy_stats is None:
            return True

        current_energy = None
        try:
            current_energy = float(energy_stats["Mean"]["real"])
        except Exception:
            pass

        if current_energy is None:
            try:
                current_energy = float(np.real(driver.energy.mean))
            except Exception:
                return True

        self.energies.append(current_energy)

        if len(self.energies) < 2 * self.window or step < self.min_steps:
            return True

        arr = np.array(self.energies, dtype=float)
        prev_mean = np.mean(arr[:self.window])
        curr_mean = np.mean(arr[self.window:])
        delta = abs(curr_mean - prev_mean)

        log_data["warmup_deltaE"] = delta

        if delta < self.tol:
            self.small_change_count += 1
        else:
            self.small_change_count = 0

        if self.small_change_count >= self.patience:
            print(f"Warmup early stop at step {step}: ΔE(window) = {delta:.3e}")
            return False

        return True

# ============================================================
# Multi-start warmup using SR
# ============================================================
print(
    f"\n=== Multi-start SR warmup: {NUM_STARTS} starts, "
    f"up to {NUM_ITERS_WARM} iterations each ===\n"
)

start_summaries = []
all_outs: List[str] = []

for k in range(NUM_STARTS):
    print(f"--- Start {k+1}/{NUM_STARTS} ---")

    model_k = build_model()
    sampler_k = make_parallel_tempering_local(hi, NUM_SAMPLES_WARMUP)

    vstate_k = nk.vqs.MCState(
        sampler=sampler_k,
        model=model_k,
        n_samples=NUM_SAMPLES_WARMUP,
        chunk_size=CHUNK_SIZE,
        n_discard_per_chain=N_DISCARD_PER_CHAIN,
        seed=1234 + k,
    )

    print("Parameters:", vstate_k.n_parameters)
    print("n_chains:", vstate_k.sampler.n_chains)
    print("chain_length:", vstate_k.chain_length)

    warm_opt = optax.sgd(learning_rate=WARM_SR_LR)

    driver = nk.driver.VMC_SR(
        hamiltonian=ha,
        optimizer=warm_opt,
        variational_state=vstate_k,
        momentum=WARM_SR_MOMENTUM,
        linear_solver=WARM_SR_SOLVER,
        mode="complex",
        diag_shift=WARM_SR_DIAGSHIFT,
    )

    out_k = f"out_{JOB_BASE}_warm_start_{k}"

    warmup_callback = EnergyPlateauStopping(
        tol=WARMUP_ENERGY_TOL,
        patience=WARMUP_PATIENCE,
        window=WARMUP_WINDOW,
        min_steps=WARMUP_MIN_STEPS,
    )

    driver.run(
        n_iter=NUM_ITERS_WARM,
        out=out_k,
        save_params_every=10,
        callback=warmup_callback,
    )
    all_outs.append(out_k)

    best_e, energies = load_best_energy(f"{out_k}.log")
    print(f"Best energy for start {k}: {best_e}")

    start_summaries.append(
        {
            "start": k,
            "best_energy": best_e,
            "out": out_k,
            "variables": vstate_k.variables,
        }
    )

# Choose best start
start_summaries.sort(key=lambda d: d["best_energy"])
best = start_summaries[0]

print("\n=== Best start selected ===")
print("Start index:", best["start"])
print("Best warmup energy:", best["best_energy"])
print()

# ============================================================
# Main SR continuation from best start
# ============================================================
print(f"=== Main SR continuation for {NUM_ITERS_MAIN} iterations ===")

model_main = build_model()
sampler_main = make_parallel_tempering_local(hi, NUM_SAMPLES)

vstate_main = nk.vqs.MCState(
    sampler=sampler_main,
    model=model_main,
    n_samples=NUM_SAMPLES,
    variables=best["variables"],
    chunk_size=CHUNK_SIZE,
    n_discard_per_chain=N_DISCARD_PER_CHAIN,
)

print("Parameters:", vstate_main.n_parameters)
print("n_chains:", vstate_main.sampler.n_chains)
print("chain_length:", vstate_main.chain_length)

main_sr_opt = optax.sgd(learning_rate=MAIN_SR_LR)

driver_main = nk.driver.VMC_SR(
    hamiltonian=ha,
    optimizer=main_sr_opt,
    variational_state=vstate_main,
    momentum=MAIN_SR_MOMENTUM,
    linear_solver=MAIN_SR_SOLVER,
    mode="complex",
    diag_shift=MAIN_SR_DIAGSHIFT,
)

OUT_MAIN = f"out_{JOB_BASE}_main_sr"
driver_main.run(n_iter=NUM_ITERS_MAIN, out=OUT_MAIN, save_params_every=10)
all_outs.append(OUT_MAIN)

# ============================================================
# Final SR refinement with PT
# ============================================================
print(f"\n=== Final PT-SR refinement for {NUM_ITERS_SR} iterations ===")

sampler_sr = make_parallel_tempering_local(hi, NUM_SAMPLES)

vstate_sr = nk.vqs.MCState(
    sampler=sampler_sr,
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
        mode="complex",
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

# ============================================================
# Collect all energies and summarize
# ============================================================
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
plt.title(f"{NUM_SITES}-site Kitaev (best-of-{NUM_STARTS} SR restarts → SR → PT-SR)")
plt.tight_layout()
plt.savefig(f"kitaev_vit_{JOB_BASE}.png")
plt.show()

print("\n=== Warmup ranking ===")
for row in start_summaries:
    print(f"start={row['start']}  best_energy={row['best_energy']:.12f}")
