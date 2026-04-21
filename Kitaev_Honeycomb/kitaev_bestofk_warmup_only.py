import os
import json
from datetime import date
from typing import Optional, Tuple, List
from collections import deque

import jax
import jax.numpy as jnp
import netket as nk
from netket.optimizer.solver import pinv_smooth
import optax

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

from flax import serialization
import flax.linen as nn
from flax.linen.initializers import normal

from hamiltonian import kitaev_hamiltonian

print(jax.devices())
jax.config.update("jax_enable_x64", True)
os.environ["NETKET_DEBUG"] = "1"

# ============================================================
# USER OPTIONS
# ============================================================
LEARN_PHASE = True

NUM_SITES = 18
NUM_SAMPLES = 3* 2**9

# ---------- Multi-start ----------
NUM_STARTS = 10
NUM_ITERS_WARM = 1000

# ---------- Warmup early stopping ----------
WARMUP_ENERGY_TOL = 1e-4
WARMUP_PATIENCE = 20
WARMUP_WINDOW = 20
WARMUP_MIN_STEPS = 50

# ---------- Model ----------
EMBED_DIM = 16
NUM_HEADS = 8
NUM_LAYERS = 2
MLP_HIDDEN_DIM = 2 * EMBED_DIM
PATCH_SIZE = 1
CHUNK_SIZE = None

# ---------- Warmup SR ----------
WARM_SR_LR = 1e-3
WARM_SR_MOMENTUM = 0.85
WARM_SR_DIAGSHIFT = 1e-2
WARM_SR_SOLVER = pinv_smooth

# ---------- Sampling ----------
N_DISCARD_PER_CHAIN = 16

TODAY = date.today().isoformat()
JOB_BASE = (
    f"{NUM_SITES}-site_"
    f"{NUM_LAYERS}_layers_"
    f"{NUM_HEADS}_heads_"
    f"{PATCH_SIZE}_patchsize_"
    f"{NUM_SAMPLES}_samples_"
    f"{TODAY}_Kitaev_ViT_bestofK_only"
)

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
    n_chains = max(1, n_samples // 64)
    return nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=n_chains)


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


def load_best_energy(logfile):
    with open(logfile) as f:
        data = json.load(f)
    energies = np.array(data["Energy"]["Mean"]["real"], dtype=float)
    tail = energies[-min(20, len(energies)):]
    return float(np.mean(tail)), energies.tolist()


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


print(f"\n=== Multi-start SR warmup: {NUM_STARTS} starts, up to {NUM_ITERS_WARM} iterations each ===\n")

start_summaries = []
all_outs: List[str] = []

for k in range(NUM_STARTS):
    print(f"--- Start {k+1}/{NUM_STARTS} ---")

    vstate_k = nk.vqs.MCState(
        sampler=make_metropolis_local(hi, NUM_SAMPLES),
        model=build_model(),
        n_samples=NUM_SAMPLES,
        chunk_size=CHUNK_SIZE,
        n_discard_per_chain=N_DISCARD_PER_CHAIN,
        seed=1234 + k,
    )

    print("Parameters:", vstate_k.n_parameters)
    print("n_chains:", vstate_k.sampler.n_chains)
    print("chain_length:", vstate_k.chain_length)

    driver = nk.driver.VMC_SR(
        hamiltonian=ha,
        optimizer=optax.sgd(learning_rate=WARM_SR_LR),
        variational_state=vstate_k,
        momentum=WARM_SR_MOMENTUM,
        linear_solver=WARM_SR_SOLVER,
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
    final_params_file = f"{out_k}_final_variables.mpack"
    with open(final_params_file, "wb") as f:
        f.write(serialization.to_bytes(vstate_k.variables))

    print(f"Best energy for start {k}: {best_e}")

    start_summaries.append(
        {
            "start": k,
            "best_energy": best_e,
            "out": out_k,
            "final_variables_file": final_params_file,
            "n_logged_energies": len(energies),
        }
    )

start_summaries.sort(key=lambda d: d["best_energy"])
best = start_summaries[0]

best_params_file = f"best_warm_params_{JOB_BASE}.mpack"
with open(best["final_variables_file"], "rb") as f_in, open(best_params_file, "wb") as f_out:
    f_out.write(f_in.read())

summary = {
    "job_base": JOB_BASE,
    "num_sites": NUM_SITES,
    "num_samples": NUM_SAMPLES,
    "num_starts": NUM_STARTS,
    "num_iters_warm": NUM_ITERS_WARM,
    "model": {
	"embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "mlp_hidden_dim": MLP_HIDDEN_DIM,
        "patch_size": PATCH_SIZE,
        "chunk_size": CHUNK_SIZE,
        "learn_phase": LEARN_PHASE,
        "permutation": list(perm),
    },
    "sampling": {
        "n_discard_per_chain": N_DISCARD_PER_CHAIN,
    },
    "warmup": {
        "lr": WARM_SR_LR,
        "momentum": WARM_SR_MOMENTUM,
        "diagshift": WARM_SR_DIAGSHIFT,
        "energy_tol": WARMUP_ENERGY_TOL,
        "patience": WARMUP_PATIENCE,
        "window": WARMUP_WINDOW,
        "min_steps": WARMUP_MIN_STEPS,
    },
    "best_start": best,
    "best_params_file": best_params_file,
    "ranking": start_summaries,
}

summary_file = f"bestofk_summary_{JOB_BASE}.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Best start selected ===")
print("Start index:", best["start"])
print("Best warmup energy:", best["best_energy"])
print("Best params file:", best_params_file)
print("Summary file:", summary_file)

warm_energy = []
for out in all_outs:
    logf = f"{out}.log"
    if os.path.exists(logf):
        with open(logf) as f:
            data = json.load(f)
            warm_energy.extend(data["Energy"]["Mean"]["real"])

with open(f"mean_energy_warmup_{JOB_BASE}.txt", "w") as f:
    for e in warm_energy:
        f.write(f"{e}\n")


print("\n=== Warmup ranking ===")
for row in start_summaries:
    print(f"start={row['start']}  best_energy={row['best_energy']:.12f}  params={row['final_variables_file']}")
