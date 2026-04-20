mport os
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
NUM_SAMPLES_WARMUP = 2**9
NUM_SAMPLES = 2**12

# ---------- Multi-start ----------
NUM_STARTS = 20
NUM_ITERS_WARM = 1000      # max SR warmup iterations for each restart
NUM_ITERS_MAIN = 1000     # main SR continuation of best restart
NUM_ITERS_SR = 5000	  # final PT-SR refinement

# ---------- Warmup early stopping ----------
WARMUP_ENERGY_TOL = 1e-3
WARMUP_PATIENCE = 40
WARMUP_WINDOW = 150
WARMUP_MIN_STEPS = 50

# ---------- Model ----------
EMBED_DIM = 48
NUM_HEADS = 4
NUM_LAYERS = 2
MLP_HIDDEN_DIM = 2 * EMBED_DIM
PATCH_SIZE = 3
CHUNK_SIZE = None

# ---------- Warmup SR ----------
WARM_SR_LR = 1e-2
WARM_SR_MOMENTUM = 0.9
WARM_SR_DIAGSHIFT = 1e-2
WARM_SR_SOLVER = pinv_smooth

# ---------- Main SR ----------
MAIN_SR_LR = 5e-3
MAIN_SR_MOMENTUM = 0.9
MAIN_SR_DIAGSHIFT = 1e-3
MAIN_SR_SOLVER = pinv_smooth

# ---------- Final PT-SR ----------
SR_LR = 1e-3
SR_MOMENTUM = 0.9
SR_SOLVER = pinv_smooth
SR_DIAGSHIFT_SCHEDULE = [
    (0.30, 1e-2, "sr_shift1e-2"),
    (0.40, 1e-2, "sr_shift5e-4"),
    (0.30, 2e-4, "sr_shift2e-4"),
]

# ---------- Sampling ----------
N_DISCARD_PER_CHAIN = 64

TODAY = date.today().isoformat()

JOB_BASE = (
    f"{NUM_SITES}-site_"
    f"{NUM_LAYERS}_layers_"
    f"{NUM_HEADS}_heads_"
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


