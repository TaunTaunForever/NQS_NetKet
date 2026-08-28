#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

# Let JAX choose the available backend, so cluster runs can use GPU without
# requiring environment variables while CPU-only machines still fall back.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPRO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPRO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nir_paper_reproduction.hamiltonians import make_square_j1j2
from nir_paper_reproduction.models import PaperSquareJ1J2ResNetLogPsi
from nir_paper_reproduction.nir_core import (
    NIRSettings,
    exact_ground_state_energy,
    run_single_state_nir,
    safe_hilbert_n_states,
)


PAPER_LENGTH = 10
PAPER_J1 = 1.0
PAPER_J2 = 0.5
PAPER_NUM_STEPS = 10000
PAPER_BATCH_SIZE = 512
PAPER_PROPOSAL_BATCH = 512
PAPER_MAX_PROPOSAL_BATCHES = 64
PAPER_MAX_ADAPTIVE_ROUNDS = 8
PAPER_ALPHA_ESS = 2.0
PAPER_ALPHA_EFF = 0.1
PAPER_TARGET_LR = 1e-3
PAPER_PROPOSAL_LR = 1e-3
PAPER_SR_DIAG_SHIFT = 1e-3
PAPER_PROPOSAL_EMBED_DIM = 32
PAPER_PROPOSAL_HEADS = 4
PAPER_PROPOSAL_LAYERS = 4
PAPER_RESNET_CHANNELS = 16
PAPER_RESNET_BLOCKS = 4
PAPER_LOG_EVERY = 10


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.lower() in {"", "none", "null"}:
        return None
    return float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def main() -> None:
    length = env_int("NIR_PAPER_J1J2_LENGTH", PAPER_LENGTH)
    j1 = env_float("NIR_PAPER_J1", PAPER_J1)
    j2 = env_float("NIR_PAPER_J2", PAPER_J2)
    hilbert, _graph, hamiltonian = make_square_j1j2(length=length, j1=j1, j2=j2)
    exact_energy = exact_ground_state_energy(
        hamiltonian,
        max_states=env_int("NIR_PAPER_MAX_ED_STATES", 65536),
    )
    if exact_energy is None:
        n_states = safe_hilbert_n_states(hilbert)
        if n_states is None:
            print("Exact ground-state energy: skipped (Hilbert space is too large to index)")
        else:
            print(
                "Exact ground-state energy: skipped "
                f"(Hilbert size {n_states} exceeds NIR_PAPER_MAX_ED_STATES)"
            )
        if length == 10 and abs(j1 - 1.0) < 1e-12 and abs(j2 - 0.5) < 1e-12:
            print("Paper comparison range: E/N ≈ -0.4966 to -0.4969")
    else:
        print(f"Exact ground-state energy: {exact_energy:.12f}")
        print(f"Exact ground-state energy per site: {exact_energy / hilbert.size:.12f}")

    target_model = PaperSquareJ1J2ResNetLogPsi(
        length=length,
        channels=env_int("NIR_PAPER_RESNET_CHANNELS", PAPER_RESNET_CHANNELS),
        num_blocks=env_int("NIR_PAPER_RESNET_BLOCKS", PAPER_RESNET_BLOCKS),
    )
    settings = NIRSettings(
        num_steps=env_int("NIR_PAPER_NUM_STEPS", PAPER_NUM_STEPS),
        n_samples=env_int("NIR_PAPER_BATCH_SIZE", PAPER_BATCH_SIZE),
        proposal_batch=env_int("NIR_PAPER_PROPOSAL_BATCH", PAPER_PROPOSAL_BATCH),
        max_proposal_batches=env_int(
            "NIR_PAPER_MAX_PROPOSAL_BATCHES",
            PAPER_MAX_PROPOSAL_BATCHES,
        ),
        max_adaptive_rounds=env_int(
            "NIR_PAPER_MAX_ADAPTIVE_ROUNDS",
            PAPER_MAX_ADAPTIVE_ROUNDS,
        ),
        alpha_ess=env_float("NIR_PAPER_ALPHA_ESS", PAPER_ALPHA_ESS),
        alpha_eff=env_float("NIR_PAPER_ALPHA_EFF", PAPER_ALPHA_EFF),
        target_lr=env_float("NIR_PAPER_TARGET_LR", PAPER_TARGET_LR),
        proposal_lr=env_float("NIR_PAPER_PROPOSAL_LR", PAPER_PROPOSAL_LR),
        target_preconditioner=os.environ.get("NIR_PAPER_TARGET_PRECONDITIONER", "minsr"),
        sr_diag_shift=env_float("NIR_PAPER_SR_DIAG_SHIFT", PAPER_SR_DIAG_SHIFT),
        proposal_embed_dim=env_int(
            "NIR_PAPER_PROPOSAL_EMBED_DIM",
            PAPER_PROPOSAL_EMBED_DIM,
        ),
        proposal_heads=env_int("NIR_PAPER_PROPOSAL_HEADS", PAPER_PROPOSAL_HEADS),
        proposal_layers=env_int("NIR_PAPER_PROPOSAL_LAYERS", PAPER_PROPOSAL_LAYERS),
        prob_floor=env_float("NIR_PAPER_PROB_FLOOR", 1e-6),
        seed=env_int("NIR_PAPER_SEED", 1234),
        chunk_size=None,
        log_every=env_int("NIR_PAPER_LOG_EVERY", PAPER_LOG_EVERY),
        run_name=os.environ.get(
            "NIR_PAPER_RUN_NAME",
            f"j1j2_square_{length}x{length}_j2={j2}_paper_resnet_nir",
        ),
        reject_nonfinite_updates=env_bool("NIR_PAPER_REJECT_NONFINITE_UPDATES", True),
        target_update_norm_clip=env_optional_float("NIR_PAPER_TARGET_UPDATE_NORM_CLIP"),
        target_update_check_samples=env_int("NIR_PAPER_TARGET_UPDATE_CHECK_SAMPLES", 64),
    )
    result = run_single_state_nir(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        target_model=target_model,
        settings=settings,
        output_root=REPRO_ROOT / "runs",
    )
    if not result["history"]:
        print("No optimization iterations requested; initialization check complete.")
        return

    final_energy = result["history"][-1]["energy"]
    tail100_energy = sum(row["energy"] for row in result["history"][-100:]) / min(
        len(result["history"]),
        100,
    )
    print(f"Final variational energy: {final_energy:.12f}")
    print(f"Final variational energy per site: {final_energy / hilbert.size:.12f}")
    print(f"Final-100 mean energy: {tail100_energy:.12f}")
    print(f"Final-100 mean energy per site: {tail100_energy / hilbert.size:.12f}")
    if exact_energy is not None:
        print(f"Final energy gap to ED: {final_energy - exact_energy:.12e}")
        print(f"Final-100 mean gap to ED: {tail100_energy - exact_energy:.12e}")


if __name__ == "__main__":
    main()
