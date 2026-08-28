#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPRO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPRO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nir_paper_reproduction.hamiltonians import make_tfi
from nir_paper_reproduction.models import PaperMLPLogPsi
from nir_paper_reproduction.nir_core import (
    NIRSettings,
    empirical_distribution,
    exact_ground_state_energy,
    exact_probabilities,
    jensen_shannon_divergence,
    run_single_state_nir,
    safe_hilbert_n_states,
)


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
    lx = env_int("NIR_PAPER_TFI_LX", 2)
    ly = env_int("NIR_PAPER_TFI_LY", 3)
    g = env_float("NIR_PAPER_TFI_G", 0.01)
    hidden_width = env_int("NIR_PAPER_TFI_MLP_WIDTH", 128)

    hilbert, _graph, hamiltonian = make_tfi((lx, ly), g)
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
    else:
        print(f"Exact ground-state energy: {exact_energy:.12f}")

    target_model = PaperMLPLogPsi(
        hidden_width=hidden_width,
        num_layers=env_int("NIR_PAPER_TFI_MLP_LAYERS", 4),
    )
    settings = NIRSettings(
        num_steps=env_int("NIR_PAPER_NUM_STEPS", 20),
        n_samples=env_int("NIR_PAPER_BATCH_SIZE", 128),
        proposal_batch=env_int("NIR_PAPER_PROPOSAL_BATCH", 128),
        max_proposal_batches=env_int("NIR_PAPER_MAX_PROPOSAL_BATCHES", 16),
        max_adaptive_rounds=env_int("NIR_PAPER_MAX_ADAPTIVE_ROUNDS", 4),
        alpha_ess=env_float("NIR_PAPER_ALPHA_ESS", 2.0),
        alpha_eff=env_float("NIR_PAPER_ALPHA_EFF", 0.1),
        target_lr=env_float("NIR_PAPER_TARGET_LR", 1e-3),
        proposal_lr=env_float("NIR_PAPER_PROPOSAL_LR", 1e-3),
        target_preconditioner=os.environ.get("NIR_PAPER_TARGET_PRECONDITIONER", "minsr"),
        sr_diag_shift=env_float("NIR_PAPER_SR_DIAG_SHIFT", 1e-3),
        proposal_embed_dim=env_int("NIR_PAPER_PROPOSAL_EMBED_DIM", 32),
        proposal_heads=env_int("NIR_PAPER_PROPOSAL_HEADS", 4),
        proposal_layers=env_int("NIR_PAPER_PROPOSAL_LAYERS", 4),
        prob_floor=env_float("NIR_PAPER_PROB_FLOOR", 1e-6),
        seed=env_int("NIR_PAPER_SEED", 1234),
        chunk_size=None,
        log_every=env_int("NIR_PAPER_LOG_EVERY", 1),
        run_name=os.environ.get(
            "NIR_PAPER_RUN_NAME",
            f"tfi_single_state_{lx}x{ly}_g={g}_paper_nir_smoke",
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

    n_states = safe_hilbert_n_states(hilbert)
    if n_states is not None and n_states <= env_int("NIR_PAPER_MAX_EXACT_STATES", 4096):
        _states, exact_probs = exact_probabilities(result["vstate"])
        empirical = empirical_distribution(result["last_resampled"], hilbert.size)
        jsd = jensen_shannon_divergence(exact_probs, empirical)
        print(f"Final resampled-vs-exact JSD: {jsd:.8e}")

    final_energy = result["history"][-1]["energy"]
    tail100_energy = sum(row["energy"] for row in result["history"][-100:]) / min(
        len(result["history"]),
        100,
    )
    print(f"Final variational energy: {final_energy:.12f}")
    print(f"Final-100 mean energy: {tail100_energy:.12f}")
    if exact_energy is not None:
        print(f"Final energy gap to ED: {final_energy - exact_energy:.12e}")
        print(f"Final-100 mean gap to ED: {tail100_energy - exact_energy:.12e}")


if __name__ == "__main__":
    main()
