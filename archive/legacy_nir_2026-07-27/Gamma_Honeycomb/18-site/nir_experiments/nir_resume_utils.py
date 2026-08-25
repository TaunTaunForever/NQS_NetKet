from __future__ import annotations

from pathlib import Path

from flax import serialization


def save_training_state(
    path: str | Path,
    *,
    variables,
    proposal_params,
    proposal_opt_state,
    target_opt_state,
    target_update_state,
    rng,
    completed_iterations: int,
    best_energy,
    best_energy_distance_to_exact,
    best_iteration,
):
    state = {
        "variables": variables,
        "proposal_params": proposal_params,
        "proposal_opt_state": proposal_opt_state,
        "target_opt_state": target_opt_state,
        "target_update_state": target_update_state,
        "rng": rng,
        "completed_iterations": int(completed_iterations),
        "best_energy": best_energy,
        "best_energy_distance_to_exact": best_energy_distance_to_exact,
        "best_iteration": best_iteration,
    }
    with open(Path(path), "wb") as f:
        f.write(serialization.to_bytes(state))


def load_training_state(path: str | Path, template):
    with open(Path(path), "rb") as f:
        payload = f.read()
    return serialization.from_bytes(template, payload)
