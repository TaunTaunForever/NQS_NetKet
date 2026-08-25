"""Independent continuation/refinement of the 8-site site-relation NIS run.

Each invocation restores the best available source parameters into a *new*
timestamped output directory. This avoids silently restarting from the base
run and appending unrelated metrics to an earlier refinement log.
"""
import os
from datetime import datetime
from pathlib import Path

# Backend selection must happen before importing the JAX-based implementation.
NIS_DEVICE = None  # None = automatic; use "gpu" or "cpu" only when needed.
if NIS_DEVICE is not None:
    os.environ["JAX_PLATFORM_NAME"] = NIS_DEVICE
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("NETKET_EXPERIMENTAL_SHARDING", "1")

from run_gamma_nis import NISRunConfig, run_experiment


HERE = Path(__file__).resolve().parent
BASE_RESULTS = HERE / "results/nis/gamma_8site_site_relation"
PRIOR_REFINEMENT_RESULTS = HERE / "results/nis/gamma_8site_site_relation_refine"


def source_checkpoint(name: str) -> Path:
    """Prefer the preceding refinement's best validated checkpoint."""
    for directory in (PRIOR_REFINEMENT_RESULTS, BASE_RESULTS):
        for filename in (f"best_{name}.msgpack", f"{name}.msgpack"):
            path = directory / filename
            if path.is_file():
                return path
    raise FileNotFoundError(
        "No refinement source checkpoint found. Expected target/proposal.msgpack "
        f"under {PRIOR_REFINEMENT_RESULTS} or {BASE_RESULTS}."
    )


RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_RESULTS = HERE / f"results/nis/gamma_8site_site_relation_refine_continue_{RUN_TAG}"

# -------------------- Refinement parameters: edit here --------------------
CONFIG = NISRunConfig(
    variant="site_relation",
    execution_backend="netket",
    num_iterations=5000,
    seed=1,
    diagnostics_dir=str(OUTPUT_RESULTS),
    resume_target_checkpoint=str(source_checkpoint("target")),
    resume_proposal_checkpoint=str(source_checkpoint("proposal")),

    # Keep this architecture identical to the source checkpoint.
    embed_dim=8,
    num_heads=2,
    num_layers=2,
    mlp_hidden_dim=16,
    patch_size=1,
    proposal_embed_dim=8,

    # Weighted NIS on the native three-GPU mesh.
    n_proposals=4098,
    num_samples=512,
    use_multi_gpu=True,
    ess_threshold=0.05,
    always_update_target=True,
    proposal_lr=3.0e-4,
    proposal_train_steps=1,
    proposal_train_batch_size=512,
    # ESS is already high in the source run. Keep the proposal fixed so that
    # late-stage target refinement is measured against a stable sampler.
    proposal_update_interval=1,
    proposal_freeze_after=0,

    # Conservative, annealed NetKet-style weighted minimum-SR refinement.
    # Exact 8-site values are logged solely for validation; they are never
    # used in the gradient or QGT. The solver policy is internal.
    target_update="netket_sr",
    target_lr=1.0e-2,
    target_lr_final=2.0e-5,
    target_lr_decay_steps=5000,
    sr_diag_shift=1.0e-4,
    sr_chunk_size=1366,
    sr_trust_region=5.0e-1,
    sr_momentum=0.8,  # Set to None to disable SPRING memory.
    sr_proj_reg=None,  # Optional weighted null-mode regularisation.
    target_grad_batch_size=1366,
    local_energy_chunk_size=16384,

    # Exact calculations are 8-site validation only, never part of the update.
    compute_exact_ground_energy=True,
    compute_exact_diagnostics=True,
    exact_diagnostics_every=25,
    checkpoint_every=25,
    # Independent weighted pool; diagnostic only, never used for updates.
    heldout_diagnostics_every=100,
    # Reject target updates that degrade the paired weighted estimate.
    post_update_energy_guard=True,
    post_update_energy_guard_sigmas=2.0,
    post_update_min_ess_fraction=0.10,
)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment(CONFIG)
