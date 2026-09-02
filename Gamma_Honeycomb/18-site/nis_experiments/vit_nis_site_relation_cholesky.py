"""Short multi-GPU dense-kernel Cholesky-versus-CG 18-site ablation."""
from pathlib import Path
import os
import sys

# The weighted kernel is constructed from local Jacobian blocks on all visible
# GPUs. The final 6144 x 6144 Cholesky factorisation is replicated because
# standard JAX does not provide a distributed factorisation primitive.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

THIS_DIR = Path(__file__).resolve().parent
SITE_DIR = THIS_DIR.parent
SHARED_NIS_DIR = SITE_DIR.parent / "8-site" / "nis_experiments"
if str(SHARED_NIS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_NIS_DIR))

from run_gamma_nis import NISRunConfig, run_experiment


# ---------------- Cholesky ablation parameters: edit here ----------------
PREVIOUS_RUN = THIS_DIR / "results/nis/gamma_18site_site_relation"

CONFIG = NISRunConfig(
    # Use a fresh target and proposal initialization. This isolates the dense
    # solver experiment from the stopped matrix-free run and its checkpoints.
    num_sites=18,
    site_dir=str(SITE_DIR),
    variant="site_relation",
    execution_backend="netket",
    embed_dim=16,
    num_heads=2,
    num_layers=6,
    mlp_hidden_dim=32,
    patch_size=1,
    proposal_embed_dim=16,
    resume_target_checkpoint=None,
    resume_proposal_checkpoint=None,

    # A short diagnostic run: compare its energy, wall time, and acceptance
    # against the stopped matrix-free run before considering any longer run.
    num_iterations=1000,
    seed=0,
    diagnostics_dir="results/nis/gamma_18site_site_relation_distributed_cholesky",

    # Keep the same global pool. The Jacobian rows are divided across the
    # visible GPUs, preserving a fair comparison with the matrix-free run.
    n_proposals=3 * 1024,
    num_samples=512,
    ess_threshold=0.05,
    always_update_target=True,
    resample_method="systematic",
    use_multi_gpu=True,

    proposal_lr=1.0e-2,
    proposal_train_steps=1,
    proposal_train_batch_size=1024,

    # Continue at the learning rate reached at iteration 211 of the stopped
    # run. Construct the dense kernel in distributed local-Jacobian blocks;
    # standard JAX then factorises the small dense kernel independently on
    # each GPU.
    target_update="netket_sr",
    target_lr=3e-2,
    target_lr_final=None,
    target_lr_decay_steps=0,
    sr_diag_shift=1.0e-4,
    sr_chunk_size=2048,
    sr_trust_region=5.0e-1,
    sr_momentum=0.8,
    sr_proj_reg=None,
    sr_direct_solver="distributed_cholesky",
    sr_dense_jacobian_chunk_size=64,
    target_grad_batch_size=2048,
    local_energy_chunk_size=16384,

    post_update_energy_guard=True,
    post_update_energy_guard_sigmas=2.0,
    post_update_min_ess_fraction=0.10,
    compute_exact_ground_energy=True,
    compute_exact_diagnostics=False,
    heldout_diagnostics_every=50,
    checkpoint_every=10,
)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    run_experiment(CONFIG)
