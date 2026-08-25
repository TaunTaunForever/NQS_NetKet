"""Direct, editable Gamma 18-site site-relation weighted-NIS experiment."""
from pathlib import Path
import os
import sys

# Backend selection must happen before importing the JAX-based implementation.
NIS_DEVICE = None  # None = automatic; use "gpu" or "cpu" only when needed.
if NIS_DEVICE is not None:
    os.environ["JAX_PLATFORM_NAME"] = NIS_DEVICE
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
# Use NetKet's native SPMD mesh for the proposal pool and the matrix-free
# weighted quantum-geometric-tensor solve. Set this to "0" only when testing
# the legacy explicit-JAX/pmap backend.
os.environ.setdefault("NETKET_EXPERIMENTAL_SHARDING", "1")

THIS_DIR = Path(__file__).resolve().parent
SITE_DIR = THIS_DIR.parent
SHARED_NIS_DIR = SITE_DIR.parent / "8-site" / "nis_experiments"
if str(SHARED_NIS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_NIS_DIR))

from run_gamma_nis import NISRunConfig, run_experiment


# -------------------- Experiment parameters: edit here --------------------
CONFIG = NISRunConfig(
    # Geometry and target model
    num_sites=18,
    site_dir=str(SITE_DIR),
    variant="site_relation",
    # Default: NetKet VariationalState + NetKet-compatible NIS driver.
    # Use "explicit_jax" only to opt into the legacy experimental pmap backend.
    execution_backend="netket",
    embed_dim=16,
    num_heads=4,
    num_layers=2,
    mlp_hidden_dim=32,
    patch_size=1,

    # Run length and output. Change these paths when keeping multiple runs.
    num_iterations=10000,
    seed=0,
    diagnostics_dir="results/nis/gamma_18site_site_relation",
    resume_target_checkpoint=None,
    resume_proposal_checkpoint=None,

    # NIS sampling. 3072 is 3 x 1024, matching the usual three-GPU setup and
    # the established 18-site sampling budget. Keep this divisible by the
    # number of visible GPUs when native sharding is enabled.
    n_proposals= 3 * 512,
    num_samples= 1024,
    ess_threshold=0.05,
    always_update_target=True,
    resample_method="systematic",
    use_multi_gpu=True,

    # Autoregressive proposal training
    proposal_embed_dim=16,
    proposal_lr=1.0e-2,
    proposal_train_steps=1,
    proposal_train_batch_size=1024,

    # NetKet-style weighted minimum stochastic reconfiguration. Plain SGD
    # follows the solve so the trust-region bound has its intended
    # quantum-geometric meaning. The solver policy is maintained internally.
    target_update="netket_sr",
    target_lr=3.0e-2,
    target_lr_final=1.0e-2,
    target_lr_decay_steps=5000,
    sr_diag_shift=1.0e-4,
    sr_chunk_size=2048,
    sr_trust_region=5.0e-1,
    # NetKet-style SPRING memory. Set to None to disable it.
    sr_momentum=0.8,
    # Optional weighted null-mode regularisation; leave disabled by default.
    sr_proj_reg=None,
    target_grad_batch_size=2048,
    local_energy_chunk_size=16384,

    # Protect the accepted target state. A candidate update is evaluated on
    # the current proposal pool and rolled back only when it is significantly
    # worse than the pre-update estimate or loses proposal coverage.
    post_update_energy_guard=True,
    post_update_energy_guard_sigmas=2.0,
    post_update_min_ess_fraction=0.10,

    # The 18-site Lanczos energy is printed once before optimisation. Full
    # Hilbert-space diagnostics would enumerate 2**18 states repeatedly, so
    # keep them disabled in this scalable training launcher.
    compute_exact_ground_energy=True,
    compute_exact_diagnostics=False,
    # Independent weighted-pool check; it never supplies an optimization
    # update, but reports whether the SR direction generalizes off-pool.
    heldout_diagnostics_every=50,
    checkpoint_every=10,
)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    run_experiment(CONFIG)
