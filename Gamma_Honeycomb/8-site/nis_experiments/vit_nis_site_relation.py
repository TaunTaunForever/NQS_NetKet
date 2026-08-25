"""Direct, editable Gamma 8-site site-relation weighted-NIS experiment."""
import os

# Backend selection must happen before importing the JAX-based implementation.
NIS_DEVICE = None  # None = automatic; use "gpu" or "cpu" only when needed.
if NIS_DEVICE is not None:
    os.environ["JAX_PLATFORM_NAME"] = NIS_DEVICE
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
# Use NetKet's native SPMD mesh for weighted proposal pools and matrix-free SR.
# This must be set before importing JAX or NetKet. Set it to "0" only when
# deliberately running the legacy explicit-JAX/pmap backend.
os.environ.setdefault("NETKET_EXPERIMENTAL_SHARDING", "1")

from run_gamma_nis import NISRunConfig, run_experiment

# -------------------- Experiment parameters: edit here --------------------
CONFIG = NISRunConfig(
    variant="site_relation",
    # Default: NetKet VariationalState + NetKet-compatible NIS driver.
    # Use "explicit_jax" only to opt into the experimental pmap backend.
    execution_backend="netket",
    num_iterations=1000,
    seed=0,
    diagnostics_dir="results/nis/gamma_8site_site_relation",
    # Leave these ``None`` for a fresh run. The dedicated refinement launcher
    # restores them from a previous checkpoint at a lower learning rate.
    resume_target_checkpoint=None,
    resume_proposal_checkpoint=None,

    # NIS sampling
    n_proposals=3*512,
    num_samples=256,
    ess_threshold=0.05,
    always_update_target=True,
    resample_method="systematic",
    # NetKet shards one global weighted proposal pool across all visible GPUs.
    # Keep n_proposals divisible by the GPU count (4098 = 3 × 1366 here).
    use_multi_gpu=True,

    # Reject target updates that degrade the paired weighted estimate.
    post_update_energy_guard=True,
    post_update_energy_guard_sigmas=2.0,
    post_update_min_ess_fraction=0.10,

    # Site-relation target model
    embed_dim=8,
    num_heads=2,
    num_layers=4,
    mlp_hidden_dim=16,
    patch_size=1,

    # Proposal model and optimisation
    proposal_embed_dim=8,
    proposal_lr=1.0e-2,
    proposal_train_steps=1,
    proposal_train_batch_size=512,
    # NetKet-style weighted minimum SR. Plain SGD follows the solve so the
    # QGT trust-region bound has its intended geometric meaning. The
    # conjugate-gradient policy is maintained internally.
    target_update="netket_sr",
    target_lr=3.0e-2,
    # Exponentially anneal stochastic SR updates into a low-noise final phase.
    target_lr_final=1.0e-2,
    target_lr_decay_steps=10000,
    sr_diag_shift=1.0e-4,
    sr_chunk_size=1366,
    sr_trust_region=5.0e-1,
    # NetKet-style SPRING memory. Set to None to disable it.
    sr_momentum=0.8,
    # Optional weighted null-mode regularisation; leave disabled by default.
    sr_proj_reg=None,
    target_grad_batch_size=1366,
    local_energy_chunk_size=16384,

    # Exact methods are optional 8-site validation only. Leave both false for
    # the scalable NIS path used at larger systems.
    compute_exact_ground_energy=True,
    compute_exact_diagnostics=True,
    exact_diagnostics_every=10,
    checkpoint_every=10
)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment(CONFIG)
