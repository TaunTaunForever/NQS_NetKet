"""Direct, editable Gamma 8-site input-projection weighted-NIS experiment."""
import os

# Backend selection must happen before importing the JAX-based implementation.
NIS_DEVICE = None  # None = automatic; use "gpu" or "cpu" only when needed.
if NIS_DEVICE is not None:
    os.environ["JAX_PLATFORM_NAME"] = NIS_DEVICE
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

from run_gamma_nis import NISRunConfig, run_experiment

# -------------------- Experiment parameters: edit here --------------------
CONFIG = NISRunConfig(
    variant="inputproj",
    execution_backend="netket",
    num_iterations=10000,
    seed=0,
    diagnostics_dir="results/nis/gamma_8site_inputproj",

    # NIS sampling. Keep n_proposals divisible by the GPU count.
    n_proposals=4098,
    num_samples=512,
    ess_threshold=0.05,
    always_update_target=True,
    resample_method="systematic",
    use_multi_gpu=True,

    # Reject target updates that degrade the paired weighted estimate.
    post_update_energy_guard=True,
    post_update_energy_guard_sigmas=2.0,
    post_update_min_ess_fraction=0.10,

    # Input-projection target model
    embed_dim=8,
    num_heads=2,
    num_layers=2,
    mlp_hidden_dim=16,
    patch_size=1,

    # Proposal model
    proposal_embed_dim=8,
    proposal_lr=1.0e-2,
    proposal_train_steps=1,
    proposal_train_batch_size=512,
    # NetKet-style weighted minimum SR. The linear-solver details are kept
    # internal; the three settings below are the intended SR controls.
    target_update="netket_sr",
    target_lr=3.0e-2,
    target_lr_final=None,  # Set a positive value to enable exponential annealing.
    target_lr_decay_steps=0,
    sr_diag_shift=1.0e-4,
    sr_chunk_size=512,
    sr_trust_region=5.0e-2,
    sr_momentum=0.8,  # Set to None to disable SPRING memory.
    sr_proj_reg=None,  # Optional weighted null-mode regularisation.
    target_grad_batch_size=256,
    local_energy_chunk_size=16384,

    # Optional 8-site validation only; leave disabled for scalable runs.
    compute_exact_ground_energy=False,
    compute_exact_diagnostics=False,
    exact_diagnostics_every=25,
    checkpoint_every=25,
)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment(CONFIG)
