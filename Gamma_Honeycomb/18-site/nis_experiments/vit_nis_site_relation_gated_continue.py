"""Continue the converged 18-site gated site-relation NIS run.

This is intentionally a normal editable experiment script. It restores the
target and proposal network parameters from the completed 32-dimensional
gated run, while writing all new checkpoints and diagnostics to a separate
directory.

The restore is parameter-only: optimiser, SR-momentum, sampler, and random-key
states start afresh.  This makes the continuation an independently reproducible
second optimisation stage rather than appending state to the original process.
"""

from pathlib import Path
import os
import sys


# Backend selection must happen before importing the JAX-based implementation.
# None selects the available accelerator automatically; use "gpu" or "cpu"
# only when a specific backend is required.
NIS_DEVICE = None
if NIS_DEVICE is not None:
    os.environ["JAX_PLATFORM_NAME"] = NIS_DEVICE
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


# ---------------------------------------------------------------------------
# Imports shared by all Gamma-honeycomb NIS experiment launchers
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
SITE_DIR = THIS_DIR.parent
SHARED_NIS_DIR = SITE_DIR.parent / "8-site" / "nis_experiments"

if str(SHARED_NIS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_NIS_DIR))

from run_gamma_nis import NISRunConfig, run_experiment  # noqa: E402


# ---------------------------------------------------------------------------
# Continuation source and output location
# ---------------------------------------------------------------------------
# Change SOURCE_RUN only when deliberately continuing a different completed run.
SOURCE_RUN = THIS_DIR / "results" / "nis" / "gamma_18site_site_relation_gated"

# Always use a new directory for a continuation so the source checkpoints and
# logs remain intact.
CONTINUATION_DIR = THIS_DIR / "results" / "nis" / "gamma_18site_site_relation_gated_d32_continue"


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
# These reproduce the completed source run. Modify values here only to make a
# deliberately different second-stage schedule.
CONFIG = NISRunConfig(
    num_sites=18,
    site_dir=str(SITE_DIR),
    variant="site_relation_gated",
    execution_backend="netket",
    embed_dim=32,
    num_heads=4,
    num_layers=6,
    # This must match the saved target checkpoint; without it the shared
    # runner falls back to 16 and cannot load the saved 32-by-64 MLP weights.
    mlp_hidden_dim=64,
    patch_size=1,
    num_iterations=1000,
    seed=0,
    diagnostics_dir=str(CONTINUATION_DIR),
    resume_target_checkpoint=str(SOURCE_RUN / "target.msgpack"),
    resume_proposal_checkpoint=str(SOURCE_RUN / "proposal.msgpack"),
    n_proposals=3 * 1024,
    num_samples=512,
    ess_threshold=0.05,
    always_update_target=True,
    resample_method="systematic",
    use_multi_gpu=True,
    proposal_embed_dim=32,
    # The completed proposal checkpoint has two layers. Keep this at two to
    # restore it exactly. To test the new four-layer proposal instead, set
    # this to 4 and set ``resume_proposal_checkpoint=None``.
    proposal_num_layers=2,
    proposal_lr=1.0e-2,
    proposal_train_steps=1,
    proposal_train_batch_size=1024,
    target_update="netket_sr",
    target_lr=3.0e-2,
    target_lr_final=1.0e-2,
    target_lr_decay_steps=10000,
    sr_diag_shift=1.0e-3,
    sr_chunk_size=2048,
    sr_trust_region=5.0e-1,
    sr_momentum=0.8,
    sr_proj_reg=None,
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


if __name__ == "__main__":
    run_experiment(CONFIG)
