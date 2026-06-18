import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
os.chdir(THIS_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vit_srt_common import run_srt_experiment


run_srt_experiment(
    site_dir=THIS_DIR,
    num_sites=8,
    num_samples_warmup=int(os.environ.get("GAMMA_NUM_SAMPLES_WARMUP", str(3 * 2**8))),
    num_samples=int(os.environ.get("GAMMA_NUM_SAMPLES", str(3 * 2**9))),
    num_iters_warm=int(os.environ.get("GAMMA_NUM_ITERS_WARM", "100")),
    num_iters_main=int(os.environ.get("GAMMA_NUM_ITERS_MAIN", "1000")),
    num_iters_refine=int(os.environ.get("GAMMA_NUM_ITERS_REFINE", "2000")),
    patch_size=int(os.environ.get("GAMMA_PATCH_SIZE", "1")),
    embed_dim=int(os.environ.get("GAMMA_EMBED_DIM", "32")),
    num_heads=int(os.environ.get("GAMMA_NUM_HEADS", "4")),
    num_layers=int(os.environ.get("GAMMA_NUM_LAYERS", "4")),
    mlp_hidden_dim=(
        int(os.environ["GAMMA_MLP_HIDDEN_DIM"])
        if "GAMMA_MLP_HIDDEN_DIM" in os.environ
        else None
    ),
    learn_phase_warmup=os.environ.get("GAMMA_LEARN_PHASE_WARMUP", "false").lower() in {"1", "true", "yes", "on"},
    learn_phase_main=os.environ.get("GAMMA_LEARN_PHASE_MAIN", "true").lower() in {"1", "true", "yes", "on"},
    warm_sr_lr=float(os.environ.get("GAMMA_WARM_SR_LR", "1e-3")),
    warm_sr_momentum=float(os.environ.get("GAMMA_WARM_SR_MOMENTUM", "0.7")),
    warm_sr_diagshift=float(os.environ.get("GAMMA_WARM_SR_DIAGSHIFT", "1e-2")),
    main_sr_lr=float(os.environ.get("GAMMA_MAIN_SR_LR", "1e-2")),
    main_sr_momentum=float(os.environ.get("GAMMA_MAIN_SR_MOMENTUM", "0.9")),
    main_sr_diagshift=float(os.environ.get("GAMMA_MAIN_SR_DIAGSHIFT", "1e-4")),
    refine_sr_lr=float(os.environ.get("GAMMA_REFINE_SR_LR", "1e-4")),
    refine_sr_momentum=float(os.environ.get("GAMMA_REFINE_SR_MOMENTUM", "0.0")),
    n_discard_per_chain=int(os.environ.get("GAMMA_N_DISCARD_PER_CHAIN", "4")),
    target_chain_length=int(os.environ.get("GAMMA_TARGET_CHAIN_LENGTH", "64")),
    pt_sweep_size=int(os.environ.get("GAMMA_PT_SWEEP_SIZE", str(8 * 2))),
    chunk_size=(
        None
        if os.environ.get("GAMMA_CHUNK_SIZE", "none").strip().lower() in {"", "none"}
        else int(os.environ["GAMMA_CHUNK_SIZE"])
    ),
    chunk_size_bwd=(
        None
        if os.environ.get("GAMMA_CHUNK_SIZE_BWD", "none").strip().lower() in {"", "none"}
        else int(os.environ["GAMMA_CHUNK_SIZE_BWD"])
    ),
    num_starts=int(os.environ.get("GAMMA_NUM_STARTS", "1")),
    netket_debug=os.environ.get("GAMMA_NETKET_DEBUG", "false").lower() in {"1", "true", "yes", "on"},
    profile_time=os.environ.get("GAMMA_PROFILE_TIME", "false").lower() in {"1", "true", "yes", "on"},
    log_step_size=int(os.environ.get("GAMMA_LOG_STEP_SIZE", "1")),
    write_every=int(os.environ.get("GAMMA_WRITE_EVERY", "1")),
    save_params_every=int(os.environ.get("GAMMA_SAVE_PARAMS_EVERY", "25")),
    use_experimental_vmc_sr=os.environ.get("GAMMA_USE_EXPERIMENTAL_VMC_SR", "false").lower() in {"1", "true", "yes", "on"},
    experimental_use_ntk=os.environ.get("GAMMA_EXPERIMENTAL_USE_NTK", "true").lower() in {"1", "true", "yes", "on"},
    experimental_on_the_fly=os.environ.get("GAMMA_EXPERIMENTAL_ON_THE_FLY", "true").lower() in {"1", "true", "yes", "on"},
    sampler_name=os.environ.get("GAMMA_SAMPLER", "local"),
    sampler_name_refine=os.environ.get("GAMMA_SAMPLER_REFINE", "pt_local"),
    model_type=os.environ.get("GAMMA_MODEL_TYPE", "site_type_relation"),
    run_tag=os.environ.get("GAMMA_RUN_TAG"),
)
