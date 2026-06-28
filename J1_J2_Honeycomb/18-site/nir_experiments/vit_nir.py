import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
os.chdir(THIS_DIR)
if str(ROOT / "nir_experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "nir_experiments"))

from vit_nir_common import run_nir_experiment


embed_dim = int(os.environ.get("J1J2_NIR_EMBED_DIM", "16"))
mlp_hidden = int(os.environ.get("J1J2_NIR_MLP_HIDDEN", str(2 * embed_dim)))
num_iters_total = int(os.environ.get("J1J2_NIR_NUM_ITERS", "1000"))
train_lr_stage_1_iters = int(
    os.environ.get("J1J2_NIR_LR_STAGE_1_ITERS", "100")
)
train_lr_stage_2_iters = int(
    os.environ.get("J1J2_NIR_LR_STAGE_2_ITERS", "600")
)
nir_proposal_embed_dim = int(os.environ.get("J1J2_NIR_PROPOSAL_EMBED_DIM", "16"))
nir_proposal_mlp = int(
    os.environ.get("J1J2_NIR_PROPOSAL_MLP", str(2 * nir_proposal_embed_dim))
)
chunk_size_env = os.environ.get("J1J2_NIR_CHUNK_SIZE", "none").strip().lower()
chunk_size = None if chunk_size_env in {"", "none"} else int(chunk_size_env)


run_nir_experiment(
    num_sites=18,
    j1=float(os.environ.get("J1J2_J1", "1.0")),
    j2=float(os.environ.get("J1J2_J2", "0.0")),
    num_samples_stage_1=int(os.environ.get("J1J2_NIR_SAMPLES_STAGE_1", str(3 * 2**8))),
    num_samples_stage_2=int(os.environ.get("J1J2_NIR_SAMPLES_STAGE_2", str(3 * 2**8))),
    num_samples_stage_3=int(os.environ.get("J1J2_NIR_SAMPLES_STAGE_3", str(3 * 2**9))),
    num_iters_total=num_iters_total,
    patch_size=int(os.environ.get("J1J2_NIR_PATCH_SIZE", "1")),
    embed_dim=embed_dim,
    num_heads=int(os.environ.get("J1J2_NIR_NUM_HEADS", "2")),
    num_layers=int(os.environ.get("J1J2_NIR_NUM_LAYERS", "4")),
    mlp_hidden=mlp_hidden,
    chunk_size=chunk_size,
    train_lr_stage_1=float(os.environ.get("J1J2_NIR_LR_STAGE_1", "1e-2")),
    train_lr_stage_2=float(os.environ.get("J1J2_NIR_LR_STAGE_2", "1e-2")),
    train_lr_stage_3=float(os.environ.get("J1J2_NIR_LR_STAGE_3", "1e-3")),
    train_lr_stage_1_iters=train_lr_stage_1_iters,
    train_lr_stage_2_iters=train_lr_stage_2_iters,
    train_lr_stage_3_iters=int(os.environ.get("J1J2_NIR_LR_STAGE_3_ITERS", "200")),
    learn_phase_stage_1=os.environ.get("J1J2_NIR_LEARN_PHASE_STAGE_1", "true").lower() in {"1","true","yes","on"},
    learn_phase_stage_2=os.environ.get("J1J2_NIR_LEARN_PHASE_STAGE_2", "true").lower() in {"1","true","yes","on"},
    learn_phase_stage_3=os.environ.get("J1J2_NIR_LEARN_PHASE_STAGE_3", "true").lower() in {"1","true","yes","on"},
    target_sampler_name=os.environ.get("J1J2_NIR_TARGET_SAMPLER", "local"),
    target_optimizer_name=os.environ.get("J1J2_NIR_TARGET_OPTIMIZER", "sgd"),
    target_sgd_momentum=float(os.environ.get("J1J2_NIR_TARGET_MOMENTUM", "0.0")),
    target_preconditioner=os.environ.get("J1J2_NIR_TARGET_PRECONDITIONER", "minsr"),
    target_sr_diag_shift=float(os.environ.get("J1J2_NIR_TARGET_SR_DIAG_SHIFT", "1e-3")),
    target_sr_diag_shift_stage_1=float(os.environ.get("J1J2_NIR_TARGET_SR_DIAG_SHIFT_STAGE_1", "1e-3")),
    target_sr_diag_shift_stage_2=float(os.environ.get("J1J2_NIR_TARGET_SR_DIAG_SHIFT_STAGE_2", "1e-4")),
    target_sr_diag_shift_stage_3=float(os.environ.get("J1J2_NIR_TARGET_SR_DIAG_SHIFT_STAGE_3", "1e-5")),
    target_sr_proj_reg=(
        None
        if os.environ.get("J1J2_NIR_TARGET_SR_PROJ_REG") in {None, "", "none", "None"}
        else float(os.environ["J1J2_NIR_TARGET_SR_PROJ_REG"])
    ),
    target_sr_momentum=(
        None
        if os.environ.get("J1J2_NIR_TARGET_SR_MOMENTUM", "0.9") in {"", "none", "None"}
        else float(os.environ.get("J1J2_NIR_TARGET_SR_MOMENTUM", "0.9"))
    ),
    target_sr_mode=os.environ.get("J1J2_NIR_TARGET_SR_MODE", "complex"),
    nir_proposal_batch=int(os.environ.get("J1J2_NIR_PROPOSAL_BATCH", str(3 * 2**10))),
    nir_max_proposal_batches=int(os.environ.get("J1J2_NIR_MAX_PROPOSAL_BATCHES", "8")),
    nir_max_adaptive_rounds=int(os.environ.get("J1J2_NIR_MAX_ADAPTIVE_ROUNDS", "4")),
    nir_ess_threshold_frac=float(os.environ.get("J1J2_NIR_ESS_THRESHOLD_FRAC", "0.4")),
    nir_efficiency_threshold_stage_1=float(os.environ.get("J1J2_NIR_EFF_STAGE_1", "0.15")),
    nir_efficiency_threshold_stage_2=float(os.environ.get("J1J2_NIR_EFF_STAGE_2", "0.15")),
    nir_efficiency_threshold_stage_3=float(os.environ.get("J1J2_NIR_EFF_STAGE_3", "0.20")),
    nir_adapt_metric=os.environ.get("J1J2_NIR_ADAPT_METRIC", "efficiency"),
    nir_log_ratio_std_threshold_stage_1=(
        None
        if os.environ.get("J1J2_NIR_LOGPQ_STD_STAGE_1") in {None, "", "none", "None"}
        else float(os.environ["J1J2_NIR_LOGPQ_STD_STAGE_1"])
    ),
    nir_log_ratio_std_threshold_stage_2=(
        None
        if os.environ.get("J1J2_NIR_LOGPQ_STD_STAGE_2") in {None, "", "none", "None"}
        else float(os.environ["J1J2_NIR_LOGPQ_STD_STAGE_2"])
    ),
    nir_log_ratio_std_threshold_stage_3=(
        None
        if os.environ.get("J1J2_NIR_LOGPQ_STD_STAGE_3") in {None, "", "none", "None"}
        else float(os.environ["J1J2_NIR_LOGPQ_STD_STAGE_3"])
    ),
    nir_proposal_lr_stage_1=float(os.environ.get("J1J2_NIR_PROPOSAL_LR_STAGE_1", "3e-3")),
    nir_proposal_lr_stage_2=float(os.environ.get("J1J2_NIR_PROPOSAL_LR_STAGE_2", "3e-3")),
    nir_proposal_lr_stage_3=float(os.environ.get("J1J2_NIR_PROPOSAL_LR_STAGE_3", "3e-4")),
    nir_proposal_steps_stage_1=int(os.environ.get("J1J2_NIR_PROPOSAL_STEPS_STAGE_1", "8")),
    nir_proposal_steps_stage_2=int(os.environ.get("J1J2_NIR_PROPOSAL_STEPS_STAGE_2", "2")),
    nir_proposal_steps_stage_3=int(os.environ.get("J1J2_NIR_PROPOSAL_STEPS_STAGE_3", "1")),
    nir_proposal_embed_dim=nir_proposal_embed_dim,
    nir_proposal_heads=int(os.environ.get("J1J2_NIR_PROPOSAL_HEADS", "4")),
    nir_proposal_layers=int(os.environ.get("J1J2_NIR_PROPOSAL_LAYERS", "2")),
    nir_proposal_mlp=nir_proposal_mlp,
    nir_proposal_constrain_total_sz=os.environ.get(
        "J1J2_NIR_PROPOSAL_CONSTRAIN_SZ", "true"
    ).lower() in {"1", "true", "yes", "on"},
    nir_proposal_training_mode=os.environ.get(
        "J1J2_NIR_PROPOSAL_TRAINING_MODE", "resampled"
    ),
    nir_proposal_weight_power=float(
        os.environ.get("J1J2_NIR_PROPOSAL_WEIGHT_POWER", "1.0")
    ),
    nir_proposal_weight_clip_factor=(
        None
        if os.environ.get("J1J2_NIR_PROPOSAL_WEIGHT_CLIP_FACTOR")
        in {None, "", "none", "None"}
        else float(os.environ["J1J2_NIR_PROPOSAL_WEIGHT_CLIP_FACTOR"])
    ),
    nir_prob_floor=float(os.environ.get("J1J2_NIR_PROB_FLOOR", "1e-6")),
    nir_resampling_method=os.environ.get("J1J2_NIR_RESAMPLING_METHOD", "multinomial"),
    nir_weighted_pool_cap_factor=(
        None
        if os.environ.get("J1J2_NIR_WEIGHTED_POOL_CAP_FACTOR") in {None, "", "none", "None"}
        else float(os.environ.get("J1J2_NIR_WEIGHTED_POOL_CAP_FACTOR", "2.0"))
    ),
    nir_target_update_mode=os.environ.get("J1J2_NIR_TARGET_UPDATE_MODE", "resample"),
    nir_exact_refine_after_iter=(
        None
        if os.environ.get("J1J2_NIR_EXACT_REFINE_AFTER_ITER") in {None, "", "none", "None"}
        else int(os.environ["J1J2_NIR_EXACT_REFINE_AFTER_ITER"])
    ),
    nir_exact_refine_max_states=int(
        os.environ.get("J1J2_NIR_EXACT_REFINE_MAX_STATES", "4096")
    ),
    nir_sampler_refine_after_iter=(
        None
        if os.environ.get("J1J2_NIR_SAMPLER_REFINE_AFTER_ITER") in {None, "", "none", "None"}
        else int(os.environ["J1J2_NIR_SAMPLER_REFINE_AFTER_ITER"])
    ),
    nir_reset_state_on_refine=os.environ.get(
        "J1J2_NIR_RESET_STATE_ON_REFINE", "true"
    ).lower() in {"1", "true", "yes", "on"},
    resume_checkpoint_path=os.environ.get("J1J2_NIR_RESUME_CHECKPOINT"),
    resume_proposal_checkpoint_path=os.environ.get(
        "J1J2_NIR_RESUME_PROPOSAL_CHECKPOINT"
    ),
    rng_seed=(
        None
        if os.environ.get("J1J2_NIR_SEED") in {None, "", "none", "None"}
        else int(os.environ["J1J2_NIR_SEED"])
    ),
    model_type=os.environ.get("J1J2_NIR_MODEL_TYPE", "site_type_relation_gated_pool_bond"),
    run_tag=os.environ.get("J1J2_NIR_RUN_TAG"),
)
