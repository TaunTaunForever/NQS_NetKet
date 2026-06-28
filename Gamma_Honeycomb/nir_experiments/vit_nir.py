import os
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
MODEL_ROOT = ROOT / "18-site"

os.chdir(THIS_DIR)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

if "GAMMA_NIR_DEVICE" in os.environ:
    os.environ.setdefault("JAX_PLATFORM_NAME", os.environ["GAMMA_NIR_DEVICE"])

from vit_nir_common import run_nir_experiment


def env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def env_optional_int(name: str):
    value = os.environ.get(name)
    if value in {None, "", "none", "None"}:
        return None
    return int(value)


def env_optional_float(name: str):
    value = os.environ.get(name)
    if value in {None, "", "none", "None"}:
        return None
    return float(value)


num_sites = int(os.environ.get("GAMMA_NIR_NUM_SITES", "18"))
embed_dim = int(os.environ.get("GAMMA_NIR_EMBED_DIM", "24"))
mlp_hidden = int(os.environ.get("GAMMA_NIR_MLP_HIDDEN", str(2 * embed_dim)))
num_iters_total = int(os.environ.get("GAMMA_NIR_NUM_ITERS", "1000"))
train_lr_stage_1_iters = int(os.environ.get("GAMMA_NIR_LR_STAGE_1_ITERS", "100"))
train_lr_stage_2_iters = int(os.environ.get("GAMMA_NIR_LR_STAGE_2_ITERS", "600"))
nir_proposal_embed_dim = int(os.environ.get("GAMMA_NIR_PROPOSAL_EMBED_DIM", "32"))
nir_proposal_mlp = int(
    os.environ.get("GAMMA_NIR_PROPOSAL_MLP", str(2 * nir_proposal_embed_dim))
)
chunk_size_env = os.environ.get("GAMMA_NIR_CHUNK_SIZE", "1024").strip().lower()
chunk_size = None if chunk_size_env in {"", "none"} else int(chunk_size_env)


run_nir_experiment(
    num_sites=num_sites,
    num_samples_stage_1=int(os.environ.get("GAMMA_NIR_SAMPLES_STAGE_1", str(3 * 2**9))),
    num_samples_stage_2=int(os.environ.get("GAMMA_NIR_SAMPLES_STAGE_2", str(3 * 2**9))),
    num_samples_stage_3=int(os.environ.get("GAMMA_NIR_SAMPLES_STAGE_3", str(3 * 2**10))),
    num_samples_stage_4=env_optional_int("GAMMA_NIR_SAMPLES_STAGE_4"),
    num_iters_total=num_iters_total,
    patch_size=int(os.environ.get("GAMMA_NIR_PATCH_SIZE", "1")),
    embed_dim=embed_dim,
    num_heads=int(os.environ.get("GAMMA_NIR_NUM_HEADS", "4")),
    num_layers=int(os.environ.get("GAMMA_NIR_NUM_LAYERS", "2")),
    mlp_hidden=mlp_hidden,
    chunk_size=chunk_size,
    train_lr_stage_1=float(os.environ.get("GAMMA_NIR_LR_STAGE_1", "1e-3")),
    train_lr_stage_2=float(os.environ.get("GAMMA_NIR_LR_STAGE_2", "1e-3")),
    train_lr_stage_3=float(os.environ.get("GAMMA_NIR_LR_STAGE_3", "1e-3")),
    train_lr_stage_4=env_optional_float("GAMMA_NIR_LR_STAGE_4"),
    train_lr_stage_1_iters=train_lr_stage_1_iters,
    train_lr_stage_2_iters=train_lr_stage_2_iters,
    train_lr_stage_3_iters=env_optional_int("GAMMA_NIR_LR_STAGE_3_ITERS"),
    learn_phase_stage_1=env_bool("GAMMA_NIR_LEARN_PHASE_STAGE_1", "true"),
    learn_phase_stage_2=env_bool("GAMMA_NIR_LEARN_PHASE_STAGE_2", "true"),
    learn_phase_stage_3=env_bool("GAMMA_NIR_LEARN_PHASE_STAGE_3", "true"),
    learn_phase_stage_4=(
        None
        if os.environ.get("GAMMA_NIR_LEARN_PHASE_STAGE_4") in {None, "", "none", "None"}
        else env_bool("GAMMA_NIR_LEARN_PHASE_STAGE_4", "true")
    ),
    target_sampler_name=os.environ.get("GAMMA_NIR_TARGET_SAMPLER", "local"),
    target_optimizer_name=os.environ.get("GAMMA_NIR_TARGET_OPTIMIZER", "adam"),
    target_sgd_momentum=float(os.environ.get("GAMMA_NIR_TARGET_MOMENTUM", "0.0")),
    target_preconditioner=os.environ.get("GAMMA_NIR_TARGET_PRECONDITIONER", "minsr"),
    target_sr_diag_shift=float(os.environ.get("GAMMA_NIR_TARGET_SR_DIAG_SHIFT", "1e-4")),
    target_sr_diag_shift_stage_1=float(
        os.environ.get("GAMMA_NIR_TARGET_SR_DIAG_SHIFT_STAGE_1", "1e-3")
    ),
    target_sr_diag_shift_stage_2=float(
        os.environ.get("GAMMA_NIR_TARGET_SR_DIAG_SHIFT_STAGE_2", "1e-4")
    ),
    target_sr_diag_shift_stage_3=float(
        os.environ.get("GAMMA_NIR_TARGET_SR_DIAG_SHIFT_STAGE_3", "1e-4")
    ),
    target_sr_diag_shift_stage_4=env_optional_float("GAMMA_NIR_TARGET_SR_DIAG_SHIFT_STAGE_4"),
    target_sr_proj_reg=env_optional_float("GAMMA_NIR_TARGET_SR_PROJ_REG"),
    target_sr_momentum=env_optional_float("GAMMA_NIR_TARGET_SR_MOMENTUM"),
    target_sr_mode=os.environ.get("GAMMA_NIR_TARGET_SR_MODE", "complex"),
    nir_proposal_batch=int(os.environ.get("GAMMA_NIR_PROPOSAL_BATCH", str(3 * 2**9))),
    nir_max_proposal_batches=int(os.environ.get("GAMMA_NIR_MAX_PROPOSAL_BATCHES", "12")),
    nir_max_adaptive_rounds=int(os.environ.get("GAMMA_NIR_MAX_ADAPTIVE_ROUNDS", "6")),
    nir_min_adaptive_rounds=int(os.environ.get("GAMMA_NIR_MIN_ADAPTIVE_ROUNDS", "1")),
    nir_force_adapt_until_iter=int(os.environ.get("GAMMA_NIR_FORCE_ADAPT_UNTIL_ITER", "0")),
    nir_ess_threshold_frac=float(os.environ.get("GAMMA_NIR_ESS_THRESHOLD_FRAC", "0.4")),
    nir_efficiency_threshold_stage_1=float(os.environ.get("GAMMA_NIR_EFF_STAGE_1", "0.10")),
    nir_efficiency_threshold_stage_2=float(os.environ.get("GAMMA_NIR_EFF_STAGE_2", "0.10")),
    nir_efficiency_threshold_stage_3=float(os.environ.get("GAMMA_NIR_EFF_STAGE_3", "0.10")),
    nir_efficiency_threshold_stage_4=env_optional_float("GAMMA_NIR_EFF_STAGE_4"),
    nir_adapt_metric=os.environ.get("GAMMA_NIR_ADAPT_METRIC", "efficiency"),
    nir_log_ratio_std_threshold_stage_1=env_optional_float("GAMMA_NIR_LOGPQ_STD_STAGE_1"),
    nir_log_ratio_std_threshold_stage_2=env_optional_float("GAMMA_NIR_LOGPQ_STD_STAGE_2"),
    nir_log_ratio_std_threshold_stage_3=env_optional_float("GAMMA_NIR_LOGPQ_STD_STAGE_3"),
    nir_log_ratio_std_threshold_stage_4=env_optional_float("GAMMA_NIR_LOGPQ_STD_STAGE_4"),
    nir_proposal_lr_stage_1=float(os.environ.get("GAMMA_NIR_PROPOSAL_LR_STAGE_1", "1e-3")),
    nir_proposal_lr_stage_2=float(os.environ.get("GAMMA_NIR_PROPOSAL_LR_STAGE_2", "1e-3")),
    nir_proposal_lr_stage_3=float(os.environ.get("GAMMA_NIR_PROPOSAL_LR_STAGE_3", "1e-3")),
    nir_proposal_lr_stage_4=env_optional_float("GAMMA_NIR_PROPOSAL_LR_STAGE_4"),
    nir_proposal_steps_stage_1=int(os.environ.get("GAMMA_NIR_PROPOSAL_STEPS_STAGE_1", "1")),
    nir_proposal_steps_stage_2=int(os.environ.get("GAMMA_NIR_PROPOSAL_STEPS_STAGE_2", "1")),
    nir_proposal_steps_stage_3=int(os.environ.get("GAMMA_NIR_PROPOSAL_STEPS_STAGE_3", "1")),
    nir_proposal_steps_stage_4=env_optional_int("GAMMA_NIR_PROPOSAL_STEPS_STAGE_4"),
    nir_proposal_embed_dim=nir_proposal_embed_dim,
    nir_proposal_heads=int(os.environ.get("GAMMA_NIR_PROPOSAL_HEADS", "4")),
    nir_proposal_layers=int(os.environ.get("GAMMA_NIR_PROPOSAL_LAYERS", "4")),
    nir_proposal_mlp=nir_proposal_mlp,
    nir_proposal_constrain_total_sz=env_bool("GAMMA_NIR_PROPOSAL_CONSTRAIN_SZ", "false"),
    nir_proposal_training_mode=os.environ.get("GAMMA_NIR_PROPOSAL_TRAINING_MODE", "resampled"),
    nir_proposal_weight_power=float(os.environ.get("GAMMA_NIR_PROPOSAL_WEIGHT_POWER", "1.0")),
    nir_proposal_weight_clip_factor=env_optional_float("GAMMA_NIR_PROPOSAL_WEIGHT_CLIP_FACTOR"),
    nir_prob_floor=float(os.environ.get("GAMMA_NIR_PROB_FLOOR", "1e-6")),
    nir_resampling_method=os.environ.get("GAMMA_NIR_RESAMPLING_METHOD", "multinomial"),
    nir_weighted_pool_cap_factor=env_optional_float("GAMMA_NIR_WEIGHTED_POOL_CAP_FACTOR"),
    nir_target_update_mode=os.environ.get("GAMMA_NIR_TARGET_UPDATE_MODE", "resample"),
    nir_exact_refine_after_iter=env_optional_int("GAMMA_NIR_EXACT_REFINE_AFTER_ITER"),
    nir_exact_refine_max_states=int(os.environ.get("GAMMA_NIR_EXACT_REFINE_MAX_STATES", "4096")),
    nir_sampler_refine_after_iter=env_optional_int("GAMMA_NIR_SAMPLER_REFINE_AFTER_ITER"),
    nir_reset_state_on_refine=env_bool("GAMMA_NIR_RESET_STATE_ON_REFINE", "true"),
    resume_checkpoint_path=os.environ.get("GAMMA_NIR_RESUME_CHECKPOINT"),
    resume_proposal_checkpoint_path=os.environ.get("GAMMA_NIR_RESUME_PROPOSAL_CHECKPOINT"),
    rng_seed=env_optional_int("GAMMA_NIR_SEED"),
    model_type=os.environ.get("GAMMA_NIR_MODEL_TYPE", "site_type_relation_gated_pool_bond"),
    run_tag=os.environ.get("GAMMA_NIR_RUN_TAG"),
)
