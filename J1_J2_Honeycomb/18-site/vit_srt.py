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
    num_sites=18,
    j1=float(os.environ.get("J1J2_J1", "1.0")),
    j2=float(os.environ.get("J1J2_J2", "0.0")),
    num_samples=int(os.environ.get("J1J2_NUM_SAMPLES", str(3*1024))),
    num_iters_warm=int(os.environ.get("J1J2_NUM_ITERS_WARM", "0")),
    num_iters=int(os.environ.get("J1J2_NUM_ITERS", "1000")),
    num_samples_refine=int(os.environ.get("J1J2_NUM_SAMPLES_REFINE", str(3 * 512))),
    num_iters_refine=int(os.environ.get("J1J2_NUM_ITERS_REFINE", "1000")),
    chunk_size=int(os.environ.get("J1J2_CHUNK_SIZE", "1024")),
    embed_dim=int(os.environ.get("J1J2_EMBED_DIM", "8")),
    num_heads=int(os.environ.get("J1J2_NUM_HEADS", "2")),
    num_layers=int(os.environ.get("J1J2_NUM_LAYERS", "4")),
    learn_phase_main=os.environ.get("J1J2_LEARN_PHASE_MAIN", "true").lower() in {"1", "true", "yes", "on"},
    learning_rate_warm=float(os.environ.get("J1J2_LR_WARM", "5e-3")),
    learning_rate_main=float(os.environ.get("J1J2_LR_MAIN", "1e-2")),
    learning_rate_refine=float(os.environ.get("J1J2_LR_REFINE", "3e-4")),
    momentum=float(os.environ.get("J1J2_MOMENTUM", "0.0")),
    diag_shift_warm=float(os.environ.get("J1J2_DIAG_SHIFT_WARM", "1e-2")),
    diag_shift_main=float(os.environ.get("J1J2_DIAG_SHIFT_MAIN", "1e-4")),
    diag_shift_refine=float(os.environ.get("J1J2_DIAG_SHIFT_REFINE", "1e-5")),
    sampler_name=os.environ.get("J1J2_SAMPLER", "pt_local"),
    sampler_name_refine=os.environ.get("J1J2_SAMPLER_REFINE", "pt_local"),
    observable_num_samples=(
        int(os.environ["J1J2_OBS_NUM_SAMPLES"])
        if "J1J2_OBS_NUM_SAMPLES" in os.environ
        else None
    ),
    observable_chunk_size=(
        int(os.environ["J1J2_OBS_CHUNK_SIZE"])
        if "J1J2_OBS_CHUNK_SIZE" in os.environ
        else None
    ),
    observable_sampler_name=os.environ.get("J1J2_OBS_SAMPLER"),
    observable_d_max=(
        int(os.environ["J1J2_OBS_D_MAX"])
        if "J1J2_OBS_D_MAX" in os.environ
        else None
    ),
    d_max=int(os.environ.get("J1J2_D_MAX", "1")),
    d_max_refine=int(os.environ.get("J1J2_D_MAX_REFINE", "1")),
    optimizer_name=os.environ.get("J1J2_OPTIMIZER", "sgd"),
    resume_checkpoint_path=os.environ.get("J1J2_RESUME_CHECKPOINT"),
    run_tag=os.environ.get("J1J2_RUN_TAG", "staged_local_refine_4h"),
    model_type=os.environ.get("J1J2_MODEL_TYPE", "site_type_relation"),
)
