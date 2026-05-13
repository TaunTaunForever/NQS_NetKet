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
    num_samples=int(os.environ.get("J1J2_NUM_SAMPLES", "1024")),
    num_iters_warm=int(os.environ.get("J1J2_NUM_ITERS_WARM", "0")),
    num_iters=int(os.environ.get("J1J2_NUM_ITERS", "1000")),
    chunk_size=int(os.environ.get("J1J2_CHUNK_SIZE", "1056")),
    embed_dim=int(os.environ.get("J1J2_EMBED_DIM", "24")),
    num_heads=int(os.environ.get("J1J2_NUM_HEADS", "4")),
    num_layers=int(os.environ.get("J1J2_NUM_LAYERS", "4")),
    learn_phase_main=os.environ.get("J1J2_LEARN_PHASE_MAIN", "false").lower() in {"1", "true", "yes", "on"},
    learning_rate_warm=float(os.environ.get("J1J2_LR_WARM", "5e-3")),
    learning_rate_main=float(os.environ.get("J1J2_LR_MAIN", "1e-2")),
    momentum=float(os.environ.get("J1J2_MOMENTUM", "0.0")),
    diag_shift_warm=float(os.environ.get("J1J2_DIAG_SHIFT_WARM", "1e-2")),
    diag_shift_main=float(os.environ.get("J1J2_DIAG_SHIFT_MAIN", "1e-2")),
    sampler_name=os.environ.get("J1J2_SAMPLER", "local"),
    optimizer_name=os.environ.get("J1J2_OPTIMIZER", "sgd"),
)
