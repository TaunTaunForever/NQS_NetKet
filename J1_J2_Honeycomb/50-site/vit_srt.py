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
    num_sites=50,
    j1=float(os.environ.get("J1J2_J1", "1.0")),
    j2=float(os.environ.get("J1J2_J2", "0.2")),
)

