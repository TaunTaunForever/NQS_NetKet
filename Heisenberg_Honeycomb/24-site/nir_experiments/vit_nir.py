import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
os.chdir(THIS_DIR)
if str(ROOT / "nir_experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "nir_experiments"))

from vit_nir_common import run_nir_experiment


run_nir_experiment(num_sites=24)

