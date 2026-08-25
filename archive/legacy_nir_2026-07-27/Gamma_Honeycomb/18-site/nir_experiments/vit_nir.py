import runpy
from pathlib import Path


GAMMA_ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(
    str(GAMMA_ROOT / "nir_experiments" / "vit_nir.py"),
    run_name="__main__",
)
