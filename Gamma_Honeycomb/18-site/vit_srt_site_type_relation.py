import runpy
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
runpy.run_path(str(THIS_DIR / "vit_srt.py"), run_name="__main__")
