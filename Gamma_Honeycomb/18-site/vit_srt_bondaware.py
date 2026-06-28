"""Compatibility entry point for the current Gamma ViT SRt model.

This script name used to launch the older bond-aware ViT.  Keep the entry point
available, but route it through vit_srt.py so all Gamma SRt runs use the shared
gated/site-aware/attention-pooled/bond-relation model by default.
"""

import os
import runpy
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
os.environ.setdefault("GAMMA_MODEL_TYPE", "site_type_relation_gated_pool_bond")
runpy.run_path(str(THIS_DIR / "vit_srt.py"), run_name="__main__")
