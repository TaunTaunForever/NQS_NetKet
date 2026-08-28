#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPRO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPRO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nir_paper_reproduction import print_plan


if __name__ == "__main__":
    print_plan()
