from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPRO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPRO_ROOT / "configs"

BENCHMARKS = {
    "tfi_2d": CONFIG_DIR / "tfi_2d_paper.json",
    "j1j2_square_10x10": CONFIG_DIR / "j1j2_square_10x10_paper.json",
}


def load_config(name: str) -> dict[str, Any]:
    try:
        path = BENCHMARKS[name]
    except KeyError as exc:
        known = ", ".join(sorted(BENCHMARKS))
        raise KeyError(f"Unknown benchmark {name!r}. Known benchmarks: {known}") from exc
    return json.loads(path.read_text())


def print_plan() -> None:
    print("NIR paper reproduction plan")
    print("===========================")
    for name, path in BENCHMARKS.items():
        cfg = load_config(name)
        print(f"\n{name}")
        print(f"  config: {path.relative_to(REPRO_ROOT)}")
        print(f"  benchmark: {cfg['benchmark']}")
        print(f"  status: {cfg['status']}")
        if name == "tfi_2d":
            print(f"  lattices: {cfg['lattices']}")
            print(f"  target NQS: {cfg['target_nqs']['type']}")
            print(
                "  proposal: "
                f"d={cfg['proposal']['embed_dim']}, "
                f"heads={cfg['proposal']['num_heads']}, "
                f"layers={cfg['proposal']['num_layers']}"
            )
            print(
                "  ambiguity: text width="
                f"{cfg['target_nqs']['hidden_width_text']}, appendix width="
                f"{cfg['target_nqs']['hidden_width_appendix']}"
            )
        if name == "j1j2_square_10x10":
            completed = [
                row for row in cfg["proposal_baselines"] if row["paper_status"] == "completed"
            ]
            print(f"  lattice: {cfg['lattice']}, J2/J1={cfg['j2'] / cfg['j1']}")
            print(f"  target NQS: {cfg['target_nqs']['type']}")
            print("  completed proposal baselines:")
            for row in completed:
                print(
                    "    "
                    f"d={row['embed_dim']}, layers={row['num_layers']}, "
                    f"E/N={row['energy_per_site']}, "
                    f"eff={row['mean_sampling_efficiency']}"
                )


if __name__ == "__main__":
    print_plan()
