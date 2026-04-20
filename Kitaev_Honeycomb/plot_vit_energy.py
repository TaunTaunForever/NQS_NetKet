from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Kitaev ViT energy traces from a run directory or plain text file."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a run directory containing NetKet .log files, or a plain text file with one energy per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults beside the input source.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--exact-energy",
        type=float,
        default=None,
        help="Optional exact ground-state energy to overlay as a horizontal line.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        help="Matplotlib backend to use. Defaults to 'Agg' for save-only and 'WebAgg' for --show.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=100,
        help="Number of final iterations to use for the tail energy summary.",
    )
    return parser.parse_args()


def ordered_log_paths(run_dir: Path):
    log_paths = sorted(run_dir.glob("*.log"))
    priority = [
        "_warm_start_",
        "_warm_sr",
        "_main_sr",
        "_sr_shift1e-2",
        "_sr_shift5e-4",
        "_sr_shift2e-4",
        "_sr_shift1e-4",
        "_sr_tail",
    ]

    def sort_key(path: Path):
        name = path.name
        for idx, token in enumerate(priority):
            if token in name:
                return (idx, name)
        return (len(priority), name)

    return sorted(log_paths, key=sort_key)


def load_energies(run_dir: Path):
    energies = []
    used_logs = []
    for log_path in ordered_log_paths(run_dir):
        with open(log_path) as f:
            data = json.load(f)
        if "Energy" not in data:
            continue
        segment = data["Energy"]["Mean"]["real"]
        if not segment:
            continue
        energies.extend(segment)
        used_logs.append(log_path)
    return energies, used_logs


def load_energies_from_txt(txt_path: Path):
    energies = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            energies.append(float(line))
    return energies


def default_output_path(input_path: Path):
    if input_path.is_dir():
        return input_path / f"energy_vs_iterations_{input_path.name}.png"
    return input_path.with_suffix(".png")


def main():
    args = parse_args()
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "matplotlib-codex-cache"),
    )
    import matplotlib

    if args.backend != "auto":
        matplotlib.use(args.backend)
    elif args.show:
        matplotlib.use("WebAgg")
    else:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    input_path = args.input_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_dir():
        energies, used_logs = load_energies(input_path)
        if not energies:
            raise RuntimeError(f"No energy data found in {input_path}")
        title_source = input_path.name
        used_logs_count = len(used_logs)
    else:
        energies = load_energies_from_txt(input_path)
        if not energies:
            raise RuntimeError(f"No energy data found in {input_path}")
        used_logs = []
        title_source = input_path.stem
        used_logs_count = 0

    iterations = list(range(1, len(energies) + 1))
    output_path = args.output.resolve() if args.output else default_output_path(input_path)
    title = args.title or f"{title_source} Energy vs Iterations"
    tail_len = min(args.tail, len(energies))
    tail = np.asarray(energies[-tail_len:], dtype=float)
    tail_mean = float(np.mean(tail))
    tail_std = float(np.std(tail, ddof=1)) if tail_len > 1 else 0.0
    tail_sem = float(tail_std / np.sqrt(tail_len)) if tail_len > 0 else 0.0

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, energies, linewidth=1.5, label="Variational energy")
    if args.exact_energy is not None:
        plt.axhline(
            args.exact_energy,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Exact ground state",
        )
        plt.legend()
    plt.xscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if args.show:
        plt.show()

    print(f"Saved plot: {output_path}")
    if input_path.is_dir():
        print(f"Used {used_logs_count} log files")
    else:
        print("Used 1 plain text energy file")
    print(f"Total points: {len(energies)}")
    print(f"Final energy: {energies[-1]}")
    print(
        f"Last {tail_len} iterations mean energy: {tail_mean} +/- {tail_sem} (SEM)"
    )
    print(
        f"Last {tail_len} iterations standard deviation: {tail_std}"
    )


if __name__ == "__main__":
    main()
