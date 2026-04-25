from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def require_summary(summary_file: str | Path) -> Path:
    path = Path(summary_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Set SOURCE_SUMMARY_FILE to a finished run summary JSON before launching."
        )
    return path


def load_summary(summary_file: str | Path) -> dict:
    path = require_summary(summary_file)
    with open(path) as f:
        return json.load(f)


def is_continuation_summary(summary: dict) -> bool:
    return "source_summary_file" in summary and "runs" in summary


def select_continuation_mode(summary: dict, preferred_mode: str | None = None) -> str:
    runs = summary.get("runs", {})
    if not runs:
        raise ValueError("Continuation summary does not contain any continuation runs.")

    if preferred_mode is not None:
        preferred_mode = preferred_mode.lower()
        if preferred_mode not in runs:
            raise ValueError(
                f"Requested SOURCE_SAMPLER_MODE '{preferred_mode}' was not found in continuation summary."
            )
        return preferred_mode

    sampler_modes = summary.get("sampler_modes", [])
    if len(runs) == 1:
        return next(iter(runs))
    if len(sampler_modes) == 1 and sampler_modes[0] in runs:
        return sampler_modes[0]
    for mode in reversed(sampler_modes):
        if mode in runs:
            return mode
    return next(iter(runs))


def resolve_base_summary(summary: dict, summary_path: Path) -> tuple[dict, Path]:
    if not is_continuation_summary(summary):
        return summary, summary_path
    base_summary_path = require_summary(summary["source_summary_file"])
    base_summary = load_summary(base_summary_path)
    return resolve_base_summary(base_summary, base_summary_path)


def resolve_source_run_dir(summary: dict, summary_path: Path) -> Path:
    if is_continuation_summary(summary):
        run_dir = Path(
            summary.get("continuation_run_dir", summary_path.parent)
        ).expanduser().resolve()
    else:
        run_dir = Path(summary.get("run_dir", summary_path.parent)).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def find_latest_checkpoint(
    summary: dict, run_dir: Path, preferred_mode: str | None = None
) -> Path:
    if is_continuation_summary(summary):
        mode = select_continuation_mode(summary, preferred_mode)
        run_summary = summary["runs"][mode]
        if "output" in run_summary:
            candidate = Path(run_summary["output"]).with_suffix(".mpack")
            if candidate.is_file():
                return candidate.resolve()
        if "checkpoint_file" in run_summary:
            candidate = Path(run_summary["checkpoint_file"])
            if candidate.is_file():
                return candidate.resolve()
        if "checkpoint_file" in summary:
            candidate = Path(summary["checkpoint_file"])
            if candidate.is_file():
                return candidate.resolve()

    for out in reversed(summary.get("outputs", [])):
        candidate = Path(out).with_suffix(".mpack")
        if candidate.is_file():
            return candidate.resolve()

    candidates = sorted(run_dir.glob("*.mpack"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1].resolve()

    raise FileNotFoundError(
        f"No .mpack checkpoint was found in {run_dir}. "
        "The source run must have a saved checkpoint."
    )


def _first_existing_path(candidates: list[str | Path | None]) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            return path
    return None


def find_latest_training_state(
    summary: dict, run_dir: Path, preferred_mode: str | None = None
) -> Path | None:
    if is_continuation_summary(summary):
        mode = select_continuation_mode(summary, preferred_mode)
        run_summary = summary["runs"][mode]
        path = _first_existing_path(
            [
                run_summary.get("training_state_file"),
                run_summary.get("latest_training_state_file"),
                summary.get("training_state_file"),
                summary.get("latest_training_state_file"),
            ]
        )
        if path is not None:
            return path
    else:
        path = _first_existing_path(
            [
                summary.get("training_state_file"),
                summary.get("latest_training_state_file"),
            ]
        )
        if path is not None:
            return path

    candidates = sorted(run_dir.glob("*_training_state.mpack"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1].resolve()
    return None


def find_best_training_state(
    summary: dict, run_dir: Path, preferred_mode: str | None = None
) -> Path | None:
    if is_continuation_summary(summary):
        mode = select_continuation_mode(summary, preferred_mode)
        run_summary = summary["runs"][mode]
        path = _first_existing_path(
            [
                run_summary.get("best_training_state_file"),
                summary.get("best_training_state_file"),
            ]
        )
        if path is not None:
            return path
    else:
        path = _first_existing_path([summary.get("best_training_state_file")])
        if path is not None:
            return path

    candidates = sorted(
        run_dir.glob("*_training_state_best.mpack"), key=lambda p: p.stat().st_mtime
    )
    if candidates:
        return candidates[-1].resolve()
    return None


def read_energy_trace(log_path: Path) -> list[float]:
    if not log_path.is_file():
        return []
    with open(log_path) as f:
        data = json.load(f)
    return list(np.asarray(data["Energy"]["Mean"]["real"], dtype=float))


def collect_previous_energy(
    summary: dict,
    summary_path: Path | None = None,
    preferred_mode: str | None = None,
) -> list[float]:
    if is_continuation_summary(summary):
        if summary_path is None:
            raise ValueError("summary_path is required for continuation summaries.")
        source_summary_path = require_summary(summary["source_summary_file"])
        source_summary = load_summary(source_summary_path)
        energy = collect_previous_energy(
            source_summary,
            summary_path=source_summary_path,
            preferred_mode=preferred_mode,
        )
        mode = select_continuation_mode(summary, preferred_mode)
        run_summary = summary["runs"][mode]
        if "output" in run_summary:
            energy.extend(read_energy_trace(Path(run_summary["output"]).with_suffix(".log")))
        return energy

    energy: list[float] = []
    for out in summary.get("outputs", []):
        energy.extend(read_energy_trace(Path(out).with_suffix(".log")))
    return energy


def make_continuation_dir(source_run_dir: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = source_run_dir / f"{label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
