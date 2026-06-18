from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("NETKET_DEBUG", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import jax
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
import optax
from netket.optimizer.solver import cholesky
from scipy.sparse.linalg import eigsh


jax.config.update("jax_enable_x64", True)


def _extract_real_series(log_data: dict[str, Any], observable: str = "Energy") -> list[float]:
    series = log_data[observable]["Mean"]
    if isinstance(series, dict):
        if "real" in series:
            return [float(x) for x in series["real"]]
        if "value" in series:
            return [float(x) for x in series["value"]]
    return [float(x) for x in series]


def _load_site_modules(site_dir: Path):
    site_dir = site_dir.resolve()
    module_path = str(site_dir)
    fallback_model_path = str((site_dir.parent / "18-site").resolve())
    removed = {}

    for name in (
        "hamiltonian",
        "vit_model",
        "vit_site_type_relation_model",
        "vit_site_type_relation_gated_pool_model",
    ):
        removed[name] = sys.modules.pop(name, None)

    sys.path.insert(0, module_path)
    try:
        hamiltonian = importlib.import_module("hamiltonian")
        try:
            vit_site_type_relation_model = importlib.import_module(
                "vit_site_type_relation_model"
            )
        except ModuleNotFoundError:
            sys.path.insert(0, fallback_model_path)
            try:
                vit_site_type_relation_model = importlib.import_module(
                    "vit_site_type_relation_model"
                )
            finally:
                if sys.path and sys.path[0] == fallback_model_path:
                    sys.path.pop(0)
        return hamiltonian, vit_site_type_relation_model
    finally:
        if sys.path and sys.path[0] == module_path:
            sys.path.pop(0)
        for name, module in removed.items():
            if module is not None and name not in sys.modules:
                sys.modules[name] = module


def run_srt_experiment(
    *,
    site_dir: str | Path,
    num_sites: int,
    num_samples_warmup: int,
    num_samples: int,
    num_iters_warm: int,
    num_iters_main: int,
    num_iters_refine: int,
    patch_size: int = 1,
    embed_dim: int = 32,
    num_heads: int = 4,
    num_layers: int = 4,
    mlp_hidden_dim: int | None = None,
    learn_phase_warmup: bool = False,
    learn_phase_main: bool = True,
    warm_sr_lr: float = 1e-3,
    warm_sr_momentum: float = 0.7,
    warm_sr_diagshift: float = 1e-2,
    main_sr_lr: float = 1e-2,
    main_sr_momentum: float = 0.9,
    main_sr_diagshift: float = 1e-4,
    refine_sr_lr: float = 1e-4,
    refine_sr_momentum: float = 0.0,
    refine_sr_diagshift_schedule: list[tuple[float, float, str]] | None = None,
    n_discard_per_chain: int = 4,
    target_chain_length: int = 64,
    pt_sweep_size: int | None = None,
    chunk_size: int | None = None,
    chunk_size_bwd: int | None = None,
    num_starts: int = 1,
    netket_debug: bool = False,
    profile_time: bool = False,
    log_step_size: int = 1,
    write_every: int = 1,
    save_params_every: int = 25,
    use_experimental_vmc_sr: bool = False,
    experimental_use_ntk: bool = True,
    experimental_on_the_fly: bool = True,
    sampler_name: str = "local",
    sampler_name_refine: str | None = "pt_local",
    model_type: str = "site_type_relation",
    run_tag: str | None = None,
):
    site_dir = Path(site_dir).resolve()
    os.chdir(site_dir)
    sampler_name_refine = sampler_name if sampler_name_refine is None else sampler_name_refine

    if refine_sr_diagshift_schedule is None:
        refine_sr_diagshift_schedule = [
            (0.30, 1e-2, "sr_shift1e-2"),
            (0.40, 5e-4, "sr_shift5e-4"),
            (0.30, 2e-4, "sr_shift2e-4"),
        ]

    mlp_hidden_dim = 2 * embed_dim if mlp_hidden_dim is None else mlp_hidden_dim
    pt_sweep_size = num_sites * 2 if pt_sweep_size is None else pt_sweep_size

    os.environ["NETKET_DEBUG"] = "1" if netket_debug else "0"

    hamiltonian_module, model_module = _load_site_modules(site_dir)
    gamma_hamiltonian = hamiltonian_module.gamma_hamiltonian
    KitaevSiteTypeRelationHoneycombViT = model_module.KitaevSiteTypeRelationHoneycombViT
    build_bipartite_site_type_ids = model_module.build_bipartite_site_type_ids
    build_extended_kitaev_relation_matrix = model_module.build_extended_kitaev_relation_matrix
    site_relation_to_patch_relation_expanded = model_module.site_relation_to_patch_relation_expanded
    site_type_ids_to_patch_type_ids = model_module.site_type_ids_to_patch_type_ids
    gated_model_module = None
    if model_type == "site_type_relation_gated_pool_bond":
        try:
            gated_model_module = importlib.import_module(
                "vit_site_type_relation_gated_pool_model"
            )
        except ModuleNotFoundError:
            fallback_model_path = str((site_dir.parent / "18-site").resolve())
            sys.path.insert(0, fallback_model_path)
            try:
                gated_model_module = importlib.import_module(
                    "vit_site_type_relation_gated_pool_model"
                )
            finally:
                if sys.path and sys.path[0] == fallback_model_path:
                    sys.path.pop(0)
        KitaevSiteTypeRelationGatedPoolViT = gated_model_module.KitaevSiteTypeRelationGatedPoolViT
    elif model_type != "site_type_relation":
        raise ValueError(f"Unsupported model_type={model_type!r}")

    today = date.today().isoformat()
    job_name = (
        f"{num_sites}-site_"
        f"{num_layers}L_{num_heads}H_{patch_size}p_"
        f"{num_samples}_samples_{today}_Gamma_Honeycomb_ViT_SRt_{model_type}"
    )
    if run_tag:
        job_name = f"{job_name}_{run_tag}"

    graph, _symm_group, hi, ha = gamma_hamiltonian(num_sites)
    perm = tuple(range(graph.n_nodes))

    site_type_ids = build_bipartite_site_type_ids(graph, permutation=perm)
    token_site_type_ids = (
        site_type_ids
        if patch_size == 1
        else site_type_ids_to_patch_type_ids(site_type_ids, patch_size)
    )
    site_relation_matrix = build_extended_kitaev_relation_matrix(graph, permutation=perm)
    relation_matrix = (
        site_relation_matrix
        if patch_size == 1
        else site_relation_to_patch_relation_expanded(site_relation_matrix, patch_size)
    )

    num_relation_types = max(max(row) for row in relation_matrix) + 1
    num_site_types = max(token_site_type_ids) + 1

    head_dim = embed_dim // num_heads
    print("Model type:", model_type)
    print("Permutation:", perm)
    print("Site type ids:", token_site_type_ids)
    print("Number of site types:", num_site_types)
    print("Relation matrix shape:", (len(relation_matrix), len(relation_matrix[0])))
    print("Number of relation types:", num_relation_types)
    print("Embed dim:", embed_dim)
    print("Head dim:", head_dim)

    exact_gs = None
    if num_sites <= 18:
        sp_h = ha.to_sparse()
        eig_vals, _ = eigsh(sp_h, k=2, which="SA")
        exact_gs = float(eig_vals[0])
        print("Exact ground-state energy:", exact_gs)
        print()

    def make_metropolis_local(hilbert, n_samples):
        n_chains = max(3, n_samples // target_chain_length)
        n_chains -= n_chains % 3
        n_chains = max(3, n_chains)
        return nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=n_chains)

    def make_parallel_tempering_local(hilbert, n_samples):
        n_chains = max(3, n_samples // target_chain_length)
        n_chains -= n_chains % 3
        n_chains = max(3, n_chains)
        return nk.sampler.ParallelTemperingLocal(
            hilbert=hilbert,
            n_chains=n_chains,
            sweep_size=pt_sweep_size,
        )

    def build_sampler(name: str, n_samples: int):
        if name == "local":
            return make_metropolis_local(hi, n_samples)
        if name == "pt_local":
            return make_parallel_tempering_local(hi, n_samples)
        if name == "exact":
            return nk.sampler.ExactSampler(hi)
        raise ValueError(f"Unsupported sampler_name={name!r}")

    def build_model(*, learn_phase: bool):
        if model_type == "site_type_relation_gated_pool_bond":
            return KitaevSiteTypeRelationGatedPoolViT(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_hidden_dim=mlp_hidden_dim,
                patch_size=patch_size,
                learn_phase=learn_phase,
                relation_matrix=relation_matrix,
                site_type_ids=token_site_type_ids,
                permutation=perm,
            )
        return KitaevSiteTypeRelationHoneycombViT(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            patch_size=patch_size,
            learn_phase=learn_phase,
            relation_matrix=relation_matrix,
            site_type_ids=token_site_type_ids,
            permutation=perm,
        )

    def _print_sampler_info(vstate):
        print("Parameters:", vstate.n_parameters)
        if hasattr(vstate.sampler, "n_chains"):
            print("n_chains:", vstate.sampler.n_chains)
        print("chain_length:", vstate.chain_length)

    def build_vmc_sr_driver(*, variational_state, optimizer, momentum, diag_shift):
        if use_experimental_vmc_sr:
            return nk.experimental.driver.VMC_SR(
                hamiltonian=ha,
                optimizer=optimizer,
                variational_state=variational_state,
                momentum=momentum,
                linear_solver_fn=cholesky,
                mode="complex",
                diag_shift=diag_shift,
                use_ntk=experimental_use_ntk,
                on_the_fly=experimental_on_the_fly,
                chunk_size_bwd=chunk_size_bwd,
            )

        return nk.driver.VMC_SR(
            hamiltonian=ha,
            optimizer=optimizer,
            variational_state=variational_state,
            momentum=momentum,
            linear_solver=cholesky,
            mode="complex",
            diag_shift=diag_shift,
            use_ntk=experimental_use_ntk,
            on_the_fly=experimental_on_the_fly,
            chunk_size_bwd=chunk_size_bwd,
        )

    warmup_summaries = []
    energy: list[float] = []

    if num_iters_warm > 0:
        print(
            f"\n=== Multi-start amplitude-only SR: {num_starts} starts, "
            f"{num_iters_warm} iterations each ===\n"
        )

        for start_idx in range(num_starts):
            print(f"--- Amplitude-only start {start_idx + 1}/{num_starts} ---")
            vstate_warm = nk.vqs.MCState(
                sampler=build_sampler(sampler_name, num_samples_warmup),
                model=build_model(learn_phase=learn_phase_warmup),
                n_samples=num_samples_warmup,
                chunk_size=chunk_size,
                n_discard_per_chain=n_discard_per_chain,
                seed=1234 + start_idx,
            )
            _print_sampler_info(vstate_warm)

            out_warm = f"out_{job_name}_warm_start_{start_idx}"
            driver_warm = build_vmc_sr_driver(
                variational_state=vstate_warm,
                optimizer=optax.sgd(learning_rate=warm_sr_lr),
                momentum=warm_sr_momentum,
                diag_shift=warm_sr_diagshift,
            )
            driver_warm.run(
                n_iter=num_iters_warm,
                out=out_warm,
                step_size=log_step_size,
                write_every=write_every,
                save_params_every=save_params_every,
                timeit=profile_time,
            )

            with open(f"{out_warm}.log") as f:
                warm_data = json.load(f)
            warm_energy = _extract_real_series(warm_data, "Energy")
            energy.extend(warm_energy)
            warmup_energy = float(np.mean(warm_energy[-min(20, len(warm_energy)) :]))
            print("Amplitude-only energy:", warmup_energy)
            if exact_gs is not None:
                print("Amplitude-only gap to exact:", warmup_energy - exact_gs)
            print()

            warmup_summaries.append(
                {
                    "start": start_idx,
                    "warmup_energy": warmup_energy,
                    "variables": vstate_warm.variables,
                }
            )

    if warmup_summaries:
        best_warmup = min(warmup_summaries, key=lambda row: row["warmup_energy"])
        initial_variables = best_warmup["variables"]
        selected_start = int(best_warmup["start"])
        selected_warmup_energy = float(best_warmup["warmup_energy"])
        print("Selected amplitude-only start:", selected_start)
        print("Selected amplitude-only energy:", selected_warmup_energy)
        print()
    else:
        initial_variables = None
        selected_start = None
        selected_warmup_energy = None

    print(f"\n=== Main SR continuation for {num_iters_main} iterations ===")
    vstate_main = nk.vqs.MCState(
        sampler=build_sampler(sampler_name, num_samples),
        model=build_model(learn_phase=learn_phase_main),
        n_samples=num_samples,
        variables=initial_variables,
        chunk_size=chunk_size,
        n_discard_per_chain=n_discard_per_chain,
    )
    _print_sampler_info(vstate_main)

    stage2_out = f"out_{job_name}_stage2"
    driver_main = build_vmc_sr_driver(
        variational_state=vstate_main,
        optimizer=optax.sgd(learning_rate=main_sr_lr),
        momentum=main_sr_momentum,
        diag_shift=main_sr_diagshift,
    )
    driver_main.run(
        n_iter=num_iters_main,
        out=stage2_out,
        step_size=log_step_size,
        write_every=write_every,
        save_params_every=save_params_every,
        timeit=profile_time,
    )
    with open(f"{stage2_out}.log") as f:
        data2 = json.load(f)
    energy.extend(_extract_real_series(data2, "Energy"))

    final_vstate = vstate_main
    refine_log_file = None
    if num_iters_refine > 0:
        print(f"\n=== Final PT-SR refinement for {num_iters_refine} iterations ===")
        vstate_refine = nk.vqs.MCState(
            sampler=build_sampler(sampler_name_refine, num_samples),
            model=build_model(learn_phase=learn_phase_main),
            n_samples=num_samples,
            variables=vstate_main.variables,
            chunk_size=chunk_size,
            n_discard_per_chain=n_discard_per_chain,
        )
        _print_sampler_info(vstate_refine)

        def run_sr_segment(n_iter: int, diag_shift: float, tag: str):
            out = f"out_{job_name}_{tag}"
            driver_refine = build_vmc_sr_driver(
                variational_state=vstate_refine,
                optimizer=optax.sgd(learning_rate=refine_sr_lr),
                momentum=refine_sr_momentum,
                diag_shift=diag_shift,
            )
            driver_refine.run(
                n_iter=n_iter,
                out=out,
                step_size=log_step_size,
                write_every=write_every,
                save_params_every=save_params_every,
                timeit=profile_time,
            )
            return out

        remaining = num_iters_refine
        for frac, shift, tag in refine_sr_diagshift_schedule:
            n_seg = int(round(frac * num_iters_refine))
            n_seg = max(1, min(n_seg, remaining))
            remaining -= n_seg
            out = run_sr_segment(n_seg, shift, tag)
            with open(f"{out}.log") as f:
                data_refine = json.load(f)
            energy.extend(_extract_real_series(data_refine, "Energy"))
            refine_log_file = f"{out}.log"
            if remaining <= 0:
                break

        if remaining > 0:
            out = run_sr_segment(remaining, refine_sr_diagshift_schedule[-1][1], "sr_tail")
            with open(f"{out}.log") as f:
                data_refine = json.load(f)
            energy.extend(_extract_real_series(data_refine, "Energy"))
            refine_log_file = f"{out}.log"

        final_vstate = vstate_refine

    with open(f"mean_energy_run_{job_name}.txt", "w") as f:
        for item in energy:
            f.write(f"{item}\n")

    plt.figure(figsize=(12, 8))
    plt.plot(energy)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"Gamma Honeycomb {num_sites}-site (ViT SRt)")
    plt.tight_layout()
    plt.savefig(f"energy_{job_name}.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(energy)
    plt.xscale("log")
    plt.xlabel("Iteration (log)")
    plt.ylabel("Energy")
    plt.title(f"Gamma Honeycomb {num_sites}-site (ViT SRt)")
    plt.tight_layout()
    plt.savefig(f"energy_log_{job_name}.png")
    plt.close()

    summary = {
        "job_name": job_name,
        "num_sites": num_sites,
        "num_samples_warmup": num_samples_warmup,
        "num_samples": num_samples,
        "num_iters_warm": num_iters_warm,
        "num_iters_main": num_iters_main,
        "num_iters_refine": num_iters_refine,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "mlp_hidden_dim": mlp_hidden_dim,
        "learn_phase_warmup": learn_phase_warmup,
        "learn_phase_main": learn_phase_main,
        "warm_sr_lr": warm_sr_lr,
        "warm_sr_momentum": warm_sr_momentum,
        "warm_sr_diagshift": warm_sr_diagshift,
        "main_sr_lr": main_sr_lr,
        "main_sr_momentum": main_sr_momentum,
        "main_sr_diagshift": main_sr_diagshift,
        "refine_sr_lr": refine_sr_lr,
        "refine_sr_momentum": refine_sr_momentum,
        "refine_sr_diagshift_schedule": [
            [float(frac), float(shift), str(tag)]
            for frac, shift, tag in refine_sr_diagshift_schedule
        ],
        "n_discard_per_chain": n_discard_per_chain,
        "target_chain_length": target_chain_length,
        "pt_sweep_size": pt_sweep_size,
        "chunk_size": chunk_size,
        "chunk_size_bwd": chunk_size_bwd,
        "num_starts": num_starts,
        "netket_debug": netket_debug,
        "profile_time": profile_time,
        "log_step_size": log_step_size,
        "write_every": write_every,
        "save_params_every": save_params_every,
        "use_experimental_vmc_sr": use_experimental_vmc_sr,
        "experimental_use_ntk": experimental_use_ntk,
        "experimental_on_the_fly": experimental_on_the_fly,
        "sampler_name": sampler_name,
        "sampler_name_refine": sampler_name_refine,
        "model_type": model_type,
        "run_tag": run_tag,
        "permutation": list(perm),
        "site_type_ids": list(token_site_type_ids),
        "num_site_types": num_site_types,
        "relation_matrix": [list(row) for row in relation_matrix],
        "num_relation_types": num_relation_types,
        "head_dim": head_dim,
        "exact_ground_state_energy": exact_gs,
        "final_energy": float(energy[-1]) if energy else None,
        "best_energy_seen": float(min(energy)) if energy else None,
        "tail_mean_last_20": float(np.mean(energy[-20:])) if len(energy) >= 20 else None,
        "tail_mean_last_50": float(np.mean(energy[-50:])) if len(energy) >= 50 else None,
        "tail_mean_last_100": float(np.mean(energy[-100:])) if len(energy) >= 100 else None,
        "tail_std_last_50": float(np.std(energy[-50:])) if len(energy) >= 50 else None,
        "selected_warmup_start": selected_start,
        "warmup_energy": selected_warmup_energy,
        "mean_energy_file": f"mean_energy_run_{job_name}.txt",
        "warm_log_file": (
            f"out_{job_name}_warm_start_{selected_start}.log"
            if num_iters_warm > 0 and selected_start is not None
            else None
        ),
        "log_file": f"{stage2_out}.log",
        "refine_log_file": refine_log_file,
        "plot_file": f"energy_{job_name}.png",
        "plot_log_file": f"energy_log_{job_name}.png",
    }

    with open(f"summary_{job_name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    if exact_gs is not None and energy:
        print("Final gap to exact:", float(energy[-1]) - exact_gs)

    return summary
