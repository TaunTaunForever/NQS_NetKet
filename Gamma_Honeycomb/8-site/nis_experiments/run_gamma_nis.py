"""Reusable implementation for Gamma-honeycomb weighted-NIS experiments.

For ordinary runs edit the site-size-specific ``vit_nis_site_relation.py``
launcher. This module deliberately keeps numerical work in small named
functions and only retains a CLI for backward-compatible automation.
"""
from __future__ import annotations

import argparse
import functools
import importlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SITE_DIR = THIS_DIR.parent
GAMMA_DIR = SITE_DIR.parent
REPO_ROOT = SITE_DIR.parents[1]
for path in (REPO_ROOT, GAMMA_DIR, SITE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax
from flax import serialization
from netket.jax.sharding import gather

from diagnostics.nis_logging import NISLogger
from diagnostics.nis_metrics import (
    compare_distributions_smallN,
    summarize_weight_batch,
    weighted_variational_statistics,
)
from samplers.minimal_autoregressive import MinimalAutoregressiveTransformer
from samplers.neural_importance_sampling import NeuralImportanceSampler, sample_proposals_and_metrics
from samplers.proposal_wrappers import AutoregressiveProposalWrapper
from samplers.resampling import resample_batch
from samplers.importance_weights import stable_log_weights
from optim.weighted_sr import WeightedMinSR, WeightedSR
from vqs import WeightedNISState, WeightedNISVMC
from vit_model import HoneycombPatchViT
from vit_srt_common import _load_site_modules
from vit_symm_model import CanonicalRepresentativeHoneycombViT, SymmetryProjectedHoneycombViT


@dataclass
class NISRunConfig:
    """All experiment controls. The direct launcher owns the usual values."""

    variant: str = "site_relation"
    # The reusable NIS runner normally operates on this module's 8-site
    # directory.  An explicit directory lets a thin launcher reuse the same
    # weighted-NIS implementation for another Gamma-honeycomb size without
    # accidentally importing the 8-site Hamiltonian or relation builder.
    num_sites: int = 8
    site_dir: str | None = None
    # ``netket`` is the default scalable weighted-VQS integration and uses
    # NetKet's native mesh when ``use_multi_gpu`` is enabled. The retained
    # ``explicit_jax`` backend is the legacy experimental pmap implementation.
    execution_backend: str = "netket"
    n_proposals: int = 4096
    num_samples: int = 1024
    num_iterations: int = 1000
    seed: int = 0
    diagnostics_dir: str = "results/nis/gamma_8site_site_relation"
    # Optional parameter-only restore. Optimizer and sampler state deliberately
    # restart, which makes this useful for a conservative refinement phase.
    resume_target_checkpoint: str | None = None
    resume_proposal_checkpoint: str | None = None

    embed_dim: int = 8
    num_heads: int = 2
    num_layers: int = 2
    mlp_hidden_dim: int = 16
    patch_size: int = 1
    # The proposal is deliberately independent of the target ansatz.  Keep
    # this explicit here (rather than relying on the model default) so the
    # size-specific launchers can tune it without touching sampler code.
    proposal_num_layers: int = 4
    proposal_embed_dim: int = 16

    target_lr: float = 1.0e-3
    # Optional exponential annealing. ``None`` keeps a constant target LR;
    # otherwise interpolate from ``target_lr`` to this value over the stated
    # number of optimiser updates.
    target_lr_final: float | None = None
    target_lr_decay_steps: int = 0
    # ``netket_sr`` is the production weighted minimum-SR backend. It follows
    # NetKet's sample-space formulation while retaining the NIS weights in
    # both the force and metric. Its solver controls are deliberately
    # compact: the damping and trust cap below, plus the two NetKet-style
    # stability controls that follow.
    # ``weighted_sr`` retains the previous, more experimental backend.
    target_update: str = "weighted_sr"
    sr_diag_shift: float = 1.0e-3
    # The remaining CG, adaptive, and diagonal controls below apply only to
    # ``weighted_sr`` and are kept for backward-compatible experiments.
    sr_cg_tol: float = 1.0e-5
    sr_cg_maxiter: int = 12
    sr_chunk_size: int | None = 512
    # Maximum Fubini--Study/QGT norm of one outer SGD target update.
    sr_trust_region: float | None = 0.05
    # SPRING momentum retains an accepted untrust-clipped SR update and uses
    # it only in directions unsupported by the current weighted pool.  0.8
    # is NetKet's empirical recommendation; ``None`` disables it.
    sr_momentum: float | None = 0.8
    # Weighted extension of NetKet's optional projection regularisation.
    # It acts only on the weighted-centering null mode, so leave it disabled
    # unless numerical diagnostics motivate it.
    sr_proj_reg: float | None = None
    # Keep production NIS on the scalable matrix-free multi-GPU path.  The
    # automatic dense-Cholesky heuristic can select a memory-heavy Jacobian
    # path for *smaller* pools, because their realified sample space happens
    # to fall below its threshold. ``cholesky`` and ``distributed_cholesky``
    # remain explicit small-system diagnostics; ``auto`` is retained only for
    # controlled solver experiments.
    sr_direct_solver: str = "matrix_free"
    # Internal batch size for the distributed dense-kernel diagnostic. It
    # bounds reverse-mode Jacobian workspace rather than changing the NIS pool.
    sr_dense_jacobian_chunk_size: int = 64
    # Adaptive weighted-SR controller. It extends useful CG solves up to this
    # ceiling, raises damping after unreliable pools/solves, and restores the
    # user-selected trust cap only after reliable iterations.
    sr_adaptive: bool = True
    sr_adaptive_maxiter: int = 32
    sr_adaptive_diag_shift_factor: float = 4.0
    sr_adaptive_max_diag_shift: float = 1.0e-1
    sr_adaptive_diag_shift_decay: float = 1.5
    sr_adaptive_ess_threshold: float = 0.10
    sr_adaptive_healthy_residual: float = 1.0e-3
    sr_adaptive_trust_region_min_scale: float = 0.10
    sr_adaptive_trust_region_growth: float = 1.25
    sr_adaptive_trust_region_shrink: float = 0.5
    # Reuse the previous un-clipped SR solve when its damping is unchanged.
    # This reduces late-stage Krylov work without changing the linear system.
    sr_cg_warm_start: bool = False
    # Optional matrix-free diagonal preconditioning. The diagonal is estimated
    # with Rademacher probes of the weighted quantum geometric tensor and can
    # define relative damping: S + lambda * diag(S).
    sr_diagonal_preconditioner: bool = False
    sr_relative_damping: bool = False
    sr_diagonal_probes: int = 4
    sr_diagonal_update_interval: int = 25
    sr_diagonal_ema: float = 0.9
    sr_diagonal_floor: float = 1.0e-6
    # ``per_leaf`` is lower variance for correlated transformer parameters;
    # ``per_parameter`` keeps the literal Hutchinson diagonal.
    sr_diagonal_mode: str = "per_parameter"
    # Stop once this true relative residual is reached; the adaptive maximum
    # remains a safety ceiling. ``None`` uses ``sr_cg_tol``.
    sr_residual_target: float | None = None
    sr_residual_replacement_interval: int = 0
    proposal_lr: float = 3.0e-3
    proposal_train_steps: int = 4
    proposal_train_batch_size: int = 512
    # Proposal MLE can be slowed or frozen after it has reached good coverage.
    # ``None`` preserves continual updates; zero freezes immediately.
    proposal_update_interval: int = 1
    proposal_freeze_after: int | None = None
    always_update_target: bool = True
    ess_threshold: float = 0.05
    target_grad_batch_size: int = 256
    local_energy_chunk_size: int = 16384
    use_multi_gpu: bool = True
    compute_exact_ground_energy: bool = False
    exact_diagnostics_every: int = 25
    checkpoint_every: int = 25
    # Draw a second, independent proposal pool every N iterations. It reports
    # weighted energy/variance/force and an SR residual without affecting the
    # optimisation pool. Zero disables this production diagnostic.
    heldout_diagnostics_every: int = 0
    # Optional paired post-update energy check. It evaluates the candidate
    # target on the proposal pool already drawn for the update, rejects a
    # statistically significant degradation, and restores the target and its
    # optimizer state. This is implemented by the NetKet weighted-NIS backend.
    post_update_energy_guard: bool = False
    post_update_energy_guard_sigmas: float = 2.0
    post_update_min_ess_fraction: float = 0.10
    resample_method: str = "systematic"
    # Full-Hilbert-space checks are strictly a small-N validation mode.
    compute_exact_diagnostics: bool = False


def resolve_site_dir(config: NISRunConfig) -> Path:
    """Return the geometry/model directory selected by one run configuration."""
    path = SITE_DIR if config.site_dir is None else Path(config.site_dir).expanduser()
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Gamma-honeycomb site directory does not exist: {path}")
    return path


def build_hamiltonian(config: NISRunConfig):
    """Build the configured Gamma-honeycomb Hamiltonian from its site directory."""
    hamiltonian_module, _, _ = _load_site_modules(resolve_site_dir(config))
    return hamiltonian_module.gamma_hamiltonian(config.num_sites)


def make_target_learning_rate(config: NISRunConfig):
    """Return a constant or exponential target-learning-rate schedule."""
    if config.target_lr <= 0.0:
        raise ValueError("target_lr must be positive")
    if config.target_lr_final is None:
        return float(config.target_lr)
    if config.target_lr_final <= 0.0:
        raise ValueError("target_lr_final must be positive or None")
    if config.target_lr_decay_steps < 1:
        raise ValueError("target_lr_decay_steps must be positive with target_lr_final")

    initial = float(config.target_lr)
    final = float(config.target_lr_final)
    decay_steps = int(config.target_lr_decay_steps)

    def schedule(step):
        fraction = jnp.minimum(jnp.asarray(step, dtype=jnp.float64) / decay_steps, 1.0)
        return initial * jnp.exp(jnp.log(final / initial) * fraction)

    return schedule


def restore_checkpoint(template, checkpoint_path: str | None, *, label: str):
    """Restore variables into a matching Flax template when requested."""
    if checkpoint_path is None:
        return template
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")
    print(f"Restoring {label} parameters from: {path}")
    return serialization.from_bytes(template, path.read_bytes())


def _symmetry_permutations(group, n_sites: int):
    """Convert NetKet point-group elements to immutable input permutations."""
    permutations = []
    for element in group:
        permutation = getattr(element, "inverse_permutation_array", None)
        if permutation is None:
            permutation = getattr(element, "permutation_array", None)
        if permutation is None:
            # Older/custom group elements may only expose element-wise lookup.
            permutation = [element(index) for index in range(n_sites)]
        permutations.append(tuple(map(int, jnp.asarray(permutation))))
    return tuple(permutations)


def _load_gated_relation_module(site_dir: Path):
    """Load the optional gated relation-aware target for a site directory.

    The standard relation model is loaded through ``_load_site_modules``.
    Keep this separate because the gated model is an optional NIS target, and
    the 8-site directory deliberately does not need to provide it.
    """
    module_name = "vit_site_type_relation_gated_pool_model"
    module_path = str(site_dir.resolve())
    fallback_model_path = str((site_dir.parent / "18-site").resolve())
    previous_module = sys.modules.pop(module_name, None)

    sys.path.insert(0, module_path)
    try:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
            sys.path.insert(0, fallback_model_path)
            try:
                return importlib.import_module(module_name)
            finally:
                if sys.path and sys.path[0] == fallback_model_path:
                    sys.path.pop(0)
    finally:
        if sys.path and sys.path[0] == module_path:
            sys.path.pop(0)
        if previous_module is not None and module_name not in sys.modules:
            sys.modules[module_name] = previous_module


def build_target_model(config: NISRunConfig, graph, symmetry_group):
    common = dict(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        mlp_hidden_dim=config.mlp_hidden_dim,
        patch_size=config.patch_size,
        learn_phase=True,
        permutation=tuple(range(graph.n_nodes)),
    )
    site_dir = resolve_site_dir(config)
    _, site_vit_model, relation_module = _load_site_modules(site_dir)
    if config.variant == "plain":
        return site_vit_model.HoneycombPatchViT(**common)
    if config.variant in {"site_relation", "site_relation_gated"}:
        permutation = tuple(range(graph.n_nodes))
        site_types = relation_module.build_bipartite_site_type_ids(graph, permutation=permutation)
        relations = relation_module.build_extended_kitaev_relation_matrix(graph, permutation=permutation)
        if config.patch_size != 1:
            site_types = relation_module.site_type_ids_to_patch_type_ids(site_types, config.patch_size)
            relations = relation_module.site_relation_to_patch_relation_expanded(relations, config.patch_size)
        if config.variant == "site_relation_gated":
            gated_module = _load_gated_relation_module(site_dir)
            return gated_module.KitaevSiteTypeRelationGatedPoolViT(
                **common, relation_matrix=relations, site_type_ids=site_types
            )
        return relation_module.KitaevSiteTypeRelationHoneycombViT(
            **common, relation_matrix=relations, site_type_ids=site_types
        )
    symmetries = _symmetry_permutations(symmetry_group, graph.n_nodes)
    if site_dir != SITE_DIR:
        raise ValueError(
            "The reusable 18-site NIS launcher currently supports the plain "
            "and site-relation (including gated site-relation) target variants."
        )
    common["symmetries"] = symmetries
    if config.variant == "inputproj":
        return CanonicalRepresentativeHoneycombViT(**common)
    if config.variant == "symmproj":
        return SymmetryProjectedHoneycombViT(**common)
    raise ValueError(f"Unknown target variant {config.variant!r}")


def local_energy(operator, model, variables, sigma, *, log_psi=None, chunk_size: int | None = None):
    connected, matrix_elements = operator.get_conn_padded(sigma)
    if log_psi is None:
        log_psi = model.apply(variables, sigma)
    flat_connected = connected.reshape(-1, connected.shape[-1])
    if chunk_size is None or flat_connected.shape[0] <= chunk_size:
        connected_log_psi = model.apply(variables, flat_connected)
    else:
        chunks = [
            model.apply(variables, flat_connected[start:start + chunk_size])
            for start in range(0, flat_connected.shape[0], chunk_size)
        ]
        connected_log_psi = jnp.concatenate(chunks, axis=0)
    connected_log_psi = connected_log_psi.reshape(connected.shape[:-1])
    return jnp.sum(matrix_elements * jnp.exp(connected_log_psi - log_psi[:, None]), axis=1)


def make_chunk_force(model):
    """Compile one reverse-mode force kernel for re-use on every iteration."""
    @jax.jit
    def chunk_force(parameters, sigma_chunk, coefficients):
        def force_objective(params):
            log_psi = model.apply({"params": params}, sigma_chunk)
            return jnp.sum(
                jnp.real(coefficients) * jnp.real(log_psi)
                + jnp.imag(coefficients) * jnp.imag(log_psi)
            )
        return jax.grad(force_objective)(parameters)
    return chunk_force


def make_proposal_step(proposal_wrapper, optimizer):
    """Compile the proposal maximum-likelihood update once."""
    @jax.jit
    def proposal_step(parameters, optimizer_state, training_sigma):
        def proposal_loss(params):
            return -jnp.mean(proposal_wrapper.log_prob({"params": params}, training_sigma))
        loss, gradient = jax.value_and_grad(proposal_loss)(parameters)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, parameters)
        return optax.apply_updates(parameters, updates), optimizer_state, loss
    return proposal_step


def make_multi_gpu_kernels(target, proposal_wrapper, n_proposals_per_device: int, grad_chunk_size: int):
    """Build single-host data-parallel kernels across all local JAX devices."""
    axis_name = "nis_devices"
    proposal_model = proposal_wrapper.model
    n_sites = proposal_wrapper.hilbert.size
    proposal_dtype = proposal_wrapper.dtype
    temperature = proposal_wrapper.temperature
    probability_floor = proposal_wrapper.probability_floor

    @functools.partial(jax.pmap, axis_name=axis_name)
    def sample_proposals(proposal_parameters, keys):
        # Keep the full causal loop inside this pmap. Calling the cached
        # single-device jitted sampler here creates a nested mesh conflict.
        sigma = -jnp.ones((n_proposals_per_device, n_sites), dtype=proposal_dtype)
        logq = jnp.zeros((n_proposals_per_device,), dtype=jnp.float64)
        site_keys = jax.random.split(keys, n_sites)
        for site, site_key in enumerate(site_keys):
            logits = jnp.asarray(
                proposal_model.apply({"params": proposal_parameters}, sigma),
                dtype=jnp.float64,
            ) / temperature
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            if probability_floor:
                probabilities = jnp.maximum(jnp.exp(log_probs), probability_floor)
                log_probs = jnp.log(probabilities / jnp.sum(probabilities, axis=-1, keepdims=True))
            token = jax.random.categorical(site_key, log_probs[:, site, :])
            logq = logq + jnp.take_along_axis(log_probs[:, site, :], token[:, None], axis=-1)[:, 0]
            spins = jnp.asarray(2 * token - 1, dtype=proposal_dtype)
            sigma = sigma.at[:, site].set(spins)
        return sigma, logq

    @functools.partial(jax.pmap, axis_name=axis_name)
    def target_logpsi(target_parameters, sigma):
        return target.apply({"params": target_parameters}, sigma)

    def bind_hamiltonian(hamiltonian):
        @functools.partial(jax.pmap, axis_name=axis_name)
        def local_energies(target_parameters, sigma):
            return local_energy(
                hamiltonian, target, {"params": target_parameters}, sigma
            )

        @functools.partial(jax.pmap, axis_name=axis_name)
        def target_force(target_parameters, sigma, weights, local_energies, global_energy):
            total = jax.tree_util.tree_map(jnp.zeros_like, target_parameters)
            for start in range(0, n_proposals_per_device, grad_chunk_size):
                stop = min(start + grad_chunk_size, n_proposals_per_device)
                sigma_chunk = sigma[start:stop]
                coefficients = weights[start:stop] * jax.lax.stop_gradient(
                    local_energies[start:stop] - global_energy
                )

                def force_objective(params):
                    log_psi = target.apply({"params": params}, sigma_chunk)
                    return jnp.sum(
                        jnp.real(coefficients) * jnp.real(log_psi)
                        + jnp.imag(coefficients) * jnp.imag(log_psi)
                    )

                partial = jax.grad(force_objective)(target_parameters)
                total = jax.tree_util.tree_map(lambda old, new: old + 2.0 * new, total, partial)
            return jax.tree_util.tree_map(lambda value: jax.lax.psum(value, axis_name), total)

        @functools.partial(jax.pmap, axis_name=axis_name)
        def proposal_gradient(proposal_parameters, training_sigma):
            def loss(params):
                return -jnp.mean(proposal_wrapper.log_prob({"params": params}, training_sigma))
            value, gradient = jax.value_and_grad(loss)(proposal_parameters)
            return (
                jax.lax.pmean(value, axis_name),
                jax.tree_util.tree_map(lambda grad: jax.lax.pmean(grad, axis_name), gradient),
            )

        return local_energies, target_force, proposal_gradient

    return sample_proposals, target_logpsi, bind_hamiltonian


def first_replica(tree):
    """Take a replicated pmap result back to one host-side parameter pytree."""
    return jax.tree_util.tree_map(lambda value: jax.device_get(value[0]), tree)


def replicate_for_pmap(tree, n_devices: int):
    """Add the pmap leading axis without pre-sharding under a different mesh."""
    return jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(jnp.asarray(value), (n_devices,) + value.shape), tree
    )


def weighted_target_grad(parameters, sigma, weights, local_energies, chunk_size: int, chunk_force):
    """Evaluate the complex SNIS VMC force using chunked reverse-mode AD.

    This avoids constructing the prohibitive [batch, parameter] Jacobian.
    """
    weighted_energy = jnp.sum(weights * local_energies)
    total = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    for start in range(0, sigma.shape[0], chunk_size):
        stop = min(start + chunk_size, sigma.shape[0])
        sigma_chunk = sigma[start:stop]
        coefficients = weights[start:stop] * jax.lax.stop_gradient(
            local_energies[start:stop] - weighted_energy
        )

        partial = chunk_force(parameters, sigma_chunk, coefficients)
        total = jax.tree_util.tree_map(lambda old, new: old + 2.0 * new, total, partial)
    return total


def tree_norm(tree) -> float:
    return float(jnp.sqrt(sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in jax.tree_util.tree_leaves(tree))))


def exact_variational_energy(hilbert, hamiltonian, target, variables) -> float:
    states = hilbert.all_states()
    log_probabilities = 2.0 * jnp.real(target.apply(variables, states))
    probabilities = jax.nn.softmax(log_probabilities)
    return float(jnp.real(jnp.sum(probabilities * local_energy(hamiltonian, target, variables, states))))


def save_energy_plot(output_dir: Path, config: NISRunConfig) -> Path | None:
    """Save standard linear and log-scale NIS energy plots.

    The diagnostics logger appends when a directory is reused.  Iteration
    numbers therefore identify the final contiguous run, rather than letting
    values from an earlier restart leak into the plot.  Plotting is deliberately
    best-effort: an unavailable plotting backend must not invalidate a finished
    optimisation or its checkpoints.
    """
    metrics_path = output_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        print("Energy plot skipped: metrics.jsonl was not created.")
        return None

    try:
        entries: list[tuple[int, float, float | None]] = []
        for line in metrics_path.read_text().splitlines():
            try:
                record = json.loads(line)
                iteration = record.get("iteration")
                energy = float(record["energy_mean"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if not isinstance(iteration, int) or not math.isfinite(energy):
                continue
            exact_energy = record.get("exact_ground_energy")
            exact_energy = (
                float(exact_energy)
                if exact_energy is not None and math.isfinite(float(exact_energy))
                else None
            )
            entries.append((iteration, energy, exact_energy))

        if not entries:
            print("Energy plot skipped: no finite energy records were found.")
            return None

        segments: list[list[tuple[int, float, float | None]]] = []
        segment: list[tuple[int, float, float | None]] = []
        previous_iteration = -1
        for entry in entries:
            if segment and entry[0] < previous_iteration:
                segments.append(segment)
                segment = []
            segment.append(entry)
            previous_iteration = entry[0]
        segments.append(segment)

        # Keep the final record if an iteration was logged more than once.
        by_iteration = {
            iteration: (energy, exact_energy)
            for iteration, energy, exact_energy in segments[-1]
        }
        iterations = np.asarray(sorted(by_iteration), dtype=int)
        energies = np.asarray([by_iteration[iteration][0] for iteration in iterations])
        exact_energy = next(
            (
                by_iteration[iteration][1]
                for iteration in reversed(iterations)
                if by_iteration[iteration][1] is not None
            ),
            None,
        )

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        rolling_window = min(100, len(energies))
        rolling = np.convolve(
            energies,
            np.full(rolling_window, 1.0 / rolling_window),
            mode="valid",
        )
        rolling_iterations = iterations[rolling_window - 1 :]
        title = (
            f"{config.num_sites}-site {config.variant.replace('_', ' ')} weighted-NIS: "
            f"iterations {iterations[0]}–{iterations[-1]}"
        )

        figure, axes = plt.subplots(
            2,
            1,
            figsize=(13, 10),
            constrained_layout=True,
        )
        figure.suptitle(title, fontsize=16)
        for axis in axes:
            axis.plot(
                iterations,
                energies,
                color="#4c78a8",
                alpha=0.16,
                linewidth=0.55,
                label="Instantaneous NIS energy",
            )
            axis.plot(
                rolling_iterations,
                rolling,
                color="#0b3c6d",
                linewidth=2.0,
                label=f"{rolling_window}-iteration rolling mean",
            )
            if exact_energy is not None:
                axis.axhline(
                    exact_energy,
                    color="#c62828",
                    linestyle="--",
                    linewidth=1.8,
                    label=f"ED ground energy = {exact_energy:.8f}",
                )
            axis.grid(alpha=0.22)
            axis.set_ylabel("Energy")

        axes[0].set_title("Full optimization trajectory")
        axes[0].set_xlim(iterations[0], iterations[-1])
        axes[0].legend(frameon=False, loc="lower right")

        late_start = max(iterations[0], int(0.05 * iterations[-1]))
        late_mask = iterations >= late_start
        late_energies = energies[late_mask]
        if exact_energy is not None:
            late_energies = np.append(late_energies, exact_energy)
        late_range = max(float(np.ptp(late_energies)), 1.0e-3)
        late_margin = max(0.02, 0.15 * late_range)
        axes[1].set_title("Late-stage view")
        axes[1].set_xlim(late_start, iterations[-1])
        axes[1].set_ylim(
            float(np.min(late_energies) - late_margin),
            float(np.max(late_energies) + late_margin),
        )
        axes[1].set_xlabel("Iteration")
        axes[1].legend(frameon=False, loc="lower right")

        plot_path = output_dir / "energy_vs_iteration.png"
        figure.savefig(plot_path, dpi=180)
        plt.close(figure)

        log_figure, log_axis = plt.subplots(
            figsize=(13, 7),
            constrained_layout=True,
        )
        log_figure.suptitle(f"{title} (log iteration axis)", fontsize=16)
        log_axis.plot(
            iterations,
            energies,
            color="#1f4e79",
            alpha=0.70,
            linewidth=0.65,
            label="Instantaneous NIS energy",
        )
        if exact_energy is not None:
            log_axis.axhline(
                exact_energy,
                color="#c62828",
                linestyle="--",
                linewidth=1.8,
                label=f"ED ground energy = {exact_energy:.8f}",
            )
        log_axis.set_xscale("log")
        log_axis.set_xlim(max(iterations[0], 1), iterations[-1])
        log_axis.set(xlabel="Iteration (log scale)", ylabel="Energy")
        log_axis.grid(alpha=0.22, which="both")
        log_axis.legend(frameon=False, loc="lower right")

        inset = log_axis.inset_axes([0.60, 0.55, 0.35, 0.36])
        inset_start = max(iterations[0], iterations[-1] - 999)
        inset_mask = iterations >= inset_start
        inset_energies = energies[inset_mask]
        inset.plot(
            iterations[inset_mask],
            inset_energies,
            color="#1f4e79",
            alpha=0.85,
            linewidth=0.65,
        )
        if exact_energy is not None:
            inset.axhline(
                exact_energy,
                color="#c62828",
                linestyle="--",
                linewidth=1.1,
            )
            inset_energies = np.append(inset_energies, exact_energy)
        inset_range = max(float(np.ptp(inset_energies)), 1.0e-3)
        inset_margin = max(0.01, 0.15 * inset_range)
        inset.set_xlim(inset_start, iterations[-1])
        inset.set_ylim(
            float(np.min(inset_energies) - inset_margin),
            float(np.max(inset_energies) + inset_margin),
        )
        inset.set_title("Final 1,000 iterations", fontsize=9)
        inset.tick_params(labelsize=8)
        inset.grid(alpha=0.20)
        log_plot_path = output_dir / "energy_vs_log_iteration.png"
        log_figure.savefig(log_plot_path, dpi=180)
        plt.close(log_figure)
        print(f"Saved energy plots: {plot_path}, {log_plot_path}")
        return plot_path
    except Exception as error:  # A completed NIS run must remain usable.
        print(f"Energy plot skipped: {error}")
        return None


def run_netket_experiment(config: NISRunConfig):
    """Run the Gamma experiment through NetKet's VQS/driver interfaces.

    With ``use_multi_gpu=True``, proposal samples are sharded across NetKet's
    native JAX mesh while model parameters remain replicated. No exact
    calculation is used unless the small-system validation flags are explicitly
    enabled.
    """
    if config.post_update_energy_guard_sigmas <= 0.0:
        raise ValueError("post_update_energy_guard_sigmas must be positive")
    if not 0.0 <= config.post_update_min_ess_fraction <= 1.0:
        raise ValueError("post_update_min_ess_fraction must be between zero and one")

    graph, symmetry_group, hilbert, hamiltonian = build_hamiltonian(config)
    key = jax.random.PRNGKey(config.seed)
    key, target_key, proposal_key = jax.random.split(key, 3)

    target = build_target_model(config, graph, symmetry_group)
    proposal = MinimalAutoregressiveTransformer(
        hilbert.size,
        config.proposal_embed_dim,
        num_layers=config.proposal_num_layers,
    )
    initial_sigma = jnp.ones((2, hilbert.size), dtype=jnp.float64)
    target_variables = target.init(target_key, initial_sigma)
    proposal_variables = proposal.init(proposal_key, initial_sigma)
    target_variables = restore_checkpoint(
        target_variables,
        config.resume_target_checkpoint,
        label="target",
    )
    proposal_variables = restore_checkpoint(
        proposal_variables,
        config.resume_proposal_checkpoint,
        label="proposal",
    )
    proposal_wrapper = AutoregressiveProposalWrapper(hilbert, proposal, probability_floor=1e-8)
    sampler = NeuralImportanceSampler(
        hilbert,
        proposal,
        proposal_variables,
        proposal_wrapper,
        n_proposals=config.n_proposals,
        resample_size=config.num_samples,
        resample_method=config.resample_method,
        ESS_threshold=config.ess_threshold,
    )
    state = WeightedNISState(
        hilbert,
        target,
        sampler,
        variables=target_variables,
        proposal_variables=proposal_variables,
        n_samples=config.n_proposals,
        seed=config.seed,
        chunk_size=config.target_grad_batch_size,
        local_energy_chunk_size=config.local_energy_chunk_size,
        use_sharding=config.use_multi_gpu,
    )
    target_learning_rate = make_target_learning_rate(config)
    if config.target_update == "adam":
        target_optimizer = optax.adam(target_learning_rate)
        target_preconditioner = None
    elif config.target_update == "weighted_sr":
        # WeightedSR's trust-region bound applies to this plain-SGD update.
        target_optimizer = optax.sgd(target_learning_rate)
        target_preconditioner = WeightedSR(
            diag_shift=config.sr_diag_shift,
            maxiter=config.sr_cg_maxiter,
            tol=config.sr_cg_tol,
            chunk_size=config.sr_chunk_size,
            trust_region=config.sr_trust_region,
            learning_rate=target_learning_rate,
            adaptive=config.sr_adaptive,
            adaptive_maxiter=config.sr_adaptive_maxiter,
            adaptive_diag_shift_factor=config.sr_adaptive_diag_shift_factor,
            adaptive_max_diag_shift=config.sr_adaptive_max_diag_shift,
            adaptive_diag_shift_decay=config.sr_adaptive_diag_shift_decay,
            adaptive_ess_threshold=config.sr_adaptive_ess_threshold,
            adaptive_healthy_residual=config.sr_adaptive_healthy_residual,
            adaptive_trust_region_min_scale=config.sr_adaptive_trust_region_min_scale,
            adaptive_trust_region_growth=config.sr_adaptive_trust_region_growth,
            adaptive_trust_region_shrink=config.sr_adaptive_trust_region_shrink,
            warm_start=config.sr_cg_warm_start,
            diagonal_preconditioner=config.sr_diagonal_preconditioner,
            relative_damping=config.sr_relative_damping,
            diagonal_probes=config.sr_diagonal_probes,
            diagonal_update_interval=config.sr_diagonal_update_interval,
            diagonal_ema=config.sr_diagonal_ema,
            diagonal_floor=config.sr_diagonal_floor,
            diagonal_mode=config.sr_diagonal_mode,
            residual_target=config.sr_residual_target,
            residual_replacement_interval=config.sr_residual_replacement_interval,
        )
    elif config.target_update in {"netket_sr", "weighted_minsr"}:
        # NetKet-style minimum SR, adapted so NIS weights remain in both the
        # force and the sample-space metric.  The existing launcher controls
        # provide the physically meaningful choices; its on-device solve
        # budget and residual target stay internal rather than adding a second
        # page of solver parameters.
        target_optimizer = optax.sgd(target_learning_rate)
        target_preconditioner = WeightedMinSR(
            diag_shift=config.sr_diag_shift,
            chunk_size=config.sr_chunk_size,
            trust_region=config.sr_trust_region,
            learning_rate=target_learning_rate,
            momentum=config.sr_momentum,
            proj_reg=config.sr_proj_reg,
            direct_solver=config.sr_direct_solver,
            distributed_jacobian_chunk_size=config.sr_dense_jacobian_chunk_size,
        )
    else:
        raise ValueError(
            "target_update must be 'adam', 'weighted_sr', or 'netket_sr'"
        )

    driver = WeightedNISVMC(
        hamiltonian,
        target_optimizer,
        variational_state=state,
        proposal_optimizer=optax.adam(config.proposal_lr),
        proposal_train_steps=config.proposal_train_steps,
        proposal_train_batch_size=config.proposal_train_batch_size,
        always_update_target=config.always_update_target,
        ess_threshold=config.ess_threshold,
        preconditioner=target_preconditioner,
        proposal_update_interval=config.proposal_update_interval,
        proposal_freeze_after=config.proposal_freeze_after,
        heldout_diagnostics_every=config.heldout_diagnostics_every,
    )

    logger = NISLogger(config.diagnostics_dir)
    logger.save_config(asdict(config))
    output_dir = Path(config.diagnostics_dir)

    if config.target_lr_final is not None:
        print(
            "Target LR schedule: exponential "
            f"{config.target_lr:.3e} -> {config.target_lr_final:.3e} over "
            f"{config.target_lr_decay_steps} updates"
        )

    exact_ground_energy = None
    if config.compute_exact_ground_energy and graph.n_nodes < 24:
        exact_ground_energy = float(nk.exact.lanczos_ed(hamiltonian, k=1)[0])
        print("Exact ground-state energy (NetKet Lanczos ED):", exact_ground_energy)
    print("JAX devices:", jax.devices())
    if state.uses_native_sharding:
        print(
            "NetKet weighted-NIS backend: native sharding over "
            f"{jax.device_count()} devices ({state.n_samples_per_device} proposals/device)."
        )
    elif config.use_multi_gpu and len(jax.local_devices()) > 1:
        print(
            "NetKet weighted-NIS backend: sharding was requested but is disabled. "
            "Ensure that one process can see all requested GPUs."
        )
    else:
        print("NetKet weighted-NIS backend: single-device execution")

    previous_energy = None
    best_exact_target_energy = float("inf")
    for iteration in range(config.num_iterations):
        iteration_start = time.perf_counter()
        # Optax transformations are functional, so retaining these references
        # is sufficient to restore the full target-update state if the paired
        # post-update energy check rejects the candidate.
        parameters_before_update = (
            state.parameters if config.post_update_energy_guard else None
        )
        optimizer_state_before_update = (
            driver._optimizer_state if config.post_update_energy_guard else None
        )
        driver.advance(1)
        stats = driver.energy
        batch = state.last_batch
        if batch is None:  # Defensive: a driver iteration must generate one pool.
            raise RuntimeError("NetKet NIS driver did not retain its proposal pool")

        selected_stats = stats
        selected_batch = batch
        energy_before_update = float(jnp.real(stats.mean))
        error_before_update = float(stats.error_of_mean)
        post_update_guard = None
        if config.post_update_energy_guard and driver.target_updated:
            candidate_stats, candidate_batch, candidate_diagnostics, finite_candidate = (
                state.evaluate_current_target_on_last_batch(hamiltonian)
            )
            candidate_energy = float(jnp.real(candidate_stats.mean))
            candidate_error = float(candidate_stats.error_of_mean)
            candidate_ess_fraction = float(
                jax.device_get(candidate_diagnostics["ESSFrac"])
            )
            paired_uncertainty = math.hypot(error_before_update, candidate_error)
            permitted_increase = (
                config.post_update_energy_guard_sigmas * paired_uncertainty
            )
            energy_increase = candidate_energy - energy_before_update
            accepted = (
                finite_candidate
                and candidate_ess_fraction >= config.post_update_min_ess_fraction
                and energy_increase <= permitted_increase
            )
            post_update_guard = {
                "pre_update_energy": energy_before_update,
                "pre_update_energy_error": error_before_update,
                "post_update_energy": candidate_energy,
                "post_update_energy_error": candidate_error,
                "post_update_energy_delta": energy_increase,
                "post_update_energy_uncertainty": paired_uncertainty,
                "post_update_energy_permitted_increase": permitted_increase,
                "post_update_candidate_ESSFrac": candidate_ess_fraction,
                "post_update_candidate_finite": finite_candidate,
                "post_update_accepted": accepted,
            }
            if accepted:
                selected_stats = candidate_stats
                selected_batch = candidate_batch
            else:
                state.parameters = parameters_before_update
                driver._optimizer_state = optimizer_state_before_update
                notify_rejected_update = getattr(
                    driver.preconditioner, "notify_rejected_update", None
                )
                if notify_rejected_update is not None:
                    notify_rejected_update()

        weights = selected_batch["weight_info"]["weights_normalized"]
        # Sorting/resampling are end-of-iteration diagnostics, unlike the
        # weighted energy/force/SR path above. Gather only here so the global
        # proposal order is well-defined for checkpointed resamples.
        if state.uses_native_sharding:
            logging_sigma = gather(selected_batch["sigma_prop"])
            logging_weights = gather(weights)
            logging_logw = gather(selected_batch["weight_info"]["logw_raw"])
        else:
            logging_sigma = selected_batch["sigma_prop"]
            logging_weights = weights
            logging_logw = selected_batch["weight_info"]["logw_raw"]
        metrics = summarize_weight_batch(logging_logw, logging_weights)
        energy = float(jnp.real(selected_stats.mean))
        energy_delta = None if previous_energy is None else energy - previous_energy
        proposal_loss = driver._proposal_loss
        target_update_accepted = bool(
            driver.target_updated
            and (
                post_update_guard is None
                or post_update_guard["post_update_accepted"]
            )
        )
        mode = (
            "nis"
            if target_update_accepted
            else "target_update_rejected"
            if driver.target_updated
            else "proposal_recovery"
        )
        metrics.update({
            "mode": mode,
            "iteration": iteration + 1,
            "number_of_proposals": config.n_proposals,
            "number_of_returned_samples": config.num_samples,
            "Mean": energy,
            "Variance": float(selected_stats.variance),
            "ErrorOfMean": float(selected_stats.error_of_mean),
            "energy_mean": energy,
            "energy_variance": float(selected_stats.variance),
            "energy_error": float(selected_stats.error_of_mean),
            "energy_delta_from_previous_iteration": energy_delta,
            # ``target_updated`` reports an accepted state change. Keep the
            # attempted flag as a separate diagnostic so rejected proposals
            # cannot be mistaken for a completed target update.
            "target_updated": target_update_accepted,
            "target_update_attempted": driver.target_updated,
            "target_update_accepted": target_update_accepted,
            "proposal_updated": driver.proposal_updated,
            "gradient_norm": tree_norm(driver._dp),
            "proposal_loss": None if proposal_loss is None else float(proposal_loss),
            "exact_ground_energy": exact_ground_energy,
            "native_sharding": state.uses_native_sharding,
            "number_of_jax_devices": jax.device_count(),
        })
        if post_update_guard is not None:
            metrics.update(post_update_guard)
        sr_info = getattr(driver.preconditioner, "last_info", None)
        if sr_info is not None:
            metrics.update({f"sr_{key}": value for key, value in sr_info.items()})
        heldout = driver.heldout_diagnostics
        if heldout:
            metrics.update({f"heldout_{key}": value for key, value in heldout.items()})

        should_compute_exact = graph.n_nodes < 24 and config.compute_exact_diagnostics and (
            iteration == 0 or (iteration + 1) % config.exact_diagnostics_every == 0
        )
        if should_compute_exact:
            exact_target_energy = exact_variational_energy(
                hilbert, hamiltonian, target, state.variables
            )
            states = hilbert.all_states()
            target_logp = 2.0 * jnp.real(target.apply(state.variables, states))
            proposal_logq = proposal_wrapper.log_prob(state.proposal_variables, states)
            metrics.update(compare_distributions_smallN(target_logp, proposal_logq))
            metrics["exact_target_energy"] = exact_target_energy
            if exact_target_energy < best_exact_target_energy:
                best_exact_target_energy = exact_target_energy
                (output_dir / "best_target.msgpack").write_bytes(
                    serialization.to_bytes(state.variables)
                )
                (output_dir / "best_proposal.msgpack").write_bytes(
                    serialization.to_bytes(state.proposal_variables)
                )
                (output_dir / "best_checkpoint.json").write_text(
                    json.dumps(
                        {
                            "iteration": iteration + 1,
                            "exact_target_energy": exact_target_energy,
                        },
                        indent=2,
                    )
                )
                metrics["is_best_exact_checkpoint"] = True
            else:
                metrics["is_best_exact_checkpoint"] = False
            metrics["best_exact_target_energy"] = best_exact_target_energy

        key, resample_key = jax.random.split(key)
        resampled, indices = resample_batch(
            resample_key,
            logging_sigma,
            logging_weights,
            n_samples=config.num_samples,
            method=config.resample_method,
        )
        metrics["runtime_iteration_sec"] = time.perf_counter() - iteration_start
        logger.log(metrics)
        print(
            f"it={iteration + 1:4d} Energy = {energy:.17g} "
            f"Error = {float(selected_stats.error_of_mean):.6g} "
            f"ESS = {float(metrics['ESS']):.1f} ({float(metrics['ESS_frac']):.3f})"
        )
        previous_energy = energy

        if (iteration + 1) % config.checkpoint_every == 0 or iteration + 1 == config.num_iterations:
            (output_dir / "target.msgpack").write_bytes(serialization.to_bytes(state.variables))
            (output_dir / "proposal.msgpack").write_bytes(serialization.to_bytes(state.proposal_variables))
            jnp.save(output_dir / "resampled_sigma.npy", resampled)
            jnp.save(output_dir / "resample_indices.npy", indices)

    save_energy_plot(output_dir, config)


def _run_explicit_jax_experiment(config: NISRunConfig):
    """Run the multi-step proposal/target optimisation experiment."""
    graph, symmetry_group, hilbert, hamiltonian = build_hamiltonian(config)
    key = jax.random.PRNGKey(config.seed)
    key, target_key, proposal_key = jax.random.split(key, 3)

    target = build_target_model(config, graph, symmetry_group)
    proposal = MinimalAutoregressiveTransformer(
        hilbert.size,
        config.proposal_embed_dim,
        num_layers=config.proposal_num_layers,
    )
    initial_sigma = jnp.ones((2, hilbert.size), dtype=jnp.float64)
    target_parameters = target.init(target_key, initial_sigma)["params"]
    proposal_parameters = proposal.init(proposal_key, initial_sigma)["params"]
    proposal_wrapper = AutoregressiveProposalWrapper(hilbert, proposal, probability_floor=1e-8)

    target_optimizer = optax.adam(config.target_lr)
    proposal_optimizer = optax.adam(config.proposal_lr)
    target_optimizer_state = target_optimizer.init(target_parameters)
    proposal_optimizer_state = proposal_optimizer.init(proposal_parameters)
    chunk_force = make_chunk_force(target)
    proposal_step = make_proposal_step(proposal_wrapper, proposal_optimizer)

    logger = NISLogger(config.diagnostics_dir)
    logger.save_config(asdict(config))
    output_dir = Path(config.diagnostics_dir)

    # Optional small-system validation only; production NIS never depends on ED.
    exact_ground_energy = None
    if config.compute_exact_ground_energy and graph.n_nodes < 24:
        exact_ground_energy = float(nk.exact.lanczos_ed(hamiltonian, k=1)[0])
        print("Exact ground-state energy (NetKet Lanczos ED):", exact_ground_energy)
    print("JAX devices:", jax.devices())

    devices = jax.local_devices()
    multi_gpu = config.use_multi_gpu and len(devices) > 1
    if multi_gpu:
        n_proposals_per_device = (config.n_proposals + len(devices) - 1) // len(devices)
        total_proposals = n_proposals_per_device * len(devices)
        sample_kernel, logpsi_kernel, bind_hamiltonian = make_multi_gpu_kernels(
            target, proposal_wrapper, n_proposals_per_device, config.target_grad_batch_size
        )
        local_energy_kernel, target_force_kernel, proposal_gradient_kernel = bind_hamiltonian(hamiltonian)
        print(
            f"NIS multi-GPU mode: {len(devices)} devices × {n_proposals_per_device} proposals "
            f"= {total_proposals} total proposals per iteration"
        )
    else:
        total_proposals = config.n_proposals
        print("NIS execution mode: single device")

    previous_instantaneous_energy = None
    for iteration in range(config.num_iterations):
        iteration_start = time.perf_counter()
        key, draw_key, proposal_batch_key, resample_key = jax.random.split(key, 4)
        target_variables = {"params": target_parameters}
        proposal_variables = {"params": proposal_parameters}

        if multi_gpu:
            target_replicas = replicate_for_pmap(target_parameters, len(devices))
            proposal_replicas = replicate_for_pmap(proposal_parameters, len(devices))
            device_keys = jax.random.split(draw_key, len(devices))
            sigma_devices, logq_devices = sample_kernel(proposal_replicas, device_keys)
            logpsi_devices = logpsi_kernel(target_replicas, sigma_devices)
            sigma = jnp.asarray(np.asarray(sigma_devices).reshape(total_proposals, graph.n_nodes))
            logq = jnp.asarray(np.asarray(logq_devices).reshape(total_proposals))
            logp = 2.0 * jnp.real(jnp.asarray(np.asarray(logpsi_devices).reshape(total_proposals)))
            weight_info = stable_log_weights(logp, logq)
            weights = weight_info["weights_normalized"]
            weight_devices = jnp.asarray(weights).reshape(len(devices), n_proposals_per_device)
            local_energy_devices = local_energy_kernel(target_replicas, sigma_devices)
            local_energies = jnp.asarray(np.asarray(local_energy_devices).reshape(total_proposals))
            batch = {"sigma_prop": sigma, "logq_prop": logq, "logp_tilde_prop": logp, "weight_info": weight_info}
        else:
            batch = sample_proposals_and_metrics(
                target.apply, target_variables, proposal_wrapper, proposal_variables,
                draw_key, n_proposals=config.n_proposals,
            )
            sigma = batch["sigma_prop"]
            weights = batch["weight_info"]["weights_normalized"]
            local_energies = local_energy(
                hamiltonian, target, target_variables, sigma,
                log_psi=batch["logpsi_prop"], chunk_size=config.local_energy_chunk_size,
            )
        variational_stats = weighted_variational_statistics(local_energies, weights)
        snis_energy = variational_stats["Mean"]
        snis_variance = variational_stats["Variance"]
        ess_fraction = float(batch["weight_info"]["ess_frac"])

        proposal_batch_size = min(config.proposal_train_batch_size, total_proposals)
        proposal_indices = jax.random.choice(
            proposal_batch_key, total_proposals, shape=(proposal_batch_size,),
            replace=True, p=weights,
        )
        proposal_training_sigma = sigma[proposal_indices]

        loss = None
        if multi_gpu:
            local_train_size = (proposal_batch_size + len(devices) - 1) // len(devices)
            padded_size = local_train_size * len(devices)
            if padded_size != proposal_batch_size:
                extra = proposal_training_sigma[:padded_size - proposal_batch_size]
                proposal_training_sigma = jnp.concatenate([proposal_training_sigma, extra], axis=0)
            training_devices = jnp.asarray(proposal_training_sigma).reshape(
                len(devices), local_train_size, graph.n_nodes
            )
            for _ in range(config.proposal_train_steps):
                replicas = replicate_for_pmap(proposal_parameters, len(devices))
                loss_devices, gradient_devices = proposal_gradient_kernel(replicas, training_devices)
                proposal_gradient = first_replica(gradient_devices)
                loss = float(np.asarray(loss_devices[0]))
                updates, proposal_optimizer_state = proposal_optimizer.update(
                    proposal_gradient, proposal_optimizer_state, proposal_parameters
                )
                proposal_parameters = optax.apply_updates(proposal_parameters, updates)
        else:
            for _ in range(config.proposal_train_steps):
                proposal_parameters, proposal_optimizer_state, loss = proposal_step(
                    proposal_parameters, proposal_optimizer_state, proposal_training_sigma
                )

        target_updated = config.always_update_target or ess_fraction >= config.ess_threshold
        mode = "nis" if target_updated else "proposal_recovery"
        gradient_norm = 0.0
        target_gradient_seconds = 0.0
        if target_updated:
            gradient_start = time.perf_counter()
            if multi_gpu:
                target_replicas = replicate_for_pmap(target_parameters, len(devices))
                energy_replicas = jnp.broadcast_to(snis_energy, (len(devices),))
                target_gradient = first_replica(target_force_kernel(
                    target_replicas, sigma_devices, weight_devices,
                    local_energy_devices, energy_replicas,
                ))
            else:
                target_gradient = weighted_target_grad(
                    target_parameters, sigma, weights, jax.lax.stop_gradient(local_energies),
                    config.target_grad_batch_size, chunk_force,
                )
            target_gradient_seconds = time.perf_counter() - gradient_start
            gradient_norm = tree_norm(target_gradient)
            updates, target_optimizer_state = target_optimizer.update(
                target_gradient, target_optimizer_state, target_parameters
            )
            target_parameters = optax.apply_updates(target_parameters, updates)

        metrics = summarize_weight_batch(batch["weight_info"]["logw_raw"], weights)
        metrics.update({
            "mode": mode,
            "iteration": iteration + 1,
            "number_of_proposals": total_proposals,
            "number_of_returned_samples": config.num_samples,
            # NetKet-style variational expectation fields, adapted to ESS.
            "Mean": float(jnp.real(variational_stats["Mean"])),
            "Variance": float(variational_stats["Variance"]),
            "ErrorOfMean": float(variational_stats["ErrorOfMean"]),
            "energy_mean": float(jnp.real(snis_energy)),
            "energy_variance": float(snis_variance),
            "energy_error": float(variational_stats["ErrorOfMean"]),
            "target_updated": bool(target_updated),
            "gradient_norm": gradient_norm,
            "proposal_loss": float(loss),
            "exact_ground_energy": exact_ground_energy,
            "runtime_target_gradient_sec": target_gradient_seconds,
        })

        should_compute_exact = config.compute_exact_diagnostics and (
            iteration == 0 or (iteration + 1) % config.exact_diagnostics_every == 0
        )
        if should_compute_exact:
            energy_before = exact_variational_energy(hilbert, hamiltonian, target, target_variables)
            updated_target_variables = {"params": target_parameters}
            energy_after = exact_variational_energy(hilbert, hamiltonian, target, updated_target_variables)
            states = hilbert.all_states()
            target_logp = 2.0 * jnp.real(target.apply(updated_target_variables, states))
            proposal_logq = proposal_wrapper.log_prob({"params": proposal_parameters}, states)
            metrics.update(compare_distributions_smallN(target_logp, proposal_logq))
            metrics.update({
                "variational_energy_before_update": energy_before,
                "variational_energy_after_update": energy_after,
                "variational_energy_delta": energy_after - energy_before,
                "exact_target_energy": energy_after,
            })

        # This is the current SNIS variational-energy estimate. Exact values,
        # when enabled, are logged on the configured diagnostic cadence.
        instantaneous_energy = metrics["energy_mean"]
        energy_delta = (
            None
            if previous_instantaneous_energy is None
            else instantaneous_energy - previous_instantaneous_energy
        )
        metrics["energy_delta_from_previous_iteration"] = energy_delta
        metrics["runtime_iteration_sec"] = time.perf_counter() - iteration_start
        resampled, indices = resample_batch(
            resample_key, sigma, weights,
            n_samples=config.num_samples, method=config.resample_method,
        )
        logger.log(metrics)
        print(
            f"it={iteration + 1:4d} Energy = {instantaneous_energy:.17g} "
            f"Error = {metrics['energy_error']:.6g} "
            f"ESS = {metrics['ESS']:.1f} ({metrics['ESS_frac']:.3f})"
        )
        previous_instantaneous_energy = instantaneous_energy

        if (iteration + 1) % config.checkpoint_every == 0 or iteration + 1 == config.num_iterations:
            (output_dir / "target.msgpack").write_bytes(serialization.to_bytes({"params": target_parameters}))
            (output_dir / "proposal.msgpack").write_bytes(serialization.to_bytes({"params": proposal_parameters}))
            jnp.save(output_dir / "resampled_sigma.npy", resampled)
            jnp.save(output_dir / "resample_indices.npy", indices)

    save_energy_plot(output_dir, config)


def run_experiment(config: NISRunConfig):
    """Run the selected NIS implementation backend.

    ``netket`` is the production default for the 8-site launcher.  The
    preserved ``explicit_jax`` backend is the experimental multi-GPU pmap
    implementation and is intentionally selected explicitly.
    """
    if config.execution_backend == "netket":
        return run_netket_experiment(config)
    if config.execution_backend == "explicit_jax":
        if config.target_update != "adam":
            raise ValueError(
                "weighted_sr is currently implemented for execution_backend='netket' only"
            )
        return _run_explicit_jax_experiment(config)
    raise ValueError(
        "execution_backend must be 'netket' or 'explicit_jax', "
        f"got {config.execution_backend!r}"
    )


def main(default_variant: str = "site_relation"):
    """Small compatibility CLI. Prefer the direct configuration launcher."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("plain", "site_relation", "site_relation_gated", "inputproj", "symmproj"),
        default=default_variant,
    )
    parser.add_argument("--n-proposals", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diagnostics-dir", default="results/nis/gamma_8site")
    parser.add_argument(
        "--exact-validation", action="store_true",
        help="Enable full-Hilbert-space diagnostics and ED; use only for small systems.",
    )
    args = parser.parse_args()
    run_experiment(NISRunConfig(
        variant=args.variant, n_proposals=args.n_proposals, num_samples=args.num_samples,
        num_iterations=args.num_iterations, seed=args.seed, diagnostics_dir=args.diagnostics_dir,
        compute_exact_diagnostics=args.exact_validation,
        compute_exact_ground_energy=args.exact_validation,
    ))


if __name__ == "__main__":
    main()
