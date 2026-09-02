"""Fixed-pool validation of the 8-site weighted NIS SR implementation.

This is a small-system diagnostic, not a production optimiser.  It restores a
saved target/proposal pair, draws one fixed NIS pool, and compares:

* the matrix-free weighted-QGT action against an explicit sampled QGT;
* matrix-free CG directions against a dense solve of that same sampled QGT;
* symmetry and positive-semidefiniteness of the real weighted QGT.

No checkpoint or existing experiment log is changed.  Edit only the constants
in the section below, then run this file directly.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

# Keep the explicit dense validation on one device. Native sharding is useful
# in production, but gathering a dense sample-by-parameter Jacobian is a
# deliberately small-N debugging operation.
NIS_DEVICE = None  # None = automatic; use "gpu" or "cpu" only when needed.
if NIS_DEVICE is not None:
    os.environ["JAX_PLATFORM_NAME"] = NIS_DEVICE
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import netket as nk
from flax import serialization

# Import the experiment module first: it establishes the repository's module
# search path for direct execution from this 8-site directory.
from run_gamma_nis import NISRunConfig, build_target_model, exact_variational_energy

from hamiltonian import gamma_hamiltonian
from optim.weighted_sr import WeightedSR, weighted_qgt
from samplers.minimal_autoregressive import MinimalAutoregressiveTransformer
from samplers.neural_importance_sampling import NeuralImportanceSampler
from samplers.proposal_wrappers import AutoregressiveProposalWrapper
from vqs import WeightedNISState

HERE = Path(__file__).resolve().parent

# ---------------------- Diagnostic parameters: edit here ----------------------
# This is the final checkpoint from the completed conservative refinement. Set
# this to another run directory to diagnose a different late-stage state.
CHECKPOINT_DIRECTORY = HERE / "results/nis/gamma_8site_site_relation_refine"
TARGET_CHECKPOINT = CHECKPOINT_DIRECTORY / "target.msgpack"
PROPOSAL_CHECKPOINT = CHECKPOINT_DIRECTORY / "proposal.msgpack"

# Use the same proposal-pool size as the refinement run. Reducing this can be
# useful for a quick smoke test, but the decisive comparison should use 4098.
# ``NIS_DIAGNOSTIC_PROPOSALS=256`` is useful for a quick smoke test; omit it
# for the full 4098-proposal diagnostic.
N_PROPOSALS = int(os.environ.get("NIS_DIAGNOSTIC_PROPOSALS", "4098"))
SEED = 20260805

# Evaluate the actual late-stage damping and an enlarged Krylov budget. The
# dense reference always uses precisely the same shift as the CG solve.
DIAG_SHIFT = 1.0e-1
# ``NIS_DIAGNOSTIC_CG_MAXITERS=64`` skips the expensive enlarged-solve check
# during a smoke test. The default runs both comparisons.
CG_ITERATION_BUDGETS = tuple(
    int(value) for value in os.environ.get("NIS_DIAGNOSTIC_CG_MAXITERS", "64,256").split(",")
)
CG_TOLERANCE = 1.0e-10
CHUNK_SIZE = 1366
# Set ``NIS_DIAGNOSTIC_DIAGONAL_PRECONDITIONER=1`` to validate the new
# production solve: diagonal-preconditioned conjugate gradient with relative
# damping and periodically recomputed true residuals.
USE_DIAGONAL_PRECONDITIONER = (
    os.environ.get("NIS_DIAGNOSTIC_DIAGONAL_PRECONDITIONER", "0") == "1"
)
DIAGONAL_PROBES = int(os.environ.get("NIS_DIAGNOSTIC_DIAGONAL_PROBES", "4"))
DIAGONAL_MODE = os.environ.get("NIS_DIAGNOSTIC_DIAGONAL_MODE", "per_parameter")
RESIDUAL_TARGET = float(os.environ.get("NIS_DIAGNOSTIC_RESIDUAL_TARGET", "1e-3"))
RESIDUAL_REPLACEMENT_INTERVAL = 32

# Match the architecture stored in the refinement checkpoint.
MODEL_CONFIG = NISRunConfig(
    variant="site_relation",
    embed_dim=8,
    num_heads=2,
    num_layers=2,
    mlp_hidden_dim=16,
    patch_size=1,
    proposal_embed_dim=8,
)
# ------------------------------------------------------------------------------


def _flatten_complex_log_derivatives(state, sigma):
    """Return J[i, k] = d log(psi(sigma_i)) / d theta_k explicitly.

    Parameters are real while the target log wavefunction is complex, so the
    two real-output Jacobians are formed separately and combined.  This is
    intentionally not scalable; it is the reference for this 8-site check.
    """
    model_state = state.model_state

    def apply(parameters):
        return state.model.apply({"params": parameters, **model_state}, sigma)

    jacobian_real = jax.jacrev(lambda parameters: jnp.real(apply(parameters)))(
        state.parameters
    )
    jacobian_imag = jax.jacrev(lambda parameters: jnp.imag(apply(parameters)))(
        state.parameters
    )
    return jnp.concatenate(
        [
            (real + 1j * imag).reshape(sigma.shape[0], -1)
            for real, imag in zip(
                jax.tree.leaves(jacobian_real), jax.tree.leaves(jacobian_imag)
            )
        ],
        axis=1,
    )


def _norm(vector):
    return jnp.linalg.norm(vector)


def _relative_error(actual, reference):
    return _norm(actual - reference) / jnp.maximum(_norm(reference), jnp.finfo(jnp.float64).tiny)


def _cosine(left, right):
    denominator = _norm(left) * _norm(right)
    return jnp.where(denominator > 0.0, jnp.vdot(left, right) / denominator, 1.0)


def _check_dtypes(weights, local_energies, force):
    force_dtypes = {str(leaf.dtype) for leaf in jax.tree.leaves(force)}
    print(
        "Dtypes: "
        f"weights={weights.dtype}, local_energies={local_energies.dtype}, "
        f"force={sorted(force_dtypes)}"
    )
    if weights.dtype != jnp.float64:
        raise TypeError("weighted SR diagnostic requires float64 normalized weights")
    if local_energies.dtype != jnp.complex128:
        raise TypeError("weighted SR diagnostic requires complex128 local energies")
    if force_dtypes != {"float64"}:
        raise TypeError("weighted SR diagnostic requires float64 real parameter forces")


def _restore(template, path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")
    print(f"Restoring {label}: {path}")
    return serialization.from_bytes(template, path.read_bytes())


def main():
    graph, symmetry_group, hilbert, hamiltonian = gamma_hamiltonian(8)
    key = jax.random.PRNGKey(SEED)
    key, target_key, proposal_key = jax.random.split(key, 3)
    initial_sigma = jnp.ones((2, graph.n_nodes), dtype=jnp.float64)

    target = build_target_model(MODEL_CONFIG, graph, symmetry_group)
    proposal = MinimalAutoregressiveTransformer(graph.n_nodes, MODEL_CONFIG.proposal_embed_dim)
    target_variables = _restore(
        target.init(target_key, initial_sigma), TARGET_CHECKPOINT, "target"
    )
    proposal_variables = _restore(
        proposal.init(proposal_key, initial_sigma), PROPOSAL_CHECKPOINT, "proposal"
    )
    proposal_wrapper = AutoregressiveProposalWrapper(hilbert, proposal, probability_floor=1.0e-8)
    sampler = NeuralImportanceSampler(
        hilbert,
        proposal,
        proposal_variables,
        proposal_wrapper,
        n_proposals=N_PROPOSALS,
        resample_size=min(1024, N_PROPOSALS),
    )
    state = WeightedNISState(
        hilbert,
        target,
        sampler,
        variables=target_variables,
        proposal_variables=proposal_variables,
        n_samples=N_PROPOSALS,
        seed=SEED,
        chunk_size=CHUNK_SIZE,
        local_energy_chunk_size=16384,
        use_sharding=False,
    )

    print("JAX devices:", jax.devices())
    start = time.perf_counter()
    stats, force = state.expect_and_grad(hamiltonian)
    batch = state.last_batch
    assert batch is not None
    weights = jnp.asarray(batch["weight_info"]["weights_normalized"], dtype=jnp.float64)
    weights = weights / state.global_sum(weights)
    local_energies = state._last_local_energies
    assert local_energies is not None
    _check_dtypes(weights, local_energies, force)
    flat_force, unravel = jax.flatten_util.ravel_pytree(force)
    print(
        f"Fixed pool: N={N_PROPOSALS}, parameters={flat_force.size}, "
        f"Energy={float(jnp.real(stats.mean)):.10f}, "
        f"ESS fraction={float(batch['weight_info']['ess_frac']):.6f}"
    )
    print(
        "Exact variational energy: "
        f"{exact_variational_energy(hilbert, hamiltonian, target, state.variables):.10f}"
    )

    print("Forming explicit sampled weighted QGT (8-site validation only)...")
    log_derivatives = _flatten_complex_log_derivatives(state, batch["sigma_prop"])
    dense_qgt = jnp.real(weighted_qgt(log_derivatives, weights, diag_shift=0.0))
    dense_qgt = 0.5 * (dense_qgt + dense_qgt.T)
    eigenvalues = jnp.linalg.eigvalsh(dense_qgt)
    max_eigenvalue = jnp.max(eigenvalues)
    positive_eigenvalues = eigenvalues[eigenvalues > max_eigenvalue * 1.0e-12]
    condition_number = (
        jnp.inf
        if positive_eigenvalues.size == 0
        else max_eigenvalue / jnp.min(positive_eigenvalues)
    )
    damped_condition_number = (max_eigenvalue + DIAG_SHIFT) / jnp.maximum(
        jnp.min(eigenvalues) + DIAG_SHIFT, jnp.finfo(jnp.float64).tiny
    )

    random_key = jax.random.PRNGKey(SEED + 1)
    random_key, u_key, v_key = jax.random.split(random_key, 3)
    random_u = jax.random.normal(u_key, flat_force.shape, dtype=jnp.float64)
    random_v = jax.random.normal(v_key, flat_force.shape, dtype=jnp.float64)
    matvec_sr = WeightedSR(
        diag_shift=DIAG_SHIFT,
        maxiter=1,
        chunk_size=CHUNK_SIZE,
        trust_region=None,
        adaptive=False,
    )
    matrix_free_v, _ = jax.flatten_util.ravel_pytree(
        matvec_sr._weighted_qgt_matvec(
            state, batch["sigma_prop"], weights, unravel(random_v), diag_shift=0.0
        )
    )
    matrix_free_u, _ = jax.flatten_util.ravel_pytree(
        matvec_sr._weighted_qgt_matvec(
            state, batch["sigma_prop"], weights, unravel(random_u), diag_shift=0.0
        )
    )
    dense_v = dense_qgt @ random_v
    matvec_error = _relative_error(matrix_free_v, dense_v)
    symmetry_error = jnp.abs(random_u @ matrix_free_v - random_v @ matrix_free_u) / jnp.maximum(
        jnp.abs(random_u @ matrix_free_v) + jnp.abs(random_v @ matrix_free_u),
        jnp.finfo(jnp.float64).tiny,
    )
    print(
        "QGT validation: "
        f"matvec relative error={float(matvec_error):.3e}, "
        f"symmetry error={float(symmetry_error):.3e}, "
        f"min eigenvalue={float(jnp.min(eigenvalues)):.3e}, "
        f"effective condition={float(condition_number):.3e}, "
        f"damped condition={float(damped_condition_number):.3e}"
    )

    for maxiter in CG_ITERATION_BUDGETS:
        sr = WeightedSR(
            diag_shift=DIAG_SHIFT,
            maxiter=maxiter,
            tol=CG_TOLERANCE,
            chunk_size=CHUNK_SIZE,
            trust_region=None,
            adaptive=False,
            adaptive_maxiter=maxiter,
            diagonal_preconditioner=USE_DIAGONAL_PRECONDITIONER,
            relative_damping=USE_DIAGONAL_PRECONDITIONER,
            diagonal_probes=DIAGONAL_PROBES,
            diagonal_update_interval=1,
            diagonal_ema=0.0,
            diagonal_mode=DIAGONAL_MODE,
            residual_target=RESIDUAL_TARGET if USE_DIAGONAL_PRECONDITIONER else None,
            residual_replacement_interval=(
                RESIDUAL_REPLACEMENT_INTERVAL if USE_DIAGONAL_PRECONDITIONER else 0
            ),
        )
        matrix_free_direction = sr(state, force)
        flat_direction, _ = jax.flatten_util.ravel_pytree(matrix_free_direction)
        if USE_DIAGONAL_PRECONDITIONER:
            assert sr._metric_diagonal is not None
            flat_diagonal, _ = jax.flatten_util.ravel_pytree(sr._metric_diagonal)
            dense_system = dense_qgt + DIAG_SHIFT * jnp.diag(flat_diagonal)
        else:
            dense_system = dense_qgt + DIAG_SHIFT * jnp.eye(
                flat_force.size, dtype=jnp.float64
            )
        dense_direction = jnp.linalg.solve(dense_system, flat_force)
        residual = _relative_error(dense_system @ flat_direction, flat_force)
        direction_error = _relative_error(flat_direction, dense_direction)
        cosine = _cosine(flat_direction, dense_direction)
        assert sr.last_info is not None
        print(
            f"CG maxiter={maxiter}: iterations={sr.last_info['CGIterations']}, "
            f"reported residual={sr.last_info['CGRelativeResidual']:.3e}, "
            f"explicit residual={float(residual):.3e}, "
            f"direction error={float(direction_error):.3e}, "
            f"cosine={float(cosine):.10f}, "
            f"converged={sr.last_info['CGConverged']}, "
            f"preconditioned={sr.last_info['DiagonalPreconditioner']}"
        )

    elapsed = time.perf_counter() - start
    print(f"Diagnostic completed in {elapsed:.1f} s.")
    print("Reference thresholds: matvec error < 1e-8, symmetry error < 1e-10,")
    print("and, for a converged CG solve, explicit residual < 1e-4 and cosine > 0.999.")


if __name__ == "__main__":
    main()
