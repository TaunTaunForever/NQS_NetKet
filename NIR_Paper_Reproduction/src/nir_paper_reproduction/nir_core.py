from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax
from jax.scipy.special import logsumexp
from netket._src.ngd.sr_srt_common import srt as compute_minsr_direction
from scipy.sparse.linalg import eigsh

from .proposal import (
    AutoregressiveProposalNet,
    proposal_log_prob,
    sample_from_proposal,
    train_proposal_step,
)


@dataclass
class NIRSettings:
    num_steps: int = 10000
    n_samples: int = 512
    proposal_batch: int = 512
    max_proposal_batches: int = 64
    max_adaptive_rounds: int = 8
    alpha_ess: float = 2.0
    alpha_eff: float = 0.1
    target_lr: float = 1e-3
    proposal_lr: float = 1e-3
    target_preconditioner: str = "minsr"
    sr_diag_shift: float = 1e-3
    sr_mode: str = "complex"
    proposal_embed_dim: int = 32
    proposal_heads: int = 4
    proposal_layers: int = 4
    proposal_mlp_hidden: int | None = None
    prob_floor: float = 1e-6
    seed: int = 1234
    chunk_size: int | None = None
    log_every: int = 10
    run_name: str = "paper_nir"
    reject_nonfinite_updates: bool = True
    target_update_norm_clip: float | None = None
    target_update_check_samples: int = 64


def count_params(tree) -> int:
    return int(sum(np.asarray(leaf).size for leaf in jax.tree.leaves(tree)))


def tree_l2_norm(tree) -> float:
    total = 0.0
    for leaf in jax.tree.leaves(tree):
        arr = jnp.asarray(leaf)
        total += float(np.asarray(jax.device_get(jnp.sum(jnp.abs(arr) ** 2))))
    return math.sqrt(total)


def tree_all_finite(tree) -> bool:
    for leaf in jax.tree.leaves(tree):
        if not bool(np.asarray(jax.device_get(jnp.all(jnp.isfinite(leaf))))):
            return False
    return True


def array_all_finite(array) -> bool:
    return bool(np.asarray(jax.device_get(jnp.all(jnp.isfinite(array)))))


def scale_tree(tree, scale: float):
    return jax.tree.map(lambda leaf: leaf * scale, tree)


def candidate_target_is_finite(vstate, params, samples, settings: NIRSettings):
    if not tree_all_finite(params):
        return False, "nonfinite_target_parameters"

    n_check = min(settings.target_update_check_samples, int(samples.shape[0]))
    check_samples = samples[:n_check]
    log_values = vstate._apply_fun(
        {"params": params, **vstate.model_state},
        check_samples,
    )
    if not array_all_finite(log_values):
        return False, "nonfinite_target_log_value"
    return True, ""


def normalised_importance_weights(log_target, log_proposal):
    log_weights = jnp.asarray(log_target - log_proposal, dtype=jnp.float64)
    return jnp.exp(log_weights - logsumexp(log_weights))


def effective_sample_size(weights):
    weights = jnp.asarray(weights, dtype=jnp.float64)
    return 1.0 / jnp.sum(jnp.square(weights))


def sampling_efficiency(weights):
    return effective_sample_size(weights) / weights.shape[0]


def multinomial_resample(samples, weights, n_samples, rng):
    rng, subkey = jax.random.split(rng)
    safe_log_weights = jnp.log(jnp.clip(weights, jnp.finfo(weights.dtype).tiny, 1.0))
    indices = jax.random.categorical(subkey, safe_log_weights, shape=(n_samples,))
    return samples[indices], indices, rng


def make_local_sampler(hilbert, n_samples):
    n_chains = max(1, n_samples // 128)
    return nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=n_chains)


def inject_external_samples(vstate, samples):
    samples = jnp.asarray(samples, dtype=jnp.float64)
    reshaped = samples.reshape(vstate.sampler.n_chains, vstate.chain_length, samples.shape[-1])
    vstate._samples = reshaped
    return reshaped


def target_log_probs(vstate, samples):
    return 2.0 * jnp.real(vstate.log_value(samples))


def compute_target_direction(vstate, hamiltonian, settings: NIRSettings, old_updates):
    if settings.target_preconditioner == "none":
        stats, grad = vstate.expect_and_grad(hamiltonian)
        return stats, grad, old_updates

    if settings.target_preconditioner != "minsr":
        raise ValueError("Only 'none' and 'minsr' are currently implemented here.")

    local_energies = vstate.local_estimators(hamiltonian, chunk_size=settings.chunk_size)
    stats = nk.stats.statistics(local_energies)
    samples = jax.lax.collapse(vstate.samples, 0, vstate.samples.ndim - 1)
    dp, old_updates, _info = compute_minsr_direction(
        vstate._apply_fun,
        local_energies,
        vstate.parameters,
        vstate.model_state,
        samples,
        diag_shift=settings.sr_diag_shift,
        solver_fn=nk.optimizer.solver.cholesky,
        mode=settings.sr_mode,
        old_updates=old_updates,
        chunk_size=settings.chunk_size,
    )
    return stats, dp, old_updates


def sample_until_ess(vstate, proposal_model, proposal_params, rng, settings: NIRSettings):
    proposal_batches = []
    log_target_batches = []
    log_proposal_batches = []
    ess_target = settings.alpha_ess * settings.n_samples

    for _ in range(settings.max_proposal_batches):
        proposal_samples, rng = sample_from_proposal(
            proposal_model,
            proposal_params,
            rng,
            settings.proposal_batch,
            vstate.hilbert.size,
            prob_floor=settings.prob_floor,
        )
        log_target = target_log_probs(vstate, proposal_samples)
        log_proposal = proposal_log_prob(
            proposal_model,
            proposal_params,
            proposal_samples,
            prob_floor=settings.prob_floor,
        )
        proposal_batches.append(proposal_samples)
        log_target_batches.append(log_target)
        log_proposal_batches.append(log_proposal)

        weights = normalised_importance_weights(
            jnp.concatenate(log_target_batches, axis=0),
            jnp.concatenate(log_proposal_batches, axis=0),
        )
        if float(effective_sample_size(weights)) >= ess_target:
            break

    all_samples = jnp.concatenate(proposal_batches, axis=0)
    all_log_target = jnp.concatenate(log_target_batches, axis=0)
    all_log_proposal = jnp.concatenate(log_proposal_batches, axis=0)
    weights = normalised_importance_weights(all_log_target, all_log_proposal)
    return all_samples, weights, rng


def guarded_proposal_update(
    *,
    proposal_model,
    proposal_params,
    proposal_opt_state,
    proposal_optimizer,
    resampled,
    settings: NIRSettings,
):
    old_params = proposal_params
    old_opt_state = proposal_opt_state
    proposal_params, proposal_opt_state, proposal_loss = train_proposal_step(
        proposal_model,
        proposal_params,
        proposal_opt_state,
        proposal_optimizer,
        resampled,
        prob_floor=settings.prob_floor,
    )

    rejected = False
    reject_reason = ""
    if settings.reject_nonfinite_updates and not tree_all_finite(proposal_params):
        proposal_params = old_params
        proposal_opt_state = old_opt_state
        rejected = True
        reject_reason = "nonfinite_proposal_parameters"

    return proposal_params, proposal_opt_state, proposal_loss, rejected, reject_reason


def guarded_target_update(
    *,
    vstate,
    hamiltonian,
    settings: NIRSettings,
    old_updates,
    target_optimizer,
    target_opt_state,
    resampled,
):
    stats, direction, old_updates_candidate = compute_target_direction(
        vstate,
        hamiltonian,
        settings,
        old_updates,
    )
    diagnostics = {
        "target_direction_norm": tree_l2_norm(direction),
        "target_update_norm": None,
        "target_update_rejected": False,
        "target_reject_reason": "",
        "target_direction_scale": 1.0,
    }

    if settings.reject_nonfinite_updates and not tree_all_finite(direction):
        diagnostics["target_update_rejected"] = True
        diagnostics["target_reject_reason"] = "nonfinite_target_direction"
        return stats, old_updates, target_opt_state, diagnostics

    direction_scale = 1.0
    if settings.target_update_norm_clip is not None:
        direction_norm = diagnostics["target_direction_norm"]
        if direction_norm > settings.target_update_norm_clip:
            direction_scale = settings.target_update_norm_clip / (direction_norm + 1e-12)
            direction = scale_tree(direction, direction_scale)
            diagnostics["target_direction_scale"] = direction_scale

    updates, target_opt_state_candidate = target_optimizer.update(
        direction,
        target_opt_state,
        vstate.parameters,
    )
    diagnostics["target_update_norm"] = tree_l2_norm(updates)
    candidate_params = optax.apply_updates(vstate.parameters, updates)

    if settings.reject_nonfinite_updates:
        is_finite, reason = candidate_target_is_finite(
            vstate,
            candidate_params,
            resampled,
            settings,
        )
        if not is_finite:
            diagnostics["target_update_rejected"] = True
            diagnostics["target_reject_reason"] = reason
            return stats, None, target_optimizer.init(vstate.parameters), diagnostics

    vstate.parameters = candidate_params
    return stats, old_updates_candidate, target_opt_state_candidate, diagnostics


def run_single_state_nir(
    *,
    hilbert,
    hamiltonian,
    target_model,
    settings: NIRSettings,
    output_root: str | Path,
):
    output_root = Path(output_root)
    run_dir = output_root / date.today().isoformat() / settings.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(settings.seed)
    rng, target_key, proposal_key = jax.random.split(rng, 3)

    sampler = make_local_sampler(hilbert, settings.n_samples)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=target_model,
        n_samples=settings.n_samples,
        seed=target_key,
        chunk_size=settings.chunk_size,
    )

    proposal_model = AutoregressiveProposalNet(
        n_sites=hilbert.size,
        embed_dim=settings.proposal_embed_dim,
        num_heads=settings.proposal_heads,
        num_layers=settings.proposal_layers,
        mlp_hidden_dim=settings.proposal_mlp_hidden,
    )
    init_sigma = jnp.ones((4, hilbert.size), dtype=jnp.float64)
    proposal_params = proposal_model.init(proposal_key, init_sigma)["params"]

    target_optimizer = optax.adam(settings.target_lr)
    target_opt_state = target_optimizer.init(vstate.parameters)
    proposal_optimizer = optax.adam(settings.proposal_lr)
    proposal_opt_state = proposal_optimizer.init(proposal_params)
    old_updates = None

    print("Run directory:", run_dir)
    print("Target parameters:", count_params(vstate.parameters))
    print("Proposal parameters:", count_params(proposal_params))
    print("Settings:", json.dumps(asdict(settings), sort_keys=True))

    history = []
    last_resampled = None
    for step in range(settings.num_steps):
        target_update_applied = False
        round_records = []
        for round_idx in range(settings.max_adaptive_rounds):
            all_samples, weights, rng = sample_until_ess(
                vstate,
                proposal_model,
                proposal_params,
                rng,
                settings,
            )
            ess = float(effective_sample_size(weights))
            eff = float(sampling_efficiency(weights))
            resampled, _indices, rng = multinomial_resample(
                all_samples,
                weights,
                settings.n_samples,
                rng,
            )
            (
                proposal_params,
                proposal_opt_state,
                proposal_loss,
                proposal_update_rejected,
                proposal_reject_reason,
            ) = guarded_proposal_update(
                proposal_model=proposal_model,
                proposal_params=proposal_params,
                proposal_opt_state=proposal_opt_state,
                proposal_optimizer=proposal_optimizer,
                resampled=resampled,
                settings=settings,
            )
            round_records.append(
                {
                    "round": round_idx,
                    "proposal_pool": int(all_samples.shape[0]),
                    "ess": ess,
                    "efficiency": eff,
                    "proposal_loss": float(proposal_loss),
                    "proposal_update_rejected": proposal_update_rejected,
                    "proposal_reject_reason": proposal_reject_reason,
                }
            )
            if eff >= settings.alpha_eff:
                target_update_applied = True
                break

        last_resampled = resampled
        inject_external_samples(vstate, resampled)
        target_diagnostics = {
            "target_direction_norm": None,
            "target_update_norm": None,
            "target_update_rejected": False,
            "target_reject_reason": "",
            "target_direction_scale": 1.0,
        }
        if target_update_applied:
            stats, old_updates, target_opt_state, target_diagnostics = (
                guarded_target_update(
                    vstate=vstate,
                    hamiltonian=hamiltonian,
                    settings=settings,
                    old_updates=old_updates,
                    target_optimizer=target_optimizer,
                    target_opt_state=target_opt_state,
                    resampled=resampled,
                )
            )
        else:
            stats = nk.stats.statistics(
                vstate.local_estimators(hamiltonian, chunk_size=settings.chunk_size)
            )

        energy = float(np.real(np.asarray(stats.mean)))
        record = {
            "iteration": step + 1,
            "energy": energy,
            "target_update_applied": target_update_applied,
            "rounds": round_records,
            **target_diagnostics,
        }
        history.append(record)
        if step == 0 or (step + 1) % settings.log_every == 0:
            last_round = round_records[-1]
            print(
                f"it={step + 1:6d} "
                f"Energy={energy:.10f} "
                f"ESS={last_round['ess']:.2f} "
                f"Eff={last_round['efficiency']:.4f} "
                f"Pool={last_round['proposal_pool']} "
                f"Update={'yes' if target_update_applied else 'no'}"
            )

    summary = {
        "settings": asdict(settings),
        "target_parameter_count": count_params(vstate.parameters),
        "proposal_parameter_count": count_params(proposal_params),
        "final_energy": history[-1]["energy"] if history else None,
        "tail100_energy_mean": (
            float(np.mean([row["energy"] for row in history[-100:]]))
            if history
            else None
        ),
        "history": history,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return {
        "run_dir": run_dir,
        "summary_path": summary_path,
        "vstate": vstate,
        "proposal_model": proposal_model,
        "proposal_params": proposal_params,
        "last_resampled": last_resampled,
        "history": history,
        "rng": rng,
    }


def exact_probabilities(vstate):
    states = vstate.hilbert.all_states()
    logp = target_log_probs(vstate, states)
    probs = jnp.exp(logp - logsumexp(logp))
    states_np = np.asarray(states)
    probs_np = np.asarray(probs)
    ordered = np.zeros(2 ** vstate.hilbert.size, dtype=np.float64)
    ordered[samples_to_keys(states_np)] = probs_np
    return states_np, ordered


def safe_hilbert_n_states(hilbert) -> int | None:
    try:
        return int(hilbert.n_states)
    except RuntimeError:
        return None


def exact_ground_state_energy(hamiltonian, *, max_states: int = 65536):
    hilbert = hamiltonian.hilbert
    n_states = safe_hilbert_n_states(hilbert)
    if n_states is None:
        return None

    if n_states > max_states:
        return None

    try:
        matrix = hamiltonian.to_sparse()
    except RuntimeError:
        return None

    if matrix.shape[0] <= 2:
        return float(np.min(np.linalg.eigvalsh(matrix.toarray())))

    values = eigsh(matrix, k=1, which="SA", return_eigenvectors=False)
    return float(np.real(values[0]))


def samples_to_keys(samples):
    samples = np.asarray(samples)
    tokens = ((samples + 1) // 2).astype(np.int64)
    powers = (1 << np.arange(tokens.shape[-1], dtype=np.int64))
    return tokens @ powers


def empirical_distribution(samples, n_sites):
    keys = samples_to_keys(samples)
    counts = np.bincount(keys, minlength=2**n_sites).astype(np.float64)
    return counts / np.maximum(np.sum(counts), 1.0)


def jensen_shannon_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * (np.log(a[mask]) - np.log(b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
