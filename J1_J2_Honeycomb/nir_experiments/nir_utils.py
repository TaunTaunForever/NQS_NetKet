from __future__ import annotations

import numpy as np


def logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    vmax = np.max(values)
    return float(vmax + np.log(np.sum(np.exp(values - vmax))))


def normalised_importance_weights_from_log_probs(
    log_target_probs: np.ndarray,
    log_proposal_probs: np.ndarray,
) -> np.ndarray:
    log_target_probs = np.asarray(log_target_probs, dtype=np.float64)
    log_proposal_probs = np.asarray(log_proposal_probs, dtype=np.float64)
    if log_target_probs.shape != log_proposal_probs.shape:
        raise ValueError("Target and proposal log-probabilities must have matching shapes.")

    log_weights = log_target_probs - log_proposal_probs
    log_norm = logsumexp(log_weights)
    return np.exp(log_weights - log_norm)


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array.")
    return float(1.0 / np.sum(np.square(weights)))


def sampling_efficiency(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    return float(effective_sample_size(weights) / len(weights))


def importance_resample(
    configs: np.ndarray,
    log_target_probs: np.ndarray,
    log_proposal_probs: np.ndarray,
    *,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    configs = np.asarray(configs)
    weights = normalised_importance_weights_from_log_probs(log_target_probs, log_proposal_probs)
    indices = rng.choice(len(configs), size=n_samples, replace=True, p=weights)
    return configs[indices], indices, weights

