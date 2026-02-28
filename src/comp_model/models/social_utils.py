"""Shared helpers for social model implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def extract_state(observation: Any) -> int:
    """Extract latent state index from observation payload."""

    if isinstance(observation, Mapping) and "state" in observation:
        return int(observation["state"])
    return 0


def maybe_extract_reward(outcome: Any) -> float | None:
    """Extract scalar reward when present, otherwise return ``None``."""

    if outcome is None:
        return None

    if hasattr(outcome, "reward"):
        return float(getattr(outcome, "reward"))

    if isinstance(outcome, Mapping) and "reward" in outcome:
        return float(outcome["reward"])

    return None


def extract_reward(outcome: Any) -> float:
    """Extract scalar reward from supported outcome payload forms."""

    reward = maybe_extract_reward(outcome)
    if reward is None:
        raise TypeError("Outcome must expose reward via attribute 'reward' or mapping key 'reward'")
    return reward


def extract_source_actor_id(observation: Any, outcome: Any) -> str | None:
    """Resolve the actor ID that generated the action/outcome when available."""

    if outcome is not None:
        if hasattr(outcome, "source_actor_id"):
            return str(getattr(outcome, "source_actor_id"))
        if isinstance(outcome, Mapping) and "source_actor_id" in outcome:
            return str(outcome["source_actor_id"])

    if isinstance(observation, Mapping) and "stage" in observation:
        stage = str(observation["stage"])
        if stage in {"subject", "demonstrator"}:
            return stage

    return None


def is_subject_stage(observation: Any, outcome: Any) -> bool:
    """Return whether current update corresponds to subject-generated action."""

    actor_id = extract_source_actor_id(observation, outcome)
    return actor_id == "subject"


def is_demonstrator_stage(observation: Any, outcome: Any) -> bool:
    """Return whether current update corresponds to demonstrator action/outcome."""

    actor_id = extract_source_actor_id(observation, outcome)
    return actor_id == "demonstrator"


def ensure_q_state(
    q_values: dict[int, dict[Any, float]],
    *,
    state: int,
    actions: tuple[Any, ...],
    initial: float,
) -> dict[Any, float]:
    """Ensure a state's action values exist for all currently available actions."""

    state_values = q_values.setdefault(int(state), {})
    for action in actions:
        state_values.setdefault(action, float(initial))
    return state_values


def ensure_policy_state(
    policy_values: dict[int, dict[Any, float]],
    *,
    state: int,
    actions: tuple[Any, ...],
) -> dict[Any, float]:
    """Ensure a state's policy estimate exists and is normalized."""

    if not actions:
        raise ValueError("actions must not be empty")

    state_values = policy_values.setdefault(int(state), {})
    if not state_values:
        uniform = 1.0 / float(len(actions))
        for action in actions:
            state_values[action] = uniform
        return state_values

    for action in actions:
        state_values.setdefault(action, 1e-8)

    total = float(sum(state_values[action] for action in actions))
    if total <= 0.0:
        uniform = 1.0 / float(len(actions))
        for action in actions:
            state_values[action] = uniform
        return state_values

    for action in actions:
        state_values[action] = float(state_values[action]) / total
    return state_values


def ensure_count_state(
    count_values: dict[int, dict[Any, float]],
    *,
    state: int,
    actions: tuple[Any, ...],
    prior: float,
) -> dict[Any, float]:
    """Ensure a state's count table exists for Dirichlet-style policy inference."""

    if prior <= 0.0:
        raise ValueError("prior must be > 0")

    state_values = count_values.setdefault(int(state), {})
    for action in actions:
        state_values.setdefault(action, float(prior))
    return state_values


def stable_softmax(values: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax probabilities."""

    if values.size == 1:
        return np.asarray([1.0], dtype=float)

    logits = np.asarray(values, dtype=float)
    logits -= float(np.max(logits))
    exp_logits = np.exp(logits)
    return exp_logits / float(np.sum(exp_logits))


def policy_vector(policy_state: dict[Any, float], *, actions: tuple[Any, ...]) -> np.ndarray:
    """Build ordered policy vector for current action set."""

    values = np.asarray([float(policy_state[action]) for action in actions], dtype=float)
    total = float(np.sum(values))
    if total <= 0.0:
        return np.ones(values.size, dtype=float) / float(values.size)
    return values / total


def count_vector(count_state: dict[Any, float], *, actions: tuple[Any, ...]) -> np.ndarray:
    """Build normalized action-frequency vector from count state."""

    values = np.asarray([float(count_state[action]) for action in actions], dtype=float)
    total = float(np.sum(values))
    if total <= 0.0:
        return np.ones(values.size, dtype=float) / float(values.size)
    return values / total


def determinism_index_reliability(probabilities: np.ndarray) -> float:
    """Compute normalized determinism reliability in ``[0, 1]``.

    The metric is 0 at uniform policy and 1 at deterministic policy.
    """

    n_actions = int(probabilities.size)
    if n_actions <= 1:
        return 1.0

    uniform_prob = 1.0 / float(n_actions)
    rel = (float(np.max(probabilities)) - uniform_prob) / (1.0 - uniform_prob)
    return float(np.clip(rel, 0.0, 1.0))


def normalized_policy_drive(probabilities: np.ndarray) -> np.ndarray:
    """Convert policy probabilities to chance-centered decision signal."""

    n_actions = int(probabilities.size)
    if n_actions <= 1:
        return np.asarray([0.0], dtype=float)

    uniform_prob = 1.0 / float(n_actions)
    return (probabilities - uniform_prob) / (1.0 - uniform_prob)


def validate_alpha(alpha: float, *, name: str) -> None:
    """Validate learning-rate parameter."""

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def validate_beta(beta: float, *, name: str = "beta") -> None:
    """Validate inverse-temperature/weight parameter."""

    if beta < 0.0:
        raise ValueError(f"{name} must be >= 0")
