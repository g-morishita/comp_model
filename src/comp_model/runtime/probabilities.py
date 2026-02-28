"""Probability utilities shared across simulation and replay.

Centralizing probability normalization prevents divergence between generation
and inference code paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def normalize_distribution(
    raw_distribution: Mapping[Any, float],
    available_actions: tuple[Any, ...],
) -> dict[Any, float]:
    """Validate and normalize action probabilities.

    Parameters
    ----------
    raw_distribution : Mapping[Any, float]
        Model-emitted action weights.
    available_actions : tuple[Any, ...]
        Legal actions for the trial.

    Returns
    -------
    dict[Any, float]
        Normalized probabilities over ``available_actions``.

    Raises
    ------
    ValueError
        If the model emits invalid keys/values or total probability is zero.
    """

    unknown_actions = set(raw_distribution.keys()) - set(available_actions)
    if unknown_actions:
        raise ValueError(f"distribution contains unknown actions: {sorted(unknown_actions)!r}")

    weights: dict[Any, float] = {}
    for action in available_actions:
        value = float(raw_distribution.get(action, 0.0))
        if value < 0:
            raise ValueError(f"distribution contains negative weight for action {action!r}")
        weights[action] = value

    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("distribution sum must be > 0 for available actions")

    return {action: value / total for action, value in weights.items()}


def sample_action(distribution: Mapping[Any, float], rng: np.random.Generator) -> Any:
    """Sample one action from a validated distribution.

    Parameters
    ----------
    distribution : Mapping[Any, float]
        Normalized action probabilities.
    rng : numpy.random.Generator
        Random generator used for sampling.

    Returns
    -------
    Any
        Sampled action.
    """

    actions = tuple(distribution.keys())
    probs = np.asarray(tuple(distribution.values()), dtype=float)
    return actions[int(rng.choice(len(actions), p=probs))]
