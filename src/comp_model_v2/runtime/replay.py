"""Replay engine for model likelihood evaluation on canonical traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from comp_model_v2.core.contracts import AgentModel, DecisionContext
from comp_model_v2.core.events import EpisodeTrace, group_events_by_trial, validate_trace
from comp_model_v2.runtime.probabilities import normalize_distribution


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """Per-trial replay likelihood record.

    Parameters
    ----------
    trial_index : int
        Zero-based trial index.
    action : Any
        Action observed in the trace.
    probability : float
        Model-assigned probability of the observed action.
    log_probability : float
        Natural log-probability of the observed action.
    """

    trial_index: int
    action: Any
    probability: float
    log_probability: float


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Aggregate replay likelihood output.

    Parameters
    ----------
    total_log_likelihood : float
        Sum of per-trial action log-probabilities.
    steps : tuple[ReplayStep, ...]
        Trial-wise replay records in chronological order.
    """

    total_log_likelihood: float
    steps: tuple[ReplayStep, ...]


def replay_episode(trace: EpisodeTrace, model: AgentModel) -> ReplayResult:
    """Replay a canonical episode trace and evaluate model action likelihood.

    Parameters
    ----------
    trace : EpisodeTrace
        Canonical event trace. Must follow the expected trial phase sequence.
    model : AgentModel
        Model instance used for replay. A fresh episode is started internally.

    Returns
    -------
    ReplayResult
        Per-trial and aggregate action log-likelihood records.

    Raises
    ------
    ValueError
        If trace structure or payload fields are invalid.

    Notes
    -----
    Replay consumes observations/outcomes from the trace and calls
    ``model.update`` to maintain state consistency with simulation.
    """

    validate_trace(trace)
    grouped = group_events_by_trial(trace)

    model.start_episode()
    total_log_likelihood = 0.0
    steps: list[ReplayStep] = []

    for trial_index in sorted(grouped):
        observation_event, decision_event, outcome_event, _update_event = grouped[trial_index]

        observation = _payload_get(observation_event.payload, "observation", trial_index)
        available_actions = tuple(_payload_get(observation_event.payload, "available_actions", trial_index))
        context = DecisionContext(trial_index=trial_index, available_actions=available_actions)

        action = _payload_get(decision_event.payload, "action", trial_index)
        if action not in available_actions:
            raise ValueError(
                f"trace action {action!r} is not available in trial {trial_index}: "
                f"{available_actions!r}"
            )

        raw_distribution = model.action_distribution(observation, context=context)
        distribution = normalize_distribution(raw_distribution, available_actions)

        probability = float(distribution[action])
        log_probability = float(np.log(probability)) if probability > 0 else float(-np.inf)
        total_log_likelihood += log_probability

        outcome = _payload_get(outcome_event.payload, "outcome", trial_index)
        model.update(observation, action, outcome, context=context)

        steps.append(
            ReplayStep(
                trial_index=trial_index,
                action=action,
                probability=probability,
                log_probability=log_probability,
            )
        )

    return ReplayResult(total_log_likelihood=total_log_likelihood, steps=tuple(steps))


def _payload_get(payload: Mapping[str, Any] | Any, key: str, trial_index: int) -> Any:
    """Read a payload field and raise a clear error if absent."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"trial {trial_index} payload must be a mapping")
    if key not in payload:
        raise ValueError(f"trial {trial_index} payload is missing required key {key!r}")
    return payload[key]
