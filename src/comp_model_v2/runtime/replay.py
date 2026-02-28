"""Replay engines for model likelihood evaluation on canonical traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from comp_model_v2.core.contracts import AgentModel, DecisionContext
from comp_model_v2.core.events import (
    EpisodeTrace,
    group_events_by_trial,
    split_trial_events_into_phase_blocks,
    validate_trace,
)
from comp_model_v2.runtime.probabilities import normalize_distribution


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """Per-decision replay likelihood record.

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
    actor_id : str, optional
        Actor that produced the decision.
    learner_id : str, optional
        Actor model that received update callback.
    decision_index : int, optional
        Decision-step index within trial.
    node_id : str | None, optional
        Optional semantic decision-node identifier.
    """

    trial_index: int
    action: Any
    probability: float
    log_probability: float
    actor_id: str = "subject"
    learner_id: str = "subject"
    decision_index: int = 0
    node_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Aggregate replay likelihood output.

    Parameters
    ----------
    total_log_likelihood : float
        Sum of per-decision action log-probabilities.
    steps : tuple[ReplayStep, ...]
        Decision-wise replay records in chronological order.
    """

    total_log_likelihood: float
    steps: tuple[ReplayStep, ...]


def replay_trial_program(
    *,
    trace: EpisodeTrace,
    models: Mapping[str, AgentModel],
) -> ReplayResult:
    """Replay a canonical trace for multi-phase trial programs.

    Parameters
    ----------
    trace : EpisodeTrace
        Canonical event trace with one-or-more phase blocks per trial.
    models : Mapping[str, AgentModel]
        Actor-ID to model mapping used for decision likelihood and updates.

    Returns
    -------
    ReplayResult
        Per-decision and aggregate action log-likelihood records.

    Raises
    ------
    ValueError
        If trace structure or payload fields are invalid.
    """

    if not models:
        raise ValueError("models mapping must include at least one actor model")

    validate_trace(trace)
    grouped = group_events_by_trial(trace)

    for model in _unique_models(models):
        model.start_episode()

    total_log_likelihood = 0.0
    steps: list[ReplayStep] = []

    for trial_index in sorted(grouped):
        phase_blocks = split_trial_events_into_phase_blocks(
            grouped[trial_index],
            trial_index=trial_index,
        )

        for block_index, block in enumerate(phase_blocks):
            observation_event, decision_event, outcome_event, update_event = block

            observation = _payload_get(observation_event.payload, "observation", trial_index)
            available_actions = tuple(_payload_get(observation_event.payload, "available_actions", trial_index))

            decision_payload = _payload_mapping(decision_event.payload, trial_index)
            action = _payload_get(decision_payload, "action", trial_index)
            actor_id = str(decision_payload.get("actor_id", "subject"))
            decision_index = int(decision_payload.get("decision_index", block_index))
            node_value = decision_payload.get("node_id")
            node_id = str(node_value) if node_value is not None else None

            context = DecisionContext(
                trial_index=trial_index,
                available_actions=available_actions,
                actor_id=actor_id,
                decision_index=decision_index,
                decision_label=node_id,
            )

            if action not in available_actions:
                raise ValueError(
                    f"trace action {action!r} is not available in trial {trial_index}: "
                    f"{available_actions!r}"
                )

            actor_model = _get_actor_model(models=models, actor_id=actor_id)
            raw_distribution = actor_model.action_distribution(observation, context=context)
            distribution = normalize_distribution(raw_distribution, available_actions)

            probability = float(distribution[action])
            log_probability = float(np.log(probability)) if probability > 0 else float(-np.inf)
            total_log_likelihood += log_probability

            outcome = _payload_get(outcome_event.payload, "outcome", trial_index)
            update_payload = _payload_mapping(update_event.payload, trial_index)
            learner_id = str(update_payload.get("learner_id", actor_id))
            learner_model = _get_actor_model(models=models, actor_id=learner_id)
            learner_context = context if learner_id == actor_id else replace(context, actor_id=learner_id)
            learner_model.update(observation, action, outcome, context=learner_context)

            steps.append(
                ReplayStep(
                    trial_index=trial_index,
                    action=action,
                    probability=probability,
                    log_probability=log_probability,
                    actor_id=actor_id,
                    learner_id=learner_id,
                    decision_index=decision_index,
                    node_id=node_id,
                )
            )

    return ReplayResult(total_log_likelihood=total_log_likelihood, steps=tuple(steps))


def replay_episode(trace: EpisodeTrace, model: AgentModel) -> ReplayResult:
    """Replay a backward-compatible single-actor episode trace.

    Parameters
    ----------
    trace : EpisodeTrace
        Canonical event trace.
    model : AgentModel
        Subject model used for replay.

    Returns
    -------
    ReplayResult
        Replay likelihood output.

    Notes
    -----
    This function is a compatibility wrapper around
    :func:`replay_trial_program` with a single ``"subject"`` actor.
    """

    return replay_trial_program(trace=trace, models={"subject": model})


def _payload_get(payload: Mapping[str, Any] | Any, key: str, trial_index: int) -> Any:
    """Read a payload field and raise a clear error if absent."""

    mapping = _payload_mapping(payload, trial_index)
    if key not in mapping:
        raise ValueError(f"trial {trial_index} payload is missing required key {key!r}")
    return mapping[key]


def _payload_mapping(payload: Mapping[str, Any] | Any, trial_index: int) -> Mapping[str, Any]:
    """Validate payload is mapping-like and return it."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"trial {trial_index} payload must be a mapping")
    return payload


def _get_actor_model(models: Mapping[str, AgentModel], actor_id: str) -> AgentModel:
    """Resolve actor model or raise a clear error."""

    model = models.get(actor_id)
    if model is None:
        available = ", ".join(sorted(models))
        raise ValueError(f"unknown actor_id {actor_id!r}; available actors: {available}")
    return model


def _unique_models(models: Mapping[str, AgentModel]) -> tuple[AgentModel, ...]:
    """Return unique model instances preserving insertion order."""

    unique: list[AgentModel] = []
    seen: set[int] = set()
    for model in models.values():
        key = id(model)
        if key in seen:
            continue
        seen.add(key)
        unique.append(model)
    return tuple(unique)
