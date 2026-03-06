"""Replay engines for model likelihood evaluation on canonical traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from comp_model.core.contracts import AgentModel, DecisionContext
from comp_model.core.events import EpisodeTrace, EventPhase, group_events_by_trial, validate_trace
from comp_model.runtime.probabilities import normalize_distribution


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
    learner_ids : tuple[str, ...], optional
        Learners updated for this decision in chronological order.
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
    learner_ids: tuple[str, ...] = ("subject",)
    decision_index: int = 0
    node_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Aggregate replay likelihood output."""

    total_log_likelihood: float
    steps: tuple[ReplayStep, ...]


@dataclass(slots=True)
class _ReplayNodeState:
    """Mutable state tracked while replaying one node."""

    observation: Any
    available_actions: tuple[Any, ...]
    actor_id: str
    node_id: str
    decision_index: int | None = None
    action: Any | None = None
    outcome: Any = None
    probability: float | None = None
    log_probability: float | None = None
    learner_ids: list[str] | None = None


def replay_trial_program(
    *,
    trace: EpisodeTrace,
    models: Mapping[str, AgentModel],
) -> ReplayResult:
    """Replay a canonical trace for multi-phase trial programs."""

    if not models:
        raise ValueError("models mapping must include at least one actor model")

    validate_trace(trace)
    grouped = group_events_by_trial(trace)

    for model in _unique_models(models):
        model.start_episode()

    total_log_likelihood = 0.0
    steps: list[ReplayStep] = []

    for trial_index in sorted(grouped):
        node_states: dict[str, _ReplayNodeState] = {}
        decision_order: list[str] = []

        for event in grouped[trial_index]:
            payload = _payload_mapping(event.payload, trial_index)
            node_id = str(payload.get("node_id"))

            if event.phase is EventPhase.OBSERVATION:
                observation = _payload_get(payload, "observation", trial_index)
                available_actions = tuple(_payload_get(payload, "available_actions", trial_index))
                actor_id = str(payload.get("actor_id", "subject"))
                node_states[node_id] = _ReplayNodeState(
                    observation=observation,
                    available_actions=available_actions,
                    actor_id=actor_id,
                    node_id=node_id,
                )
                continue

            state = node_states.get(node_id)
            if state is None:
                raise ValueError(
                    f"trial {trial_index}: event {event.phase.value!r} for node {node_id!r} "
                    "requires a prior observation"
                )

            if event.phase is EventPhase.DECISION:
                action = _payload_get(payload, "action", trial_index)
                actor_id = str(payload.get("actor_id", state.actor_id))
                decision_index = int(payload.get("decision_index", len(decision_order)))
                context = DecisionContext(
                    trial_index=trial_index,
                    available_actions=state.available_actions,
                    actor_id=actor_id,
                    decision_index=decision_index,
                    decision_label=node_id,
                )
                if action not in state.available_actions:
                    raise ValueError(
                        f"trace action {action!r} is not available in trial {trial_index}: "
                        f"{state.available_actions!r}"
                    )

                actor_model = _get_actor_model(models=models, actor_id=actor_id)
                raw_distribution = actor_model.action_distribution(state.observation, context=context)
                distribution = normalize_distribution(raw_distribution, state.available_actions)
                probability = float(distribution[action])
                log_probability = float(np.log(probability)) if probability > 0 else float(-np.inf)
                total_log_likelihood += log_probability

                state.actor_id = actor_id
                state.decision_index = decision_index
                state.action = action
                state.probability = probability
                state.log_probability = log_probability
                state.learner_ids = []
                decision_order.append(node_id)
                continue

            if event.phase is EventPhase.OUTCOME:
                state.outcome = payload.get("outcome")
                continue

            if event.phase is EventPhase.UPDATE:
                if state.action is None or state.decision_index is None:
                    raise ValueError(
                        f"trial {trial_index}: update for node {node_id!r} requires a prior decision"
                    )

                context = DecisionContext(
                    trial_index=trial_index,
                    available_actions=state.available_actions,
                    actor_id=state.actor_id,
                    decision_index=state.decision_index,
                    decision_label=node_id,
                )
                learner_id = str(payload.get("learner_id", state.actor_id))
                learner_model = _get_actor_model(models=models, actor_id=learner_id)
                learner_context = context if learner_id == state.actor_id else replace(context, actor_id=learner_id)
                learner_model.update(
                    state.observation,
                    state.action,
                    state.outcome,
                    context=learner_context,
                )
                assert state.learner_ids is not None
                state.learner_ids.append(learner_id)
                continue

            raise ValueError(f"trial {trial_index}: unsupported event phase {event.phase!r}")

        for node_id in decision_order:
            state = node_states[node_id]
            if state.action is None or state.probability is None or state.log_probability is None or state.decision_index is None:
                raise ValueError(f"trial {trial_index}: node {node_id!r} is missing replay decision state")
            learner_ids = tuple(state.learner_ids) if state.learner_ids is not None else ()
            steps.append(
                ReplayStep(
                    trial_index=trial_index,
                    action=state.action,
                    probability=state.probability,
                    log_probability=state.log_probability,
                    actor_id=state.actor_id,
                    learner_ids=learner_ids,
                    decision_index=state.decision_index,
                    node_id=node_id,
                )
            )

    return ReplayResult(total_log_likelihood=total_log_likelihood, steps=tuple(steps))


def replay_episode(trace: EpisodeTrace, model: AgentModel) -> ReplayResult:
    """Replay a single-actor episode trace."""

    return replay_trial_program(trace=trace, models={"subject": model})


def _payload_get(payload: Mapping[str, Any], key: str, trial_index: int) -> Any:
    """Read a payload field and raise a clear error if absent."""

    if key not in payload:
        raise ValueError(f"trial {trial_index} payload is missing required key {key!r}")
    return payload[key]


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
