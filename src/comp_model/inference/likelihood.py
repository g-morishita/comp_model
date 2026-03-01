"""Likelihood-program interfaces and baseline implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from comp_model.core.contracts import AgentModel
from comp_model.core.events import EpisodeTrace
from comp_model.runtime.replay import ReplayResult, replay_episode, replay_trial_program


@runtime_checkable
class LikelihoodProgram(Protocol):
    """Protocol for replay-based likelihood evaluators."""

    def evaluate(self, trace: EpisodeTrace, model: AgentModel) -> ReplayResult:
        """Evaluate model likelihood on an episode trace."""


class ActionReplayLikelihood:
    """Likelihood program based on action probabilities during replay.

    The implementation delegates to :func:`comp_model.runtime.replay.replay_episode`.
    """

    def evaluate(self, trace: EpisodeTrace, model: AgentModel) -> ReplayResult:
        """Evaluate action log-likelihood by replaying the trace."""

        return replay_episode(trace=trace, model=model)

    def evaluate_with_models(
        self,
        trace: EpisodeTrace,
        models: Mapping[str, AgentModel],
    ) -> ReplayResult:
        """Evaluate action log-likelihood for multi-actor traces.

        Parameters
        ----------
        trace : EpisodeTrace
            Canonical event trace.
        models : Mapping[str, AgentModel]
            Actor-ID to model mapping used during replay.

        Returns
        -------
        ReplayResult
            Replay likelihood result.
        """

        return replay_trial_program(trace=trace, models=models)


class ActorSubsetReplayLikelihood:
    """Replay likelihood for multi-actor traces with actor-subset scoring.

    Parameters
    ----------
    fitted_actor_id : str, optional
        Actor ID assigned to the model passed into :meth:`evaluate`.
    scored_actor_ids : Sequence[str] | None, optional
        Actor IDs included in likelihood scoring. If ``None``, all replayed
        actor decisions are scored.
    fixed_actor_models : Mapping[str, AgentModel] | None, optional
        Optional fixed models for non-fitted actors.
    auto_fill_unmodeled_actors : bool, optional
        If ``True``, any actor missing from ``fixed_actor_models`` is replayed
        with a trace-observed deterministic policy (probability 1.0 on the
        observed action) so the fitted actor can still be updated from their
        outcomes.

    Notes
    -----
    This likelihood is designed for fitting one actor (for example, a subject)
    on traces that include other actor decisions (for example, demonstrators).
    """

    def __init__(
        self,
        *,
        fitted_actor_id: str = "subject",
        scored_actor_ids: Sequence[str] | None = ("subject",),
        fixed_actor_models: Mapping[str, AgentModel] | None = None,
        auto_fill_unmodeled_actors: bool = True,
    ) -> None:
        self._fitted_actor_id = str(fitted_actor_id)
        self._scored_actor_ids = (
            None if scored_actor_ids is None else tuple(str(value) for value in scored_actor_ids)
        )
        self._fixed_actor_models = dict(fixed_actor_models) if fixed_actor_models is not None else {}
        self._auto_fill_unmodeled_actors = bool(auto_fill_unmodeled_actors)

    def evaluate(self, trace: EpisodeTrace, model: AgentModel) -> ReplayResult:
        """Evaluate trace likelihood while scoring only selected actors."""

        models: dict[str, AgentModel] = dict(self._fixed_actor_models)
        models[self._fitted_actor_id] = model

        actor_ids_in_trace = _actor_ids_from_trace(trace)
        if self._auto_fill_unmodeled_actors:
            observed_actions = _trace_observed_actions(trace)
            for actor_id in actor_ids_in_trace:
                if actor_id in models:
                    continue
                models[actor_id] = _TraceObservedActionModel(
                    actor_id=actor_id,
                    observed_actions=observed_actions,
                )

        replay = replay_trial_program(trace=trace, models=models)
        if self._scored_actor_ids is None:
            return replay

        scored_actor_ids = set(self._scored_actor_ids)
        filtered_steps = tuple(
            step
            for step in replay.steps
            if step.actor_id in scored_actor_ids
        )
        return ReplayResult(
            total_log_likelihood=float(sum(step.log_probability for step in filtered_steps)),
            steps=filtered_steps,
        )


@dataclass(slots=True)
class _TraceObservedActionModel:
    """Internal deterministic actor model that reproduces trace actions."""

    actor_id: str
    observed_actions: dict[tuple[int, str, int], Any]

    def start_episode(self) -> None:
        """No-op reset."""

    def action_distribution(self, observation: Any, *, context) -> dict[Any, float]:
        """Return one-hot distribution over observed action in trace."""

        del observation
        key = (int(context.trial_index), str(context.actor_id), int(context.decision_index))
        if key not in self.observed_actions:
            raise ValueError(
                f"missing observed action for actor {context.actor_id!r} "
                f"at trial={context.trial_index}, decision_index={context.decision_index}"
            )

        observed_action = self.observed_actions[key]
        out: dict[Any, float] = {action: 0.0 for action in context.available_actions}
        if observed_action not in out:
            raise ValueError(
                f"observed action {observed_action!r} not in available_actions "
                f"{context.available_actions!r} for actor {context.actor_id!r}"
            )
        out[observed_action] = 1.0
        return out

    def update(self, observation: Any, action: Any, outcome: Any, *, context) -> None:
        """No-op update for trace-observed auxiliary actors."""

        del observation, action, outcome, context


def _trace_observed_actions(trace: EpisodeTrace) -> dict[tuple[int, str, int], Any]:
    """Collect observed actor decisions from trace decision events."""

    actions: dict[tuple[int, str, int], Any] = {}
    per_trial_counts: dict[int, int] = {}
    for event in trace.events:
        if event.phase.value != "decision":
            continue
        actor_id = str(event.payload.get("actor_id", "subject"))
        trial_index = int(event.trial_index)
        decision_index_raw = event.payload.get("decision_index")
        if decision_index_raw is None:
            default_index = per_trial_counts.get(trial_index, 0)
            decision_index = int(default_index)
        else:
            decision_index = int(decision_index_raw)
        per_trial_counts[trial_index] = per_trial_counts.get(trial_index, 0) + 1
        key = (trial_index, actor_id, decision_index)
        actions[key] = event.payload.get("action")
    return actions


def _actor_ids_from_trace(trace: EpisodeTrace) -> tuple[str, ...]:
    """Return sorted actor IDs present in trace decision events."""

    actor_ids = {
        str(event.payload.get("actor_id", "subject"))
        for event in trace.events
        if event.phase.value == "decision"
    }
    return tuple(sorted(actor_ids))
