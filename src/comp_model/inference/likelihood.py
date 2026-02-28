"""Likelihood-program interfaces and baseline implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

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
