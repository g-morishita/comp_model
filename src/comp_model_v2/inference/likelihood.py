"""Likelihood-program interfaces and baseline implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from comp_model_v2.core.contracts import AgentModel
from comp_model_v2.core.events import EpisodeTrace
from comp_model_v2.runtime.replay import ReplayResult, replay_episode


@runtime_checkable
class LikelihoodProgram(Protocol):
    """Protocol for replay-based likelihood evaluators."""

    def evaluate(self, trace: EpisodeTrace, model: AgentModel) -> ReplayResult:
        """Evaluate model likelihood on an episode trace."""


class ActionReplayLikelihood:
    """Likelihood program based on action probabilities during replay.

    The implementation delegates to :func:`comp_model_v2.runtime.replay.replay_episode`.
    """

    def evaluate(self, trace: EpisodeTrace, model: AgentModel) -> ReplayResult:
        """Evaluate action log-likelihood by replaying the trace."""

        return replay_episode(trace=trace, model=model)
