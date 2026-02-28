"""Reinforcement-learning demonstrator implementation."""

from __future__ import annotations

from typing import Any, Callable

from comp_model.core.contracts import DecisionContext
from comp_model.models.q_learning import QLearningAgent, QLearningConfig
from comp_model.plugins import ComponentManifest


class RLDemonstrator:
    """Q-learning demonstrator for social task simulations.

    Parameters
    ----------
    alpha : float, optional
        Learning rate in ``[0, 1]``.
    beta : float, optional
        Inverse-temperature for softmax policy.
    initial_value : float, optional
        Initial action value for unseen actions.
    reward_getter : Callable[[Any], float] | None, optional
        Optional custom outcome-to-reward extractor.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.2,
        beta: float = 3.0,
        initial_value: float = 0.0,
        reward_getter: Callable[[Any], float] | None = None,
    ) -> None:
        self._agent = QLearningAgent(
            config=QLearningConfig(alpha=alpha, beta=beta, initial_value=initial_value),
            reward_getter=reward_getter,
        )

    def start_episode(self) -> None:
        """Reset internal learner state for a new episode."""

        self._agent.start_episode()

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return softmax policy over available actions."""

        return self._agent.action_distribution(observation, context=context)

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """Update learner state from observed outcome."""

        self._agent.update(observation, action, outcome, context=context)

    def q_values_snapshot(self) -> dict[Any, float]:
        """Return copy of current Q-values for diagnostics/tests."""

        return self._agent.q_values_snapshot()


def create_rl_demonstrator(
    *,
    alpha: float = 0.2,
    beta: float = 3.0,
    initial_value: float = 0.0,
) -> RLDemonstrator:
    """Factory used by plugin discovery."""

    return RLDemonstrator(alpha=alpha, beta=beta, initial_value=initial_value)


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="demonstrator",
        component_id="rl_demonstrator",
        factory=create_rl_demonstrator,
        description="Q-learning demonstrator for social simulations",
    )
]
