"""Random baseline agent.

This model is useful for smoke tests because it has deterministic semantics and
no internal learning state.
"""

from __future__ import annotations

from typing import Any

from comp_model_v2.core.contracts import DecisionContext


class RandomAgent:
    """Uniform-random action model with no-op update.

    Methods follow the :class:`comp_model_v2.core.contracts.AgentModel` protocol.
    """

    def start_episode(self) -> None:
        """Reset episode state.

        Notes
        -----
        This model has no persistent episode state.
        """

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return a uniform distribution over available actions.

        Parameters
        ----------
        observation : Any
            Unused observation object.
        context : DecisionContext[Any]
            Per-trial context containing available actions.

        Returns
        -------
        dict[Any, float]
            Uniform action probabilities.
        """

        probability = 1.0 / float(len(context.available_actions))
        return {action: probability for action in context.available_actions}

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """No-op update.

        Parameters
        ----------
        observation : Any
            Unused.
        action : Any
            Unused.
        outcome : Any
            Unused.
        context : DecisionContext[Any]
            Unused.
        """

